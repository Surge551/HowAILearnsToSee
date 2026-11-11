(() => {
const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const storageGet = (key) => { try { return localStorage.getItem(key); } catch { return null; } };
const storageSet = (key, value) => { try { localStorage.setItem(key, value); return true; } catch { return false; } };
const storageRemove = (key) => { try { localStorage.removeItem(key); } catch { /* storage is optional */ } };
const formatNumber = (number) => new Intl.NumberFormat('en-US').format(number);

// The training-size slider is logarithmic so 10, 100 and 10,000 are all easy to reach.
const MIN_TRAIN = 10;
const MAX_TRAIN = 55000;
const SLIDER_STEPS = 1000;

const state = {
  dataset: null,
  trainingOrder: null,
  model: null,
  // Screen 2 always uses this never-trained copy, so it keeps showing a random
  // guess even after the main model has been trained.
  untrainedModel: null,
  screen: 'data',
  trainCount: 10000,
  trained: false,
  accuracy: null,
  examIndexes: [],
  networkExample: 0,
  beforePixels: new Float32Array(784),
  playPixels: new Float32Array(784),
  drawings: {},
  chart: null,
  isTraining: false,
  isEvaluating: false,
  isPaused: false,
  initializationSeed: 2026,
  trainNetwork: null,
  playNetwork: null,
};

function setStatus(status) {
  $('#model-status').dataset.state = status;
  const text = {
    loading: 'Loading data…',
    ready: 'Not trained',
    training: 'Training',
    paused: 'Paused',
    evaluating: 'Final test…',
    trained: Number.isFinite(state.accuracy) ? `Trained · ${state.accuracy.toFixed(1)}% on test images` : 'Trained',
    error: 'Could not load the data',
  };
  $('#model-status-text').textContent = text[status] ?? text.ready;
}

function updateStatus() {
  if (state.isEvaluating) setStatus('evaluating');
  else if (state.isTraining) setStatus(state.isPaused ? 'paused' : 'training');
  else setStatus(state.trained ? 'trained' : 'ready');
}

function showToast(message, isError = false, duration = 3600) {
  const toast = document.createElement('div');
  toast.className = `toast${isError ? ' error' : ''}`;
  toast.textContent = message;
  $('#toast-region').append(toast);
  setTimeout(() => toast.remove(), duration);
}

function showScreen(name) {
  state.screen = name;
  $$('[data-screen-panel]').forEach((section) => {
    const active = section.dataset.screenPanel === name;
    section.hidden = !active;
    section.classList.toggle('is-active', active);
  });
  $$('.step-link').forEach((button) => button.classList.toggle('is-active', button.dataset.screen === name));
  if (name === 'training' && state.model) { renderExamBoard(); renderTrainNetwork(); }
  if (name === 'before') renderDrawingPrediction('before', state.beforePixels);
  if (name === 'play') renderDrawingPrediction('play', state.playPixels);
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function imageBytes(split, index) {
  const images = split === 'test' ? state.dataset.testImages : split === 'validation' ? state.dataset.validationImages : state.dataset.trainImages;
  return images.subarray(index * 784, index * 784 + 784);
}

const normalized = (bytes) => Float32Array.from(bytes, (v) => v / 255);

function paint28(canvas, pixels) {
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(28, 28);
  for (let i = 0; i < 784; i++) imageData.data.set([pixels[i], pixels[i], pixels[i], 255], i * 4);
  ctx.putImageData(imageData, 0, 0);
}

function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

// ---------- Screen 1: data ----------

function sliderToCount(position) {
  const raw = MIN_TRAIN * (MAX_TRAIN / MIN_TRAIN) ** (position / SLIDER_STEPS);
  const step = raw < 100 ? 5 : raw < 1000 ? 10 : raw < 10000 ? 100 : 500;
  return Math.max(MIN_TRAIN, Math.min(MAX_TRAIN, Math.round(raw / step) * step));
}

function countToSlider(count) {
  return Math.round(SLIDER_STEPS * Math.log(count / MIN_TRAIN) / Math.log(MAX_TRAIN / MIN_TRAIN));
}

function setTrainCount(count, { fromSlider = false } = {}) {
  state.trainCount = Math.max(MIN_TRAIN, Math.min(MAX_TRAIN, Math.round(count) || MIN_TRAIN));
  if (!fromSlider) $('#data-size').value = String(countToSlider(state.trainCount));
  $('#data-size-number').value = String(state.trainCount);
  $('#train-count-chip').textContent = formatNumber(state.trainCount);
  const percent = (Number($('#data-size').value) / SLIDER_STEPS) * 100;
  $('#data-size').style.setProperty('--fill', `${percent}%`);
}

function placeSizeTicks() {
  $$('.size-ticks span').forEach((tick) => {
    tick.style.left = `${(countToSlider(Number(tick.dataset.value)) / SLIDER_STEPS) * 100}%`;
    tick.addEventListener('click', () => setTrainCount(Number(tick.dataset.value)));
  });
}

function renderSamples() {
  if (!state.dataset) return;
  const grid = $('#sample-grid');
  grid.replaceChildren();
  for (let n = 0; n < 30; n++) {
    const index = Math.floor(Math.random() * state.dataset.trainCount);
    const label = state.dataset.trainLabels[index];
    const tile = document.createElement('button');
    tile.type = 'button';
    tile.className = 'sample-tile';
    const canvas = document.createElement('canvas');
    canvas.width = 28; canvas.height = 28;
    paint28(canvas, imageBytes('train', index));
    const digit = document.createElement('strong');
    digit.textContent = label;
    tile.append(canvas, digit);
    tile.addEventListener('click', () => openPixelDialog(index, label));
    grid.append(tile);
  }
}

function openPixelDialog(index, label) {
  $('#dialog-label').textContent = label;
  const matrix = $('#pixel-matrix');
  matrix.replaceChildren();
  imageBytes('train', index).forEach((value) => {
    const cell = document.createElement('span');
    cell.className = 'pixel-cell';
    cell.textContent = value;
    cell.style.backgroundColor = `rgb(${value} ${value} ${value})`;
    cell.style.color = value > 150 ? '#101714' : '#ffffff';
    matrix.append(cell);
  });
  $('#pixel-dialog').showModal();
}

// ---------- Drawing screens ----------

function createConfidenceRows(container) {
  container.replaceChildren();
  for (let digit = 0; digit < 10; digit++) {
    const row = document.createElement('div');
    row.className = 'confidence-row';
    row.dataset.digit = digit;
    row.innerHTML = `<span class="confidence-digit">${digit}</span><div class="confidence-track"><div class="confidence-fill"></div></div><span class="confidence-value">0%</span>`;
    container.append(row);
  }
}

function updateConfidenceBars(container, predictions) {
  const best = argmax(predictions);
  [...container.children].forEach((row) => {
    const digit = Number(row.dataset.digit);
    const value = Math.max(0, Math.min(1, predictions[digit] || 0));
    row.classList.toggle('is-high', digit === best);
    row.querySelector('.confidence-fill').style.width = `${value * 100}%`;
    row.querySelector('.confidence-value').textContent = `${(value * 100).toFixed(1)}%`;
  });
  return best;
}

const isBlank = (pixels) => !pixels.some((value) => value > 0);

function renderDrawingPrediction(kind, pixels, { animate = false } = {}) {
  const model = kind === 'before' ? state.untrainedModel : state.model;
  if (!model?.model) return;
  const bars = kind === 'before' ? $('#before-bars') : $('#play-bars');
  const prediction = kind === 'before' ? $('#before-prediction') : $('#play-prediction');
  if (kind === 'play') {
    renderProcessed($('#processed-preview'), pixels);
    $('#untrained-warning').hidden = state.trained;
  }
  if (isBlank(pixels)) {
    updateConfidenceBars(bars, new Float32Array(10));
    prediction.textContent = '—';
    if (kind === 'play') state.playNetwork?.update(null);
    return;
  }
  const inspection = model.inspect(pixels);
  prediction.textContent = String(updateConfidenceBars(bars, inspection.output));
  if (kind === 'play') state.playNetwork?.update({ pixels, ...inspection }, { animate });
}

// ---------- Screen 3: training ----------

function chooseExamExamples() {
  const found = Array(10).fill(0);
  const indexes = [];
  for (let i = 0; i < state.dataset.validationCount && indexes.length < 20; i++) {
    const digit = state.dataset.validationLabels[i];
    if (found[digit] < 2) { found[digit]++; indexes.push(i); }
  }
  state.examIndexes = indexes;
}

function renderExamBoard() {
  if (!state.model?.model || state.isEvaluating) return;
  const grid = $('#exam-grid');
  grid.replaceChildren();
  let correct = 0;
  state.examIndexes.forEach((index, position) => {
    const actual = state.dataset.validationLabels[index];
    const guessed = argmax(state.model.predict(normalized(imageBytes('validation', index))));
    const isCorrect = guessed === actual;
    if (isCorrect) correct++;
    const tile = document.createElement('button');
    tile.type = 'button';
    tile.className = `exam-tile ${isCorrect ? 'correct' : 'wrong'}${position === state.networkExample ? ' is-selected' : ''}`;
    tile.title = `True digit ${actual} · network says ${guessed}`;
    const canvas = document.createElement('canvas');
    canvas.width = 28; canvas.height = 28;
    paint28(canvas, imageBytes('validation', index));
    const guess = document.createElement('strong');
    guess.textContent = `${guessed}`;
    tile.append(canvas, guess);
    tile.addEventListener('click', () => selectNetworkExample(position, true));
    grid.append(tile);
  });
  $('#exam-score').textContent = `${correct} / ${state.examIndexes.length}`;
}

function selectNetworkExample(position, animate = false) {
  const count = state.examIndexes.length;
  state.networkExample = ((position % count) + count) % count;
  $$('#exam-grid .exam-tile').forEach((tile, i) => tile.classList.toggle('is-selected', i === state.networkExample));
  renderTrainNetwork({ animate });
}

function renderTrainNetwork({ animate = false } = {}) {
  if (!state.model?.model || !state.examIndexes.length || !state.trainNetwork) return;
  const index = state.examIndexes[state.networkExample];
  const truth = state.dataset.validationLabels[index];
  const pixels = normalized(imageBytes('validation', index));
  const inspection = state.model.inspect(pixels);
  const guess = argmax(inspection.output);
  state.trainNetwork.update({ pixels, truth, ...inspection }, { animate });
  const label = $('#net-example');
  label.textContent = `Image ${state.networkExample + 1} of ${state.examIndexes.length} · guess ${guess} ${guess === truth ? '✓' : '✗'}`;
  label.classList.toggle('is-wrong', guess !== truth);
}

function makeChart() {
  if (!globalThis.Chart) return;
  state.chart = new Chart($('#learning-chart').getContext('2d'), {
    type: 'line',
    data: { labels: [], datasets: [
      { label: 'accuracy', data: [], yAxisID: 'y', borderColor: '#187753', backgroundColor: '#187753', borderWidth: 2.5, pointRadius: 4, tension: .28 },
      { label: 'loss', data: [], yAxisID: 'y1', borderColor: '#bd433a', backgroundColor: '#bd433a', borderWidth: 2.5, pointRadius: 4, tension: .28 }
    ] },
    options: {
      responsive: true, maintainAspectRatio: false, animation: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { display: false }, tooltip: { backgroundColor: '#182421', padding: 10 } },
      scales: {
        x: { grid: { display: false }, title: { display: true, text: 'epoch', color: '#66736e' }, ticks: { color: '#66736e' } },
        y: { beginAtZero: true, max: 100, position: 'left', grid: { color: '#edf0ed' }, ticks: { color: '#66736e', callback: (v) => `${v}%` } },
        y1: { beginAtZero: true, position: 'right', grid: { drawOnChartArea: false }, ticks: { color: '#66736e' } }
      }
    }
  });
}

function clearChart() {
  if (!state.chart) return;
  state.chart.data.labels = [];
  state.chart.data.datasets.forEach((dataset) => { dataset.data = []; });
  state.chart.update('none');
}

function resetReadouts(epochs) {
  $('#train-progress').style.width = '0%';
  $('#epoch-value').innerHTML = `0 <small>of</small> ${epochs}`;
  $('#accuracy-label').textContent = 'ACCURACY';
  $('#accuracy-caption').textContent = 'on images it never trained on';
  $('#accuracy-value').innerHTML = '—<small>%</small>';
  $('#loss-value').textContent = '—';
  clearChart();
}

function buildTrainingOrder(count) {
  const indexes = new Uint32Array(count);
  for (let i = 0; i < count; i++) indexes[i] = i;
  // A fixed seed keeps the sample sizes nested and reproducible, so a run
  // with 100 examples uses the first 100 from the 1,000 and 10,000 sets.
  let seed = 0x51A7E;
  const random = () => {
    seed = (seed + 0x6D2B79F5) | 0;
    let value = seed;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
  for (let i = count - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1));
    [indexes[i], indexes[j]] = [indexes[j], indexes[i]];
  }
  return indexes;
}

function allocateSeed() {
  const stored = Number(storageGet('wil-next-seed'));
  const seed = Number.isInteger(stored) && stored >= 0 && stored <= 0xffffffff ? stored : 2026;
  storageSet('wil-next-seed', String((seed + 1) >>> 0));
  return seed;
}

function selectTrainingSubset(count) {
  const images = new Uint8Array(count * 784);
  const labels = new Uint8Array(count);
  for (let i = 0; i < count; i++) {
    const sourceIndex = state.trainingOrder[i];
    images.set(imageBytes('train', sourceIndex), i * 784);
    labels[i] = state.dataset.trainLabels[sourceIndex];
  }
  return { images, labels };
}

function setTrainingButtons(running) {
  $('#start-training').disabled = running;
  $('#pause-training').disabled = !running;
  $('#pause-training').hidden = false;
  $('#resume-training').hidden = true;
  $('#reset-training').disabled = running && state.isEvaluating;
}

async function startTraining() {
  if (state.isTraining || state.isEvaluating || !state.dataset || !state.model) return;
  const epochs = Math.max(1, Math.min(20, Number($('#epochs').value) || 10));
  $('#epochs').value = epochs;
  const trainingData = selectTrainingSubset(state.trainCount);
  Object.assign(state, { isTraining: true, isPaused: false, trained: false, accuracy: null });
  setTrainingButtons(true);
  resetReadouts(epochs);
  $('#batch-readout').textContent = 'Training…';
  updateStatus();

  try {
    const summary = await state.model.train({
      trainImages: trainingData.images,
      trainLabels: trainingData.labels,
      validationImages: state.dataset.validationImages,
      validationLabels: state.dataset.validationLabels,
      trainCount: state.trainCount,
      seed: allocateSeed(),
      epochs,
      batchSize: 64,
      speedMs: Number($('#training-speed').value) || 0,
      onBatch: (result) => {
        $('#train-progress').style.width = `${result.progress}%`;
        $('#epoch-value').innerHTML = `${result.epoch} <small>of</small> ${epochs}`;
        $('#batch-readout').textContent = `batch ${formatNumber(result.batch)} / ${formatNumber(result.batches)}`;
        if (result.batch % 10 === 0 || result.batch === result.batches) { renderExamBoard(); renderTrainNetwork(); }
      },
      onEpoch: (result) => {
        state.chart?.data.labels.push(String(result.epoch));
        state.chart?.data.datasets[0].data.push(result.validationAccuracy * 100);
        state.chart?.data.datasets[1].data.push(result.validationLoss);
        state.chart?.update('none');
        $('#accuracy-value').innerHTML = `${(result.validationAccuracy * 100).toFixed(1)}<small>%</small>`;
        $('#loss-value').textContent = result.validationLoss.toFixed(3);
        renderExamBoard();
        renderTrainNetwork();
      },
    });
    if (summary.aborted) return;

    // Final exam: all 10,000 test images, used once after training ends.
    state.isTraining = false;
    state.isEvaluating = true;
    state.trained = true;
    $('#train-progress').style.width = '100%';
    updateStatus();
    const result = await state.model.evaluate(state.dataset.testImages, state.dataset.testLabels, {
      count: state.dataset.testCount,
      batchSize: 128,
      onProgress: ({ progress }) => { $('#batch-readout').textContent = `Final test on 10,000 new images… ${Math.round(progress)}%`; },
    });
    showFinalAccuracy(result.accuracy * 100, result.correct, result.count);
    storageSet('wil-last-accuracy', String(state.accuracy));
    try { await state.model.save(); } catch { /* storage is optional; the trained model stays in memory */ }
  } catch (error) {
    console.error(error);
    showToast(`Training could not finish: ${error.message || error}`, true, 6500);
  } finally {
    state.isTraining = false;
    state.isEvaluating = false;
    state.isPaused = false;
    setTrainingButtons(false);
    $('#pause-training').disabled = true;
    updateStatus();
    renderExamBoard();
    renderTrainNetwork();
  }
}

function showFinalAccuracy(accuracy, correct, count) {
  state.accuracy = accuracy;
  $('#accuracy-label').textContent = 'FINAL TEST ACCURACY';
  $('#accuracy-value').innerHTML = `${accuracy.toFixed(1)}<small>%</small>`;
  $('#accuracy-caption').textContent = correct ? `${formatNumber(correct)} of ${formatNumber(count)} test images correct` : 'on 10,000 test images';
  $('#batch-readout').textContent = 'Done';
}

async function resetTraining() {
  if (!state.model || state.isEvaluating) return;
  const button = $('#reset-training');
  button.disabled = true;
  try {
    await state.model.abort();
    // Start over is a real reset: forget the saved model too, so reopening the site starts untrained.
    try { await state.model.deleteSaved(); } catch (error) { console.error(error); }
    storageRemove('wil-last-accuracy');
    state.model.model?.dispose();
    state.model.model = null;
    state.initializationSeed = allocateSeed();
    await state.model.initialize({ seed: state.initializationSeed });
    Object.assign(state, { trained: false, accuracy: null, isTraining: false, isPaused: false });
    setTrainingButtons(false);
    $('#pause-training').disabled = true;
    resetReadouts($('#epochs').value);
    $('#batch-readout').textContent = 'Fresh random weights';
    renderExamBoard();
    renderTrainNetwork();
    updateStatus();
  } finally { button.disabled = false; }
}

function pauseTraining() {
  state.model.pause(); state.isPaused = true;
  $('#pause-training').hidden = true; $('#resume-training').hidden = false;
  updateStatus();
}

function resumeTraining() {
  state.model.resume(); state.isPaused = false;
  $('#pause-training').hidden = false; $('#resume-training').hidden = true;
  updateStatus();
}

// ---------- Setup ----------

function initializeEvents() {
  $$('.step-link').forEach((button) => button.addEventListener('click', () => showScreen(button.dataset.screen)));
  $$('[data-go]').forEach((button) => button.addEventListener('click', () => showScreen(button.dataset.go)));
  $('#data-size').addEventListener('input', (event) => setTrainCount(sliderToCount(Number(event.target.value)), { fromSlider: true }));
  $('#data-size-number').addEventListener('change', (event) => setTrainCount(Number(event.target.value)));
  $('#data-size-number').addEventListener('keydown', (event) => { if (event.key === 'Enter') setTrainCount(Number(event.target.value)); });
  $('#refresh-examples').addEventListener('click', renderSamples);
  $('#start-training').addEventListener('click', startTraining);
  $('#pause-training').addEventListener('click', pauseTraining);
  $('#resume-training').addEventListener('click', resumeTraining);
  $('#reset-training').addEventListener('click', resetTraining);
  $('#net-prev').addEventListener('click', () => selectNetworkExample(state.networkExample - 1, true));
  $('#net-next').addEventListener('click', () => selectNetworkExample(state.networkExample + 1, true));
  $('#net-replay').addEventListener('click', () => state.trainNetwork.replay());
  $('#play-replay').addEventListener('click', () => state.playNetwork.replay());
  $$('.clear-button').forEach((button) => button.addEventListener('click', () => state.drawings[button.dataset.canvas]?.clear()));
  $('#pixel-dialog').addEventListener('click', (event) => { if (event.target === $('#pixel-dialog')) $('#pixel-dialog').close(); });
  document.addEventListener('keydown', (event) => {
    const target = event.target;
    const typing = target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement;
    if (typing || event.altKey || event.ctrlKey || event.metaKey) return;
    const screens = ['data', 'before', 'training', 'play'];
    const current = screens.indexOf(state.screen);
    if (event.key === 'ArrowRight') { event.preventDefault(); showScreen(screens[Math.min(3, current + 1)]); }
    if (event.key === 'ArrowLeft') { event.preventDefault(); showScreen(screens[Math.max(0, current - 1)]); }
    if (event.code === 'Space' && state.isTraining && !state.isEvaluating) {
      event.preventDefault();
      state.isPaused ? resumeTraining() : pauseTraining();
    }
    if (event.key.toLowerCase() === 'c' && (state.screen === 'play' || state.screen === 'before')) {
      state.drawings[state.screen === 'play' ? 'play-canvas' : 'before-canvas']?.clear();
    }
  });
}

async function initialize() {
  initializeEvents();
  placeSizeTicks();
  setTrainCount(10000);
  createConfidenceRows($('#before-bars'));
  createConfidenceRows($('#play-bars'));
  state.trainNetwork = new NetworkView($('#train-network'));
  state.playNetwork = new NetworkView($('#play-network'));
  state.drawings['before-canvas'] = attachDrawing($('#before-canvas'), (pixels) => {
    state.beforePixels = pixels;
    renderDrawingPrediction('before', pixels);
  });
  state.drawings['play-canvas'] = attachDrawing($('#play-canvas'), (pixels) => {
    state.playPixels = pixels;
    renderDrawingPrediction('play', pixels, { animate: true });
  });
  makeChart();
  try {
    if (!globalThis.tf || !globalThis.Chart) throw new Error('Local libraries are missing. Check the lib folder.');
    try { await tf.setBackend('webgl'); } catch { await tf.setBackend('cpu'); }
    await tf.ready();
    state.dataset = await loadMNIST((progress) => { $('#loading-progress').style.width = `${progress.percent}%`; });
    state.trainingOrder = buildTrainingOrder(state.dataset.trainCount);
    state.model = new DigitModel(tf);
    state.initializationSeed = allocateSeed();
    await state.model.initialize({ seed: state.initializationSeed });
    state.untrainedModel = new DigitModel(tf);
    await state.untrainedModel.initialize({ seed: state.initializationSeed });
    chooseExamExamples();
    $('#loading-overlay').hidden = true;
    renderSamples();
    const restored = await state.model.loadSaved().catch(() => false);
    if (restored) {
      state.trained = true;
      const saved = Number(storageGet('wil-last-accuracy'));
      if (Number.isFinite(saved) && saved > 0) showFinalAccuracy(saved);
    }
    updateStatus();
    renderExamBoard();
    renderTrainNetwork();
  } catch (error) {
    console.error(error);
    $('#loading-overlay').hidden = true;
    setStatus('error');
    showToast(error.message || 'Could not load the data.', true, 9000);
  }
}

initialize();
})();
