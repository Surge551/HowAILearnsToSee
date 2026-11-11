/**
 * Interactive diagram of the 784 → 64 → 10 network for one input image.
 *
 * update({ pixels, hidden, output, outputWeights, truth }, { animate }) draws
 * the forward pass: pixels light up hidden neurons (ReLU), and each active
 * hidden neuron pushes the ten digit scores up (green) or down (red) by
 * activation × weight. Hovering a neuron isolates its connections and explains
 * its role; replay() animates the pass one layer at a time.
 */
class NetworkView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.data = null;
    this.hover = null;
    this.progress = 1;
    this.layout = null;
    this.frame = 0;
    this.emptyMessage = 'Draw a digit';
    this.tip = document.createElement('div');
    this.tip.className = 'net-tip';
    this.tip.hidden = true;
    canvas.parentElement.append(this.tip);
    this.inputCanvas = document.createElement('canvas');
    this.inputCanvas.width = 28;
    this.inputCanvas.height = 28;

    new ResizeObserver(() => this.resize()).observe(canvas.parentElement);
    canvas.addEventListener('pointermove', (event) => this.onPointer(event));
    canvas.addEventListener('pointerleave', () => { this.hover = null; this.tip.hidden = true; this.draw(); });
  }

  update(data, { animate = false } = {}) {
    this.data = data;
    if (data) this.paintInput(data.pixels);
    if (animate) this.replay();
    else { cancelAnimationFrame(this.frame); this.progress = 1; this.draw(); }
    if (this.hover) this.showTip();
  }

  replay(duration = 1800) {
    if (!this.data) return;
    cancelAnimationFrame(this.frame);
    const start = performance.now();
    const step = (now) => {
      this.progress = Math.min(1, (now - start) / duration);
      this.draw();
      if (this.progress < 1) this.frame = requestAnimationFrame(step);
    };
    this.frame = requestAnimationFrame(step);
  }

  resize() {
    const width = this.canvas.parentElement.clientWidth;
    if (!width) return;
    const height = Math.round(Math.max(340, Math.min(520, width * 0.5)));
    const ratio = window.devicePixelRatio || 1;
    this.canvas.style.height = `${height}px`;
    this.canvas.width = Math.round(width * ratio);
    this.canvas.height = Math.round(height * ratio);
    this.ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    this.width = width;
    this.height = height;
    this.computeLayout();
    this.draw();
  }

  computeLayout() {
    const W = this.width, H = this.height;
    const top = 44, bottom = H - 16;
    const imgSize = Math.min(H * 0.46, W * 0.19);
    const hiddenX = W * 0.4, outputX = W * 0.64;
    const hiddenStep = (bottom - top) / 63;
    const outputStep = (bottom - top) / 9;
    this.layout = {
      top, imgSize,
      img: { x: 8, y: (H - imgSize) / 2 + 12 },
      hidden: Array.from({ length: 64 }, (_, i) => ({ x: hiddenX, y: top + i * hiddenStep })),
      hiddenR: Math.max(2, Math.min(5, hiddenStep * 0.42)),
      output: Array.from({ length: 10 }, (_, j) => ({ x: outputX, y: top + j * outputStep })),
      outputR: Math.max(9, Math.min(17, outputStep * 0.36)),
      barX: outputX + Math.min(17, outputStep * 0.36) + 12,
      // Leave room after each bar for the percentage and the "correct answer" tag.
      barW: Math.max(40, W - outputX - Math.min(17, outputStep * 0.36) - 12 - 160),
    };
  }

  paintInput(pixels) {
    const ctx = this.inputCanvas.getContext('2d');
    const image = ctx.createImageData(28, 28);
    for (let i = 0; i < 784; i++) {
      const shade = Math.round(Math.max(0, Math.min(1, pixels[i])) * 255);
      image.data.set([shade, shade, shade, 255], i * 4);
    }
    ctx.putImageData(image, 0, 0);
  }

  draw() {
    const { ctx, layout: L } = this;
    if (!L) return;
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.font = '700 11px ProjectCairo, Segoe UI, Arial, sans-serif';
    ctx.fillStyle = '#688074';
    ctx.textAlign = 'left';
    ctx.fillText('INPUT · 784 PIXELS', L.img.x, 18);
    ctx.textAlign = 'center';
    ctx.fillText('HIDDEN · 64 NEURONS', L.hidden[0].x, 18);
    ctx.textAlign = 'left';
    ctx.fillText('OUTPUT · 10 DIGITS', L.output[0].x - L.outputR, 18);

    const d = this.data;
    const p = this.progress;
    const stage = (from, to) => Math.max(0, Math.min(1, (p - from) / (to - from)));
    const sIn = stage(0, 0.3), sHidden = stage(0.25, 0.45), sLinks = stage(0.45, 0.8), sOut = stage(0.75, 1);

    // Input image
    ctx.fillStyle = '#101714';
    ctx.fillRect(L.img.x, L.img.y, L.imgSize, L.imgSize);
    if (d) {
      ctx.imageSmoothingEnabled = false;
      ctx.drawImage(this.inputCanvas, L.img.x, L.img.y, L.imgSize, L.imgSize);
    }
    ctx.strokeStyle = this.hover?.layer === 'input' ? '#187753' : '#27342f';
    ctx.lineWidth = this.hover?.layer === 'input' ? 3 : 1;
    ctx.strokeRect(L.img.x, L.img.y, L.imgSize, L.imgSize);

    if (!d) {
      this.drawSkeleton();
      ctx.fillStyle = '#66736e';
      ctx.font = '600 15px ProjectCairo, Segoe UI, Arial, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(this.emptyMessage, L.img.x + L.imgSize / 2, L.img.y + L.imgSize / 2 + 5);
      return;
    }

    const maxAct = Math.max(1e-6, ...d.hidden);
    const hoverHidden = this.hover?.layer === 'hidden' ? this.hover.index : -1;
    const hoverOutput = this.hover?.layer === 'output' ? this.hover.index : -1;
    const focus = hoverHidden >= 0 || hoverOutput >= 0;

    // Input → hidden: brighter fan lines reach the neurons this image switches on.
    const fanX = L.img.x + L.imgSize;
    for (let i = 0; i < 64; i++) {
      const node = L.hidden[i];
      const strength = d.hidden[i] / maxAct;
      let alpha = (strength > 0 ? 0.08 + 0.5 * strength : 0.03) * sIn;
      if (focus) alpha *= i === hoverHidden ? 2.2 : 0.25;
      ctx.strokeStyle = `rgba(24,119,83,${Math.min(1, alpha)})`;
      ctx.lineWidth = strength > 0 ? 0.6 + 1.4 * strength : 0.5;
      ctx.beginPath();
      ctx.moveTo(fanX, L.img.y + L.imgSize * (0.1 + 0.8 * i / 63));
      ctx.lineTo(node.x - L.hiddenR, node.y);
      ctx.stroke();
    }

    // Hidden → output: contribution = activation × weight.
    let maxContribution = 1e-6;
    for (let i = 0; i < 64; i++) for (let j = 0; j < 10; j++) {
      maxContribution = Math.max(maxContribution, Math.abs(d.hidden[i] * d.outputWeights[i * 10 + j]));
    }
    for (let i = 0; i < 64; i++) {
      if (d.hidden[i] <= 0) continue;
      for (let j = 0; j < 10; j++) {
        const c = d.hidden[i] * d.outputWeights[i * 10 + j];
        const strength = Math.abs(c) / maxContribution;
        const selected = i === hoverHidden || j === hoverOutput;
        if (strength < (selected ? 0.01 : 0.06)) continue;
        let alpha = 0.75 * strength * sLinks;
        if (focus) alpha = selected ? Math.max(0.25, strength) * sLinks : alpha * 0.12;
        ctx.strokeStyle = c > 0 ? `rgba(24,119,83,${alpha})` : `rgba(189,67,58,${alpha})`;
        ctx.lineWidth = 0.5 + 2.6 * strength;
        ctx.beginPath();
        ctx.moveTo(L.hidden[i].x + L.hiddenR, L.hidden[i].y);
        ctx.lineTo(L.output[j].x - L.outputR, L.output[j].y);
        ctx.stroke();
      }
    }

    // Hidden neurons
    for (let i = 0; i < 64; i++) {
      const node = L.hidden[i];
      const strength = d.hidden[i] / maxAct;
      ctx.beginPath();
      ctx.arc(node.x, node.y, L.hiddenR + (i === hoverHidden ? 2 : 0), 0, Math.PI * 2);
      if (strength > 0 && sHidden > 0) {
        ctx.fillStyle = `rgba(24,119,83,${(0.25 + 0.75 * strength) * sHidden})`;
        ctx.fill();
      } else {
        ctx.fillStyle = '#ffffff';
        ctx.fill();
        ctx.strokeStyle = '#b7c2bd';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      if (i === hoverHidden) { ctx.strokeStyle = '#182421'; ctx.lineWidth = 2; ctx.stroke(); }
    }

    // Output neurons, scores and bars
    const best = argmaxOf(d.output);
    for (let j = 0; j < 10; j++) {
      const node = L.output[j];
      const score = d.output[j] * sOut;
      const isBest = j === best && sOut >= 1;
      ctx.beginPath();
      ctx.arc(node.x, node.y, L.outputR, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
      ctx.fillStyle = `rgba(24,119,83,${0.1 + 0.9 * score})`;
      ctx.fill();
      ctx.strokeStyle = isBest ? '#187753' : j === hoverOutput ? '#182421' : '#c4cec9';
      ctx.lineWidth = isBest || j === hoverOutput ? 3 : 1;
      ctx.stroke();
      ctx.fillStyle = score > 0.45 ? '#ffffff' : '#182421';
      ctx.font = `800 ${Math.round(L.outputR * 0.95)}px Arial, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(String(j), node.x, node.y + 1);

      ctx.fillStyle = '#edf1ee';
      ctx.fillRect(L.barX, node.y - 5, L.barW, 10);
      ctx.fillStyle = isBest ? '#187753' : '#aab9b1';
      ctx.fillRect(L.barX, node.y - 5, L.barW * score, 10);
      ctx.textAlign = 'left';
      ctx.font = '600 11px Arial, sans-serif';
      ctx.fillStyle = '#596760';
      ctx.fillText(`${(score * 100).toFixed(1)}%`, L.barX + L.barW + 6, node.y + 1);
      if (d.truth === j && sOut >= 1) {
        ctx.fillStyle = best === j ? '#187753' : '#bd433a';
        ctx.font = '800 11px ProjectCairo, Segoe UI, Arial, sans-serif';
        ctx.fillText(best === j ? '✓ correct answer' : '← correct answer', L.barX + L.barW + 50, node.y + 1);
      }
      ctx.textBaseline = 'alphabetic';
    }
  }

  drawSkeleton() {
    const { ctx, layout: L } = this;
    ctx.strokeStyle = '#c4cec9';
    ctx.lineWidth = 1;
    for (const node of L.hidden) { ctx.beginPath(); ctx.arc(node.x, node.y, L.hiddenR, 0, Math.PI * 2); ctx.stroke(); }
    ctx.font = `800 ${Math.round(L.outputR * 0.95)}px Arial, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    L.output.forEach((node, j) => {
      ctx.beginPath(); ctx.arc(node.x, node.y, L.outputR, 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = '#9ca8a2'; ctx.fillText(String(j), node.x, node.y + 1);
    });
    ctx.textBaseline = 'alphabetic';
  }

  onPointer(event) {
    if (!this.layout) return;
    const rect = this.canvas.getBoundingClientRect();
    const x = event.clientX - rect.left, y = event.clientY - rect.top;
    this.pointer = { x, y };
    const L = this.layout;
    let hover = null;
    if (x >= L.img.x && x <= L.img.x + L.imgSize && y >= L.img.y && y <= L.img.y + L.imgSize) hover = { layer: 'input' };
    const hiddenStep = L.hidden[1].y - L.hidden[0].y;
    if (Math.abs(x - L.hidden[0].x) < 12) {
      const index = Math.round((y - L.top) / hiddenStep);
      if (index >= 0 && index < 64) hover = { layer: 'hidden', index };
    }
    const outputStep = L.output[1].y - L.output[0].y;
    if (x > L.output[0].x - L.outputR - 6 && x < L.barX + L.barW + 40) {
      const index = Math.round((y - L.top) / outputStep);
      if (index >= 0 && index < 10 && Math.abs(y - L.output[index].y) < outputStep / 2) hover = { layer: 'output', index };
    }
    const changed = JSON.stringify(hover) !== JSON.stringify(this.hover);
    this.hover = hover;
    this.canvas.style.cursor = hover ? 'pointer' : 'default';
    if (changed) this.draw();
    this.showTip();
  }

  showTip() {
    const d = this.data, h = this.hover;
    if (!h || !this.pointer) { this.tip.hidden = true; return; }
    let title, text;
    if (h.layer === 'input') {
      title = '784 numbers';
      text = 'The network never sees a picture — only 784 brightness values between 0 (dark) and 1 (bright).';
    } else if (!d) {
      this.tip.hidden = true; return;
    } else if (h.layer === 'hidden') {
      const act = d.hidden[h.index];
      title = `Hidden neuron ${h.index + 1} · activation ${act.toFixed(2)}`;
      if (act <= 0) {
        text = 'Silent for this image: its weighted sum was negative, so ReLU turned it into 0. It has no say in the answer.';
      } else {
        const pushes = Array.from({ length: 10 }, (_, j) => ({ j, c: act * d.outputWeights[h.index * 10 + j] })).sort((a, b) => b.c - a.c);
        const up = pushes[0], down = pushes[9];
        text = `It found a pattern it looks for. It votes most for ${up.j} (+${up.c.toFixed(2)}) and most against ${down.j} (${down.c.toFixed(2)}).`;
      }
    } else {
      const j = h.index;
      const helpers = Array.from({ length: 64 }, (_, i) => ({ i, c: d.hidden[i] * d.outputWeights[i * 10 + j] })).filter((v) => v.c > 0).sort((a, b) => b.c - a.c).slice(0, 3);
      title = `Digit ${j} · score ${(d.output[j] * 100).toFixed(1)}%`;
      text = helpers.length
        ? `Score = all green votes minus red votes, then softmax. Biggest supporters: neurons ${helpers.map((v) => v.i + 1).join(', ')}.`
        : 'No active neuron votes for this digit, so its score stays low.';
    }
    this.tip.innerHTML = '';
    const strong = document.createElement('strong');
    strong.textContent = title;
    const body = document.createElement('span');
    body.textContent = text;
    this.tip.append(strong, body);
    this.tip.hidden = false;
    const wrapWidth = this.canvas.parentElement.clientWidth;
    const left = Math.min(this.pointer.x + 14, wrapWidth - 280);
    this.tip.style.left = `${Math.max(0, left)}px`;
    this.tip.style.top = `${this.pointer.y + 16}px`;
  }
}

function argmaxOf(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}
