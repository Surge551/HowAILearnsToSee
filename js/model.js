/**
 * How AI Learns to See's small, explainable MNIST model.
 *
 * Data contract: loadMNIST (js/data.js) supplies flat row-major
 * image buffers, 784 bytes per image, and one integer label (0–9) per image.
 * Training pixels are Uint8 values in [0, 255] and are divided by 255 here.
 * predict() instead accepts a Float32Array already normalized to [0, 1], which
 * is convenient for the drawing canvas. Keep that preprocessing consistent:
 * a canvas/data mismatch can lower accuracy even when training worked well.
 *
 * Network: 784 inputs → Dense(64, ReLU) → Dense(10, softmax).
 * Optimizer/loss: Adam(0.001), categorical crossentropy.
 *
 * Public API:
 *   const model = new DigitModel();
 *   await model.initialize();
 *   const probabilities = model.predict(normalized784Float32);
 *   const inspection = model.inspect(normalized784Float32); // for the diagram
 *   const evaluation = await model.evaluate(testImages, testLabels);
 *   await model.save();
 *   const restored = await model.loadSaved();
 *   const run = model.train({ trainImages, trainLabels, validationImages, validationLabels,
 *     epochs: 5, batchSize: 64, speedMs: 0, onBatch, onEpoch, onComplete });
 *   model.pause(); model.resume();
 *   await model.abort(); // wait until the old run has stopped before restart
 *   await model.train(...);
 *
 * onBatch receives { epoch, batch, batches, loss, accuracy, progress }.
 * onEpoch receives { epoch, epochs, trainCount, validationCount, validationLoss,
 *   validationAccuracy, durationMs }.
 * onComplete receives and train() resolves to
 * { aborted, epochsCompleted, trainCount, validationCount, history }.
 *
 * This implementation intentionally creates only one training batch at a
 * time. It uses the first 1,000 validation examples in a stable order after
 * every epoch. The canonical test set is evaluated separately after training.
 */
class DigitModel {
  constructor(tf = globalThis.tf) {
    this.tf = tf;
    this.model = null;
    this._paused = false;
    this._aborted = false;
    this._pauseWaiters = [];
    this._runPromise = null;
  }

  async initialize({ seed } = {}) {
    if (this.model) return this;
    if (!this.tf || typeof this.tf.sequential !== "function") {
      throw new Error("TensorFlow.js must be loaded before initializing DigitModel.");
    }

    const tf = this.tf;
    const firstKernelInitializer = Number.isInteger(seed) ? tf.initializers.glorotUniform({ seed }) : undefined;
    const secondKernelInitializer = Number.isInteger(seed) ? tf.initializers.glorotUniform({ seed: (seed + 1) >>> 0 }) : undefined;
    this.model = tf.sequential({
      layers: [
        tf.layers.dense({ inputShape: [784], units: 64, activation: "relu", ...(firstKernelInitializer ? { kernelInitializer: firstKernelInitializer } : {}) }),
        tf.layers.dense({ units: 10, activation: "softmax", ...(secondKernelInitializer ? { kernelInitializer: secondKernelInitializer } : {}) }),
      ],
    });
    this._compileModel(this.model);
    return this;
  }

  /** Predict probabilities for one 28×28 image already normalized to [0, 1]. */
  predict(pixels) {
    this._requireModel();
    if (!(pixels instanceof Float32Array) || pixels.length !== 784) {
      throw new TypeError("predict() expects a Float32Array with exactly 784 values.");
    }
    for (let i = 0; i < pixels.length; i += 1) {
      if (!Number.isFinite(pixels[i]) || pixels[i] < 0 || pixels[i] > 1) {
        throw new RangeError("predict() pixel values must be finite and normalized to [0, 1].");
      }
    }

    return this.tf.tidy(() => {
      const input = this.tf.tensor2d(pixels, [1, 784]);
      const output = this.model.predict(input);
      return Float32Array.from(output.dataSync());
    });
  }

  /**
   * Run one normalized image through the network layer by layer and return
   * what the diagram needs: the 64 hidden ReLU activations, the 10 softmax
   * scores, and the hidden→output kernel laid out as [hidden * 10 + digit].
   */
  inspect(pixels) {
    this._requireModel();
    const tf = this.tf;
    return tf.tidy(() => {
      const [kernel1, bias1] = this.model.layers[0].getWeights();
      const [kernel2, bias2] = this.model.layers[1].getWeights();
      const input = tf.tensor2d(pixels, [1, 784]);
      const hidden = input.matMul(kernel1).add(bias1).relu();
      const output = hidden.matMul(kernel2).add(bias2).softmax();
      return {
        hidden: Float32Array.from(hidden.dataSync()),
        output: Float32Array.from(output.dataSync()),
        outputWeights: Float32Array.from(kernel2.dataSync()),
      };
    });
  }

  /**
   * Evaluate a stable prefix of a dataset without holding all converted images
   * in GPU memory. The confusion matrix uses truth rows and prediction columns.
   */
  async evaluate(images, labels, { count = labels?.length, batchSize = 128, onProgress } = {}) {
    this._requireModel();
    if (this._runPromise) throw new Error("Cannot evaluate while training is active.");
    const available = this._validateDataset(images, labels, "evaluation");
    if (!Number.isInteger(count) || count < 1) {
      throw new RangeError("count must be a positive integer.");
    }
    if (!Number.isInteger(batchSize) || batchSize < 1) {
      throw new RangeError("batchSize must be a positive integer.");
    }
    if (onProgress != null && typeof onProgress !== "function") {
      throw new TypeError("onProgress must be a function when provided.");
    }

    const evaluatedCount = Math.min(count, available);
    if (evaluatedCount < 1) throw new RangeError("The evaluation dataset is empty.");
    const confusionMatrix = Array.from({ length: 10 }, () => Array(10).fill(0));
    let correct = 0;
    let lossSum = 0;
    const batches = Math.ceil(evaluatedCount / batchSize);

    for (let batchIndex = 0; batchIndex < batches; batchIndex += 1) {
      const start = batchIndex * batchSize;
      const end = Math.min(start + batchSize, evaluatedCount);
      const size = end - start;
      const pixelStart = start * 784;
      const pixelEnd = end * 784;
      const batchResult = this.tf.tidy(() => {
        const xs = this.tf.tensor2d(images.subarray(pixelStart, pixelEnd), [size, 784]).div(255);
        const probabilities = this.model.predict(xs);
        return {
          predictions: Int32Array.from(probabilities.argMax(1).dataSync()),
          probabilities: Float32Array.from(probabilities.dataSync()),
        };
      });

      for (let offset = 0; offset < size; offset += 1) {
        const truth = labels[start + offset];
        const prediction = batchResult.predictions[offset];
        lossSum -= Math.log(Math.max(batchResult.probabilities[offset * 10 + truth], 1e-7));
        confusionMatrix[truth][prediction] += 1;
        if (truth === prediction) correct += 1;
      }
      onProgress?.({ completed: end, count: evaluatedCount, progress: (end / evaluatedCount) * 100 });
      await this._yieldToUI(0);
    }

    return { count: evaluatedCount, correct, accuracy: correct / evaluatedCount, loss: lossSum / evaluatedCount, confusionMatrix };
  }

  /** Save the current compiled model to this browser's IndexedDB storage. */
  async save() {
    this._requireModel();
    if (this._runPromise) throw new Error("Cannot save while training is active.");
    return this.model.save("indexeddb://how-ai-learns-to-see-model");
  }

  /** Delete the saved model from this browser, if there is one. Returns true if one was removed. */
  async deleteSaved() {
    const storageKey = "indexeddb://how-ai-learns-to-see-model";
    if (!this.tf?.io || typeof this.tf.io.removeModel !== "function") return false;
    const savedModels = await this.tf.io.listModels();
    if (!savedModels[storageKey]) return false;
    await this.tf.io.removeModel(storageKey);
    return true;
  }

  /** Load and compile a saved model before replacing (and disposing) the current one. */
  async loadSaved() {
    if (!this.tf || typeof this.tf.loadLayersModel !== "function") {
      throw new Error("TensorFlow.js must be loaded before loading a saved model.");
    }
    if (this._runPromise) throw new Error("Cannot load a saved model while training is active.");

    const storageKey = "indexeddb://how-ai-learns-to-see-model";
    if (this.tf.io && typeof this.tf.io.listModels === "function") {
      const savedModels = await this.tf.io.listModels();
      if (!savedModels[storageKey]) return false;
    }

    let loaded;
    try {
      loaded = await this.tf.loadLayersModel(storageKey);
    } catch (error) {
      // Translate only a missing key to false; corruption and storage failures
      // should remain visible to the caller.
      if (/not found|cannot find model|no model/i.test(String(error?.message ?? error))) return false;
      throw error;
    }

    try {
      this._compileModel(loaded);
    } catch (error) {
      loaded.dispose();
      throw error;
    }
    const previous = this.model;
    this.model = loaded;
    if (previous && previous !== loaded) previous.dispose();
    return true;
  }

  pause() {
    if (this._runPromise) this._paused = true;
  }

  resume() {
    this._paused = false;
    const waiters = this._pauseWaiters.splice(0);
    for (const resolve of waiters) resolve();
  }

  /** Abort the active run and resolve once its current batch has been cleaned up. */
  async abort() {
    this._aborted = true;
    this.resume();
    if (this._runPromise) await this._runPromise;
  }

  /**
   * Train in small batches so the page can update between them. Calls cannot
   * overlap; await abort() before starting a replacement run.
   */
  train({
    trainImages,
    trainLabels,
    validationImages,
    validationLabels,
    trainCount = trainLabels?.length,
    seed = 2026,
    epochs = 5,
    batchSize = 64,
    speedMs = 0,
    onBatch,
    onEpoch,
    onComplete,
  } = {}) {
    if (this._runPromise) {
      throw new Error("A training run is active. Await model.abort() before restarting.");
    }

    this._aborted = false;
    this._paused = false;
    const runPromise = this._train({
      trainImages, trainLabels, validationImages, validationLabels, trainCount, seed,
      epochs, batchSize, speedMs, onBatch, onEpoch, onComplete,
    });
    this._runPromise = runPromise.finally(() => {
      if (this._runPromise === wrappedPromise) this._runPromise = null;
      this._aborted = false;
      this._paused = false;
    });
    const wrappedPromise = this._runPromise;
    return wrappedPromise;
  }

  async _train(options) {
    await this.initialize();
    const { trainImages, trainLabels, validationImages, validationLabels } = options;
    const availableTrain = this._validateDataset(trainImages, trainLabels, "training");
    const availableValidation = this._validateDataset(validationImages, validationLabels, "validation");
    const trainCount = options.trainCount;
    const epochs = options.epochs;
    const batchSize = options.batchSize;
    const speedMs = options.speedMs;
    const seed = options.seed;

    if (!Number.isInteger(trainCount) || trainCount < 1 || trainCount > availableTrain) {
      throw new RangeError(`trainCount must be from 1 to ${availableTrain}.`);
    }
    if (!Number.isInteger(epochs) || epochs < 1) {
      throw new RangeError("epochs must be a positive integer.");
    }
    if (!Number.isInteger(batchSize) || batchSize < 1) {
      throw new RangeError("batchSize must be a positive integer.");
    }
    if (!Number.isFinite(speedMs) || speedMs < 0) {
      throw new RangeError("speedMs must be a non-negative number.");
    }
    if (!Number.isInteger(seed) || seed < 0 || seed > 0xffffffff) {
      throw new RangeError("seed must be an integer from 0 to 4,294,967,295.");
    }
    for (const callbackName of ["onBatch", "onEpoch", "onComplete"]) {
      if (options[callbackName] != null && typeof options[callbackName] !== "function") {
        throw new TypeError(`${callbackName} must be a function when provided.`);
      }
    }
    if (availableValidation < 1000) {
      throw new RangeError("The validation dataset must contain at least 1,000 examples.");
    }

    const validationCount = 1000;
    const batches = Math.ceil(trainCount / batchSize);
    const history = [];
    let epochsCompleted = 0;

    for (let epoch = 0; epoch < epochs && !this._aborted; epoch += 1) {
      const epochStarted = Date.now();
      const order = this._shuffledOrder(trainCount, (seed + Math.imul(epoch, 0x9e3779b1)) >>> 0);
      for (let batchIndex = 0; batchIndex < batches; batchIndex += 1) {
        await this._waitUntilRunning();
        if (this._aborted) break;

        const start = batchIndex * batchSize;
        const end = Math.min(start + batchSize, trainCount);
        const result = await this._trainBatch(trainImages, trainLabels, order, start, end);
        const values = this._readMetrics(result);
        this._disposeMetrics(result);
        const progress = ((epoch * trainCount + end) / (epochs * trainCount)) * 100;
        options.onBatch?.({
          epoch: epoch + 1,
          batch: batchIndex + 1,
          batches,
          loss: values[0] ?? null,
          accuracy: values[1] ?? null,
          progress,
        });

        // Always yield to rendering and input, even at speedMs=0.
        await this._yieldToUI(speedMs);
      }

      if (this._aborted) break;
      await this._waitUntilRunning();
      if (this._aborted) break;

      const metrics = await this._evaluateFirst(validationImages, validationLabels, validationCount);
      const epochResult = {
        epoch: epoch + 1,
        epochs,
        trainCount,
        validationCount,
        validationLoss: metrics[0],
        validationAccuracy: metrics[1],
        durationMs: Date.now() - epochStarted,
      };
      history.push(epochResult);
      epochsCompleted += 1;
      options.onEpoch?.(epochResult);
      await this._yieldToUI(0);
    }

    const summary = {
      aborted: this._aborted,
      epochsCompleted,
      trainCount,
      validationCount,
      seed,
      history,
    };
    options.onComplete?.(summary);
    return summary;
  }

  _shuffledOrder(count, seed) {
    const indexes = new Uint32Array(count);
    for (let i = 0; i < count; i += 1) indexes[i] = i;
    let value = seed >>> 0;
    const random = () => {
      value = (value + 0x6d2b79f5) >>> 0;
      let next = value;
      next = Math.imul(next ^ (next >>> 15), next | 1);
      next ^= next + Math.imul(next ^ (next >>> 7), next | 61);
      return ((next ^ (next >>> 14)) >>> 0) / 4294967296;
    };
    for (let i = count - 1; i > 0; i -= 1) {
      const j = Math.floor(random() * (i + 1));
      [indexes[i], indexes[j]] = [indexes[j], indexes[i]];
    }
    return indexes;
  }

  async _trainBatch(images, labels, order, start, end) {
    const tf = this.tf;
    const size = end - start;
    const imageBatch = new Uint8Array(size * 784);
    const labelBatch = new Int32Array(size);
    for (let i = 0; i < size; i += 1) {
      const sourceIndex = order[start + i];
      imageBatch.set(images.subarray(sourceIndex * 784, (sourceIndex + 1) * 784), i * 784);
      labelBatch[i] = labels[sourceIndex];
    }

    const { xs, ys } = tf.tidy(() => ({
      xs: tf.tensor2d(imageBatch, [size, 784]).div(255),
      ys: tf.oneHot(tf.tensor1d(labelBatch, "int32"), 10),
    }));
    try {
      return await this.model.trainOnBatch(xs, ys);
    } finally {
      xs.dispose();
      ys.dispose();
    }
  }

  async _evaluateFirst(images, labels, count) {
    const tf = this.tf;
    const labelBatch = new Int32Array(count);
    for (let i = 0; i < count; i += 1) labelBatch[i] = labels[i];
    const { xs, ys } = tf.tidy(() => ({
      xs: tf.tensor2d(images.subarray(0, count * 784), [count, 784]).div(255),
      ys: tf.oneHot(tf.tensor1d(labelBatch, "int32"), 10),
    }));
    let result;
    try {
      result = await this.model.evaluate(xs, ys, { batchSize: count, verbose: 0 });
      return this._readMetrics(result);
    } finally {
      xs.dispose();
      ys.dispose();
      this._disposeMetrics(result);
    }
  }

  async _waitUntilRunning() {
    while (this._paused && !this._aborted) {
      await new Promise((resolve) => this._pauseWaiters.push(resolve));
    }
  }

  _yieldToUI(milliseconds) {
    if (milliseconds <= 0) return new Promise((resolve) => setTimeout(resolve, 0));
    const deadline = Date.now() + milliseconds;
    return new Promise((resolve) => {
      const tick = () => {
        if (this._aborted || Date.now() >= deadline) return resolve();
        // Check cancellation often so a slow pace does not make
        // "Start over" feel unresponsive.
        setTimeout(tick, Math.min(50, deadline - Date.now()));
      };
      tick();
    });
  }

  _validateDataset(images, labels, name) {
    if (!ArrayBuffer.isView(images) || images instanceof DataView) {
      throw new TypeError(`${name} images must be a typed array with 784 values per image.`);
    }
    if (!ArrayBuffer.isView(labels) || labels instanceof DataView) {
      throw new TypeError(`${name} labels must be an integer typed array.`);
    }
    if (images.length !== labels.length * 784) {
      throw new RangeError(`${name} image and label counts do not match (expected 784 pixels per image).`);
    }
    for (let i = 0; i < labels.length; i += 1) {
      if (!Number.isInteger(labels[i]) || labels[i] < 0 || labels[i] > 9) {
        throw new RangeError(`${name} labels must be integers from 0 to 9.`);
      }
    }
    return labels.length;
  }

  _readMetrics(result) {
    const values = Array.isArray(result) ? result : [result];
    return values.map((value) => {
      if (typeof value === "number") return value;
      if (value && typeof value.dataSync === "function") return value.dataSync()[0];
      return null;
    });
  }

  _disposeMetrics(result) {
    const values = Array.isArray(result) ? result : [result];
    for (const value of values) {
      if (value && typeof value.dispose === "function") value.dispose();
    }
  }

  _requireModel() {
    if (!this.model) throw new Error("Call and await initialize() before using the model.");
  }

  _compileModel(model) {
    model.compile({
      optimizer: this.tf.train.adam(0.001),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });
  }
}

