/**
 * MNIST loader for How AI Learns to See.
 *
 * The four original MNIST IDX files are stored gzip-compressed and base64
 * encoded in data/mnist.js (built by tools/make-mnist-bundle.py). Loading them
 * through a <script> tag works both from a web server and when index.html is
 * opened directly from disk, where browsers block fetch().
 *
 * Images are returned as row-major Uint8Array views (784 bytes per 28x28
 * image), and labels as Uint8Array views containing digits 0-9.
 *
 * Standard MNIST has 60,000 training and 10,000 test examples. The first
 * 55,000 training examples form the selectable training pool, the final 5,000
 * are kept aside as validation data, and the test set stays the canonical
 * 10,000 examples.
 *
 * @param {(progress: {phase: string, completedFiles: number, totalFiles: number,
 *   percent: number, file?: string}) => void} [onProgress]
 */
async function loadMNIST(onProgress = () => {}) {
  const assets = [
    { key: 'trainImages', file: 'train-images-idx3-ubyte', kind: 'images', count: 60000 },
    { key: 'trainLabels', file: 'train-labels-idx1-ubyte', kind: 'labels', count: 60000 },
    { key: 'testImages', file: 't10k-images-idx3-ubyte', kind: 'images', count: 10000 },
    { key: 'testLabels', file: 't10k-labels-idx1-ubyte', kind: 'labels', count: 10000 },
  ];
  const loaded = {};
  let completedFiles = 0;
  onProgress({ phase: 'loading', completedFiles, totalFiles: assets.length, percent: 0 });

  await loadBundleScript('data/mnist.js');

  await Promise.all(assets.map(async (asset) => {
    const bytes = await gunzip(decodeBase64(globalThis.MNIST_GZ?.[asset.file] ?? ''));
    loaded[asset.key] = asset.kind === 'images'
      ? parseImages(bytes, asset.file, asset.count)
      : parseLabels(bytes, asset.file, asset.count);
    completedFiles += 1;
    onProgress({
      phase: 'loading',
      completedFiles,
      totalFiles: assets.length,
      percent: Math.round((completedFiles / assets.length) * 100),
      file: asset.file,
    });
  }));
  delete globalThis.MNIST_GZ;

  onProgress({ phase: 'ready', completedFiles, totalFiles: assets.length, percent: 100 });
  const splitAt = 55000;
  const imageBytesPerImage = 28 * 28;
  return {
    trainImages: loaded.trainImages.images.subarray(0, splitAt * imageBytesPerImage),
    trainLabels: loaded.trainLabels.labels.subarray(0, splitAt),
    validationImages: loaded.trainImages.images.subarray(splitAt * imageBytesPerImage),
    validationLabels: loaded.trainLabels.labels.subarray(splitAt),
    testImages: loaded.testImages.images,
    testLabels: loaded.testLabels.labels,
    trainCount: 55000,
    validationCount: 5000,
    testCount: 10000,
  };
}

function loadBundleScript(src) {
  if (globalThis.MNIST_GZ) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = src;
    script.onload = () => resolve();
    script.onerror = () => reject(new Error(`Could not load ${src}. Keep the data folder next to index.html.`));
    document.head.append(script);
  });
}

function decodeBase64(text) {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function gunzip(bytes) {
  if (typeof DecompressionStream !== 'function') {
    throw new Error('This browser cannot unpack the dataset. Please use a recent Chrome, Edge, Firefox or Safari.');
  }
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function parseImages(bytes, file, expectedCount) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const expectedLength = 16 + expectedCount * 28 * 28;
  if (bytes.byteLength !== expectedLength) {
    throw new Error(`${file} has ${bytes.byteLength} bytes; expected exactly ${expectedLength}.`);
  }
  if (view.getUint32(0, false) !== 2051) {
    throw new Error(`${file} has an invalid IDX image magic number (expected 2051).`);
  }
  const count = view.getUint32(4, false);
  const rows = view.getUint32(8, false);
  const columns = view.getUint32(12, false);
  if (count !== expectedCount || rows !== 28 || columns !== 28) {
    throw new Error(`${file} header mismatch: expected ${expectedCount} images of 28x28; found ${count} of ${rows}x${columns}.`);
  }
  return { images: bytes.subarray(16) };
}

function parseLabels(bytes, file, expectedCount) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const expectedLength = 8 + expectedCount;
  if (bytes.byteLength !== expectedLength) {
    throw new Error(`${file} has ${bytes.byteLength} bytes; expected exactly ${expectedLength}.`);
  }
  if (view.getUint32(0, false) !== 2049) {
    throw new Error(`${file} has an invalid IDX label magic number (expected 2049).`);
  }
  const count = view.getUint32(4, false);
  if (count !== expectedCount) {
    throw new Error(`${file} header mismatch: expected ${expectedCount} labels; found ${count}.`);
  }
  const labels = bytes.subarray(8);
  for (let i = 0; i < labels.length; i += 1) {
    if (labels[i] > 9) {
      throw new Error(`${file} contains invalid label ${labels[i]} at index ${i}.`);
    }
  }
  return { labels };
}
