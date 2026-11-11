const SIZE = 280;

function attachDrawing(canvas, onChange) {
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const state = { drawing: false, last: null };
  clearCanvas(canvas);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#fff';
  ctx.fillStyle = '#fff';
  ctx.lineWidth = 22;

  const pointFromEvent = (event) => {
    const rect = canvas.getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) * (canvas.width / rect.width),
      y: (event.clientY - rect.top) * (canvas.height / rect.height)
    };
  };

  canvas.addEventListener('pointerdown', (event) => {
    event.preventDefault();
    canvas.setPointerCapture(event.pointerId);
    state.drawing = true;
    state.last = pointFromEvent(event);
    ctx.beginPath();
    ctx.arc(state.last.x, state.last.y, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fill();
  });

  canvas.addEventListener('pointermove', (event) => {
    if (!state.drawing) return;
    event.preventDefault();
    const next = pointFromEvent(event);
    ctx.beginPath();
    ctx.moveTo(state.last.x, state.last.y);
    ctx.lineTo(next.x, next.y);
    ctx.stroke();
    state.last = next;
  });

  const finish = (event) => {
    if (!state.drawing) return;
    event?.preventDefault?.();
    state.drawing = false;
    state.last = null;
    onChange?.(preprocessCanvas(canvas));
  };
  canvas.addEventListener('pointerup', finish);
  canvas.addEventListener('pointercancel', finish);
  canvas.addEventListener('lostpointercapture', finish);

  return {
    clear() {
      clearCanvas(canvas);
      onChange?.(new Float32Array(28 * 28));
    },
  };
}

function clearCanvas(canvas) {
  const ctx = canvas.getContext('2d');
  ctx.save();
  ctx.fillStyle = '#101714';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.restore();
}

function preprocessCanvas(canvas) {
  const width = canvas.width || SIZE;
  const height = canvas.height || SIZE;
  const source = canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, width, height);
  let minX = width, minY = height, maxX = -1, maxY = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const value = source.data[(y * width + x) * 4];
      if (value > 24) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }
  if (maxX < minX || maxY < minY) return new Float32Array(28 * 28);

  const cropW = maxX - minX + 1;
  const cropH = maxY - minY + 1;
  const scale = 20 / Math.max(cropW, cropH);
  const targetW = Math.max(1, Math.round(cropW * scale));
  const targetH = Math.max(1, Math.round(cropH * scale));
  const scaled = document.createElement('canvas');
  scaled.width = targetW;
  scaled.height = targetH;
  const scaledCtx = scaled.getContext('2d', { willReadFrequently: true });
  scaledCtx.imageSmoothingEnabled = true;
  scaledCtx.imageSmoothingQuality = 'high';
  scaledCtx.drawImage(canvas, minX, minY, cropW, cropH, 0, 0, targetW, targetH);
  const small = scaledCtx.getImageData(0, 0, targetW, targetH);

  const pixels = new Float32Array(28 * 28);
  const left = Math.floor((28 - targetW) / 2);
  const top = Math.floor((28 - targetH) / 2);
  for (let y = 0; y < targetH; y++) {
    for (let x = 0; x < targetW; x++) {
      const value = small.data[(y * targetW + x) * 4] / 255;
      pixels[(top + y) * 28 + left + x] = value;
    }
  }

  // Center by ink mass, which is more robust than centering the bounding box alone.
  let mass = 0, sumX = 0, sumY = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const value = pixels[y * 28 + x];
      mass += value;
      sumX += x * value;
      sumY += y * value;
    }
  }
  if (mass > 0) {
    const dx = Math.round(13.5 - sumX / mass);
    const dy = Math.round(13.5 - sumY / mass);
    if (dx || dy) {
      const centered = new Float32Array(28 * 28);
      for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
          const nextX = x + dx, nextY = y + dy;
          if (nextX >= 0 && nextX < 28 && nextY >= 0 && nextY < 28) centered[nextY * 28 + nextX] = pixels[y * 28 + x];
        }
      }
      return centered;
    }
  }
  return pixels;
}

function renderProcessed(canvas, pixels) {
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(28, 28);
  for (let i = 0; i < 28 * 28; i++) {
    const shade = Math.max(0, Math.min(255, Math.round((pixels[i] || 0) * 255)));
    const offset = i * 4;
    image.data[offset] = shade;
    image.data[offset + 1] = shade;
    image.data[offset + 2] = shade;
    image.data[offset + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
}
