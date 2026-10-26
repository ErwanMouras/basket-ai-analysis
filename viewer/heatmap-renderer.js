/* Continuous scalar density first, discrete colors second. No blurred RGB tiles. */
(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.HeatmapRenderer = factory();
})(globalThis, () => {
  const background = [23, 28, 32];
  const warm = [
    background,
    [96, 57, 30],
    [165, 78, 28],
    [224, 112, 32],
    [255, 167, 55],
    [255, 216, 116],
    [255, 242, 191],
  ];
  const cool = [
    background,
    [32, 57, 76],
    [38, 91, 128],
    [49, 134, 180],
    [82, 179, 219],
    [148, 218, 242],
    [211, 242, 252],
  ];
  const levels = [0.04, 0.14, 0.3, 0.5, 0.72, 0.9];
  const sigma = 0.85; // Metres, identical for all subjects, periods and canvas sizes.
  const samplesPerMetre = 8;

  function axisKernels(bins, extent, samples) {
    const step = extent / samples;
    return Array.from({ length: bins }, (_, bin) => {
      const centre = ((bin + 0.5) * extent) / bins;
      const from = Math.max(0, Math.ceil((centre - 3 * sigma) / step - 0.5));
      const to = Math.min(
        samples - 1,
        Math.floor((centre + 3 * sigma) / step - 0.5),
      );
      const weights = [];
      let sum = 0;
      for (let index = from; index <= to; index++) {
        const distance = ((index + 0.5) * step - centre) / sigma;
        const weight = Math.exp(-0.5 * distance * distance);
        weights.push(weight);
        sum += weight;
      }
      return { from, weights, sum };
    });
  }

  function density(values, nx, ny, length, width) {
    const cols = Math.ceil(length * samplesPerMetre);
    const rows = Math.ceil(width * samplesPerMetre);
    const area = (length * width) / (cols * rows);
    const xKernels = axisKernels(nx, length, cols);
    const yKernels = axisKernels(ny, width, rows);
    const field = new Float64Array(cols * rows);
    values.forEach((value, index) => {
      if (!value) return;
      const xx = xKernels[index % nx];
      const yy = yKernels[Math.floor(index / nx)];
      // Normalize each contribution at the court edges as well: no mass is lost
      // outside the drawing, and two profiles retain comparable density units.
      const scale = value / (xx.sum * yy.sum * area);
      yy.weights.forEach((wy, y) => {
        const offset = (yy.from + y) * cols + xx.from;
        xx.weights.forEach((wx, x) => {
          field[offset + x] += scale * wx * wy;
        });
      });
    });
    return { values: field, cols, rows, area };
  }

  function subtract(first, second, available = true) {
    return {
      ...first,
      values: first.values.map((value, index) =>
        available ? value - second.values[index] : 0,
      ),
    };
  }
  function maximum(field) {
    return field.values.reduce(
      (max, value) => Math.max(max, Math.abs(value)),
      0,
    );
  }
  function sample(field, u, v) {
    const x = Math.max(0, Math.min(field.cols - 1, u * field.cols - 0.5));
    const y = Math.max(0, Math.min(field.rows - 1, v * field.rows - 0.5));
    const left = Math.floor(x),
      top = Math.floor(y);
    const right = Math.min(field.cols - 1, left + 1),
      bottom = Math.min(field.rows - 1, top + 1);
    const fx = x - left,
      fy = y - top;
    const a =
      field.values[top * field.cols + left] * (1 - fx) +
      field.values[top * field.cols + right] * fx;
    const b =
      field.values[bottom * field.cols + left] * (1 - fx) +
      field.values[bottom * field.cols + right] * fx;
    return a * (1 - fy) + b * fy;
  }
  function color(value, max, diverging = false) {
    const amount = max > 0 ? Math.min(1, Math.abs(value) / max) : 0;
    let band = 0;
    while (band < levels.length && amount >= levels[band]) band++;
    return (diverging && value < 0 ? cool : warm)[band];
  }
  function paint(ctx, box, field, max, diverging) {
    const raster = ctx.canvas.ownerDocument.createElement("canvas");
    // Supersample only to antialias contour boundaries, never to blur colors.
    raster.width = Math.ceil(box.width * 1.5);
    raster.height = Math.ceil(box.height * 1.5);
    const target = raster.getContext("2d");
    const pixels = target.createImageData(raster.width, raster.height);
    let offset = 0;
    for (let y = 0; y < raster.height; y++) {
      for (let x = 0; x < raster.width; x++) {
        const value = sample(
          field,
          (x + 0.5) / raster.width,
          (y + 0.5) / raster.height,
        );
        const rgb = color(value, max, diverging);
        pixels.data[offset++] = rgb[0];
        pixels.data[offset++] = rgb[1];
        pixels.data[offset++] = rgb[2];
        pixels.data[offset++] = 255;
      }
    }
    target.putImageData(pixels, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(raster, box.x, box.y, box.width, box.height);
  }
  return { density, subtract, maximum, sample, color, paint };
});
