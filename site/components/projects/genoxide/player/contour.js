/**
 * Scalar fields on a grid and their iso-lines, for the contour and surface
 * plots: the test functions are evaluated here, in the browser, so a trace
 * only records the population.
 */

export const FUNCTIONS = {
  rastrigin: (x, y) => 20 + x * x - 10 * Math.cos(2 * Math.PI * x) + y * y - 10 * Math.cos(2 * Math.PI * y),
  himmelblau: (x, y) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2,
  // Branin's RCOS, with the usual constants
  branin: (x, y) => {
    const b = 5.1 / (4 * Math.PI * Math.PI);
    const c = 5 / Math.PI;
    const t = 1 / (8 * Math.PI);
    return (y - b * x * x + c * x - 6) ** 2 + 10 * (1 - t) * Math.cos(x) + 10;
  },
  goldstein_price: (x, y) =>
    (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x * x - 14 * y + 6 * x * y + 3 * y * y)) *
    (30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x * x + 48 * y - 36 * x * y + 27 * y * y)),
  six_hump_camel: (x, y) => (4 - 2.1 * x * x + (x * x * x * x) / 3) * x * x + x * y + (-4 + 4 * y * y) * y * y,
  // the scalable functions in 2 dimensions, as genoxide's problems define them
  sphere: (x, y) => x * x + y * y,
  axis_parallel_ellipsoid: (x, y) => x * x + 2 * y * y,
  schwefel_1_2: (x, y) => x * x + (x + y) ** 2,
  rosenbrock: (x, y) => 100 * (y - x * x) ** 2 + (x - 1) ** 2,
  ackley: (x, y) =>
    20 - 20 * Math.exp(-0.2 * Math.sqrt((x * x + y * y) / 2)) + Math.E - Math.exp((Math.cos(2 * Math.PI * x) + Math.cos(2 * Math.PI * y)) / 2),
  griewank: (x, y) => 1 + (x * x + y * y) / 4000 - Math.cos(x) * Math.cos(y / Math.SQRT2),
  schwefel_2_26: (x, y) => -(x * Math.sin(Math.sqrt(Math.abs(x))) + y * Math.sin(Math.sqrt(Math.abs(y)))),
  levy: (x, y) => {
    const w1 = 1 + (x - 1) / 4;
    const w2 = 1 + (y - 1) / 4;
    return (
      Math.sin(Math.PI * w1) ** 2 +
      (w1 - 1) ** 2 * (1 + 10 * Math.sin(Math.PI * w1 + 1) ** 2) +
      (w2 - 1) ** 2 * (1 + Math.sin(2 * Math.PI * w2) ** 2)
    );
  },
  zakharov: (x, y) => {
    const weighted = 0.5 * x + y;
    return x * x + y * y + weighted ** 2 + weighted ** 4;
  },
  styblinski_tang: (x, y) => 0.5 * (x ** 4 - 16 * x * x + 5 * x + y ** 4 - 16 * y * y + 5 * y),
  michalewicz: (x, y) => -Math.sin(x) * Math.sin((x * x) / Math.PI) ** 20 - Math.sin(y) * Math.sin((2 * y * y) / Math.PI) ** 20,
};

/** The contour plot's legend items; a grid of contours shows them once, for all its panels. */
export function contourLegend({ best, minima, palette }) {
  return [
    { label: "population", color: palette[1], shape: "dot" },
    ...(best ? [{ label: "best", color: palette[0], shape: "diamond" }] : []),
    ...(minima ? [{ label: minima > 1 ? "global minima" : "global minimum", shape: "ring", className: "text-base-content" }] : []),
    { label: "shading: higher f (log)", shape: "square", className: "text-base-content/30" },
  ];
}

/**
 * Samples f on an n × n grid over bounds; `values[i][j]` is at
 * x = x0 + j·dx, y = y0 + i·dy (row i runs along y).
 */
export function sample(f, [[x0, x1], [y0, y1]], n) {
  const values = [];
  for (let i = 0; i < n; i++) {
    const y = y0 + ((y1 - y0) * i) / (n - 1);
    const row = new Float64Array(n);
    for (let j = 0; j < n; j++) row[j] = f(x0 + ((x1 - x0) * j) / (n - 1), y);
    values.push(row);
  }
  return values;
}

// Marching squares: for each cell's corner mask, the edges its iso-line
// crosses (edges: 0 bottom, 1 right, 2 top, 3 left; the saddles pick one way).
const CASES = [[], [[3, 0]], [[0, 1]], [[3, 1]], [[1, 2]], [[3, 2], [0, 1]], [[0, 2]], [[3, 2]], [[2, 3]], [[2, 0]], [[0, 3], [1, 2]], [[2, 1]], [[1, 3]], [[1, 0]], [[0, 3]], []];

/**
 * The iso-line of `values` at `level` as an SVG path, in grid units
 * (column, row); map it with `toX(column)` and `toY(row)`.
 */
export function isoline(values, level, toX, toY) {
  const n = values.length;
  const m = values[0]?.length ?? 0;
  let d = "";
  const t = (a, b) => (level - a) / (b - a || 1e-12);
  for (let i = 0; i < n - 1; i++) {
    for (let j = 0; j < m - 1; j++) {
      const a = values[i][j]; // (j, i)
      const b = values[i][j + 1]; // (j + 1, i)
      const c = values[i + 1][j + 1]; // (j + 1, i + 1)
      const e = values[i + 1][j]; // (j, i + 1)
      const mask = (a > level ? 1 : 0) | (b > level ? 2 : 0) | (c > level ? 4 : 0) | (e > level ? 8 : 0);
      const edges = CASES[mask];
      if (!edges.length) continue;
      const point = (edge) => {
        if (edge === 0) return [j + t(a, b), i];
        if (edge === 1) return [j + 1, i + t(b, c)];
        if (edge === 2) return [j + t(e, c), i + 1];
        return [j, i + t(a, e)];
      };
      for (const [p, q] of edges) {
        const [x1, y1] = point(p);
        const [x2, y2] = point(q);
        d += `M${toX(x1).toFixed(1)} ${toY(y1).toFixed(1)}L${toX(x2).toFixed(1)} ${toY(y2).toFixed(1)}`;
      }
    }
  }
  return d;
}

/**
 * Paints a grid of values in [0, 1] into an image (a data URL), one pixel
 * per grid point with `color(t) -> [r, g, b, a]`; row 0 is at the bottom.
 */
export function gridImage(values, color) {
  const n = values.length;
  const m = values[0]?.length ?? 0;
  const canvas = document.createElement("canvas");
  canvas.width = m;
  canvas.height = n;
  const ctx = canvas.getContext("2d");
  if (!ctx) return null;
  const image = ctx.createImageData(m, n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < m; j++) {
      const [r, g, b, a] = color(values[i][j]);
      const k = ((n - 1 - i) * m + j) * 4;
      image.data[k] = r;
      image.data[k + 1] = g;
      image.data[k + 2] = b;
      image.data[k + 3] = a;
    }
  }
  ctx.putImageData(image, 0, 0);
  return canvas.toDataURL();
}
