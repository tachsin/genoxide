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
  // the functions of batch 6: the two-dimensional ones as they are; Hartmann's in 3 dimensions
  // as its lowest value over x₃, and Shekel's in 4 on the plane x₃ = x₁, x₄ = x₂, which holds
  // all its wells but a₇ = (5, 5, 3, 3)
  easom: (x, y) => -Math.cos(x) * Math.cos(y) * Math.exp(-((x - Math.PI) ** 2 + (y - Math.PI) ** 2)),
  eggholder: (x, y) => -(y + 47) * Math.sin(Math.sqrt(Math.abs(y + x / 2 + 47))) - x * Math.sin(Math.sqrt(Math.abs(x - (y + 47)))),
  schaffer_f6: (x, y) => {
    const r2 = x * x + y * y;
    return 0.5 + (Math.sin(Math.sqrt(r2)) ** 2 - 0.5) / (1 + 0.001 * r2) ** 2;
  },
  hartmann3: (x, y) => {
    let lowest = Number.POSITIVE_INFINITY;
    for (let k = 0; k <= HARTMANN_STEPS; k++) lowest = Math.min(lowest, hartmann3([x, y, k / HARTMANN_STEPS]));
    return lowest;
  },
  shekel5: (x, y) => shekel(5, [x, y, x, y]),
  shekel7: (x, y) => shekel(7, [x, y, x, y]),
  shekel10: (x, y) => shekel(10, [x, y, x, y]),
  // the functions of batch 10a; the scalable ones in 2 dimensions
  beale: (x, y) => (1.5 - x + x * y) ** 2 + (2.25 - x + x * y * y) ** 2 + (2.625 - x + x * y ** 3) ** 2,
  booth: (x, y) => (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2,
  matyas: (x, y) => 0.26 * (x * x + y * y) - 0.48 * x * y,
  bohachevsky1: (x, y) => x * x + 2 * y * y - 0.3 * Math.cos(3 * Math.PI * x) - 0.4 * Math.cos(4 * Math.PI * y) + 0.7,
  bohachevsky2: (x, y) => x * x + 2 * y * y - 0.3 * Math.cos(3 * Math.PI * x) * Math.cos(4 * Math.PI * y) + 0.3,
  bohachevsky3: (x, y) => x * x + 2 * y * y - 0.3 * Math.cos(3 * Math.PI * x + 4 * Math.PI * y) + 0.3,
  three_hump_camel: (x, y) => 2 * x * x - 1.05 * x ** 4 + x ** 6 / 6 + x * y + y * y,
  langermann: (x, y) => {
    let sum = 0;
    for (let i = 0; i < 5; i++) {
      const d = (x - LANGERMANN_A[i][0]) ** 2 + (y - LANGERMANN_A[i][1]) ** 2;
      sum += LANGERMANN_C[i] * Math.exp(-d / Math.PI) * Math.cos(Math.PI * d);
    }
    return sum;
  },
  shekel_foxholes: (x, y) => {
    let sum = 1 / 500;
    for (let j = 0; j < 25; j++) sum += 1 / (j + 1 + (x - FOXHOLES[j % 5]) ** 6 + (y - FOXHOLES[Math.floor(j / 5)]) ** 6);
    return 1 / sum;
  },
  schwefel_2_21: (x, y) => Math.max(Math.abs(x), Math.abs(y)),
  schwefel_2_22: (x, y) => Math.abs(x) + Math.abs(y) + Math.abs(x * y),
  trid: (x, y) => (x - 1) ** 2 + (y - 1) ** 2 - x * y,
  // the functions of batch 10b, in 2 dimensions
  sum_of_different_powers: (x, y) => Math.abs(x) ** 2 + Math.abs(y) ** 3,
  step: (x, y) => Math.floor(x + 0.5) ** 2 + Math.floor(y + 0.5) ** 2,
  quartic: (x, y) => x ** 4 + 2 * y ** 4,
  penalized1: (x, y) => {
    const [y1, y2] = [1 + (x + 1) / 4, 1 + (y + 1) / 4];
    const levy = 10 * Math.sin(Math.PI * y1) ** 2 + (y1 - 1) ** 2 * (1 + 10 * Math.sin(Math.PI * y2) ** 2) + (y2 - 1) ** 2;
    return (Math.PI / 2) * levy + penalty(x, 10, 100, 4) + penalty(y, 10, 100, 4);
  },
  penalized2: (x, y) =>
    0.1 * (Math.sin(3 * Math.PI * x) ** 2 + (x - 1) ** 2 * (1 + Math.sin(3 * Math.PI * y) ** 2) + (y - 1) ** 2 * (1 + Math.sin(2 * Math.PI * y) ** 2)) +
    penalty(x, 5, 100, 4) +
    penalty(y, 5, 100, 4),
  high_conditioned_elliptic: (x, y) => x * x + 1e6 * y * y,
  bent_cigar: (x, y) => x * x + 1e6 * y * y,
  discus: (x, y) => 1e6 * x * x + y * y,
  different_powers: (x, y) => Math.sqrt(Math.abs(x) ** 2 + Math.abs(y) ** 6),
  buche_rastrigin: (x, y) => {
    // the first gene's scale is 10 on its positive side; the second's is √10
    const z1 = (oscillation(x) > 0 ? 10 : 1) * oscillation(x);
    const z2 = Math.sqrt(10) * oscillation(y);
    const outside = Math.max(0, Math.abs(x) - 5) ** 2 + Math.max(0, Math.abs(y) - 5) ** 2;
    return 10 * (2 - Math.cos(2 * Math.PI * z1) - Math.cos(2 * Math.PI * z2)) + z1 * z1 + z2 * z2 + 100 * outside;
  },
  non_continuous_rastrigin: (x, y) => {
    const step = (v) => (Math.abs(v) < 0.5 ? v : Math.sign(v) * Math.round(Math.abs(2 * v)) / 2);
    const [a, b] = [step(x), step(y)];
    return 20 + a * a - 10 * Math.cos(2 * Math.PI * a) + b * b - 10 * Math.cos(2 * Math.PI * b);
  },
  weierstrass: (x, y) => weierstrass(x) + weierstrass(y) - 2 * weierstrass(0),
  katsuura: (x, y) => {
    const factor = (v, i) => {
      let sum = 0;
      for (let j = 1; j <= 32; j++) {
        const scaled = 2 ** j * v;
        sum += Math.abs(scaled - Math.sign(scaled) * Math.round(Math.abs(scaled))) / 2 ** j;
      }
      return (1 + i * sum) ** (10 / 2 ** 1.2);
    };
    return 2.5 * factor(x, 1) * factor(y, 2) - 2.5;
  },
  happy_cat: (x, y) => {
    const squares = x * x + y * y;
    return Math.abs(squares - 2) ** 0.25 + (0.5 * squares + x + y) / 2 + 0.5;
  },
  hg_bat: (x, y) => {
    const squares = x * x + y * y;
    return Math.sqrt(Math.abs(squares * squares - (x + y) ** 2)) + (0.5 * squares + x + y) / 2 + 0.5;
  },
  schaffer_f7: (x, y) => {
    const s = Math.sqrt(x * x + y * y);
    return (Math.sqrt(s) * (1 + Math.sin(50 * s ** 0.2) ** 2)) ** 2;
  },
  rotated_hyper_ellipsoid: (x, y) => 2 * x * x + y * y,
};

/**
 * `f` rotated as genoxide's `problems::Rotated` rotates it: at `c + M (p − c)`, for the
 * `rotation` of a trace's problem (`{ matrix, center }`, a 2 × 2 matrix by rows); `f` itself
 * without one.
 */
export function rotated(f, rotation) {
  if (!f || !rotation) return f;
  const [[a, b], [c, d]] = rotation.matrix;
  const [cx, cy] = rotation.center;
  return (x, y) => f(cx + a * (x - cx) + b * (y - cy), cy + c * (x - cx) + d * (y - cy));
}

// Yao, Liu and Lin's penalty u(x, a, k, m)
function penalty(x, a, k, m) {
  return Math.abs(x) > a ? k * (Math.abs(x) - a) ** m : 0;
}

// BBOB's oscillation T_osz of one value
function oscillation(x) {
  if (x === 0) return 0;
  const logarithm = Math.log(Math.abs(x));
  const [c1, c2] = x > 0 ? [10, 7.9] : [5.5, 3.1];
  return Math.sign(x) * Math.exp(logarithm + 0.049 * (Math.sin(c1 * logarithm) + Math.sin(c2 * logarithm)));
}

// a gene's part of the Weierstrass function, a = 0.5, b = 3, k from 0 to 20
function weierstrass(x) {
  let sum = 0;
  for (let k = 0; k <= 20; k++) sum += 0.5 ** k * Math.cos(2 * Math.PI * 3 ** k * (x + 0.5));
  return sum;
}

// Langermann's function in 2 dimensions (Molga and Smutnicki's constants), and the centers of
// Shekel's foxholes (De Jong's F5), x₁ varying first
const LANGERMANN_A = [
  [3, 5],
  [5, 2],
  [2, 1],
  [1, 4],
  [7, 9],
];
const LANGERMANN_C = [1, 2, 5, 2, 3];
const FOXHOLES = [-32, -16, 0, 16, 32];

// Hartmann's function in 3 dimensions (Hartman 1973, with Dixon and Szegö's constants), and the
// points of x₃ in [0, 1] over which the contour takes its lowest value
const HARTMANN_STEPS = 40;
const HARTMANN_C = [1, 1.2, 3, 3.2];
const HARTMANN_3_A = [
  [3, 10, 30],
  [0.1, 10, 35],
  [3, 10, 30],
  [0.1, 10, 35],
];
const HARTMANN_3_P = [
  [0.3689, 0.117, 0.2673],
  [0.4699, 0.4387, 0.747],
  [0.1091, 0.8732, 0.5547],
  [0.03815, 0.5743, 0.8828],
];

function hartmann3(x) {
  let sum = 0;
  for (let i = 0; i < 4; i++) {
    let distance = 0;
    for (let j = 0; j < 3; j++) distance += HARTMANN_3_A[i][j] * (x[j] - HARTMANN_3_P[i][j]) ** 2;
    sum += HARTMANN_C[i] * Math.exp(-distance);
  }
  return -sum;
}

// Shekel's function with its first m wells (Shekel 1971, with Dixon and Szegö's constants)
const SHEKEL_A = [
  [4, 4, 4, 4],
  [1, 1, 1, 1],
  [8, 8, 8, 8],
  [6, 6, 6, 6],
  [3, 7, 3, 7],
  [2, 9, 2, 9],
  [5, 5, 3, 3],
  [8, 1, 8, 1],
  [6, 2, 6, 2],
  [7, 3.6, 7, 3.6],
];
const SHEKEL_C = [0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5];

function shekel(m, x) {
  let sum = 0;
  for (let i = 0; i < m; i++) {
    let distance = SHEKEL_C[i];
    for (let j = 0; j < 4; j++) distance += (x[j] - SHEKEL_A[i][j]) ** 2;
    sum += 1 / distance;
  }
  return -sum;
}

/**
 * The contour plot's legend items; a grid of contours shows them once, for all its panels.
 * `minimaLabel` names the marked minima when they aren't proven global ("best known minimum").
 */
export function contourLegend({ best, minima, palette, minimaLabel, populationLabel, ends = false }) {
  return [
    { label: populationLabel ?? "population", color: palette[1], shape: "dot" },
    ...(ends ? [{ label: "converged runs", color: palette[2], shape: "ring" }] : []),
    ...(best ? [{ label: "best", color: palette[0], shape: "diamond" }] : []),
    ...(minima ? [{ label: minimaLabel ?? (minima > 1 ? "global minima" : "global minimum"), shape: "ring", className: "text-base-content" }] : []),
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
