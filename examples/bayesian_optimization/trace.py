"""The trace of the run for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: a frame per step, with the points evaluated so far and, from the first
point the model chose, the model that chose it on a grid of 25 x 25 points: its posterior mean,
shaded on a log scale as the page shades the function, and the log expected improvement, the 25
nats below its highest value shaded. Both are rounded to thousandths of their range, which is all
the page draws. A last frame shows the polished point. The Rust example writes the same file."""

import json
import math
import os

import numpy as np

import genoxide as gx

# the points per side of the grid, and the span of the acquisition that is shaded, in nats
GRID = 25
SPAN = 25.0


class Trace:
    """Records the run through ``control`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self, minima, minimum):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.minima = minima
        self.minimum = minimum
        self.frames = []

    def record(self, algorithm, progress):
        """Records a step: the points so far, the newest one, the best, and the model that chose
        the newest on the grid."""
        if not self.path:
            return
        points = progress.population.tolist()
        state = {
            "population": points,
            "newest": points[-1],
            "best": progress.best_genome.tolist(),
        }
        model = algorithm.model
        if model is not None:
            grid = points_of_grid()
            mean = model.predict(grid)[0].reshape(GRID, GRID)
            acquisition = algorithm.acquisition_at(grid).reshape(GRID, GRID)
            state["mean"] = shaded(mean, lambda lo, hi, v: log1p(v - lo) / log1p(hi - lo))
            state["acquisition"] = shaded(acquisition, acquisition_shade)
        self.frames.append(
            {
                "generation": progress.generation,
                "evaluations": progress.evaluations,
                "best": progress.best_fitness - self.minimum,
                "state": state,
            }
        )

    def write(self, x, value):
        """Writes the trace, if there's one, with a last frame for the polished point ``x`` and
        its value."""
        if not self.path:
            return
        last = json.loads(json.dumps(self.frames[-1]))
        last["generation"] += 1
        last["evaluations"] += 1
        last["best"] = min(last["best"], value - self.minimum)
        last["state"]["polished"] = [float(x[0]), float(x[1])]
        self.frames.append(last)
        settings = {
            "format": 1,
            "example": "bayesian_optimization",
            "objective": "minimize",
            "x_label": "evaluations",
            "y_label": "error to the global minimum",
            "log_y": True,
            "optimum": 0.0,
            "plot": "surrogate",
            "problem": {
                "function": "branin",
                "bounds": [[-5.0, 10.0], [0.0, 15.0]],
                "minima": self.minima.tolist(),
                "grid": GRID,
            },
        }
        write(self.path, settings, self.frames)


def points_of_grid():
    """The grid over Branin's box, a point per row: x2 from 0 to 15 by row of the grid, x1 from
    -5 to 10 by column, as the Rust example orders them."""

    def at(low, high, i):
        return low + (high - low) * i / (GRID - 1)

    return np.array(
        [[at(-5.0, 10.0, column), at(0.0, 15.0, row)] for row in range(GRID) for column in range(GRID)]
    )


def log1p(x):
    """genoxide's portable log1p, the Rust example's."""
    return float(gx.math.log1p(np.array([x]))[0])


def acquisition_shade(lo, hi, v):
    lo = max(lo, hi - SPAN)
    return max((v - lo) / (hi - lo), 0.0)


def shaded(grid, shade):
    """The grid's values as thousandths of their range: ``shade(lo, hi, v)`` in [0, 1],
    rounded."""
    lo, hi = float(grid.min()), float(grid.max())
    return [[math.floor(1000.0 * shade(lo, hi, float(v)) + 0.5) for v in row] for row in grid]


# ---- the same in every example's trace ----------------------------------------------------------


def write(path, settings, frames):
    """Writes the settings and the frames to ``path``, a frame per line."""
    lines = ",\n".join(map(to_json, frames))
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write(f'{to_json(settings)[:-1]},"frames":[\n{lines}\n]}}\n')


def to_json(value):
    """Compact JSON with sorted keys, and numbers rounded to 6 significant digits, as the Rust
    example writes it."""
    return json.dumps(rounded(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def rounded(value):
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item) for item in value]
    if isinstance(value, float):
        return float(f"{value:.5e}") if math.isfinite(value) else None
    return value
