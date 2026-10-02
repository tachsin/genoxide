"""The trace of the run for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: a frame per step, with the points evaluated so far and, from the first
point the models chose, the models that chose it on a grid of 25 x 25 points: the probability that
a point is feasible under the constraints' models, and the acquisition, the log expected
improvement plus the logarithm of that probability (before a feasible point, the logarithm alone),
the 25 nats below its highest value shaded. Both are rounded to thousandths of their range, which
is all the page draws. The Rust example writes the same file."""

import json
import math
import os

import numpy as np

# the points per side of the grid, and the span of the acquisition that is shaded, in nats
GRID = 25
SPAN = 25.0


class Trace:
    """Records the run through ``control`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self, minimizer, minimum, constraints):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.minimizer = minimizer
        self.minimum = minimum
        self.constraints = constraints
        self.frames = []

    def record(self, algorithm, progress):
        """Records a step: the points so far, the newest one, the best, and the models that
        chose the newest on the grid."""
        if not self.path:
            return
        points = progress.population.tolist()
        best = progress.best_genome
        state = {
            "population": points,
            "newest": points[-1],
            "best": best.tolist(),
        }
        if algorithm.model is not None:
            grid = points_of_grid()
            feasible = algorithm.probability_of_feasibility_at(grid).reshape(GRID, GRID)
            acquisition = algorithm.acquisition_at(grid).reshape(GRID, GRID)
            state["mean"] = shaded(feasible, lambda lo, hi, v: v)
            state["acquisition"] = shaded(acquisition, acquisition_shade)
        # the best feasible value's distance above the minimum, none before a feasible point
        error = None
        if np.all(self.constraints(best) <= 0.0):
            error = float(best[0] + best[1]) - self.minimum
        self.frames.append(
            {
                "generation": progress.generation,
                "evaluations": progress.evaluations,
                "best": error,
                "state": state,
            }
        )

    def write(self):
        """Writes the trace, if there's one."""
        if not self.path:
            return
        settings = {
            "format": 1,
            "example": "bo_constrained",
            "objective": "minimize",
            "x_label": "evaluations",
            "y_label": "error of the best feasible point",
            "log_y": True,
            "optimum": 0.0,
            "plot": "surrogate",
            "problem": {
                "bounds": [[0.0, 1.0], [0.0, 1.0]],
                "minima": [self.minimizer],
                "minima_label": "the global minimum",
                "grid": GRID,
                "panels": [
                    {
                        "title": "The probability of feasibility",
                        "shading": "shading: more likely feasible",
                    },
                    {
                        "title": "Log-EI + log P(feasible)",
                        "shading": "shading: more worth evaluating",
                    },
                ],
            },
        }
        write(self.path, settings, self.frames)


def points_of_grid():
    """The grid over the box, a point per row: x2 from 0 to 1 by row of the grid, x1 by column,
    as the Rust example orders them."""
    return np.array(
        [[column / (GRID - 1), row / (GRID - 1)] for row in range(GRID) for column in range(GRID)]
    )


def acquisition_shade(lo, hi, v):
    lo = max(lo, hi - SPAN)
    return max((v - lo) / (hi - lo), 0.0)


def shaded(grid, shade):
    """The grid's values as thousandths: ``shade(lo, hi, v)`` in [0, 1], rounded."""
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
