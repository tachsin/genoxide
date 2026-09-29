"""The trace of the run for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: the front, the infeasible solutions of the population and its feasible
share, and the front's hypervolume in scaled objectives, in at most 100 generations, over the
problem's feasible region. The Rust example writes the same file."""

import json
import math
import os

import numpy as np

import genoxide as gx

SERIES = "NSGA-II"
# the feasible region over f₁ in [0, 1] and f₂ in [0, TOP], in COLUMNS × ROWS cells
TOP = 2.0
COLUMNS = 400
ROWS = 400


def scaled(points):
    """The objectives scaled to [0, 1] on the optimal front, by its ideal and nadir points."""
    problem = gx.problems.Ctp1()
    return (np.asarray(points) - problem.ideal_point) / (problem.nadir_point - problem.ideal_point)


def whole_front_hypervolume():
    """The hypervolume of the whole optimal front, from 100,000 of its points, in scaled
    objectives with the reference point (1.1, 1.1)."""
    front = scaled(gx.problems.Ctp1().optimal_front(100_000))
    return gx.indicators.hypervolume(front, [1.1, 1.1])


def pieces(front):
    """The pieces of a front sorted by f₁, split where neighbors are more than 0.01 apart."""
    gaps = np.flatnonzero(np.hypot(*np.diff(front, axis=0).T) > 0.01) + 1
    return np.split(front, gaps)


def region(problem):
    """The feasible region: a cell is feasible when the genome with its center's objectives meets
    the constraints. x₁ = f₁, and x₂ is found by bisection, as f₂ grows with x₂; rows from the
    bottom up, each the pairs of columns [start, end) where it's feasible."""
    high = problem.genome._describe()["bounds"][1][1]
    x1 = np.tile((np.arange(COLUMNS) + 0.5) / COLUMNS, ROWS)
    target = np.repeat(TOP * (np.arange(ROWS) + 0.5) / ROWS, COLUMNS)

    def f2(x2):
        return problem.evaluate(np.column_stack([x1, x2]))[0][:, 1]

    reachable = (target >= f2(np.zeros_like(x1))) & (target <= f2(np.full_like(x1, high)))
    low, up = np.zeros_like(x1), np.full_like(x1, high)
    for _ in range(50):
        middle = 0.5 * (low + up)
        below = f2(middle) < target
        low, up = np.where(below, middle, low), np.where(below, up, middle)
    feasible = reachable & (problem.evaluate(np.column_stack([x1, up]))[1] == 0)
    rows = []
    for row in feasible.reshape(ROWS, COLUMNS):
        before = np.concatenate([[False], row[:-1]])
        runs = np.flatnonzero(row != before).tolist() + ([COLUMNS] if row[-1] else [])
        rows.append(runs)
    return {"x": [0.0, 1.0], "y": [0.0, TOP], "columns": COLUMNS, "rows": rows}


class Trace:
    """Records the run through ``on_generation`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self, problem):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.frames = Frames(100)
        self.problem = problem

    @property
    def on_generation(self):
        """The callback for ``run``: None without a trace to record."""
        return self.record if self.path else None

    def record(self, progress):
        """Records a generation: the front and its hypervolume, and the population's infeasible
        solutions and feasible share."""
        front = progress.front_objectives[progress.front_violations == 0]
        feasible = progress.violations == 0
        state = {
            "fronts": {SERIES: front.tolist()},
            "infeasible": {SERIES: progress.objectives[~feasible].tolist()},
            "hypervolume": {SERIES: gx.indicators.hypervolume(scaled(front), [1.1, 1.1])},
            "feasible": {SERIES: feasible.sum() / len(feasible)},
        }
        self.frames.push(frame(progress, state))

    def write(self):
        """Writes the trace, if there's one."""
        if self.path:
            front = self.problem.optimal_front(400)
            # the optimal front, in its pieces, over the feasible region
            problem = {
                "objectives": ["f1", "f2"],
                "true_front": [piece.tolist() for piece in pieces(front)],
                "feasible_region": region(self.problem),
                "series": [SERIES],
            }
            settings = {
                "format": 1,
                "example": "ctp1",
                "objective": ["minimize", "minimize"],
                "x_label": "generations",
                "y_label": "hypervolume (scaled objectives)",
                "log_y": False,
                "optimum": whole_front_hypervolume(),
                "plot": "front-2d",
                "problem": problem,
            }
            write(self.path, settings, self.frames.to_list())


# ---- the same in every example's trace ----------------------------------------------------------


class Frames:
    """The frames of at most ``most`` generations: every ``every``-th one, with ``every`` doubling
    whenever there are ``most``, and the last one."""

    def __init__(self, most):
        self.most, self.every, self.kept, self.last = most, 1, [], None

    def push(self, frame):
        if frame["generation"] % self.every:
            self.last = frame
            return
        self.kept.append(frame)
        self.last = None
        if len(self.kept) == self.most:
            self.every *= 2
            self.kept = [kept for kept in self.kept if kept["generation"] % self.every == 0]

    def to_list(self):
        return self.kept + ([self.last] if self.last else [])


def frame(progress, state):
    """The frame of a generation: its progress and ``state``, whose hypervolume is the curve."""
    return {
        "generation": progress.generation,
        "evaluations": progress.evaluations,
        "best": None,
        "median": None,
        "state": state,
    }


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
