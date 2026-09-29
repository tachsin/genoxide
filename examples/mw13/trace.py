"""The trace of the runs for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: both fronts, each population's infeasible solutions and feasible share,
and the fronts' hypervolumes with normalized objectives, in at most 64 generations; and the
optimal front, in its pieces, and the feasible region, sampled. A frame's evaluations are the two
runs' together. The Rust example writes the same file."""

import json
import math
import os

import numpy as np

import genoxide as gx

# the samples of the feasible region along x₁ and along the last variable, and its grid
SAMPLES = 400
CELLS = 80


class Trace:
    """Records the runs through ``on_generation`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self, problem, normalized, reference):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.problem = problem
        self.normalized = normalized
        self.reference = reference
        # per run, after each generation: the evaluations, the feasible front, the infeasible
        # solutions, the hypervolume and the feasible share
        self.series = {}

    def fronts(self, name):
        """The callback for ``run`` that records the run ``name`` after each generation: None
        without a trace to record."""
        history = self.series.setdefault(name, [])

        def record(progress):
            feasible = progress.front_violations == 0
            front = progress.front_objectives[feasible]
            infeasible = progress.objectives[progress.violations != 0]
            volume = gx.indicators.hypervolume(self.normalized(front), self.reference)
            share = (progress.violations == 0).sum() / len(progress.violations)
            history.append(
                (progress.evaluations, front.tolist(), infeasible.tolist(), volume, share)
            )

        return record if self.path else None

    def write(self):
        """Writes the trace, if there's one: the runs side by side, a frame per generation."""
        if not self.path:
            return
        frames = Frames(64)
        for generation in range(max(map(len, self.series.values()), default=0)):
            at = {name: runs[min(generation, len(runs) - 1)] for name, runs in self.series.items()}
            frames.push(
                {
                    "generation": generation,
                    "evaluations": sum(record[0] for record in at.values()),
                    "best": None,
                    "median": None,
                    "state": {
                        "fronts": {name: record[1] for name, record in at.items()},
                        "infeasible": {name: record[2] for name, record in at.items()},
                        "hypervolume": {name: record[3] for name, record in at.items()},
                        "feasible": {name: record[4] for name, record in at.items()},
                    },
                }
            )
        whole = self.normalized(self.problem.optimal_front(20_000))
        settings = {
            "format": 1,
            "example": "mw13",
            "objective": ["minimize", "minimize"],
            "x_label": "generations",
            "y_label": "hypervolume",
            "log_y": False,
            "optimum": gx.indicators.hypervolume(whole, self.reference),
            "plot": "front-2d",
            "problem": {
                "objectives": ["f1", "f2"],
                "true_front": pieces(self.problem.optimal_front(400).tolist()),
                "series": list(self.series),
                "region": region(self.problem),
            },
        }
        write(self.path, settings, frames.to_list())


def pieces(front):
    """The front in pieces, split where two points are more than 2% of its extent apart, each
    point once."""
    first, last = front[0], front[-1]
    extent = math.hypot(last[0] - first[0], first[1] - last[1])
    split = []
    for point in front:
        if not split:
            split.append([point])
            continue
        previous = split[-1][-1]
        if math.hypot(point[0] - previous[0], point[1] - previous[1]) > 0.02 * extent:
            split.append([point])
        elif point != previous:
            split[-1].append(point)
    return split


def region(problem):
    """The feasible region in objective space: genomes with x₁ evenly spread over its range and
    the last variable swept over its range, the other distance variables at their optimal values,
    which moves g from 1 up; a cell of a grid of 80 × 80 over [0, 1.25 f₁ᵐᵃˣ] × [0, 1.25 f₂ᵐᵃˣ]
    (the nadir point's) is feasible if a feasible genome lands in it, and the rows go up from f₂
    = 0."""
    n = problem.dimensions
    upper = 1.5
    nadir = problem.nadir_point
    width, height = 1.25 * nadir[0], 1.25 * nadir[1]
    genomes = []
    for a in range(SAMPLES):
        x1 = upper * a / (SAMPLES - 1)
        x = [x1, *optimal(x1, n)]
        for b in range(SAMPLES):
            x[n - 1] = upper * b / (SAMPLES - 1)
            genomes.append(list(x))
    objectives, violations = problem.evaluate(np.array(genomes))
    i = np.floor(objectives[:, 0] / width * CELLS)
    j = np.floor(objectives[:, 1] / height * CELLS)
    inside = (violations == 0) & (i >= 0) & (j >= 0) & (i < CELLS) & (j < CELLS)
    cells = [["0"] * CELLS for _ in range(CELLS)]
    for column, row in zip(i[inside].astype(int), j[inside].astype(int)):
        cells[row][column] = "1"
    return {"x": [0.0, width], "y": [0.0, height], "rows": ["".join(row) for row in cells]}


def optimal(x1, n):
    """The distance variables where g₂ is 1: xᵢ = (i − 1)/n."""
    return [j / n for j in range(1, n)]


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
