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
            "example": "mw12",
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
                "feasible_region": region(self.problem),
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
    upper = 1.0
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
    cells = [[False] * CELLS for _ in range(CELLS)]
    for column, row in zip(i[inside].astype(int), j[inside].astype(int)):
        cells[row][column] = True
    # each row as the pairs of columns [start, end) where it's feasible
    rows = [runs(row) for row in cells]
    return {"x": [0.0, width], "y": [0.0, height], "columns": CELLS, "rows": rows}


def runs(row):
    """The columns where a row of cells turns feasible or infeasible, as pairs [start, end)."""
    turns = [
        column for column in range(len(row)) if row[column] != (column > 0 and row[column - 1])
    ]
    return turns + ([len(row)] if row[-1] else [])


_ROOTS = {}


def optimal(x1, n):
    """The distance variables where g₁ is 1: xᵢ^(n−2) = 0.5 + (i − 1)/(2n), each root found by
    bisection with the power as repeated products, as the Rust trace does (the same for every
    x₁, so found once)."""
    if n not in _ROOTS:
        _ROOTS[n] = [root(0.5 + j / (2 * n), n - 2) for j in range(1, n)]
    return _ROOTS[n]


def root(value, power):
    """The x in [0, 1] with x^power = value, by bisection to the precision of floats, the power
    computed as repeated products (the same in Rust)."""
    low, high = 0.0, 1.0
    while True:
        middle = 0.5 * (low + high)
        if middle <= low or middle >= high:
            return high
        product = 1.0
        for _ in range(power):
            product *= middle
        if product < value:
            low = middle
        else:
            high = middle


# ---- the same in every example's trace ----------------------------------------------------------


class Frames:
    """The frames of at most ``most`` generations, from the part of the run where what the page
    plots changes: the frames after the last change are left out (a run that reached its target,
    or a front that no longer moves), and the rest are spread evenly over the generations up to
    it. While the run goes, up to 8 × ``most`` frames are kept: every ``every``-th generation,
    with ``every`` doubling whenever there are that many, and the last one."""

    def __init__(self, most):
        self.most, self.every, self.kept, self.last = most, 1, [], None

    def push(self, frame):
        if frame["generation"] % self.every:
            self.last = frame
            return
        self.kept.append(frame)
        self.last = None
        if len(self.kept) == 8 * self.most:
            self.every *= 2
            self.kept = [kept for kept in self.kept if kept["generation"] % self.every == 0]

    def to_list(self):
        frames = self.kept + ([self.last] if self.last else [])
        active = frames[: last_change(frames) + 1]
        count, most = len(active), max(self.most, 2)
        if count <= most:
            return active
        return [active[(i * (count - 1) + (most - 1) // 2) // (most - 1)] for i in range(most)]


def last_change(frames):
    """The index of the frame after which nothing the page plots changes. To 3 significant
    digits, as a plot shows them: the best, the median and, for a single objective (a numeric
    best), the state; to within a thousandth of their range over the run: a front's
    hypervolumes, in the state or in a grid's series."""
    if not frames:
        return 0
    last = len(frames) - 1
    number = lambda value: isinstance(value, (int, float)) and not isinstance(value, bool)
    single = any(number(frame.get("best")) for frame in frames)

    def measures(frame):
        values = []
        state = frame.get("state")
        hypervolume = state.get("hypervolume") if isinstance(state, dict) else None
        for value in (hypervolume, frame.get("series")):
            if number(value):
                values.append(float(value))
            elif isinstance(value, dict):
                values.extend(float(v) for _, v in sorted(value.items()) if number(v))
        return values

    measured = [measures(frame) for frame in frames]
    end = measured[last]
    tolerance = []
    for k in range(len(end)):
        values = [values[k] for values in measured if k < len(values)]
        tolerance.append((max(values) - min(values)) / 1000.0)

    def same(frame, final, key, flush=False):
        return coarse(frame.get(key), flush) == coarse(final.get(key), flush)

    def settled(i):
        frame, final = frames[i], frames[last]
        return (
            same(frame, final, "best")
            and same(frame, final, "median")
            and (not single or same(frame, final, "state", flush=True))
            and len(measured[i]) == len(end)
            and all(abs(v - e) <= t for v, e, t in zip(measured[i], end, tolerance))
        )

    first = last
    while first > 0 and settled(first - 1):
        first -= 1
    return first


def coarse(value, flush=False):
    """``value`` with its numbers to 3 significant digits, as precisely as a plot shows them: two
    frames whose plotted values agree to that precision look the same. With ``flush``, for the
    solutions a plot draws on their ranges, numbers below 1e-6 in size count as 0."""
    if isinstance(value, float):
        if flush and abs(value) < 1e-6:
            value = 0.0
        return f"{value:.2e}"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(coarse(item, flush) for item in value) + "]"
    if isinstance(value, dict):
        items = sorted(value.items())
        return "{" + ",".join(f"{key}:{coarse(item, flush)}" for key, item in items) + "}"
    return json.dumps(value)


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
