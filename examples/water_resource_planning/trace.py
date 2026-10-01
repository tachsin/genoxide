"""The trace of the run for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: the front's feasible solutions in two panels, f₁, f₂ and f₃, and f₃, f₄
and f₅, and the front's IGD+ to the optimal front in scaled objectives, in at most 40 generations.
The Rust example writes the same file."""

import json
import math
import os

import numpy as np

import genoxide as gx

# the objectives of each panel
PANELS = [[0, 1, 2], [2, 3, 4]]


def scaled(problem, points):
    """The objectives scaled to [0, 1] on the front, by its ideal and nadir points."""
    return (np.asarray(points) - problem.ideal_point) / (problem.nadir_point - problem.ideal_point)


def optimal_front(problem):
    """At least 5,000 points of the optimal front, scaled."""
    return scaled(problem, problem.optimal_front(5_000))


class Trace:
    """Records the run through ``on_generation`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self, problem):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.frames = Frames(40)
        self.problem = problem
        self.optimal = optimal_front(problem) if self.path else None

    @property
    def on_generation(self):
        """The callback for ``run``: None without a trace to record."""
        return self.record if self.path else None

    def record(self, progress):
        """Records a generation: the front's feasible solutions, in both panels, and their
        IGD+."""
        front = progress.front_objectives[progress.front_violations == 0]
        distance = gx.indicators.igd_plus(scaled(self.problem, front), self.optimal)
        panels = [{"front": front[:, axes].tolist()} for axes in PANELS]
        # the curve is the IGD+, as the best value of the frame
        entry = frame(progress, {"panels": panels})
        entry["best"] = distance
        self.frames.push(entry)

    def write(self):
        """Writes the trace, if there's one."""
        if self.path:
            labels = ["f1", "f2", "f3", "f4", "f5"]
            panels = []
            for axes in PANELS:
                objectives = [labels[j] for j in axes]
                panel = {"title": ", ".join(objectives), "problem": {"objectives": objectives}}
                panels.append(panel)
            settings = {
                "format": 1,
                "example": "water_resource_planning",
                "objective": ["minimize"] * 5,
                "x_label": "generations",
                "y_label": "IGD+ (scaled objectives)",
                "log_y": True,
                "optimum": None,
                "plot": "grid",
                "problem": {"panel_plot": "front-3d", "panels": panels},
            }
            write(self.path, settings, self.frames.to_list())

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
    best), the state; to within a hundredth of their range over the run: a front's
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
        tolerance.append((max(values) - min(values)) / 100.0)

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


def frame(progress, state):
    """The frame of a generation: its progress and ``state``, whose hypervolume is the curve."""
    return {
        "generation": progress.generation,
        "evaluations": progress.evaluations,
        "best": None,
        "median": None,
        "state": state,
    }


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
