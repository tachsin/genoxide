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
