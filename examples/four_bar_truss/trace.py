"""The trace of the run for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: the front's feasible solutions, the population's feasible share, and the
front's hypervolume in scaled objectives, in at most 100 generations, over the optimal front. The
Rust example writes the same file."""

import json
import math
import os

import numpy as np

import genoxide as gx

SERIES = "NSGA-II"


def scaled(problem, points):
    """The objectives scaled to [0, 1] on the front, by its ideal and nadir points."""
    return (np.asarray(points) - problem.ideal_point) / (problem.nadir_point - problem.ideal_point)


def whole_front_hypervolume(problem):
    """The hypervolume of the whole optimal front, from 100,000 of its points, in scaled
    objectives with the reference point (1.1, 1.1)."""
    front = scaled(problem, problem.optimal_front(100_000))
    return gx.indicators.hypervolume(front, [1.1, 1.1])


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
        """Records a generation: the front's feasible solutions and their hypervolume, and the
        population's feasible share."""
        front = progress.front_objectives[progress.front_violations == 0]
        feasible = progress.violations == 0
        volume = gx.indicators.hypervolume(scaled(self.problem, front), [1.1, 1.1])
        state = {
            "fronts": {SERIES: front.tolist()},
            "hypervolume": {SERIES: volume},
            "feasible": {SERIES: feasible.sum() / len(feasible)},
        }
        self.frames.push(frame(progress, state))

    def write(self):
        """Writes the trace, if there's one."""
        if self.path:
            # the optimal front, 400 of its points
            true_front = self.problem.optimal_front(400).tolist()
            problem = {
                "objectives": ["volume (cm³)", "displacement (cm)"],
                "true_front": true_front,
                "series": [SERIES],
            }
            settings = {
                "format": 1,
                "example": "four_bar_truss",
                "objective": ["minimize", "minimize"],
                "x_label": "generations",
                "y_label": "hypervolume (scaled objectives)",
                "log_y": False,
                "optimum": whole_front_hypervolume(self.problem),
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
