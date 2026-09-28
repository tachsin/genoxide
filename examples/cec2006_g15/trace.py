"""The trace of the run for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: the best solution so far and its constraints, in at most 100
generations. The Rust example writes the same file.

The curve is the error f − f* of the best feasible solution and of the population's median, on a
log scale: null while they're infeasible, whose values aren't comparable to f*."""

import json
import math
import os

from genoxide.problems.cec2006 import EQUALITY_TOLERANCE


class Trace:
    """Records the run through ``on_generation`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self, problem):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.frames = Frames(100)
        self.problem = problem

    def record(self, progress):
        """Records a generation: the errors of the best and the median, the best solution so far and
        its equalities, each as its excess max(0, |h(x)| - 0.0001), 0 (active) when met (the page
        shows each one's state)."""
        if self.path:
            best = progress.best_genome
            score, violation = self.problem(best)
            best_error = self.error(score) if violation == 0 else None
            # the population in the order of Deb's rules: the feasible solutions by value, then
            # the infeasible ones
            feasible = sorted(
                self.error(float(score))
                for score, violation in zip(progress.scores, progress.violations)
                if violation == 0
            )
            middle = median(feasible, len(progress.scores))
            h = self.problem.constraints(best).tolist()
            values = [max(abs(hj) - EQUALITY_TOLERANCE, 0.0) for hj in h]
            state = {"best": best.tolist(), "violations": values}
            self.frames.push(frame(progress, best_error, middle, state))

    def error(self, score):
        """The error f - f* of a feasible solution, 0 at or below f*."""
        return max(score - self.problem.optimum.value, 0.0)

    def write(self):
        """Writes the trace, if there's one."""
        if self.path:
            settings = {
                "format": 1,
                "example": "cec2006_g15",
                "objective": "minimize",
                "x_label": "evaluations",
                "y_label": "error f - f* of the best feasible solution",
                "log_y": True,
                "optimum": 0.0,
                "plot": "design",
                "problem": {
                    "variables": [
                        {"name": f"x{i}", "unit": "", "bounds": [0.0, 10.0]} for i in range(1, 4)
                    ],
                    "constraints": ["h1", "h2"],
                    # equalities only, recorded as their excess over the tolerance: 0 when met,
                    # which is active
                    "signed": True,
                },
            }
            write(self.path, settings, self.frames.to_list())


def median(feasible, size):
    """The median error of a population of ``size`` in the order of Deb's rules, given the sorted
    errors of its feasible solutions: None if the median is infeasible."""
    middle = size // 2
    if size == 0 or middle >= len(feasible):
        return None
    return feasible[middle] if size % 2 else (feasible[middle - 1] + feasible[middle]) / 2


def frame(progress, best, middle, state):
    """The frame of a generation: its progress, the errors of the best and the median, and
    ``state``."""
    return {
        "generation": progress.generation,
        "evaluations": progress.evaluations,
        "best": best,
        "median": middle,
        "state": state,
    }


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
