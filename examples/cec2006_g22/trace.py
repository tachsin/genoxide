"""The trace of the run for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: the best solution so far and its constraints, in at most 100
generations. The Rust example writes the same file.

The run finds no feasible solution of g22, and the curve is the constraint violation of the best
solution and of the population's median, on a log scale."""

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

    @property
    def on_generation(self):
        """The callback for ``run``: None without a trace to record."""
        return self.record if self.path else None

    def record(self, progress):
        """Records a generation: the violations of the best and the median, the best solution so far
        and its constraints, g(x) for the inequalities, satisfied at or below 0 and active at 0, and
        for the equalities the excess max(0, |h(x)| - 0.0001), 0 (active) when met (the page shows
        each one's state)."""
        if self.path:
            best = progress.best_genome
            score, violation = self.problem(best)
            best_violation = None if math.isnan(score) else violation
            # the population in the order of Deb's rules, none of it feasible in this run: by
            # violation, then the invalid solutions
            valid = sorted(
                float(violation)
                for score, violation in zip(progress.scores, progress.violations)
                if not math.isnan(score)
            )
            middle = median(valid, len(progress.scores))
            constraints = self.problem.constraints(best).tolist()
            g, h = constraints[:1], constraints[1:]
            values = g + [max(abs(hj) - EQUALITY_TOLERANCE, 0.0) for hj in h]
            state = {"best": best.tolist(), "violations": values}
            self.frames.push(frame(progress, best_violation, middle, state))

    def write(self):
        """Writes the trace, if there's one."""
        if self.path:
            settings = {
                "format": 1,
                "example": "cec2006_g22",
                "objective": "minimize",
                "x_label": "evaluations",
                "y_label": "constraint violation of the best solution",
                "log_y": True,
                "optimum": None,
                "plot": "design",
                # the inequalities as g(x), the equalities as their excess over the tolerance:
                # 0 when met, which is active
                "problem": {
                    "variables": [
                        {"name": f"x{i}", "unit": "", "bounds": [float(low), float(high)]}
                        for i, (low, high) in enumerate(bounds(self.problem.genome), 1)
                    ],
                    "constraints": ["g1"] + [f"h{i}" for i in range(1, 20)],
                    "signed": True,
                },
            }
            write(self.path, settings, self.frames.to_list())


def bounds(genome):
    """Each gene's bounds: a pair of the genome's bounds for every gene when they're the same."""
    if isinstance(genome.bounds[0], (int, float)):
        return [genome.bounds] * genome.length
    return genome.bounds


def median(valid, size):
    """The median violation of a population of ``size`` in the order of Deb's rules, given the
    sorted violations of its valid solutions: None if the median is invalid."""
    middle = size // 2
    if size == 0 or middle >= len(valid):
        return None
    return valid[middle] if size % 2 else (valid[middle - 1] + valid[middle]) / 2


def frame(progress, best, middle, state):
    """The frame of a generation: its progress, the violations of the best and the median, and
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
