"""The trace of the run for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: the best design so far and its constraint violations, in at most 200
generations. The Rust example writes the same file."""

import json
import math
import os


class Trace:
    """Records the run through ``on_generation`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self, problem):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.frames = Frames(200)
        self.problem = problem

    @property
    def on_generation(self):
        """The callback for ``run``: None without a trace to record."""
        return self.record if self.path else None

    def record(self, progress):
        """Records a generation: the best design so far and its constraint violations."""
        if self.path:
            best = progress.best_genome
            violations = [max(g, 0.0) for g in self.problem.constraints(best).tolist()]
            state = {"best": self.problem.design(best).tolist(), "violations": violations}
            self.frames.push(frame(progress, state))

    def write(self):
        """Writes the trace, if there's one."""
        if self.path:
            plates, inches = [0.0625, 6.1875], [10.0, 200.0]
            settings = {
                "format": 1,
                "example": "pressure_vessel",
                "objective": "minimize",
                "x_label": "evaluations",
                "y_label": "cost",
                "log_y": False,
                "optimum": self.problem.optimum.value,
                "plot": "design",
                "problem": {
                    "variables": [
                        {"name": name, "unit": "in", "bounds": plates if name[0] == "T" else inches}
                        for name in ("Ts", "Th", "R", "L")
                    ],
                    "constraints": ["g1", "g2", "g3", "g4"],
                },
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
    """The frame of a generation: its progress, the median score of its population and
    ``state``."""
    return {
        "generation": progress.generation,
        "evaluations": progress.evaluations,
        "best": progress.best_fitness,
        "median": median(progress.scores),
        "state": state,
    }


def median(scores):
    """The median of the valid scores, None without any."""
    scores = sorted(float(score) for score in scores if not math.isnan(score))
    middle = len(scores) // 2
    if not scores:
        return None
    return scores[middle] if len(scores) % 2 else (scores[middle - 1] + scores[middle]) / 2


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
