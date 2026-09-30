"""The trace of a run of the particle swarm with a ring topology in 10 dimensions for the plot on
the example's page, written to the file that ``GENOXIDE_TRACE`` names: the best point so far,
each gene on its range with the minimum's value marked, and the error of the best and of the
population's median to the minimum, in at most 100 generations. The Rust example writes the same
file."""

import json
import math
import os


class Trace:
    """Records a run through ``on_generation`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self, problem):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.problem = problem
        self.minimum = problem.optimum.value
        self.frames = Frames(100)

    @property
    def on_generation(self):
        """The callback for ``run``: None without a trace to record."""
        return self.record if self.path else None

    def error(self, value):
        """The error to the minimum: rounding can put a solution a few ulps below it."""
        return None if value is None else max(value - self.minimum, 0.0)

    def record(self, progress):
        """Records a generation: the errors of the best so far and of the population's median,
        and the best point so far."""
        errors = sorted(self.error(float(score)) for score in progress.scores if score == score)
        self.frames.push(
            {
                "generation": progress.generation,
                "evaluations": progress.evaluations,
                "best": self.error(progress.best_fitness),
                "median": median(errors),
                "state": {"best": progress.best_genome.tolist()},
            }
        )

    def write(self):
        """Writes the trace, if there's one."""
        if not self.path:
            return
        low, high = self.problem.genome.bounds
        minimum = self.problem.optimum.solutions[0]
        variables = [
            {"name": f"x{i + 1}", "unit": "", "bounds": [low, high], "optimum": float(value)}
            for i, value in enumerate(minimum)
        ]
        settings = {
            "format": 1,
            "example": "dixon_price",
            "objective": "minimize",
            "x_label": "evaluations",
            "y_label": "error to the minimum",
            "log_y": True,
            "optimum": 0.0,
            "plot": "design",
            "problem": {
                "variables": variables,
                "constraints": [],
                "optimum_label": "minimum",
            },
        }
        write(self.path, settings, self.frames.to_list())


def median(errors):
    """The median of sorted errors, None without any."""
    middle = len(errors) // 2
    if not errors:
        return None
    return errors[middle] if len(errors) % 2 else (errors[middle - 1] + errors[middle]) / 2


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
