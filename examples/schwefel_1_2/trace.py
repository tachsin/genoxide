"""The trace for the plot on the example's page, written to the file that ``GENOXIDE_TRACE`` names.
The plot draws the population on the function's contour, which needs two dimensions: the trace is
of a separate run of CMA-ES in 2 dimensions, with a budget of 10,000 evaluations per dimension, in
at most 100 frames. The Rust example writes the same file."""

import json
import math
import os

import genoxide as gx


def record_2d():
    """Runs CMA-ES in 2 dimensions and writes its trace, if ``GENOXIDE_TRACE`` is set."""
    path = os.environ.get("GENOXIDE_TRACE")
    if not path:
        return
    problem = gx.problems.Schwefel1_2(2)
    budget = 20_000
    cmaes = gx.Cmaes(problem.genome, objective="minimize", seed=1)
    frames = Frames(100)

    def record(progress):
        state = {"population": progress.population.tolist(), "best": progress.best_genome.tolist()}
        frames.push(frame(progress, state))

    cmaes.run(problem, target=1e-8, evaluations=budget, on_generation=record)
    settings = {
        "format": 1,
        "example": "schwefel_1_2",
        "objective": "minimize",
        "x_label": "evaluations",
        "y_label": "error",
        "log_y": True,
        "optimum": 0.0,
        "plot": "contour",
        "problem": {
            "function": "schwefel_1_2",
            "bounds": [[-100.0, 100.0], [-100.0, 100.0]],
            "minima": [[0.0, 0.0]],
        },
    }
    write(path, settings, frames.to_list())


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
