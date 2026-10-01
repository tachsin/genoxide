"""The trace of the run for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: the simplex on the contour of Rosenbrock's function, a frame per round.
The Rust example writes the same file."""

import json
import math
import os


class Trace:
    """Records the run through ``on_generation`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.frames = []

    def record(self, progress):
        """Records a round: the vertices of the simplex, best first, and the best point so far."""
        if not self.path:
            return
        vertices = progress.population.tolist()
        self.frames.append(
            {
                "generation": progress.generation,
                "evaluations": progress.evaluations,
                "best": progress.best_fitness,
                "median": median(progress.scores),
                "state": {
                    "population": vertices,
                    "simplex": vertices,
                    "best": progress.best_genome.tolist(),
                },
            }
        )

    def write(self):
        """Writes the trace, if there's one, with at most 200 of its frames."""
        if not self.path:
            return
        frames = Frames(200)
        for frame in self.frames:
            frames.push(frame)
        settings = {
            "format": 1,
            "example": "nelder_mead",
            "objective": "minimize",
            "x_label": "rounds",
            "y_label": "best value",
            "log_y": True,
            "optimum": 0.0,
            "plot": "contour",
            "problem": {
                "function": "rosenbrock",
                "bounds": [[-2.0, 2.0], [-1.0, 3.0]],
                "minima": [[1.0, 1.0]],
                "population_label": "simplex",
            },
        }
        write(self.path, settings, frames.to_list())


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
