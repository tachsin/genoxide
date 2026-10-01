"""The trace of the runs for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: the simplex on the contour of Himmelblau's function, a frame per round,
with the points where the runs so far converged. The Rust example writes the same file."""

import json
import math
import os


class Trace:
    """Records the runs through ``on_generation`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self, minima):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.minima = minima
        self.frames = []
        # where the runs so far converged
        self.ends = []

    @property
    def on_generation(self):
        """The callback for ``run``: None without a trace to record."""
        return self.record if self.path else None

    def record(self, progress):
        """Records a round: the vertices of the simplex, best first, the best point so far, and
        where the runs before converged."""
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
                    "ends": list(self.ends),
                },
            }
        )

    def end(self, point):
        """Records where a run converged, shown from the next round on."""
        self.ends.append(point.tolist())

    def write(self):
        """Writes the trace, if there's one, with at most 300 of its frames."""
        if not self.path:
            return
        frames = Frames(300)
        for frame in self.frames:
            frames.push(frame)
        settings = {
            "format": 1,
            "example": "nelder_mead_himmelblau",
            "objective": "minimize",
            "x_label": "rounds",
            "y_label": "best value",
            "log_y": True,
            "optimum": 0.0,
            "plot": "contour",
            "problem": {
                "function": "himmelblau",
                "bounds": [[-5.0, 5.0], [-5.0, 5.0]],
                "minima": self.minima.tolist(),
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
