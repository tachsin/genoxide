"""The trace of the searches for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: their points side by side on the contour, in at most 200 of their
generations. The Rust example writes the same file."""

import json
import math
import os


class Trace:
    """Records the searches through ``on_generation`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self, minima):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.minima = minima
        # per search, after each generation: its evaluations, and its best value and point
        self.searches = []

    @property
    def on_generation(self):
        """The callback for ``run``: None without a trace to record."""
        return self.record if self.path else None

    def record(self, progress):
        """Records a generation of a search, which starts at generation 0: its evaluations, and
        its best value and point so far."""
        if progress.generation == 0:
            self.searches.append([])
        point = progress.best_genome.tolist()
        self.searches[-1].append((progress.evaluations, progress.best_fitness, point))

    def write(self):
        """Writes the trace, if there's one: the searches side by side, a frame per generation,
        with their points, the best value and the median."""
        if not self.path:
            return
        frames = Frames(200)
        for generation in range(max(map(len, self.searches), default=0)):
            searches = [search[min(generation, len(search) - 1)] for search in self.searches]
            valid = [search for search in searches if search[1] is not None]
            best = min(valid, key=lambda search: search[1], default=None)
            frames.push(
                {
                    "generation": generation,
                    "evaluations": sum(search[0] for search in searches),
                    "best": best[1] if best else None,
                    "median": median([search[1] for search in valid]),
                    "state": {
                        "population": [search[2] for search in searches],
                        "best": best[2] if best else None,
                    },
                }
            )
        settings = {
            "format": 1,
            "example": "himmelblau",
            "objective": "minimize",
            "x_label": "generations",
            "y_label": "best value",
            "log_y": True,
            "optimum": 0.0,
            "plot": "contour",
            "problem": {
                "function": "himmelblau",
                "bounds": [[-5.0, 5.0], [-5.0, 5.0]],
                "minima": self.minima.tolist(),
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
