"""The trace of CMA-ES's runs for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: the best point of each run side by side on the contour, and the best
value's and the median's error to the minimum, in at most 100 of their generations. The Rust
example writes the same file."""

import json
import math
import os


class Trace:
    """Records the runs through ``on_generation`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self, problem):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.bounds = [[float(low), float(high)] for low, high in [problem.genome.bounds] * 2]
        self.optimum = problem.optimum
        # per run, after each generation: its evaluations, and its best value and point
        self.runs = []

    @property
    def on_generation(self):
        """The callback for ``run``: None without a trace to record."""
        return self.record if self.path else None

    def record(self, progress):
        """Records a generation of a run, which starts at generation 0: its evaluations, and
        its best value and point so far."""
        if progress.generation == 0:
            self.runs.append([])
        point = progress.best_genome.tolist()
        self.runs[-1].append((progress.evaluations, progress.best_fitness, point))

    def write(self):
        """Writes the trace, if there's one: the runs side by side, a frame per generation,
        with their points, and the best value's and the median's error to the minimum."""
        if not self.path:
            return
        minimum = self.optimum.value

        def error(value):
            # rounding can put a solution a few ulps below the minimum
            return None if value is None else max(value - minimum, 0.0)

        frames = Frames(100)
        for generation in range(max(map(len, self.runs), default=0)):
            runs = [run[min(generation, len(run) - 1)] for run in self.runs]
            valid = [run for run in runs if run[1] is not None]
            best = min(valid, key=lambda run: run[1], default=None)
            frames.push(
                {
                    "generation": generation,
                    "evaluations": sum(run[0] for run in runs),
                    "best": error(best[1]) if best else None,
                    "median": error(median([run[1] for run in valid])),
                    "state": {
                        "population": [run[2] for run in runs],
                        "best": best[2] if best else None,
                    },
                }
            )
        settings = {
            "format": 1,
            "example": "matyas",
            "objective": "minimize",
            "x_label": "generations",
            "y_label": "error to the minimum",
            "log_y": True,
            "optimum": 0.0,
            "plot": "contour",
            "problem": {
                "function": "matyas",
                "bounds": self.bounds,
                "minima": self.optimum.solutions.tolist(),
            },
        }
        write(self.path, settings, frames.to_list())


def median(values):
    """The median of the valid values, None without any."""
    values = sorted(float(value) for value in values if not math.isnan(value))
    middle = len(values) // 2
    if not values:
        return None
    return values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


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
