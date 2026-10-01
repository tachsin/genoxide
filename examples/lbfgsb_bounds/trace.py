"""The trace of the run for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: the iterates of L-BFGS-B so far on the contour of Rosenbrock's function
in the box, a frame per round. The Rust example writes the same file."""

import json
import math
import os


class Trace:
    """Records the run through ``on_generation`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.frames = []
        # the points the run has stood at, in order
        self.iterates = []

    def record(self, progress):
        """Records a round: every point the run has stood at so far, and the best one."""
        if not self.path:
            return
        point = progress.population[0].tolist()
        if not self.iterates or self.iterates[-1] != point:
            self.iterates.append(point)
        self.frames.append(
            {
                "generation": progress.generation,
                "evaluations": progress.evaluations,
                "best": progress.best_fitness,
                "state": {
                    "population": list(self.iterates),
                    "best": progress.best_genome.tolist(),
                },
            }
        )

    def write(self):
        """Writes the trace, if there's one."""
        if not self.path:
            return
        settings = {
            "format": 1,
            "example": "lbfgsb_bounds",
            "objective": "minimize",
            "x_label": "rounds",
            "y_label": "best value",
            "log_y": False,
            "optimum": 0.25,
            "plot": "contour",
            "problem": {
                "function": "rosenbrock",
                "bounds": [[-2.0, 0.5], [-1.0, 3.0]],
                "minima": [[0.5, 0.25]],
                "minima_label": "minimum in the box",
                "population_label": "iterates",
            },
        }
        write(self.path, settings, self.frames)


# ---- the same in every example's trace ----------------------------------------------------------


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
