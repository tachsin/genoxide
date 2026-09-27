"""The trace of the runs for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: the error of each function and algorithm after every 1,000 evaluations.
The Rust example writes the same file."""

import json
import math
import os

# the frames, evenly spaced over the budget
FRAMES = 100


class Trace:
    """Records the runs through ``on_generation`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self, budget):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.budget = budget
        # per function and algorithm: the evaluations and the error after each generation
        self.series = {}

    def errors(self, name, minimum):
        """The callback for ``run`` that records the run ``name``'s error to ``minimum`` after
        each generation: None without a trace to record."""
        history = self.series.setdefault(name, [])

        def record(progress):
            if progress.best_fitness is not None:
                history.append((progress.evaluations, max(progress.best_fitness - minimum, 0.0)))

        return record if self.path else None

    def write(self):
        """Writes the trace, if there's one: the runs side by side, a frame per 1,000 evaluations
        with each run's error after its last generation within them."""
        if not self.path:
            return

        def frame(evaluations):
            values = {}
            for name, history in self.series.items():
                done = [error for done, error in history if done <= evaluations]
                values[name] = done[-1] if done else None
            return {
                "generation": None,
                "evaluations": evaluations,
                "best": None,
                "median": None,
                "state": {"values": values},
            }

        frames = [frame(k * self.budget // FRAMES) for k in range(1, FRAMES + 1)]
        settings = {
            "format": 1,
            "example": "function_suite",
            "objective": "minimize",
            "x_label": "evaluations",
            "y_label": "error to the minimum",
            "log_y": True,
            "optimum": 0.0,
            "plot": "multi-curve",
            "problem": {"series": list(self.series)},
        }
        write(self.path, settings, frames)


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
