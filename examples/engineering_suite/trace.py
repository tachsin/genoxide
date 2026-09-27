"""The trace of the runs for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: each problem's error f - f* of the best feasible solution, every 1,000
evaluations up to 50,000, then every 9,000 up to 500,000. The Rust example writes the same file."""

import json
import math
import os


def frame_evaluations():
    """The evaluations of the frames: every 1,000 up to 50,000, where the engineering designs'
    runs end, then every 9,000 up to the CEC 2006 budget of 500,000."""
    early = [k * 1_000 for k in range(1, 51)]
    late = [50_000 + k * 9_000 for k in range(1, 51)]
    return early + late


class Trace:
    """Records the runs through ``on_generation`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self):
        self.path = os.environ.get("GENOXIDE_TRACE")
        # per problem: the evaluations and the error after each generation, None while no
        # solution is feasible
        self.series = {}

    def errors(self, name, problem, best_known):
        """The callback for ``run`` that records the run ``name``'s error to ``best_known`` after
        each generation: None without a trace to record."""
        history = self.series.setdefault(name, [])

        def record(progress):
            # under Deb's rules, the best is feasible once any solution is; a solution below a
            # best known value counts as an error of 0
            error = None
            if progress.best_fitness is not None:
                score, violation = problem(progress.best_genome)
                if violation == 0.0:
                    error = max(score - best_known, 0.0)
            history.append((progress.evaluations, error))

        return record if self.path else None

    def write(self):
        """Writes the trace, if there's one: the runs side by side, each run's error after its last
        generation within a frame's evaluations, and null from the frame after the run's end."""
        if not self.path:
            return
        evaluations = frame_evaluations()

        def frame(k):
            previous = evaluations[k - 1] if k else 0
            values = {}
            for name, history in self.series.items():
                end = history[-1][0] if history else 0
                done = [error for done, error in history if done <= evaluations[k]]
                values[name] = done[-1] if done and previous < end else None
            return {
                "generation": None,
                "evaluations": evaluations[k],
                "best": None,
                "median": None,
                "state": {"values": values},
            }

        frames = [frame(k) for k in range(len(evaluations))]
        settings = {
            "format": 1,
            "example": "engineering_suite",
            "objective": "minimize",
            "x_label": "evaluations",
            "y_label": "error f - f* of the best feasible solution",
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
