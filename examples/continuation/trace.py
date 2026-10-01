"""The trace of the runs for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: the distance to the global minimum and the stage's σ at every round, of
the run through the stages, of σ = 0 from the start, and of the stages with L-BFGS-B's pairs kept.
The Rust example writes the same file."""

import json
import math
import os

# the lines of the plot: a panel per quantity, a line per run
RUNS = ("stages", "σ = 0 from the start", "pairs kept")
QUANTITIES = ("distance to the global minimum", "σ")


class Trace:
    """Records the runs' steps through ``control`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.runs = ([], [], [])

    def record(self, run, step, distance, sigma):
        """Records a round of run ``run``."""
        if self.path:
            assert len(self.runs[run]) == step
            self.runs[run].append((distance, sigma))

    def write(self):
        """Writes the trace, if there's one, with at most 200 of its frames."""
        if not self.path:
            return
        steps = max(len(run) for run in self.runs)
        frames = Frames(200)
        for step in range(steps):
            values = {}
            for run, name in zip(self.runs, RUNS):
                at = run[step] if step < len(run) else None
                values[f"{QUANTITIES[0]}/{name}"] = None if at is None else at[0]
                values[f"{QUANTITIES[1]}/{name}"] = None if at is None else at[1]
            kept = self.runs[0][step][0] if step < len(self.runs[0]) else None
            frames.push(
                {
                    "generation": step,
                    "evaluations": None,
                    "best": kept,
                    "state": {"values": values},
                }
            )
        settings = {
            "format": 1,
            "example": "continuation",
            "objective": "minimize",
            "x_label": "rounds",
            "y_label": "distance to the global minimum",
            "log_y": True,
            "optimum": 0.0,
            "plot": "multi-curve",
            "problem": {"series": [f"{q}/{run}" for q in QUANTITIES for run in RUNS]},
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
