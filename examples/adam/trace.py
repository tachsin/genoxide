"""The trace of the runs for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: the distance to the answer and the learning rate at every step, of the
run with the schedule and of the run with a constant learning rate. The Rust example writes the
same file."""

import json
import math
import os

# the lines of the plot: a panel per quantity, a line per run
SCHEDULED = "halved every 500 steps"
CONSTANT = "constant"
QUANTITIES = ("distance to the answer", "learning rate")


class Trace:
    """Records the runs' steps through ``control`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.runs = {SCHEDULED: [], CONSTANT: []}

    def scheduled(self, step, distance, rate):
        """Records a step of the run with the schedule."""
        self._record(SCHEDULED, step, distance, rate)

    def constant(self, step, distance, rate):
        """Records a step of the run with a constant learning rate."""
        self._record(CONSTANT, step, distance, rate)

    def _record(self, run, step, distance, rate):
        if self.path:
            assert len(self.runs[run]) == step
            self.runs[run].append((distance, rate))

    def write(self):
        """Writes the trace, if there's one, with at most 200 of its frames."""
        if not self.path:
            return
        frames = Frames(200)
        for step, (scheduled, constant) in enumerate(zip(self.runs[SCHEDULED], self.runs[CONSTANT])):
            values = {}
            for k, quantity in enumerate(QUANTITIES):
                values[f"{quantity}/{SCHEDULED}"] = scheduled[k]
                values[f"{quantity}/{CONSTANT}"] = constant[k]
            frames.push(
                {
                    "generation": step,
                    "evaluations": step + 1,
                    "best": scheduled[0],
                    "state": {"values": values},
                }
            )
        settings = {
            "format": 1,
            "example": "adam",
            "objective": "minimize",
            "x_label": "steps",
            "y_label": "distance to the answer",
            "log_y": True,
            "optimum": 0.0,
            "plot": "multi-curve",
            "problem": {
                "series": [f"{q}/{line}" for q in QUANTITIES for line in (SCHEDULED, CONSTANT)]
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
