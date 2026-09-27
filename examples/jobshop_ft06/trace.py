"""The trace of the run for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: the best schedule so far, in at most 200 generations. The Rust example
writes the same file."""

import json
import math
import os


class Trace:
    """Records the run through ``on_generation`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self, instance, machines, optimum, schedule):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.frames = Frames(200)
        self.instance, self.machines, self.optimum = instance, machines, optimum
        self.schedule = schedule

    @property
    def on_generation(self):
        """The callback for ``run``: None without a trace to record."""
        return self.record if self.path else None

    def record(self, progress):
        """Records a generation: the best schedule so far."""
        if self.path:
            self.frames.push(frame(progress, {"best": self.operations(progress.best_genome)}))

    def write(self):
        """Writes the trace, if there's one."""
        if self.path:
            settings = {
                "format": 1,
                "example": "jobshop_ft06",
                "objective": "minimize",
                "x_label": "generations",
                "y_label": "makespan",
                "log_y": False,
                "optimum": float(self.optimum),
                "plot": "gantt",
                "problem": {"machines": self.machines, "jobs": self.instance},
            }
            write(self.path, settings, self.frames.to_list())

    def operations(self, order):
        """The operations of the schedule of ``order``: [job, operation, machine, start, end]."""
        starts = self.schedule(order)
        return [
            [job, step, machine, starts[job][step], starts[job][step] + duration]
            for job, steps in enumerate(self.instance)
            for step, (machine, duration) in enumerate(steps)
        ]


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
