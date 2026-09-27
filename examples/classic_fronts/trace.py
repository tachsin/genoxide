"""The trace of the runs for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: a panel per problem with its front, and each front's normalized
hypervolume, in at most 50 generations. A frame's evaluations are the four runs' together. The
Rust example writes the same file."""

import json
import math
import os


class Trace:
    """Records the runs through ``on_generation`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self):
        self.path = os.environ.get("GENOXIDE_TRACE")
        # per problem: its optimal front if it's known, and after each generation the
        # evaluations, the front and its normalized hypervolume
        self.optimal = {}
        self.series = {}

    def fronts(self, name, optimal, volume):
        """The callback for ``run`` that records the run of problem ``name``, whose optimal front
        is ``optimal``: its front and the front's ``volume`` after each generation. None without a
        trace to record."""
        self.optimal[name] = optimal
        history = self.series.setdefault(name, [])

        def record(progress):
            front = progress.front_objectives
            history.append((progress.evaluations, front.tolist(), volume(front)))

        return record if self.path else None

    def write(self):
        """Writes the trace, if there's one: the runs side by side, a frame per generation."""
        if not self.path:
            return
        frames = Frames(50)
        for generation in range(max(map(len, self.series.values()), default=0)):
            at = {name: runs[min(generation, len(runs) - 1)] for name, runs in self.series.items()}
            frames.push(
                {
                    "generation": generation,
                    "evaluations": sum(evaluations for evaluations, _, _ in at.values()),
                    "series": {name: volume for name, (_, _, volume) in at.items()},
                    "state": {
                        "panels": [{"fronts": {"NSGA-II": front}} for _, front, _ in at.values()]
                    },
                }
            )
        settings = {
            "format": 1,
            "example": "classic_fronts",
            "objective": ["minimize", "minimize"],
            "x_label": "generations",
            "y_label": "normalized hypervolume",
            "log_y": False,
            "optimum": None,
            "plot": "grid",
            "problem": {
                "panel_plot": "front-2d",
                "panels": [panel(name, optimal) for name, optimal in self.optimal.items()],
                "series": list(self.series),
            },
        }
        write(self.path, settings, frames.to_list())


def panel(name, optimal):
    """The panel of problem ``name``, with its optimal front if it's known."""
    problem = {"objectives": ["f1", "f2"], "series": ["NSGA-II"]}
    if optimal is not None:
        problem["true_front"] = optimal.tolist()
    return {"title": name, "problem": problem}


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
