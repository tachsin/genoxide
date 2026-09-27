"""The trace of the runs for the plot on the example's page, written to the file that
``GENOXIDE_TRACE`` names: a panel per problem with its front, the infeasible solutions of the
population and its feasible share, and a curve of each front's normalized hypervolume, in at most
50 generations. The Rust example writes the same file."""

import json
import math
import os


class Trace:
    """Records the runs through ``on_generation`` when ``GENOXIDE_TRACE`` is set."""

    def __init__(self, normalized_hypervolume):
        self.path = os.environ.get("GENOXIDE_TRACE")
        self.normalized_hypervolume = normalized_hypervolume
        # per problem, the frames of its run; the runs are as long, so each keeps the same
        # generations
        self.runs = []

    def panel(self, problem):
        """The callback for ``run`` on ``problem``: None without a trace to record."""
        if not self.path:
            return None
        frames = Frames(50)
        self.runs.append(frames)

        def record(progress):
            """Records a generation: the front, the population's infeasible solutions and
            feasible share, and the feasible front's normalized hypervolume."""
            front = progress.front_objectives
            feasible_front = front[progress.front_violations == 0]
            feasible = progress.violations == 0
            state = {
                "fronts": {"NSGA-II": front.tolist()},
                "infeasible": {"NSGA-II": progress.objectives[~feasible].tolist()},
                "feasible": {"NSGA-II": feasible.sum() / len(feasible)},
            }
            frames.push(
                {
                    "generation": progress.generation,
                    "evaluations": progress.evaluations,
                    "hypervolume": self.normalized_hypervolume(problem, feasible_front),
                    "state": state,
                }
            )

        return record

    def write(self, problems):
        """Writes the trace, if there's one: a frame per recorded generation, with each
        problem's normalized hypervolume and panel."""
        if not self.path:
            return
        names = [problem.name for problem in problems]
        runs = [frames.to_list() for frames in self.runs]
        frames = [
            {
                "generation": runs[0][k]["generation"],
                # the four runs' together
                "evaluations": sum(run[k]["evaluations"] for run in runs),
                "series": {name: run[k]["hypervolume"] for name, run in zip(names, runs)},
                "state": {"panels": [run[k]["state"] for run in runs]},
            }
            for k in range(len(runs[0]))
        ]
        panels = [
            {
                "title": problem.name,
                "problem": {
                    "objectives": ["f1", "f2"],
                    "true_front": problem.optimal_front(100).tolist(),
                    "series": ["NSGA-II"],
                },
            }
            for problem in problems
        ]
        settings = {
            "format": 1,
            "example": "constrained_fronts",
            "objective": ["minimize", "minimize"],
            "x_label": "generations",
            "y_label": "normalized hypervolume",
            "log_y": False,
            "optimum": None,
            "plot": "grid",
            "problem": {"panel_plot": "front-2d", "panels": panels, "series": names},
        }
        write(self.path, settings, frames)


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
