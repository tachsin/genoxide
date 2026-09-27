"""Kursawe: minimize two objectives whose Pareto front is in disconnected pieces, with SPEA2 and
NSGA-II.

Kursawe's problem in 3 variables, from genoxide's problems.Kursawe; run evaluates it in Rust. Its
front isn't known in closed form, so the example compares the two algorithms' fronts by their
hypervolume, and counts the pieces each finds: a new piece starts where two neighbors on the
front are more than 0.5 apart.

With ``GENOXIDE_TRACE=<file>``, it also writes the runs' trace for the plot on the example's page:
both fronts and their hypervolumes, in at most 64 generations. A frame's evaluations are the two
runs' together.

    python examples/kursawe/main.py
"""

import json
import math
import os

import numpy as np

import genoxide as gx

REFERENCE = [-14.0, 1.0]


# ---- the trace of the runs, for the plot on the example's page ----------------------------------


class Trace:
    """A frame per recorded generation, at most ``most``: every ``every``-th generation, with
    ``every`` doubling whenever there are ``most``, and the last generation."""

    def __init__(self, most):
        self.most, self.every, self.frames, self.last = most, 1, [], None

    def push(self, frame):
        """Keeps ``frame`` if it's of the ``every``-th generation, or as the last one."""
        if frame["generation"] % self.every:
            self.last = frame
            return
        self.frames.append(frame)
        self.last = None
        if len(self.frames) == self.most:
            self.every *= 2
            self.frames = [frame for frame in self.frames if frame["generation"] % self.every == 0]

    def write(self, path, settings):
        """Writes the settings and the frames to ``path``, a frame per line."""
        frames = ",\n".join(to_json(frame) for frame in self.frames + [self.last] if frame)
        with open(path, "w", encoding="utf-8", newline="\n") as file:
            file.write(f'{to_json(settings)[:-1]},"frames":[\n{frames}\n]}}\n')


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


# -------------------------------------------------------------------------------------------------

def report(name, front):
    """Prints the size of the front, its pieces and its hypervolume."""
    ordered = front[np.argsort(front[:, 0], kind="stable")]
    gaps = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    pieces = 1 + int((gaps > 0.5).sum())
    volume = gx.indicators.hypervolume(front, REFERENCE)
    print(f"{name:<8} {len(front)} solutions in {pieces} pieces, hypervolume {volume:.4f}")


problem = gx.problems.Kursawe(3)
settings = dict(
    objectives=problem.objectives,
    population_size=100,
    crossover=gx.SimulatedBinaryCrossover(15),
    mutation=gx.PolynomialMutation(20, rate=1 / 3),
    seed=1,
)
trace = "GENOXIDE_TRACE" in os.environ
# per algorithm, for the trace: the evaluations, the front and its hypervolume after each
# generation
histories = {"SPEA2": [], "NSGA-II": []}


def fronts(history):
    """Records the front after each generation, for the trace."""

    def record(progress):
        if trace:
            front = progress.front_objectives
            volume = gx.indicators.hypervolume(front, REFERENCE)
            history.append((progress.evaluations, front.tolist(), volume))

    return record


def write_trace(path, histories):
    """The runs side by side, a frame per generation."""
    trace = Trace(64)
    for generation in range(max(map(len, histories.values()), default=0)):
        at = {name: runs[min(generation, len(runs) - 1)] for name, runs in histories.items()}
        frame = {
            "generation": generation,
            "evaluations": sum(evaluations for evaluations, _, _ in at.values()),
            "best": None,
            "median": None,
            "state": {
                "fronts": {name: front for name, (_, front, _) in at.items()},
                "hypervolume": {name: volume for name, (_, _, volume) in at.items()},
            },
        }
        trace.push(frame)
    trace.write(
        path,
        {
            "format": 1,
            "example": "kursawe",
            "objective": ["minimize", "minimize"],
            "x_label": "generations",
            "y_label": "hypervolume",
            "log_y": False,
            "optimum": None,
            "plot": "front-2d",
            "problem": {
                "objectives": ["f1", "f2"],
                "true_front": None,
                "series": list(histories),
            },
        },
    )


spea2 = gx.Spea2(problem.genome, **settings)
result = spea2.run(problem, generations=250, on_generation=fronts(histories["SPEA2"]))
report("SPEA2", result.front_objectives)
nsga2 = gx.Nsga2(problem.genome, **settings)
result = nsga2.run(problem, generations=250, on_generation=fronts(histories["NSGA-II"]))
report("NSGA-II", result.front_objectives)
if trace:
    write_trace(os.environ["GENOXIDE_TRACE"], histories)
