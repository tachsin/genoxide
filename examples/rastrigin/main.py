"""Rastrigin: minimize a real-valued function with many local minima, in 30 dimensions.

Compares CMA-ES with IPOP restarts (a population that doubles at each restart) and L-SHADE
(differential evolution with a population that shrinks over the budget). The global minimum is 0,
at the origin. The function is genoxide's problems.Rastrigin, which run evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace for the plot on the example's page, which
draws the population on the function's contour: that needs two dimensions, so the trace is of a
separate run of L-SHADE in 2 dimensions, made only then.

    python examples/rastrigin/main.py
"""

import json
import math
import os

import genoxide as gx

DIMENSIONS = 30
BUDGET = 1_000_000


# ---- the trace of the run, for the plot on the example's page -----------------------------------


class Trace:
    """A frame per recorded generation, at most ``most``: every ``every``-th generation, with
    ``every`` doubling whenever there are ``most``, and the last generation."""

    def __init__(self, most):
        self.most, self.every, self.frames, self.last = most, 1, [], None

    def record(self, progress, state):
        """The generation's progress, the median score of its population and the plot's
        ``state``."""
        frame = {
            "generation": progress.generation,
            "evaluations": progress.evaluations,
            "best": progress.best_fitness,
            "median": median(progress.scores),
            "state": state,
        }
        self.push(frame)

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


def median(scores):
    """The median of the valid scores, None without any."""
    scores = sorted(float(score) for score in scores if not math.isnan(score))
    middle = len(scores) // 2
    if not scores:
        return None
    return scores[middle] if len(scores) % 2 else (scores[middle - 1] + scores[middle]) / 2


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

def trace_2d(path):
    """L-SHADE in 2 dimensions, with a budget of 10,000 evaluations per dimension, recorded for
    the plot of the population on the contour."""
    problem = gx.problems.Rastrigin(2)
    budget = 20_000
    l_shade = gx.De(problem.genome, l_shade=budget, objective="minimize", seed=1)
    trace = Trace(200)

    def record(progress):
        state = {"population": progress.population.tolist(), "best": progress.best_genome.tolist()}
        trace.record(progress, state)

    l_shade.run(problem, target=1e-8, evaluations=budget, on_generation=record)
    trace.write(
        path,
        {
            "format": 1,
            "example": "rastrigin",
            "objective": "minimize",
            "x_label": "evaluations",
            "y_label": "error",
            "log_y": True,
            "optimum": 0.0,
            "plot": "contour",
            "problem": {
                "function": "rastrigin",
                "bounds": [[-5.12, 5.12], [-5.12, 5.12]],
                "minima": [[0.0, 0.0]],
            },
        },
    )


problem = gx.problems.Rastrigin(DIMENSIONS)
target = problem.optimum.value + 1e-8
for name, algorithm in (
    ("CMA-ES", gx.Cmaes(problem.genome, restarts="ipop", objective="minimize", seed=1)),
    ("L-SHADE", gx.De(problem.genome, l_shade=BUDGET, objective="minimize", seed=1)),
):
    result = algorithm.run(problem, target=target, evaluations=BUDGET)
    print(f"{name}: {result.best_fitness:.6f} after {result.evaluations} evaluations")

if "GENOXIDE_TRACE" in os.environ:
    trace_2d(os.environ["GENOXIDE_TRACE"])
