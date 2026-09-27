"""Himmelblau: find the four global minima of a two-dimensional function by restarting a local
search from random points.

Each search is a hill climber with Gaussian steps; it ends in the minimum whose basin it started
in. The known minima come from genoxide's problems.Himmelblau.

With ``GENOXIDE_TRACE=<file>``, it also writes the searches' trace for the plot on the example's
page: their points side by side on the contour, in at most 200 of their generations.

    python examples/himmelblau/main.py
"""


import json
import math
import os

import numpy as np

import genoxide as gx

SEARCHES = 20


def scientific(value):
    """Two significant digits, e.g. 1.2e-7."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


# ---- the trace of the run, for the plot on the example's page -----------------------------------


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

problem = gx.problems.Himmelblau()
minima = problem.optimum.solutions
# per known minimum: the searches that ended nearest it, and the best and worst values they
# reached
found = [[0, math.inf, 0.0] for _ in minima]
trace = "GENOXIDE_TRACE" in os.environ
# per search, for the trace: its evaluations, best value and point after each generation
histories = []
for seed in range(1, SEARCHES + 1):
    history = []

    def record(progress):
        if trace:
            point = progress.best_genome.tolist()
            history.append((progress.evaluations, progress.best_fitness, point))

    search = gx.LocalSearch(
        problem.genome,
        neighbor=gx.GaussianMutation(0.001, rate=1.0),
        neighbors=10,
        acceptance=gx.Improving(),
        objective="minimize",
        seed=seed,
    )
    result = search.run(problem, generations=1_000, on_generation=record)
    histories.append(history)
    nearest = int(np.argmin(np.linalg.norm(minima - result.best_genome, axis=1)))
    found[nearest][0] += 1
    found[nearest][1] = min(found[nearest][1], result.best_fitness)
    found[nearest][2] = max(found[nearest][2], result.best_fitness)

print(f"{SEARCHES} local searches from random points in [-5, 5] x [-5, 5]")
print("minimum                  searches  values reached")
for minimum, (searches, best, worst) in zip(minima, found):
    print(
        f"({minimum[0]:>9.6f}, {minimum[1]:>9.6f})  {searches:>8}  "
        f"{scientific(best)} to {scientific(worst)}"
    )


def write_trace(path, histories, minima):
    """The searches side by side, a frame per generation: their points, the best value and the
    median."""
    trace = Trace(200)
    for generation in range(max(map(len, histories), default=0)):
        searches = [history[min(generation, len(history) - 1)] for history in histories]
        valid = [search for search in searches if search[1] is not None]
        best = min(valid, key=lambda search: search[1], default=None)
        frame = {
            "generation": generation,
            "evaluations": sum(search[0] for search in searches),
            "best": best[1] if best else None,
            "median": median([search[1] for search in valid]),
            "state": {
                "population": [search[2] for search in searches],
                "best": best[2] if best else None,
            },
        }
        trace.push(frame)
    trace.write(
        path,
        {
            "format": 1,
            "example": "himmelblau",
            "objective": "minimize",
            "x_label": "generations",
            "y_label": "best value",
            "log_y": True,
            "optimum": 0.0,
            "plot": "contour",
            "problem": {
                "function": "himmelblau",
                "bounds": [[-5.0, 5.0], [-5.0, 5.0]],
                "minima": minima.tolist(),
            },
        },
    )


if trace:
    write_trace(os.environ["GENOXIDE_TRACE"], histories, minima)
