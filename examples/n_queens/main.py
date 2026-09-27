"""N-Queens: place N queens on an N×N board so that no two attack each other.

A permutation genome puts one queen in each row and each column (queen ``row`` is in column
``order[row]``), so only the diagonals can conflict. Permutations have no position-wise crossover,
so this uses (μ+λ) with swap mutation only. The fitness function takes a generation at a time.

With ``GENOXIDE_TRACE=<file>``, it also writes the run's trace for the plot on the example's page:
the best board so far and its attacking pairs, in at most 200 generations.

    python examples/n_queens/main.py
"""

import json
import math
import os

import numpy as np

import genoxide as gx

N = 64
ROWS = np.arange(N)


def conflicts(orders):
    """The number of pairs of queens on the same diagonal, for a genome per row."""
    # a separate range of 2N counters for each genome
    offsets = 2 * N * np.arange(len(orders))[:, None]
    pairs = 0
    for diagonal in (ROWS + N - orders, ROWS + orders):
        counts = np.bincount((diagonal + offsets).ravel(), minlength=2 * N * len(orders))
        pairs = pairs + (counts * (counts - 1) // 2).reshape(len(orders), 2 * N).sum(axis=1)
    return pairs


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

ga = gx.Ga(
    gx.Permutation(N),
    population_size=20,
    select=gx.Tournament(2),
    crossover=gx.NoCrossover(),
    mutation=gx.SwapMutation(),
    scheme=gx.MuPlusLambda(20),
    objective="minimize",
    seed=1,
)


def attacks(order):
    """The pairs of rows whose queens attack each other, on a diagonal."""
    return [[a, b] for a in range(N) for b in range(a + 1, N) if abs(order[a] - order[b]) == b - a]


trace = Trace(200) if "GENOXIDE_TRACE" in os.environ else None


def record(progress):
    if trace:
        best = progress.best_genome.tolist()
        trace.record(progress, {"best": best, "attacks": attacks(best)})


result = ga.run(conflicts, batch=True, target=0, generations=50_000, on_generation=record)

print(
    f"{result.best_fitness:.0f} conflicts after {result.generations} generations and "
    f"{result.evaluations} evaluations"
)
print(f"columns {result.best_genome.tolist()}")
if trace:
    trace.write(
        os.environ["GENOXIDE_TRACE"],
        {
            "format": 1,
            "example": "n_queens",
            "objective": "minimize",
            "x_label": "generations",
            "y_label": "conflicts",
            "log_y": False,
            "optimum": 0.0,
            "plot": "board",
            "problem": {"n": N},
        },
    )
