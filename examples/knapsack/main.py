"""0/1 knapsack: choose items with the highest total value that fit in the knapsack.

Shows a constraint with Deb's feasibility rules: the fitness function returns the value and how
far the weight exceeds the capacity, so overweight selections still guide the search towards the
feasible ones. The result is checked against the optimum found by dynamic programming.

    python examples/knapsack/main.py
"""

import json
import math
import os

import numpy as np

import genoxide as gx

# (weight, value)
ITEMS = [
    (23, 92),
    (31, 57),
    (29, 49),
    (44, 68),
    (53, 60),
    (38, 43),
    (63, 67),
    (85, 84),
    (89, 87),
    (82, 72),
    (12, 31),
    (17, 29),
    (41, 52),
    (35, 38),
    (27, 44),
    (58, 61),
    (19, 26),
    (46, 55),
    (71, 70),
    (33, 41),
]
CAPACITY = 400
WEIGHTS = np.array([weight for weight, _ in ITEMS])
VALUES = np.array([value for _, value in ITEMS])


def value(selection):
    """The total value, and how much the weight exceeds the capacity (0 if the items fit)."""
    weight = WEIGHTS[selection].sum()
    return float(VALUES[selection].sum()), float(max(0, weight - CAPACITY))


def optimum():
    """The best value that fits, by dynamic programming over the capacities."""
    best = [0] * (CAPACITY + 1)
    for weight, value in ITEMS:
        for capacity in range(CAPACITY, weight - 1, -1):
            best[capacity] = max(best[capacity], best[capacity - weight] + value)
    return best[CAPACITY]


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
    gx.Binary(len(ITEMS)),
    population_size=60,
    select=gx.Tournament(3),
    crossover=gx.PointCrossover(2),
    mutation=gx.BitFlip(rate=1 / len(ITEMS)),
    seed=7,
)
trace = Trace(200) if "GENOXIDE_TRACE" in os.environ else None


def record(progress):
    if trace:
        trace.record(progress, {"best": progress.best_genome.astype(int).tolist()})


result = ga.run(value, stagnation=200, generations=2_000, on_generation=record)

best = result.best_genome
print(f"items {np.flatnonzero(best).tolist()}")
print(f"value {VALUES[best].sum()}, weight {WEIGHTS[best].sum()} of {CAPACITY}")
print(f"after {result.evaluations} evaluations; the optimum is {optimum()}")
if trace:
    trace.write(
        os.environ["GENOXIDE_TRACE"],
        {
            "format": 1,
            "example": "knapsack",
            "objective": "maximize",
            "x_label": "generations",
            "y_label": "value",
            "log_y": False,
            "optimum": float(optimum()),
            "plot": "knapsack",
            "problem": {
                "capacity": CAPACITY,
                "items": [{"weight": weight, "value": value} for weight, value in ITEMS],
            },
        },
    )
