"""Gear train design (Sandgren, 1990): the numbers of teeth of a compound gear train of four gears,
from 12 to 60 each, whose ratio is closest to 1/6.931. An integer problem.

The problem is genoxide's ``GearTrain``, on integer genes; its score is the squared error of the
ratio. A genetic algorithm with uniform crossover and a mutation that redraws each gene with
probability 0.25 searches the 49⁴ ≈ 5.8 million designs, until it reaches the minimum,
2.700857e-12, known by evaluating them all.

With ``GENOXIDE_TRACE=<file>``, it also writes the run's trace for the plot on the example's page:
the best design so far, in at most 200 generations.

    python examples/gear_train/main.py
"""

import json
import math
import os

import genoxide as gx

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

problem = gx.problems.engineering.GearTrain()
minimum = problem.optimum.value
ga = gx.Ga(
    problem.genome,
    objective=problem.objective,
    population_size=100,
    select=gx.Tournament(2),
    crossover=gx.UniformCrossover(),
    mutation=gx.UniformMutation(rate=0.25),
    seed=1,
)
trace = Trace(200) if "GENOXIDE_TRACE" in os.environ else None


def record(progress):
    if trace:
        # no constraints: no violations
        trace.record(progress, {"best": progress.best_genome.tolist(), "violations": []})


result = ga.run(problem, target=minimum, generations=2_000, on_generation=record)

teeth = result.best_genome.tolist()
ratio = teeth[0] * teeth[1] / (teeth[2] * teeth[3])
print(
    f"error {result.best_fitness:.6e} after {result.generations} generations "
    f"(the minimum: {minimum:.6e})"
)
print(f"teeth {tuple(teeth)}, ratio {ratio:.8f} (the target: {1 / 6.931:.8f})")
if trace:
    trace.write(
        os.environ["GENOXIDE_TRACE"],
        {
            "format": 1,
            "example": "gear_train",
            "objective": "minimize",
            "x_label": "generations",
            "y_label": "squared error of the ratio",
            "log_y": True,
            "optimum": minimum,
            "plot": "design",
            "problem": {
                "variables": [
                    {"name": name, "unit": "teeth", "bounds": [12, 60]}
                    for name in ("Td", "Tb", "Ta", "Tf")
                ],
                "constraints": [],
            },
        },
    )
