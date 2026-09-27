"""OneMax: find the bit string with the most ones.

The "hello world" of genetic algorithms: a binary genome, tournament selection, uniform crossover
and bit-flip mutation, with the best count printed every 50 generations.

With ``GENOXIDE_TRACE=<file>``, it also writes the run's trace for the plot on the example's page:
the first 16 genomes of the population, in at most 32 generations.

    python examples/one_max/main.py
"""

import json
import math
import os

import genoxide as gx

LEN = 500

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
    gx.Binary(LEN),
    population_size=100,
    select=gx.Tournament(3),
    crossover=gx.UniformCrossover(),
    mutation=gx.BitFlip(rate=1 / LEN),
    seed=42,
)

trace = Trace(32) if "GENOXIDE_TRACE" in os.environ else None


def progress(progress):
    if progress.generation % 50 == 0:
        print(f"{progress.generation:>10}  {progress.best_fitness:>4.0f}")
    if trace:
        rows = ["".join("1" if one else "0" for one in row) for row in progress.population[:16]]
        trace.record(progress, {"population": rows})


print("generation  best")
result = ga.run(lambda bits: bits.sum(), target=LEN, generations=10_000, on_generation=progress)
print(
    f"\n{result.best_fitness:.0f} ones after {result.generations} generations and "
    f"{result.evaluations} evaluations (the optimum: {LEN})"
)
if trace:
    trace.write(
        os.environ["GENOXIDE_TRACE"],
        {
            "format": 1,
            "example": "one_max",
            "objective": "maximize",
            "x_label": "generations",
            "y_label": "ones",
            "log_y": False,
            "optimum": float(LEN),
            "plot": "bits",
            "problem": {"length": LEN},
        },
    )
