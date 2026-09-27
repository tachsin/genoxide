"""Pressure vessel design (Sandgren, 1990): the cheapest cylindrical vessel with hemispherical heads
that holds 1,296,000 cubic inches, a constrained mixed discrete-continuous problem. The minimum
cost is 6059.714335.

The variables are the thickness of the shell and of the heads, multiples of 0.0625 inch, and the
inner radius and the length of the shell. genoxide's ``PressureVessel`` rounds the first two genes
to whole plates, and its fitness is the cost and the violation of the four constraints, which
Deb's feasibility rules compare. SHADE, a differential evolution, searches the genes.

With ``GENOXIDE_TRACE=<file>``, it also writes the run's trace for the plot on the example's page:
the best design so far and its constraint violations, in at most 200 generations.

    python examples/pressure_vessel/main.py
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

problem = gx.problems.engineering.PressureVessel()
minimum = problem.optimum.value
de = gx.De(problem.genome, objective=problem.objective, seed=1)
trace = Trace(200) if "GENOXIDE_TRACE" in os.environ else None


def record(progress):
    if trace:
        best = progress.best_genome
        violations = [max(g, 0.0) for g in problem.constraints(best).tolist()]
        trace.record(progress, {"best": problem.design(best).tolist(), "violations": violations})


result = de.run(problem, evaluations=50_000, on_generation=record)

shell, head, radius, length = problem.design(result.best_genome).tolist()
print(
    f"cost {result.best_fitness:.6f} after {result.evaluations} evaluations "
    f"(the minimum: {minimum:.6f})"
)
print(f"violation {result.violation:.6f}")
print(f"shell {shell:.4f}, heads {head:.4f}, radius {radius:.6f}, length {length:.6f}")
if trace:
    plates, inches = [0.0625, 6.1875], [10.0, 200.0]
    trace.write(
        os.environ["GENOXIDE_TRACE"],
        {
            "format": 1,
            "example": "pressure_vessel",
            "objective": "minimize",
            "x_label": "evaluations",
            "y_label": "cost",
            "log_y": False,
            "optimum": minimum,
            "plot": "design",
            "problem": {
                "variables": [
                    {"name": name, "unit": "in", "bounds": plates if name[0] == "T" else inches}
                    for name in ("Ts", "Th", "R", "L")
                ],
                "constraints": ["g1", "g2", "g3", "g4"],
            },
        },
    )
