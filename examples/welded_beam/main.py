"""Welded beam design: the cheapest beam welded to a support that carries 6000 lb at 14 inches,
subject to its weld's shear stress, its bending stress, its buckling load and its deflection. A
constrained continuous problem, in the two forms of the literature.

``WeldedBeam`` is the form with seven constraints (Rao, 1996, as restated by Coello Coello, 2000),
``WeldedBeamRagsdell`` the one with five (Ragsdell and Phillips, 1976, as restated by Deb, 2000).
Their fitness is the cost and the constraint violation, which Deb's feasibility rules compare.
SHADE, a differential evolution, solves each with the same budget, and the example prints the best
design next to the best known cost.

With ``GENOXIDE_TRACE=<file>``, it also writes the trace of the first form's run for the plot on the
example's page: the best design so far and its constraint violations, in at most 200 generations.

    python examples/welded_beam/main.py
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

forms = [gx.problems.engineering.WeldedBeam(), gx.problems.engineering.WeldedBeamRagsdell()]
# the trace is of the first form's run
trace = Trace(200) if "GENOXIDE_TRACE" in os.environ else None
for form, problem in enumerate(forms):
    best_known = problem.optimum.value
    de = gx.De(problem.genome, objective=problem.objective, seed=1)

    def record(progress):
        if trace and form == 0:
            best = progress.best_genome
            violations = [max(g, 0.0) for g in problem.constraints(best).tolist()]
            trace.record(progress, {"best": best.tolist(), "violations": violations})

    result = de.run(problem, evaluations=40_000, on_generation=record)
    h, l, t, b = result.best_genome.tolist()
    print(
        f"{problem.name}: cost {result.best_fitness:.6f}, violation {result.violation:.6f} "
        f"(the best known: {best_known})"
    )
    print(f"  h {h:.6f}, l {l:.6f}, t {t:.6f}, b {b:.6f}")
if trace:
    short, long = [0.1, 2.0], [0.1, 10.0]
    trace.write(
        os.environ["GENOXIDE_TRACE"],
        {
            "format": 1,
            "example": "welded_beam",
            "objective": "minimize",
            "x_label": "evaluations",
            "y_label": "cost",
            "log_y": False,
            "optimum": forms[0].optimum.value,
            "plot": "design",
            "problem": {
                "variables": [
                    {"name": name, "unit": "in", "bounds": bounds}
                    for name, bounds in (("h", short), ("l", long), ("t", long), ("b", short))
                ],
                "constraints": [f"g{g}" for g in range(1, 8)],
            },
        },
    )
