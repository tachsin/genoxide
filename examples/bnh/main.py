"""BNH: minimize two objectives subject to two constraints, with NSGA-II and Deb's rules.

Binh and Korn's problem, from genoxide's problems.Bnh, whose fitness is the two objectives and the
constraint violation; run evaluates it in Rust. Prints how many solutions of the final front are
feasible, their IGD+ to 500 points of the optimal front, and the front's hypervolume.

With ``GENOXIDE_TRACE=<file>``, it also writes the run's trace for the plot on the example's page:
the front, the infeasible solutions of the population and its feasible share, and the front's
hypervolume, in at most 100 generations.

    python examples/bnh/main.py
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
        """The generation's progress and the plot's ``state``, whose hypervolume is the curve."""
        frame = {
            "generation": progress.generation,
            "evaluations": progress.evaluations,
            "best": None,
            "median": None,
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

problem = gx.problems.Bnh()
nsga2 = gx.Nsga2(
    problem.genome,
    objectives=problem.objectives,
    population_size=100,
    crossover=gx.SimulatedBinaryCrossover(20),
    mutation=gx.PolynomialMutation(20, rate=0.5),
    seed=1,
)
trace = Trace(100) if "GENOXIDE_TRACE" in os.environ else None


def record(progress):
    if trace:
        front = progress.front_objectives
        feasible = progress.violations == 0
        state = {
            "fronts": {"NSGA-II": front.tolist()},
            "infeasible": {"NSGA-II": progress.objectives[~feasible].tolist()},
            "hypervolume": {"NSGA-II": gx.indicators.hypervolume(front, [210.0, 55.0])},
            "feasible": {"NSGA-II": feasible.sum() / len(feasible)},
        }
        trace.record(progress, state)


result = nsga2.run(problem, generations=250, on_generation=record)

front = result.front_objectives
feasible = int((result.front_violations == 0).sum())
print(f"{len(front)} solutions on the front, {feasible} feasible")

# IGD+ to the optimal front, x1 = x2 from 0 to 5
optimal = problem.optimal_front(500)
distance = gx.indicators.igd_plus(front, optimal)
print(f"IGD+ to the optimal front: {distance:.4f}")

# the hypervolume with the reference point (210, 55); the whole front's is 210 × 55 − 5000/3, the
# area above f2 = 2 (√(f1/8) − 5)²
volume = gx.indicators.hypervolume(front, [210.0, 55.0])
print(f"hypervolume {volume:.2f} (the whole front: 9883.33)")
if trace:
    trace.write(
        os.environ["GENOXIDE_TRACE"],
        {
            "format": 1,
            "example": "bnh",
            "objective": ["minimize", "minimize"],
            "x_label": "generations",
            "y_label": "hypervolume",
            "log_y": False,
            "optimum": 9883.33,
            "plot": "front-2d",
            "problem": {
                "objectives": ["f1", "f2"],
                "true_front": problem.optimal_front(100).tolist(),
                "series": ["NSGA-II"],
            },
        },
    )
