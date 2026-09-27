"""Function suite: CMA-ES, SHADE and PSO on twelve classic test functions in 10 dimensions.

Each algorithm has a budget of 10,000 evaluations per dimension and stops early within 1e-8 of
the known minimum. The table gives the error to the minimum: the best value found minus the
minimum. The functions, their bounds and their minima come from genoxide's problems, which run
evaluates in Rust.

With ``GENOXIDE_TRACE=<file>``, it also writes the runs' trace for the plot on the example's page:
the error of each function and algorithm after every 1,000 evaluations.

    python examples/function_suite/main.py
"""

import json
import math
import os

import genoxide as gx

DIMENSIONS = 10
BUDGET = 10_000 * DIMENSIONS
# the frames of the trace, evenly spaced over the budget
FRAMES = 100


def scientific(value):
    """Two significant digits, e.g. 1.2e-7."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


functions = [
    gx.problems.Sphere(DIMENSIONS),
    gx.problems.AxisParallelEllipsoid(DIMENSIONS),
    gx.problems.Schwefel1_2(DIMENSIONS),
    gx.problems.Zakharov(DIMENSIONS),
    gx.problems.Rosenbrock(DIMENSIONS),
    gx.problems.Rastrigin(DIMENSIONS),
    gx.problems.Ackley(DIMENSIONS),
    gx.problems.Griewank(DIMENSIONS),
    gx.problems.Schwefel2_26(DIMENSIONS),
    gx.problems.Levy(DIMENSIONS),
    gx.problems.StyblinskiTang(DIMENSIONS),
    gx.problems.Michalewicz(DIMENSIONS),
]


# ---- the trace of the runs, for the plot on the example's page ----------------------------------


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


def write_trace(path, series):
    """The runs side by side, a frame per 1,000 evaluations: each run's error after its last
    generation within them."""

    def frame(evaluations):
        values = {}
        for name, history in series.items():
            done = [error for done, error in history if done <= evaluations]
            values[name] = done[-1] if done else None
        return {
            "generation": None,
            "evaluations": evaluations,
            "best": None,
            "median": None,
            "state": {"values": values},
        }

    frames = ",\n".join(to_json(frame(k * BUDGET // FRAMES)) for k in range(1, FRAMES + 1))
    settings = {
        "format": 1,
        "example": "function_suite",
        "objective": "minimize",
        "x_label": "evaluations",
        "y_label": "error to the minimum",
        "log_y": True,
        "optimum": 0.0,
        "plot": "multi-curve",
        "problem": {"series": list(series)},
    }
    with open(path, "w", encoding="utf-8", newline="\n") as file:
        file.write(f'{to_json(settings)[:-1]},"frames":[\n{frames}\n]}}\n')


trace = "GENOXIDE_TRACE" in os.environ
# per function and algorithm, for the trace: the evaluations and the error after each generation
series = {}
print(f"Error to the minimum in {DIMENSIONS} dimensions, {BUDGET} evaluations at most")
print(f"{'function':<22}{'CMA-ES':>10}{'SHADE':>10}{'PSO':>10}")
for problem in functions:
    minimum = problem.optimum.value
    algorithms = {
        "CMA-ES": gx.Cmaes(problem.genome, restarts="ipop", objective="minimize", seed=1),
        "SHADE": gx.De(problem.genome, objective="minimize", seed=1),
        "PSO": gx.Pso(problem.genome, population_size=40, objective="minimize", seed=1),
    }
    errors = []
    for name, algorithm in algorithms.items():
        history = series.setdefault(f"{problem.name}/{name}", [])

        def record(progress):
            if trace and progress.best_fitness is not None:
                history.append((progress.evaluations, max(progress.best_fitness - minimum, 0.0)))

        result = algorithm.run(
            problem, target=minimum + 1e-8, evaluations=BUDGET, on_generation=record
        )
        # rounding can put a solution a few ulps below the minimum
        errors.append(scientific(max(result.best_fitness - minimum, 0.0)))
    print(f"{problem.name:<22}" + "".join(f"{error:>10}" for error in errors))
if trace:
    write_trace(os.environ["GENOXIDE_TRACE"], series)
