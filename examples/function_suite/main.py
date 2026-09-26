"""Function suite: CMA-ES, SHADE and PSO on twelve classic test functions in 10 dimensions.

Each algorithm has a budget of 10,000 evaluations per dimension and stops early within 1e-8 of
the known minimum. The table gives the error to the minimum: the best value found minus the
minimum. The functions, their bounds and their minima come from genoxide's problems, which run
evaluates in Rust.

    python examples/function_suite/main.py
"""

import genoxide as gx

DIMENSIONS = 10
BUDGET = 10_000 * DIMENSIONS


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

print(f"Error to the minimum in {DIMENSIONS} dimensions, {BUDGET} evaluations at most")
print(f"{'function':<22}{'CMA-ES':>10}{'SHADE':>10}{'PSO':>10}")
for problem in functions:
    minimum = problem.optimum.value
    algorithms = [
        gx.Cmaes(problem.genome, restarts="ipop", objective="minimize", seed=1),
        gx.De(problem.genome, objective="minimize", seed=1),
        gx.Pso(problem.genome, population_size=40, objective="minimize", seed=1),
    ]
    errors = []
    for algorithm in algorithms:
        result = algorithm.run(problem, target=minimum + 1e-8, evaluations=BUDGET)
        # rounding can put a solution a few ulps below the minimum
        errors.append(scientific(max(result.best_fitness - minimum, 0.0)))
    print(f"{problem.name:<22}" + "".join(f"{error:>10}" for error in errors))
