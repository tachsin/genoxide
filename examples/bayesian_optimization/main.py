"""Bayesian optimization of Branin's function: 30 evaluations chosen by a Gaussian process and the
log expected improvement, then the model's mean minimized by L-BFGS-B and evaluated once.

The search comes within 1e-4 of one of the three global minima, and the polish of the model, for
one evaluation more, closes most of what is left.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/bayesian_optimization/main.py
"""

import numpy as np

import genoxide as gx

from trace import Trace

# the evaluations of the search: the initial design, then the points the model chooses
EVALUATIONS = 30


def scientific(value):
    """Two significant digits, e.g. 1.2e-7."""
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


problem = gx.problems.Branin()
minima = problem.optimum.solutions
minimum = problem.optimum.value
bo = gx.Bo(problem.genome, objective="minimize", seed=1)
print(f"Branin's function in [-5, 10] x [0, 15]: three global minima of {minimum:.6f}")
# the initial design's default size, 2(n + 1) for n = 2 genes
print("6 points of a Latin hypercube, then a point per step by log-EI on a Gaussian process")
print("evaluation         x1         x2            f     f - f*")
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(minima, minimum)
evaluated = {"points": None, "values": None}


def on_generation(progress):
    # the points evaluated in this generation
    printed = 0 if evaluated["points"] is None else len(evaluated["points"])
    for index in range(printed, len(progress.population)):
        x, value = progress.population[index], float(progress.scores[index])
        print(
            f"{index + 1:>10} {x[0]:>10.6f} {x[1]:>10.6f} {value:>12.6f} "
            f"{scientific(value - minimum):>10}"
        )
    evaluated["points"], evaluated["values"] = progress.population, progress.scores


result = bo.run(problem, evaluations=EVALUATIONS, on_generation=on_generation, control=trace.record)

# a Gaussian process of every evaluation, its mean minimized from the best point
points, values = evaluated["points"], evaluated["values"]
model = gx.model.gp.GaussianProcess.fit(problem.genome, points, values)
lbfgsb = gx.Lbfgsb(problem.genome, initial_genome=result.best_genome, objective="minimize", seed=1)
polished = lbfgsb.run(
    lambda x: model.predict(x)[0][0],
    gradient=lambda x: model.predict_with_gradient(x)[2],
    evaluations=1_000,
)
x = polished.best_genome
# one evaluation of the function there
value = float(problem(x))
nearest = minima[int(np.argmin(np.linalg.norm(minima - x, axis=1)))]
print(f"the model of the {len(points)} evaluations, its mean minimized by L-BFGS-B from the best point:")
print(
    f"({x[0]:.6f}, {x[1]:.6f}): predicted {model.predict(x)[0][0]:.6f}, evaluated {value:.6f}, "
    f"{scientific(value - minimum)} above the minimum at ({nearest[0]:.6f}, {nearest[1]:.6f})"
)
best = min(result.best_fitness, value)
print(f"{len(points) + 1} evaluations: the best {scientific(best - minimum)} above the global minimum")
assert best - minimum <= 1e-4
trace.write(x, value)
