"""Benchmark adapter for Nevergrad.

Usage: python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
Prints one JSON line per solver per seed, see ../../README.md for the fields.

Nevergrad runs rastrigin, rosenbrock and ackley (NGOpt, CMA, TwoPointsDE and PSO) and OneMax in the
idiomatic mode (NGOpt and DiscreteOnePlusOne). It prints nothing for the other problems and modes:
it has no permutation parameter for N-Queens, no GA with DEAP's operators for the matched OneMax,
and no multi-objective algorithm with the matched settings (SBX, polynomial mutation).
"""

import os

# single-threaded numpy (BLAS), before numpy is imported
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import builtins
import functools
import json
import math
import sys
import time
import warnings

import nevergrad as ng
import numpy as np
from nevergrad.optimization import metamodel

# e.g. the budget warnings of optimizers that don't use the whole budget
warnings.filterwarnings("ignore")

# Compatibility with numpy >= 2.5: the metamodel that NGOpt uses on real-valued problems calls
# float() on a one-element array (nevergrad/optimization/metamodel.py, loss_function_sm), which
# numpy 2.5 no longer allows, so NGOpt crashes with a TypeError. This restores the old conversion
# in that module only; the algorithm is unchanged.
metamodel.float = lambda value: builtins.float(np.asarray(value).item())


# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to the other adapters (Nevergrad minimizes)
# -------------------------------------------------------------------------------------------------


def onemax(x):
    return int(sum(x))


@functools.cache
def shift(n):
    """The optimum of rastrigin and ackley, away from the origin: s_i = 2 ((37 i + 11) mod 101) / 101 - 1."""
    return tuple(2 * ((37 * i + 11) % 101) / 101 - 1 for i in range(n))


def rastrigin(x):
    """Shifted: 10 n + sum((x_i - s_i)^2 - 10 cos(2 pi (x_i - s_i)))."""
    d = [v - s for v, s in zip(x, shift(len(x)))]
    return float(10 * len(d) + sum(v * v - 10 * math.cos(2 * math.pi * v) for v in d))


def rosenbrock(x):
    return float(sum(100 * (x[i + 1] - x[i] * x[i]) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1)))


def ackley(x):
    """Shifted: Ackley of x - s."""
    n = len(x)
    d = [v - s for v, s in zip(x, shift(n))]
    return float(
        -20 * math.exp(-0.2 * math.sqrt(sum(v * v for v in d) / n))
        - math.exp(sum(math.cos(2 * math.pi * v) for v in d) / n)
        + 20 + math.e
    )


TARGET = 0.01

# (fitness function, (lower, upper) of every variable)
REAL_PROBLEMS = {
    "rastrigin": (rastrigin, (-5.12, 5.12)),
    "rosenbrock": (rosenbrock, (-5.0, 10.0)),
    "ackley": (ackley, (-32.768, 32.768)),
}
UNSUPPORTED = {"nqueens", "zdt1", "zdt2", "zdt3", "dtlz1", "dtlz2"}


# -------------------------------------------------------------------------------------------------
# Solvers
# -------------------------------------------------------------------------------------------------


def real_solvers():
    """NGOpt, the optimizer the Nevergrad docs and README recommend (it chooses an algorithm from
    the budget, the dimension and the parametrization; docs/optimization.rst), and the
    single-algorithm optimizers it is known for, all with their defaults."""
    return [
        ("ngopt", ng.optimizers.NGOpt),
        ("cma_es", ng.optimizers.CMA),
        ("de", ng.optimizers.TwoPointsDE),
        ("pso", ng.optimizers.PSO),
    ]


def onemax_solvers():
    """OneMax as in the docs' discrete example (docs/optimization.rst, test_doc.py: a
    TransitionChoice with repetitions and DiscreteOnePlusOne), and NGOpt on the same parameter."""
    return [
        ("ngopt", ng.optimizers.NGOpt),
        ("discrete_one_plus_one", ng.optimizers.DiscreteOnePlusOne),
    ]


def run(optimizer_class, parametrization, function, is_success, seed, max_evaluations, max_seconds):
    """The ask and tell loop of the docs with one worker, which stops at the target (Nevergrad
    evaluates one candidate at a time), at max_evaluations or at max_seconds.
    Returns (best loss, evaluations)."""
    parametrization.random_state = np.random.RandomState(seed)
    # the budget tells NGOpt which algorithm to choose, as a user would give it
    optimizer = optimizer_class(parametrization=parametrization, budget=max_evaluations, num_workers=1)
    deadline = time.perf_counter() + max_seconds
    best = math.inf
    evaluations = 0
    while evaluations < max_evaluations and time.perf_counter() < deadline:
        candidate = optimizer.ask()
        loss = function(candidate.value)
        evaluations += 1
        optimizer.tell(candidate, loss)
        if loss < best:
            best = loss
            if is_success(best):
                break
    return best, evaluations


def main():
    if len(sys.argv) != 8:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    problem, size, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed_from, seed_to = int(sys.argv[4]), int(sys.argv[5])
    max_evaluations, max_seconds = int(sys.argv[6]), float(sys.argv[7])

    if problem in UNSUPPORTED or (problem == "onemax" and mode != "idiomatic"):
        return
    if problem == "onemax":
        solvers = onemax_solvers()

        def parametrization():
            return ng.p.TransitionChoice(range(2), repetitions=size)

        # Nevergrad minimizes the number of zeros; best is the number of ones
        function = lambda x: size - onemax(x)
        is_success = lambda loss: loss == 0
        to_best = lambda loss: size - loss
        target = size
    elif problem in REAL_PROBLEMS:
        solvers = real_solvers()
        real_function, (lower, upper) = REAL_PROBLEMS[problem]

        def parametrization():
            # bounded arrays are the docs' way; the initial sigma spans the range
            return ng.p.Array(shape=(size,), lower=lower, upper=upper)

        function = real_function
        is_success = lambda loss: loss <= TARGET
        to_best = lambda loss: loss
        target = TARGET
    else:
        print(f"unknown problem {problem}", file=sys.stderr)
        sys.exit(2)

    for seed in range(seed_from, seed_to + 1):
        for solver, optimizer_class in solvers:
            np.random.seed(seed)
            param = parametrization()
            start = time.perf_counter()
            loss, evaluations = run(optimizer_class, param, function, is_success, seed, max_evaluations, max_seconds)
            elapsed = time.perf_counter() - start
            best = to_best(loss)
            print(json.dumps({
                "library": "nevergrad",
                "solver": solver,
                "problem": problem,
                "size": size,
                "mode": mode,
                "seed": seed,
                "time_s": round(elapsed, 6),
                # Nevergrad asks and tells one candidate at a time: one step per evaluation
                "generations": evaluations,
                "evaluations": evaluations,
                "best": best,
                "target": target,
                "success": bool(is_success(loss)),
            }), flush=True)


if __name__ == "__main__":
    main()
