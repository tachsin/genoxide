"""Benchmark adapter for Nevergrad.

Usage: python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
Prints one JSON line per solver per seed, see ../../README.md for the fields.

Nevergrad runs rastrigin, rosenbrock and ackley (NGOpt, CMA, TwoPointsDE and PSO), OneMax in the
idiomatic mode (NGOpt and DiscreteOnePlusOne), and the multi-objective problems (DE, with its own
settings, see run_front). It prints nothing for the other problems and modes: it has no permutation
parameter for N-Queens, and no GA with DEAP's operators for the matched OneMax.
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

# run.py's early stop: a solver whose first EARLY_SEEDS runs all hit the time cap (a run that took
# CAPPED of it) without reaching the target runs no more seeds
EARLY_SEEDS = 3
CAPPED = 0.98

# (fitness function, (lower, upper) of every variable)
REAL_PROBLEMS = {
    "rastrigin": (rastrigin, (-5.12, 5.12)),
    "rosenbrock": (rosenbrock, (-5.0, 10.0)),
    "ackley": (ackley, (-32.768, 32.768)),
}
UNSUPPORTED = {"nqueens"}


# multi-objective problems, minimized, all variables in [0, 1]
def zdt_g(x):
    return 1 + 9 * sum(x[1:]) / (len(x) - 1)


def zdt1(x):
    g = zdt_g(x)
    return [x[0], g * (1 - math.sqrt(x[0] / g))]


def zdt2(x):
    g = zdt_g(x)
    return [x[0], g * (1 - (x[0] / g) ** 2)]


def zdt3(x):
    g = zdt_g(x)
    return [x[0], g * (1 - math.sqrt(x[0] / g) - x[0] / g * math.sin(10 * math.pi * x[0]))]


def dtlz2(x, objectives=3):
    g = sum((v - 0.5) ** 2 for v in x[objectives - 1:])
    values = []
    for m in range(objectives):
        f = 1 + g
        for v in x[:objectives - 1 - m]:
            f *= math.cos(v * math.pi / 2)
        if m > 0:
            f *= math.sin(x[objectives - 1 - m] * math.pi / 2)
        values.append(f)
    return values


def dtlz1(x, objectives=3):
    tail = x[objectives - 1:]
    g = 100 * (len(tail) + sum((v - 0.5) ** 2 - math.cos(20 * math.pi * (v - 0.5)) for v in tail))
    values = []
    for m in range(objectives):
        f = 0.5 * (1 + g)
        for v in x[:objectives - 1 - m]:
            f *= v
        if m > 0:
            f *= 1 - x[objectives - 1 - m]
        values.append(f)
    return values


# (fitness function, variables for the size)
FRONT_PROBLEMS = {
    "zdt1": (zdt1, lambda size: size),
    "zdt2": (zdt2, lambda size: size),
    "zdt3": (zdt3, lambda size: size),
    # size: the number of objectives, with k = 10 (DTLZ2) and 5 (DTLZ1)
    "dtlz2": (dtlz2, lambda size: size + 9),
    "dtlz1": (dtlz1, lambda size: size + 4),
}


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


# -------------------------------------------------------------------------------------------------
# Multi-objective: DE
#
# An idiomatic run in a matched scenario: Nevergrad has no NSGA-II and no SBX or polynomial
# mutation, so it runs its own multi-objective optimizer with its defaults, not the matched
# settings (population 100 or 92, SBX, polynomial mutation).
#
# The optimizer is DE, the one the docs recommend (docs/optimization.rst, "Multiobjective
# minimization with Nevergrad"): most optimizers minimize a single loss derived from the
# hypervolume of the Pareto front, but "DE and its variants have however been updated to make use
# of the full multi-objective losses". NGOpt isn't used: with several losses it swaps its
# sub-optimizer for a DE (optimizerlib.py, NGOpt8._num_objectives_set_callback), but the NGOpt
# wrapper itself still turns every tell into a single loss derived from the hypervolume of its
# archive (base.py, tell; multiobjective/core.py, HypervolumePareto.add, which computes the
# hypervolume in Python for every point within its upper bounds), and passes that loss to the DE,
# which then sees one objective and doesn't use its multi-objective adaptation. That is slower
# than DE and gives up the multi-objective DE the docs recommend.
#
# DE's defaults (differentialevolution.py, DifferentialEvolution): population 30, uniform initial
# points within the bounds, current-to-best/1 with F1 = F2 = 0.8, binomial crossover with CR 0.5.
# With several losses (multiobjective_adaptation, on by default): the "best" of the donor is the
# parent if it's on the Pareto front, else a random point of the front, and the two difference
# points come from the front; a child replaces its parent with a probability equal to the share of
# the objectives it improves.
#
# No reference point (ng.p.MultiobjectiveReference): the docs advise one "for all but DE
# optimizers", since DE with multiobjective_adaptation computes no hypervolume.
#
# The front printed is optimizer.pareto_front(), the docs' way to get the result: the non-dominated
# set of every point evaluated, an unbounded archive, not a final population of 100. DE filters
# that archive at every ask, so its cost per evaluation grows with the front, and the time cap can
# stop a run before the budget.
# -------------------------------------------------------------------------------------------------


def run_front(function, variables, seed, max_evaluations, max_seconds):
    """The ask and tell loop of the docs with one worker and a list of losses, which stops at
    max_evaluations or at max_seconds. Returns (front, evaluations)."""
    parametrization = ng.p.Array(shape=(variables,), lower=0.0, upper=1.0)
    parametrization.random_state = np.random.RandomState(seed)
    optimizer = ng.optimizers.DE(parametrization=parametrization, budget=max_evaluations, num_workers=1)
    deadline = time.perf_counter() + max_seconds
    evaluations = 0
    while evaluations < max_evaluations and time.perf_counter() < deadline:
        candidate = optimizer.ask()
        losses = function(candidate.value.tolist())
        evaluations += 1
        optimizer.tell(candidate, losses)
    front = [[float(v) for v in param.losses] for param in optimizer.pareto_front()]
    return front, evaluations


def main():
    if len(sys.argv) != 8:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    problem, size, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed_from, seed_to = int(sys.argv[4]), int(sys.argv[5])
    max_evaluations, max_seconds = int(sys.argv[6]), float(sys.argv[7])

    if problem in UNSUPPORTED or (problem == "onemax" and mode != "idiomatic"):
        return
    if problem in FRONT_PROBLEMS:
        function, variables = FRONT_PROBLEMS[problem]
        capped = 0
        for index, seed in enumerate(range(seed_from, seed_to + 1)):
            if index >= EARLY_SEEDS and capped == EARLY_SEEDS:
                break
            np.random.seed(seed)
            start = time.perf_counter()
            front, evaluations = run_front(function, variables(size), seed, max_evaluations, max_seconds)
            elapsed = time.perf_counter() - start
            capped += index < EARLY_SEEDS and elapsed >= CAPPED * max_seconds
            print(json.dumps({
                "library": "nevergrad",
                "solver": "de",
                "problem": problem,
                "size": size,
                "mode": mode,
                "seed": seed,
                "time_s": round(elapsed, 6),
                # Nevergrad asks and tells one candidate at a time: one step per evaluation
                "generations": evaluations,
                "evaluations": evaluations,
                "front": front,
            }), flush=True)
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

    capped = {solver: 0 for solver, _ in solvers}
    for index, seed in enumerate(range(seed_from, seed_to + 1)):
        for solver, optimizer_class in solvers:
            if index >= EARLY_SEEDS and capped[solver] == EARLY_SEEDS:
                continue
            np.random.seed(seed)
            param = parametrization()
            start = time.perf_counter()
            loss, evaluations = run(optimizer_class, param, function, is_success, seed, max_evaluations, max_seconds)
            elapsed = time.perf_counter() - start
            best = to_best(loss)
            capped[solver] += index < EARLY_SEEDS and not is_success(loss) and elapsed >= CAPPED * max_seconds
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
