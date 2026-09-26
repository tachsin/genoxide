"""Benchmark adapter for Nevergrad.

Usage:
    python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
    python bench.py values <problem> <size>

The first prints one JSON line per solver per seed, see ../../README.md for the fields. The second
reads one JSON solution per line and prints its value with this adapter's fitness functions.

Which methods run and why, with the links to Nevergrad's docs, is on the page
docs/benchmarks/libraries/nevergrad.md:
- OneMax (idiomatic): NgIohTuned, DiscreteOnePlusOne and PortfolioDiscreteOnePlusOne, on the
  docs' discrete parameter, a TransitionChoice;
- N-Queens: NgIohTuned, RotatedTwoPointsDE and GeneticDE on a real array whose argsort is the
  permutation, Nevergrad's documented way to optimize a permutation;
- Rastrigin and Ackley: NgIohTuned, ScrHammersleySearchPlusMiddlePoint and OnePlusOne, and
  Rosenbrock: NgIohTuned, OnePlusOne and CMA, on a bounded Array.
It prints nothing for the matched OneMax (Nevergrad has no GA with the matched operators) and for
the multi-objective scenarios, which run only NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA
(rule 6.1): Nevergrad has none of them.

Every run ends only at the target, the budget or the time cap (rules 2.1 and 2.2 of
docs/benchmarks/rules.md): Nevergrad's optimizers have no stop criterion, they give a candidate at
every ask (the CMA-ES inside NgIohTuned starts again when pycma's criteria end it), and the ask and
tell loop stops at the target, the budget or the time cap.
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

# Nevergrad imports these inside the first run that uses them: pycma (with matplotlib) for the
# CMA-ES of NgIohTuned and CMA, scikit-learn for NgIohTuned's metamodel, and SciPy's COBYLA
# (which imports its PRIMA code when first called) for an optimizer inside NgIohTuned. Imported
# here, so the imports stay outside the clock (rule 4.2)
import cma  # noqa: F401
from scipy._external.pyprima import minimize as _cobyla  # noqa: F401
from sklearn.linear_model import LinearRegression  # noqa: F401
from sklearn.preprocessing import PolynomialFeatures  # noqa: F401

# e.g. the budget warnings of optimizers that don't use the whole budget
warnings.filterwarnings("ignore")

# Compatibility with numpy >= 2.5: the metamodel that NgIohTuned uses on real-valued problems calls
# float() on a one-element array (nevergrad/optimization/metamodel.py, loss_function_sm), which
# numpy 2.5 no longer allows, so it crashes with a TypeError. This restores the old conversion in
# that module only; the algorithm is unchanged.
metamodel.float = lambda value: builtins.float(np.asarray(value).item())


# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to benchmarks/problems.py (Nevergrad minimizes). Nevergrad gives
# each candidate's value on its own: a numpy array for an Array, a tuple for a TransitionChoice.
# -------------------------------------------------------------------------------------------------


def onemax(bits):
    return int(np.sum(bits))


def nqueens(order):
    """Diagonal conflicts of queens at (i, order[i]): for each diagonal, its queens minus one."""
    n = len(order)
    rows = np.arange(n)
    left = np.bincount(rows + order, minlength=2 * n - 1)
    right = np.bincount(n - 1 - rows + order, minlength=2 * n - 1)
    return int(np.sum(np.maximum(left - 1, 0)) + np.sum(np.maximum(right - 1, 0)))


@functools.cache
def shift(n, upper):
    """The optimum of rastrigin and ackley, away from the origin:
    s_i = 0.8 upper (2 ((37 i + 11) mod 101) / 101 - 1), computed in this order."""
    return np.array([0.8 * upper * (2 * ((37 * i + 11) % 101) / 101 - 1) for i in range(n)])


def rastrigin(x):
    """Shifted: 10 n + sum((x_i - s_i)^2 - 10 cos(2 pi (x_i - s_i)))."""
    d = x - shift(len(x), 5.12)
    return float(10 * len(x) + np.sum(d * d - 10 * np.cos(2 * np.pi * d)))


def rosenbrock(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def ackley(x):
    """Shifted: Ackley of x - s."""
    n = len(x)
    d = x - shift(n, 32.768)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(d * d) / n))
                 - np.exp(np.sum(np.cos(2 * np.pi * d)) / n) + 20 + math.e)


TARGET = 0.01

# run.py's early stop (rule 5.3): a solver whose first EARLY_SEEDS runs all hit the time cap (a run
# that took CAPPED of it) without reaching the target runs no more seeds
EARLY_SEEDS = 3
CAPPED = 0.98

# (fitness function, (lower, upper) of every variable)
REAL_PROBLEMS = {
    "rastrigin": (rastrigin, (-5.12, 5.12)),
    "rosenbrock": (rosenbrock, (-5.0, 10.0)),
    "ackley": (ackley, (-32.768, 32.768)),
}
UNSUPPORTED = {"zdt1", "zdt2", "zdt3", "dtlz1", "dtlz2"}


class Problem:
    """A problem as Nevergrad sees it: the parametrization, the loss of a candidate's value (to
    minimize), the solution printed for a value, the best value printed for a loss, and whether a
    value lies outside the bounds (rule 2.4; None for the problems without bounds)."""

    def __init__(self, problem, size):
        self.outside = None
        if problem == "onemax":
            # the docs' discrete example is OneMax on a TransitionChoice with repetitions
            # (docs/optimization.rst, "Basic example", test_doc.py DOC_BASE_4)
            self.parametrization = lambda: ng.p.TransitionChoice(range(2), repetitions=size)
            # Nevergrad minimizes the number of zeros; the best printed is the number of ones
            self.loss = lambda bits: size - onemax(bits)
            self.solution = lambda bits: [int(bit) for bit in bits]
            self.best = lambda loss: size - loss
            self.target = size
            self.is_success = lambda loss: loss == 0
        elif problem == "nqueens":
            # "How to optimize permutations with Nevergrad" (the docs' "Example with permutation",
            # docs/optimization.rst): a real Array, "ng.p.Array(shape=(500,)) if you consider
            # permutations in [0,1,2,3,...,499]", whose argsort is the permutation, as in
            # nevergrad/functions/stsp/core.py
            self.parametrization = lambda: ng.p.Array(shape=(size,))
            self.loss = lambda keys: nqueens(np.argsort(keys))
            self.solution = lambda keys: np.argsort(keys).tolist()
            self.best = lambda loss: loss
            self.target = 0
            self.is_success = lambda loss: loss == 0
        else:
            function, (lower, upper) = REAL_PROBLEMS[problem]
            # a bounded Array, the docs' way to give bounds (ng.p.Array's docstring: "if both lower
            # and upper bounds are provided, sigma will be adapted so that the range spans 6
            # sigma", and the initial value is the middle of the range). Its bound handling (rule
            # 2.4) is the Array's Bound layer, by default "bouncing" (set_bounds: "bounce on border
            # (at most once). This is a variant of clipping"), so every value evaluated is inside
            self.parametrization = lambda: ng.p.Array(shape=(size,), lower=lower, upper=upper)
            self.loss = function
            self.solution = lambda x: [float(v) for v in x]
            self.best = lambda loss: loss
            self.target = TARGET
            self.is_success = lambda loss: loss <= TARGET
            self.outside = lambda x: bool(np.any(x < lower) or np.any(x > upper))


def solvers(problem):
    """(solver name, optimizer) of the problem type, see the page. Every optimizer with its
    defaults.

    NgIohTuned everywhere: the docs' stated default, a "'meta'-optimizer which adapts to the
    provided settings (budget, number of workers, parametrization) and should therefore be a good
    default" (docs/optimization.rst, "Choosing an optimizer")."""
    if problem == "onemax":
        return [
            ("ngiohtuned", ng.optimizers.NgIohTuned),
            # the docs' OneMax example (docs/optimization.rst, "Basic example")
            ("discrete_one_plus_one", ng.optimizers.DiscreteOnePlusOne),
            # "excellent in discrete settings" (docs/optimization.rst, "Choosing an optimizer")
            ("portfolio_discrete_one_plus_one", ng.optimizers.PortfolioDiscreteOnePlusOne),
        ]
    if problem == "nqueens":
        return [
            ("ngiohtuned", ng.optimizers.NgIohTuned),
            # the two optimizers "How to optimize permutations with Nevergrad" names: they "will
            # cut and paste part of the permutation into other parts"
            ("rotated_two_points_de", ng.optimizers.RotatedTwoPointsDE),
            ("genetic_de", ng.optimizers.GeneticDE),
        ]
    # the continuous problems: the docs' list "Choosing an optimizer" states each optimizer's case.
    # Two match every continuous scenario here, one worker and a budget over 1000 times the
    # dimension: "OnePlusOne is a simple robust method for continuous parameters with
    # num_workers < 8", and "CMA is excellent for control (e.g. neurocontrol) when the environment
    # is not very noisy (num_workers ~50 ok) and when the budget is large (e.g. 1000 x the
    # dimension)"
    one_plus_one = ("one_plus_one", ng.optimizers.OnePlusOne)
    if problem == "rosenbrock":
        return [
            ("ngiohtuned", ng.optimizers.NgIohTuned),
            one_plus_one,
            # with bounds it is CMAbounded (MetaCMA); it starts again from the middle of the box
            # when pycma's own stop criteria end a run (optimizerlib.py, _CMA.es)
            ("cma_es", ng.optimizers.CMA),
        ]
    return [
        ("ngiohtuned", ng.optimizers.NgIohTuned),
        # "ScrHammersleySearchPlusMiddlePoint is excellent for super parallel cases (fully one-shot,
        # i.e. num_workers = budget included) or for very multimodal cases": the only one the list
        # gives for multimodal functions. A scrambled Hammersley sequence over the box, plus its
        # middle point
        ("scr_hammersley", ng.optimizers.ScrHammersleySearchPlusMiddlePoint),
        # of the two that match, the first the list gives (rule 6.2)
        one_plus_one,
    ]


def run(optimizer_class, problem, seed, max_evaluations, start, max_seconds):
    """The docs' ask and tell loop with one worker (docs/optimization.rst, "Ask and tell
    interface"), which stops at the target, at max_evaluations or at max_seconds after start.
    Returns (best loss, its candidate's value, evaluations, evaluated values outside the bounds,
    first_hit: the evaluation that reached the target and its time, or None)."""
    # the docs' two ways to seed (docs/optimization.rst, "Reproducibility"): numpy's global random
    # state and the parametrization's own
    np.random.seed(seed)
    parametrization = problem.parametrization()
    parametrization.random_state = np.random.RandomState(seed)
    # the budget tells NgIohTuned which algorithm to choose, as a user would give it
    optimizer = optimizer_class(parametrization=parametrization, budget=max_evaluations, num_workers=1)
    deadline = start + max_seconds
    best, best_value, first_hit = math.inf, None, None
    evaluations = outside = 0
    while evaluations < max_evaluations and time.perf_counter() < deadline:
        candidate = optimizer.ask()
        value = candidate.value
        loss = problem.loss(value)
        evaluations += 1
        if problem.is_success(loss):
            first_hit = {"evaluations": evaluations, "time_s": round(time.perf_counter() - start, 6)}
        if problem.outside is not None:
            outside += problem.outside(value)
        optimizer.tell(candidate, loss)
        if loss < best:
            best, best_value = loss, value
        if first_hit is not None:
            break
    return best, best_value, evaluations, outside, first_hit


def run_single(problem_name, size, mode, seed_from, seed_to, max_evaluations, max_seconds):
    problem = Problem(problem_name, size)
    solver_list = solvers(problem_name)
    capped = {solver: 0 for solver, _ in solver_list}
    for index, seed in enumerate(range(seed_from, seed_to + 1)):
        for solver, optimizer_class in solver_list:
            # rule 5.3
            if index >= EARLY_SEEDS and capped[solver] == EARLY_SEEDS:
                continue
            # the clock starts before the optimizer is created, and stops when the run ends
            # (rule 4.1)
            start = time.perf_counter()
            loss, value, evaluations, outside, first_hit = run(optimizer_class, problem, seed,
                                                               max_evaluations, start, max_seconds)
            elapsed = time.perf_counter() - start
            success = problem.is_success(loss)
            capped[solver] += index < EARLY_SEEDS and not success and elapsed >= CAPPED * max_seconds
            result = {
                "library": "nevergrad",
                "solver": solver,
                "problem": problem_name,
                "size": size,
                "mode": mode,
                "seed": seed,
                "time_s": round(elapsed, 6),
                # Nevergrad asks and tells one candidate at a time: one step per evaluation
                "generations": evaluations,
                "evaluations": evaluations,
                "last_generation": min(evaluations, 1),
                "best": problem.best(loss),
                "target": problem.target,
                "success": bool(success),
                "first_hit": first_hit,
                "solution": problem.solution(value),
            }
            if problem.outside is not None:
                result["outside"] = outside
            print(json.dumps(result), flush=True)


# -------------------------------------------------------------------------------------------------


def values(problem, size):
    """Prints the value of each solution read from stdin, one JSON list per line (rule 1.2)."""
    function = {"onemax": onemax, "nqueens": nqueens, **{k: v[0] for k, v in REAL_PROBLEMS.items()}}[problem]
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(float(function(np.array(json.loads(line))))), flush=True)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "values":
        values(sys.argv[2], int(sys.argv[3]))
        return
    if len(sys.argv) != 8:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    problem, size, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed_from, seed_to = int(sys.argv[4]), int(sys.argv[5])
    max_evaluations, max_seconds = int(sys.argv[6]), float(sys.argv[7])

    if problem in UNSUPPORTED or (problem == "onemax" and mode != "idiomatic"):
        return
    if problem in REAL_PROBLEMS or problem in ("onemax", "nqueens"):
        run_single(problem, size, mode, seed_from, seed_to, max_evaluations, max_seconds)
    else:
        print(f"unknown problem {problem}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
