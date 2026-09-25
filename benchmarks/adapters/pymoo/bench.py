"""Benchmark adapter for pymoo.

Usage:
    python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
    python bench.py values <problem> <size>

The first prints one JSON line per solver per seed, see ../../README.md for the fields. The second
reads one JSON solution per line and prints its value (or list of objectives), with the fitness
functions below.

Every method and setting, where pymoo documents it, and the separate test runs are on the library's
page: docs/benchmarks/libraries/pymoo.md. The citations "docs/source/..." are files of the pymoo
0.6.2 repository (github.com/anyoptimization/pymoo, tag 0.6.2), the sources of pymoo.org.

How a run follows the rules (docs/benchmarks/rules.md):
- The fitness functions are vectorized numpy on a population matrix, pymoo's `Problem` (rule 1.2;
  docs/source/problems/definition.md, "Problem (vectorized)"). `Counter` counts every row
  (rule 3) and keeps the best solution evaluated.
- A single-objective run stops at the target (checked after each generation), at the evaluation
  budget or at the time cap. The budget is exact: a batch that would go past it is evaluated only up
  to it, and the run stops there.
- Rule 2.2: the benchmark gives each run a budget, which in pymoo is minimize's termination
  argument, ("n_evals", N) for instance (docs/source/interface/termination.md); it replaces pymoo's
  default termination (xtol, ftol over 30 generations) entirely. So only the convergence criteria
  that belong to a method's own settings end an attempt: those a docs example sets explicitly (the
  flowshop example's DefaultSingleObjectiveTermination(period=50)), Nelder-Mead's own
  NelderAndMeadTermination, and CMA-ES's own stops, after its IPOP restarts; so does a GA whose
  mating finds no new child. Their budget limits (n_max_gen, n_max_evals, n_max_iter) are lifted.
  The method then starts again from a new random start, seed * 1000 + restart, keeping the best and
  counting every evaluation.
- Rule 2.4: every evaluated solution is inside the bounds through pymoo's own bound handling;
  `outside` counts those that aren't, as pymoo proposed them.
- A multi-objective run stops after the generation that reaches its budget, and its front is the
  non-dominated part of the final population (rule 7.2).
"""

import os

# single-threaded numpy (BLAS), before numpy is imported (rule 4.3)
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import json
import math
import sys
import time

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.algorithms.moo.spea2 import SPEA2
from pymoo.algorithms.soo.nonconvex.brkga import BRKGA
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.algorithms.soo.nonconvex.es import ES
from pymoo.algorithms.soo.nonconvex.ga import GA, comp_by_cv_and_fitness
from pymoo.algorithms.soo.nonconvex.nelder import NelderAndMeadTermination, NelderMead
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.problem import Problem
from pymoo.core.termination import TerminateIfAny, Termination
from pymoo.decomposition.pbi import PBI
from pymoo.decomposition.tchebicheff import Tchebicheff
from pymoo.operators.crossover.ox import OrderCrossover
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.mutation.inversion import InversionMutation
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.lhs import LHS
from pymoo.operators.sampling.rnd import BinaryRandomSampling, PermutationRandomSampling
from pymoo.operators.selection.tournament import TournamentSelection
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultSingleObjectiveTermination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to problems.py, vectorized: X holds one solution per row
# -------------------------------------------------------------------------------------------------

TARGET = 0.01

# the optimum of Rastrigin and Ackley: s_i = 2 ((37 i + 11) mod 101) / 101 - 1, in [-1, 1]
SHIFT = np.array([2 * ((37 * i + 11) % 101) / 101 - 1 for i in range(1000)])


def onemax(X):
    """The number of ones, negated: pymoo minimizes."""
    return -np.sum(X, axis=1).astype(float)


def nqueens(X):
    """Diagonal conflicts of queens at (i, X[i]): for each diagonal, its queens minus one. That's
    n minus the number of occupied diagonals, in each direction."""
    X = np.asarray(X, dtype=int)
    n = X.shape[1]
    column = np.arange(n)

    def occupied(diagonals):
        diagonals = np.sort(diagonals, axis=1)
        return 1 + np.count_nonzero(np.diff(diagonals, axis=1), axis=1)

    return (2 * n - occupied(X + column) - occupied(X + (n - 1 - column))).astype(float)


def rastrigin(X):
    D = X - SHIFT[:X.shape[1]]
    return 10 * X.shape[1] + np.sum(D * D - 10 * np.cos(2 * np.pi * D), axis=1)


def rosenbrock(X):
    a, b = X[:, :-1], X[:, 1:]
    return np.sum(100 * (b - a * a) ** 2 + (1 - a) ** 2, axis=1)


def ackley(X):
    n = X.shape[1]
    D = X - SHIFT[:n]
    return (-20 * np.exp(-0.2 * np.sqrt(np.sum(D * D, axis=1) / n))
            - np.exp(np.sum(np.cos(2 * np.pi * D), axis=1) / n) + 20 + math.e)


def zdt_g(X):
    return 1 + 9 * np.sum(X[:, 1:], axis=1) / (X.shape[1] - 1)


def zdt1(X):
    g = zdt_g(X)
    return np.column_stack([X[:, 0], g * (1 - np.sqrt(X[:, 0] / g))])


def zdt2(X):
    g = zdt_g(X)
    return np.column_stack([X[:, 0], g * (1 - (X[:, 0] / g) ** 2)])


def zdt3(X):
    g = zdt_g(X)
    h = 1 - np.sqrt(X[:, 0] / g) - X[:, 0] / g * np.sin(10 * np.pi * X[:, 0])
    return np.column_stack([X[:, 0], g * h])


def dtlz2(X, objectives):
    g = np.sum((X[:, objectives - 1:] - 0.5) ** 2, axis=1)
    columns = []
    for m in range(objectives):
        f = 1 + g
        for i in range(objectives - 1 - m):
            f = f * np.cos(X[:, i] * np.pi / 2)
        if m > 0:
            f = f * np.sin(X[:, objectives - 1 - m] * np.pi / 2)
        columns.append(f)
    return np.column_stack(columns)


def dtlz1(X, objectives):
    tail = X[:, objectives - 1:]
    g = 100 * (tail.shape[1] + np.sum((tail - 0.5) ** 2 - np.cos(20 * np.pi * (tail - 0.5)), axis=1))
    columns = []
    for m in range(objectives):
        f = 0.5 * (1 + g)
        for i in range(objectives - 1 - m):
            f = f * X[:, i]
        if m > 0:
            f = f * (1 - X[:, objectives - 1 - m])
        columns.append(f)
    return np.column_stack(columns)


# the real-valued problems: fitness function and bounds
REAL_PROBLEMS = {
    "rastrigin": (rastrigin, -5.12, 5.12),
    "rosenbrock": (rosenbrock, -5.0, 10.0),
    "ackley": (ackley, -32.768, 32.768),
}

# (fitness function of (X, size), variables, population size, Das-Dennis divisions)
FRONT_PROBLEMS = {
    "zdt1": (lambda X, size: zdt1(X), lambda size: size, 100, 99),
    "zdt2": (lambda X, size: zdt2(X), lambda size: size, 100, 99),
    "zdt3": (lambda X, size: zdt3(X), lambda size: size, 100, 99),
    # size: the number of objectives, with k = 10 (DTLZ2) and 5 (DTLZ1) distance variables
    "dtlz2": (dtlz2, lambda size: size + 9, 92, 12),
    "dtlz1": (dtlz1, lambda size: size + 4, 92, 12),
}


def front_objectives(problem_name, size):
    return 2 if problem_name.startswith("zdt") else size


def values(problem_name, size):
    """`values <problem> <size>`: the value of each solution read from stdin (rule 1.2)."""
    for line in sys.stdin:
        if not line.strip():
            continue
        X = np.array([json.loads(line)], dtype=float)
        if problem_name in FRONT_PROBLEMS:
            print(json.dumps([float(v) for v in FRONT_PROBLEMS[problem_name][0](X, size)[0]]))
        elif problem_name == "onemax":
            print(json.dumps(int(-onemax(X)[0])))
        elif problem_name == "nqueens":
            print(json.dumps(int(nqueens(X)[0])))
        elif problem_name in REAL_PROBLEMS:
            print(json.dumps(float(REAL_PROBLEMS[problem_name][0](X)[0])))
        else:
            print(f"unknown problem {problem_name}", file=sys.stderr)
            sys.exit(2)


# -------------------------------------------------------------------------------------------------
# Counting evaluations and stopping (rules 2 and 3)
# -------------------------------------------------------------------------------------------------


def count_outside(X, xl, xu):
    """The rows of X with a variable outside [xl, xu] (rule 2.4)."""
    X = np.asarray(X, dtype=float)
    return int(np.count_nonzero(np.any((X < xl) | (X > xu), axis=1)))


class Stop(Exception):
    """The budget or the time cap is reached in the middle of a batch."""


class Counter:
    """Counts the evaluations of a run, across its restarts, and keeps its best solution."""

    def __init__(self, max_evaluations, deadline, target):
        self.evaluations = 0
        self.max_evaluations = max_evaluations
        self.deadline = deadline
        self.target = target
        self.best = math.inf
        self.solution = None
        # evaluated solutions outside the bounds, as pymoo proposed them (rule 2.4)
        self.outside = 0

    def out_of_budget(self):
        return self.evaluations >= self.max_evaluations or time.perf_counter() >= self.deadline

    def reached(self):
        return self.best <= self.target


class SingleProblem(Problem):
    """A single-objective problem whose evaluations are counted. `decode` maps the variables to the
    solution the fitness function takes (BRKGA's random keys to a permutation)."""

    def __init__(self, counter, function, n_var, xl, xu, vtype=float, decode=None):
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu, vtype=vtype)
        self.counter = counter
        self.function = function
        self.decode = decode

    def _evaluate(self, X, out, *args, **kwargs):
        counter = self.counter
        if counter.out_of_budget():
            raise Stop
        # the budget is exact: a batch that would go past it is evaluated up to it
        stop = len(X) > counter.max_evaluations - counter.evaluations
        if stop:
            X = X[:counter.max_evaluations - counter.evaluations]
        solutions = self.decode(X) if self.decode else X
        F = self.function(solutions)
        counter.evaluations += len(X)
        counter.outside += count_outside(X, self.xl, self.xu)
        best = int(np.argmin(F))
        if F[best] < counter.best:
            counter.best = float(F[best])
            counter.solution = np.array(solutions[best], copy=True)
        if stop:
            raise Stop
        out["F"] = F
        if self.decode:
            # BRKGA's duplicate elimination compares the permutations (docs/source/algorithms/soo/brkga.md)
            out["pheno"] = solutions
            out["hash"] = [hash(s.tobytes()) for s in solutions]


class BudgetTermination(Termination):
    """Ends a run at the target, the budget or the time cap, after a generation."""

    def __init__(self, counter):
        super().__init__()
        self.counter = counter

    def _update(self, algorithm):
        return 1.0 if self.counter.reached() or self.counter.out_of_budget() else 0.0


# -------------------------------------------------------------------------------------------------
# Single-objective solvers: (name, problem factory of a counter, algorithm factory, seed offset,
# convergence factory). The convergence criteria end an attempt (rule 2.2); they're those a docs
# example sets explicitly, or those of the method itself (Nelder-Mead), without their budget limits
# (n_max_gen, n_max_evals, n_max_iter). A method whose example passes no termination, or only a
# budget such as ("n_gen", 100), has none: the benchmark's budget replaces pymoo's default
# termination, as ("n_evals", N) does for a pymoo user.
# -------------------------------------------------------------------------------------------------


def flowshop_convergence():
    """The termination of the flowshop example (docs/source/customization/permutation.md),
    DefaultSingleObjectiveTermination(period=50, n_max_gen=10000), without its generation limit:
    xtol 1e-8 and ftol 1e-6, each over a window of 50 generations."""
    return DefaultSingleObjectiveTermination(period=50, n_max_gen=math.inf, n_max_evals=math.inf)


def no_convergence():
    return None


class PermutationDuplicateElimination(ElementwiseDuplicateElimination):
    """Two random-key vectors are duplicates if they decode to the same permutation, as in
    docs/source/algorithms/soo/brkga.md."""

    def is_equal(self, a, b):
        return a.get("hash") == b.get("hash")


def decode_keys(X):
    """BRKGA's decoding of random keys to a permutation: their order (docs/source/algorithms/soo/brkga.md)."""
    return np.argsort(X, axis=1)


def onemax_solvers(size, mode):
    def problem(counter):
        return SingleProblem(counter, onemax, size, 0, 1, vtype=bool)

    if mode == "matched":
        # the matched settings: population 300, tournament of 3, two-point crossover at 0.5,
        # mutation at 0.2 of 1 / n per bit, every child evaluated. Difference: pymoo's GA survival
        # keeps the best of parents and children (FitnessSurvival), and can't be generational.
        def algorithm():
            return GA(
                pop_size=300,
                sampling=BinaryRandomSampling(),
                selection=TournamentSelection(func_comp=comp_by_cv_and_fitness, pressure=3),
                crossover=TwoPointCrossover(prob=0.5),
                mutation=BitflipMutation(prob=0.2, prob_var=1.0 / size),
                eliminate_duplicates=False,
            )
        # the termination of the docs' binary GA (customization/binary.md), ("n_gen", 100), is only
        # a budget: the GA runs to the budget
        return [("ga", problem, algorithm, 0, no_convergence)]

    # docs/source/customization/binary.md: GA for binary variables (the knapsack example). Its
    # termination, ("n_gen", 100), is only a budget: the GA runs to the budget
    def algorithm():
        return GA(
            pop_size=200,
            sampling=BinaryRandomSampling(),
            crossover=TwoPointCrossover(),
            mutation=BitflipMutation(),
            eliminate_duplicates=True,
        )
    return [("ga", problem, algorithm, 0, no_convergence)]


def nqueens_solvers(size):
    def ga_problem(counter):
        return SingleProblem(counter, nqueens, size, 0, size - 1, vtype=int)

    # docs/source/customization/permutation.md: GA with random permutations, order crossover and
    # inversion mutation, population 20, duplicates eliminated (the flowshop example, without the
    # TSP example's repair to start at city 0, which is for tours). Its termination:
    # DefaultSingleObjectiveTermination(period=50, n_max_gen=10000), the generation limit lifted
    def ga():
        return GA(
            pop_size=20,
            sampling=PermutationRandomSampling(),
            mutation=InversionMutation(),
            crossover=OrderCrossover(),
            eliminate_duplicates=True,
        )

    def brkga_problem(counter):
        return SingleProblem(counter, nqueens, size, 0.0, 1.0, decode=decode_keys)

    # docs/source/algorithms/soo/brkga.md: BRKGA, "known to perform well on combinatorial problems",
    # with its permutation example: random keys sorted into a permutation, 100 elites,
    # 300 offspring, 50 mutants, bias 0.7, duplicates eliminated by the permutation. Its
    # termination, ("n_gen", 50), is only a budget: BRKGA runs to the budget
    def brkga():
        return BRKGA(
            n_elites=100,
            n_offsprings=300,
            n_mutants=50,
            bias=0.7,
            eliminate_duplicates=PermutationDuplicateElimination(),
        )

    return [("ga", ga_problem, ga, 0, flowshop_convergence),
            ("brkga", brkga_problem, brkga, 0, no_convergence)]


def real_solvers(problem_name, size):
    function, low, high = REAL_PROBLEMS[problem_name]

    def problem(counter):
        return SingleProblem(counter, function, size, low, high)

    # docs/source/algorithms/soo/cmaes.md: "restarts can be used, which are known to work very well
    # on multi-modal functions. For instance, Rastrigin can be solved rather quickly by:
    # CMAES(restarts=10, restart_from_best=True)". Its other settings are the defaults: x0 the best
    # of 20 Latin hypercube samples, sigma 0.1 of the normalized bounds. The restarts are IPOP-CMA-ES
    # (the CMAES docstring): each doubles the population. pycma's own criteria (tolfun, tolx and
    # others) end each of its runs; after the 10th restart, the adapter starts it again. The
    # example's ("n_evals", 2500) is only a budget; CMAES has no pymoo-level criterion.
    # The seed mapping: the library seed is seed + 1, because pymoo passes its seed to pycma, where
    # 0 means a seed from the clock (not repeatable); every seed gets the next one, the same way.
    def cma_es():
        return CMAES(restarts=10, restart_from_best=True)

    if problem_name == "rosenbrock":
        # Continuous, unimodal: pymoo's local searches, which start from the best of 20 Latin
        # hypercube samples (docs/source/getting_started/preface.md: point-by-point methods "can be
        # highly efficient for rather unimodal fitness landscapes"). CMA-ES with the restarts of its
        # page, and Nelder-Mead with its defaults, as on docs/source/algorithms/soo/nelder.md.
        # Hooke and Jeeves pattern search (docs/source/algorithms/soo/pattern.md) is left out:
        # pymoo 0.6.2 draws its coordinate order from an unseeded generator (rule 5.2).
        # Nelder-Mead's own termination, NelderAndMeadTermination, part of the method (NelderMead
        # sets it itself): x_tol and f_tol of 1e-6 and a degenerate simplex end an attempt; its
        # budget limits (n_max_iter, n_max_evals) are lifted.
        return [
            ("cma_es", problem, cma_es, 1, no_convergence),
            ("nelder_mead", problem, lambda: NelderMead(), 0,
             lambda: NelderAndMeadTermination(n_max_iter=math.inf, n_max_evals=math.inf)),
        ]

    # Continuous, multimodal: the pages whose algorithm pymoo labels for multi-modal optimization.
    # docs/source/algorithms/soo/de.md (keywords "Multi-modal Optimization", "known for its good
    # results for global optimization"): its example, on Ackley. `dither="vector"` of the example
    # is left out: DE doesn't use it (see the library's page). The example passes no termination,
    # so DE runs to the budget.
    def de():
        return DE(pop_size=100, sampling=LHS(), variant="DE/rand/1/bin", CR=0.3, jitter=False)

    # docs/source/algorithms/soo/es.md (keywords "Multi-modal Optimization"): its example, on
    # Ackley, 200 offspring and the 1/7 rule, which are also the defaults. Its termination,
    # ("n_gen", 200), is only a budget: ES runs to the budget
    def es():
        return ES(n_offsprings=200, rule=1.0 / 7.0)

    return [("cma_es", problem, cma_es, 1, no_convergence), ("de", problem, de, 0, no_convergence),
            ("es", problem, es, 0, no_convergence)]


def single_solvers(problem_name, size, mode):
    if problem_name == "onemax":
        return onemax_solvers(size, mode)
    if problem_name == "nqueens":
        return nqueens_solvers(size)
    return real_solvers(problem_name, size)


def solve(make_problem, make_algorithm, make_convergence, seed, counter):
    """Runs the solver until the target, the budget or the time cap. An attempt that ends by
    itself (a convergence criterion, CMA-ES after its restarts, a GA whose mating finds no new
    child) starts again from a new random start (rule 2.2). Returns the generations of all attempts."""
    generations = 0
    restart = 0
    while True:
        run_seed = seed if restart == 0 else seed * 1000 + restart
        np.random.seed(run_seed)
        algorithm = make_algorithm()
        convergence = make_convergence()
        termination = BudgetTermination(counter)
        if convergence is not None:
            termination = TerminateIfAny(termination, convergence)
        try:
            minimize(make_problem(counter), algorithm, termination, seed=run_seed,
                     verbose=False, copy_algorithm=False, copy_termination=False)
        except Stop:
            pass
        generations += algorithm.n_gen or 0
        if counter.reached() or counter.out_of_budget():
            return generations
        restart += 1


def run_single(problem_name, size, mode, seed_from, seed_to, max_evaluations, max_seconds):
    maximize = problem_name == "onemax"
    target = -size if maximize else (0 if problem_name == "nqueens" else TARGET)
    for seed in range(seed_from, seed_to + 1):
        for solver, make_problem, make_algorithm, seed_offset, make_convergence in single_solvers(
                problem_name, size, mode):
            start = time.perf_counter()
            counter = Counter(max_evaluations, start + max_seconds, target)
            generations = solve(make_problem, make_algorithm, make_convergence, seed + seed_offset, counter)
            elapsed = time.perf_counter() - start
            if problem_name in REAL_PROBLEMS:
                best, solution = counter.best, [float(v) for v in counter.solution]
            else:
                best, solution = int(round(counter.best)), [int(v) for v in counter.solution]
            if maximize:
                best = -best
            print(json.dumps({
                "library": "pymoo",
                "solver": solver,
                "problem": problem_name,
                "size": size,
                "mode": mode,
                "seed": seed,
                "time_s": round(elapsed, 6),
                "generations": generations,
                "evaluations": counter.evaluations,
                "best": best,
                "target": -target if maximize else target,
                "success": counter.reached(),
                "solution": solution,
                **({"outside": counter.outside} if problem_name in REAL_PROBLEMS else {}),
            }), flush=True)


# -------------------------------------------------------------------------------------------------
# Multi-objective: the matched settings of every library
# -------------------------------------------------------------------------------------------------


class FrontProblem(Problem):
    """A multi-objective problem whose evaluations are counted."""

    def __init__(self, function, size, n_var, n_obj):
        super().__init__(n_var=n_var, n_obj=n_obj, xl=0.0, xu=1.0)
        self.function = function
        self.size = size
        self.evaluations = 0
        # evaluated solutions outside the bounds, as pymoo proposed them (rule 2.4)
        self.outside = 0

    def _evaluate(self, X, out, *args, **kwargs):
        self.evaluations += len(X)
        self.outside += count_outside(X, self.xl, self.xu)
        out["F"] = self.function(X, self.size)


class FrontTermination(Termination):
    """Ends a run after the generation that reaches max_evaluations, or at max_seconds."""

    def __init__(self, problem, max_evaluations, deadline):
        super().__init__()
        self.problem = problem
        self.max_evaluations = max_evaluations
        self.deadline = deadline

    def _update(self, algorithm):
        if self.problem.evaluations >= self.max_evaluations or time.perf_counter() >= self.deadline:
            return 1.0
        return 0.0


def front_solvers(problem_name, size):
    """NSGA-II, SPEA2 and SMS-EMOA: SBX with eta 15 at 0.9, polynomial mutation with eta 20 at
    1 / n. NSGA-III (3 objectives): Das-Dennis directions, SBX with eta 30 at 1. MOEA/D: 20
    neighbors, mating in the neighborhood at 0.9, Tchebycheff (PBI with theta 5 for DTLZ), SBX with
    eta 20 at 1. pymoo's defaults for the rest: its NSGA-II, NSGA-III, SPEA2 and SMS-EMOA eliminate
    duplicate children, and its SMS-EMOA makes a population of children per generation."""
    _, variables, population, divisions = FRONT_PROBLEMS[problem_name]
    n = variables(size)
    objectives = front_objectives(problem_name, size)
    ref_dirs = get_reference_directions("das-dennis", objectives, n_partitions=divisions)

    def mutation():
        return PM(eta=20, prob_var=1.0 / n)

    solvers = [
        ("nsga2", lambda: NSGA2(pop_size=population, crossover=SBX(prob=0.9, eta=15), mutation=mutation())),
        ("spea2", lambda: SPEA2(pop_size=population, crossover=SBX(prob=0.9, eta=15), mutation=mutation())),
        ("sms_emoa", lambda: SMSEMOA(pop_size=population, crossover=SBX(prob=0.9, eta=15), mutation=mutation())),
        ("moead", lambda: MOEAD(ref_dirs, n_neighbors=20, prob_neighbor_mating=0.9,
                                decomposition=Tchebicheff() if objectives == 2 else PBI(theta=5),
                                crossover=SBX(prob=1.0, eta=20), mutation=mutation())),
    ]
    if objectives > 2:
        # NSGA-III is pymoo's many-objective NSGA; U-NSGA-III is its version for 1 and 2 objectives
        # (docs/source/algorithms/list.md), so NSGA-III runs the 3-objective problems
        solvers.insert(1, ("nsga3", lambda: NSGA3(ref_dirs, pop_size=population,
                                                  crossover=SBX(prob=1.0, eta=30), mutation=mutation())))
    return solvers


def run_fronts(problem_name, size, mode, seed_from, seed_to, max_evaluations, max_seconds):
    function, variables, _, _ = FRONT_PROBLEMS[problem_name]
    n, objectives = variables(size), front_objectives(problem_name, size)
    for seed in range(seed_from, seed_to + 1):
        for solver, make_algorithm in front_solvers(problem_name, size):
            np.random.seed(seed)
            start = time.perf_counter()
            problem = FrontProblem(function, size, n, objectives)
            termination = FrontTermination(problem, max_evaluations, start + max_seconds)
            algorithm = make_algorithm()
            minimize(problem, algorithm, termination, seed=seed, verbose=False,
                     copy_algorithm=False, copy_termination=False)
            elapsed = time.perf_counter() - start
            # rule 7.2: the non-dominated part of the final population (for SPEA2, its archive),
            # not pymoo's res.opt, which for NSGA-III is only the solutions closest to its directions
            X, F = algorithm.pop.get("X", "F")
            front = NonDominatedSorting().do(F, only_non_dominated_front=True)
            print(json.dumps({
                "library": "pymoo",
                "solver": solver,
                "problem": problem_name,
                "size": size,
                "mode": mode,
                "seed": seed,
                "time_s": round(elapsed, 6),
                "generations": algorithm.n_gen,
                "evaluations": problem.evaluations,
                "front": [[float(v) for v in F[i]] for i in front],
                "solutions": [[float(v) for v in X[i]] for i in front],
                "outside": problem.outside,
            }), flush=True)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "values":
        values(sys.argv[2], int(sys.argv[3]))
        return
    if len(sys.argv) != 8:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    problem_name, size, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed_from, seed_to = int(sys.argv[4]), int(sys.argv[5])
    max_evaluations, max_seconds = int(sys.argv[6]), float(sys.argv[7])
    if problem_name in FRONT_PROBLEMS:
        run_fronts(problem_name, size, mode, seed_from, seed_to, max_evaluations, max_seconds)
    elif problem_name in ("onemax", "nqueens") or problem_name in REAL_PROBLEMS:
        run_single(problem_name, size, mode, seed_from, seed_to, max_evaluations, max_seconds)
    else:
        print(f"unknown problem {problem_name}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
