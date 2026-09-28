"""Benchmark adapter for pymoo.

Usage:
    python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
    python bench.py values <problem> <size>

The first prints one JSON line per solver per seed, see ../../README.md for the fields. The second
reads one JSON solution per line and prints its value, with the fitness functions below.

The matched suite (docs/benchmarks/rules.md): pymoo runs matched Rastrigin 30 with its DE set to
DE/rand/1/bin, and matched Rosenbrock 10 with its CMAES set to the suite's CMA-ES, and prints nothing
for any other problem, size or mode. The settings, their sources and the differences are on the
page docs/benchmarks/libraries/pymoo.md. The citations "docs/source/..." are files of the pymoo 0.6.2
repository (github.com/anyoptimization/pymoo, tag 0.6.2), the sources of pymoo.org.

How a run follows the rules:
- The fitness functions are vectorized numpy on a population matrix, pymoo's `Problem` (rule 1.2;
  docs/source/problems/definition.md, "Problem (vectorized)"). `Counter` counts every row (rule 3),
  keeps the best solution evaluated and records the first hit (rule 3.3).
- A run stops at the target (Rosenbrock 10: 0.01, checked after each generation; Rastrigin 30 has
  none, a fixed budget), at the evaluation budget or at the time cap. The budget is exact: a batch
  that would go past it is evaluated only up to it, and the run stops there.
- Rule 2.2: the budget is minimize's termination argument (docs/source/interface/termination.md),
  which replaces the algorithm's default termination. DE has no other criterion. CMAES keeps two of
  pycma's that no option turns off; if one ends the run, it ends there, not reached, and says so in
  "ended_by". No run restarts.
- Rule 2.4: every evaluated solution is inside the bounds through pymoo's own bound handling;
  `outside` counts those that aren't, as pymoo proposed them.
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
from pymoo.algorithms.soo.nonconvex.cmaes import CMAES
from pymoo.algorithms.soo.nonconvex.de import DE
from pymoo.core.problem import Problem
from pymoo.core.termination import Termination
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize

# -------------------------------------------------------------------------------------------------
# The fitness functions, identical to problems.py, vectorized: X holds one solution per row
# -------------------------------------------------------------------------------------------------

# the optimum of Rastrigin, within 80% of the box:
# s_i = 0.8 upper (2 ((37 i + 11) mod 101) / 101 - 1), computed in this order, as problems.py
RASTRIGIN_SHIFT = np.array([0.8 * 5.12 * (2 * ((37 * i + 11) % 101) / 101 - 1) for i in range(1000)])


def rastrigin(X):
    D = X - RASTRIGIN_SHIFT[:X.shape[1]]
    return 10 * X.shape[1] + np.sum(D * D - 10 * np.cos(2 * np.pi * D), axis=1)


def rosenbrock(X):
    a, b = X[:, :-1], X[:, 1:]
    return np.sum(100 * (b - a * a) ** 2 + (1 - a) ** 2, axis=1)


# the problems pymoo runs: fitness function, bounds and target (Rastrigin 30 has none: a fixed
# budget, measured by its time and final error)
PROBLEMS = {
    "rastrigin": (rastrigin, -5.12, 5.12, None),
    "rosenbrock": (rosenbrock, -5.0, 10.0, 0.01),
}


def values(problem_name, size):
    """`values <problem> <size>`: the value of each solution read from stdin (rule 1.2)."""
    if problem_name not in PROBLEMS:
        print(f"unknown problem {problem_name}", file=sys.stderr)
        sys.exit(2)
    function = PROBLEMS[problem_name][0]
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(float(function(np.array([json.loads(line)], dtype=float))[0])))


# -------------------------------------------------------------------------------------------------
# Counting evaluations and stopping (rules 2 and 3)
# -------------------------------------------------------------------------------------------------


def count_outside(X, xl, xu):
    """The rows of X with a variable outside [xl, xu] or not a number (rule 2.4)."""
    X = np.asarray(X, dtype=float)
    return int(np.count_nonzero(np.any(~((X >= xl) & (X <= xu)), axis=1)))


class Stop(Exception):
    """The budget or the time cap is reached in the middle of a batch."""


class Counter:
    """Counts the evaluations of a run and keeps its best solution. pymoo updates the termination
    after every generation, the initial population included (Algorithm._post_advance), and the
    termination marks its end here (rule 2.3)."""

    def __init__(self, start, max_evaluations, max_seconds, target):
        self.start = start
        self.target = target
        self.max_evaluations = max_evaluations
        self.deadline = start + max_seconds
        self.evaluations = 0
        self.generations = 0
        self.generation_end = 0
        self.last = 0
        self.best = math.inf
        self.solution = None
        # the first evaluation that reaches the target: (its number, seconds since the start)
        self.first_hit = None
        # evaluated solutions outside the bounds, as pymoo proposed them (rule 2.4)
        self.outside = 0

    def end_generation(self):
        self.generations += 1
        self.last = self.evaluations - self.generation_end
        self.generation_end = self.evaluations

    def all_generations(self):
        """The generations of the run, the initial population included, and the last one if it
        was cut short."""
        return self.generations + (self.evaluations > self.generation_end)

    def last_generation(self):
        """The evaluations since the start of the last generation: of one cut short, or of the
        last one that ended."""
        return self.evaluations - self.generation_end or self.last

    def out_of_budget(self):
        return self.evaluations >= self.max_evaluations or time.perf_counter() >= self.deadline

    def reached(self):
        return self.target is not None and self.best <= self.target


class CountedProblem(Problem):
    """The problem whose evaluations are counted."""

    def __init__(self, counter, function, n_var, xl, xu):
        super().__init__(n_var=n_var, n_obj=1, xl=xl, xu=xu)
        self.counter = counter
        self.function = function

    def _evaluate(self, X, out, *args, **kwargs):
        counter = self.counter
        if counter.out_of_budget():
            raise Stop
        # the budget is exact: a batch that would go past it is evaluated up to it
        stop = len(X) > counter.max_evaluations - counter.evaluations
        if stop:
            X = X[:counter.max_evaluations - counter.evaluations]
        F = self.function(X)
        if counter.first_hit is None and counter.target is not None:
            hits = np.flatnonzero(F <= counter.target)
            if hits.size:
                counter.first_hit = (counter.evaluations + int(hits[0]) + 1,
                                     time.perf_counter() - counter.start)
        counter.evaluations += len(X)
        counter.outside += count_outside(X, self.xl, self.xu)
        best = int(np.argmin(F))
        if F[best] < counter.best:
            counter.best = float(F[best])
            counter.solution = np.array(X[best], copy=True)
        if stop:
            raise Stop
        out["F"] = F


class BudgetTermination(Termination):
    """Ends a run at the target (if any), the budget or the time cap, after a generation. For
    CMAES it also keeps pycma's CMAEvolutionStrategy, from the generator pymoo runs it in (as
    pymoo's own CMAESOutput reads it), to name the criterion that ends the run, if one does."""

    def __init__(self, counter):
        super().__init__()
        self.counter = counter
        self.pycma = None

    def _update(self, algorithm):
        self.counter.end_generation()
        frame = getattr(getattr(algorithm, "es", None), "gi_frame", None)
        if frame is not None and "es" in frame.f_locals:
            self.pycma = frame.f_locals["es"]
        return 1.0 if self.counter.reached() or self.counter.out_of_budget() else 0.0


# -------------------------------------------------------------------------------------------------
# The methods
# -------------------------------------------------------------------------------------------------


def de(size, seed):
    """DE/rand/1/bin, the matched suite's (docs/benchmarks/libraries/pymoo.md), with pymoo's DE
    (docs/source/algorithms/soo/de.md): 100 individuals drawn uniformly in the box, F 0.5 fixed,
    CR 0.9. Its keyword arguments go to its Variant (pymoo/algorithms/soo/nonconvex/de.py):
    jitter=False keeps F fixed; prob_mut=0 turns off the polynomial mutation the Variant applies
    to 10% of the trials by default. The docs example's dither isn't passed: DE ignores it.
    Returns the algorithm and the seed of minimize."""
    return DE(
        pop_size=100,
        sampling=FloatRandomSampling(),
        variant="DE/rand/1/bin",
        F=0.5,
        CR=0.9,
        jitter=False,
        prob_mut=0.0,
    ), seed


def cma_es(size, seed):
    """CMA-ES, the matched suite's (docs/benchmarks/libraries/pymoo.md), with pymoo's CMAES
    (docs/source/algorithms/soo/cmaes.md), which runs pycma: the mean drawn uniformly in the box,
    sigma 0.3 of the box (CMAES normalizes the bounds to [0, 1]), 10 samples a generation, no
    restarts; CMA_active=False and the options below go to pycma ("All parameters that can be used
    there either as a keyword argument or an option can also be passed to the CMAES
    constructor"). Every pycma stop criterion with an option is off; noeffectcoord and noeffectaxis
    have none. maxstd_boundrange=inf: no cap on the standard deviations, as with pycma.
    Returns the algorithm and the seed of minimize: seed + 1, because CMAES passes it to pycma,
    which reads 0 as "seed from the clock"."""
    x0 = np.random.default_rng(seed).uniform(-5.0, 10.0, size)
    return CMAES(
        x0=x0,
        sigma=0.3,
        pop_size=10,
        restarts=0,
        CMA_active=False,
        maxstd_boundrange=np.inf,
        tolfun=0,
        tolx=0,
        tolfunhist=0,
        tolfunrel=0,
        tolstagnation=0,
        tolflatfitness=np.inf,
        tolconditioncov=0,
        tolupsigma=0,
        tolfacupx=np.inf,
        tolxstagnation=False,
        maxiter=np.inf,
    ), seed + 1


def solve(method, seed, counter, size, problem_name):
    """Runs the method once, until the target, the budget or the time cap, and returns why it
    ended otherwise (rules 2.2 and 8.4): the pycma criterion that ended CMAES, or None. CMAES
    also ends, with no criterion, on any error pycma raises (its _advance catches them all)."""
    function, lower, upper, _ = PROBLEMS[problem_name]
    algorithm, run_seed = method(size, seed)
    termination = BudgetTermination(counter)
    np.random.seed(run_seed)
    try:
        minimize(CountedProblem(counter, function, size, lower, upper), algorithm, termination,
                 seed=run_seed, verbose=False, copy_algorithm=False, copy_termination=False)
    except Stop:
        pass
    if counter.reached() or counter.out_of_budget():
        return None
    stopped = termination.pycma and dict(termination.pycma.stop(check=False))
    if stopped:
        return "pycma's " + ", ".join(sorted(stopped))
    return "the algorithm ended with no stop criterion (for CMAES: an error in pycma)"


# (problem, size) of the matched suite: (solver, method)
SCENARIOS = {("rastrigin", 30): ("de", de), ("rosenbrock", 10): ("cma_es", cma_es)}


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

    # the matched suite: pymoo runs matched Rastrigin 30 and Rosenbrock 10
    if mode != "matched" or (problem_name, size) not in SCENARIOS:
        return
    solver, method = SCENARIOS[problem_name, size]

    for seed in range(seed_from, seed_to + 1):
        # the clock starts before the initial population (rule 4.1)
        start = time.perf_counter()
        counter = Counter(start, max_evaluations, max_seconds, PROBLEMS[problem_name][3])
        ended_by = solve(method, seed, counter, size, problem_name)
        # the clock stops when the run ends (rule 4.1)
        elapsed = time.perf_counter() - start
        print(json.dumps({
            "library": "pymoo",
            "solver": solver,
            "problem": problem_name,
            "size": size,
            "mode": mode,
            "seed": seed,
            "time_s": round(elapsed, 6),
            "generations": counter.all_generations(),
            "evaluations": counter.evaluations,
            "last_generation": counter.last_generation(),
            "outside": counter.outside,
            "best": counter.best,
            "target": counter.target,
            "success": counter.reached(),
            "first_hit": counter.first_hit and {"evaluations": counter.first_hit[0],
                                                "time_s": round(counter.first_hit[1], 6)},
            **({"ended_by": ended_by} if ended_by else {}),
            "solution": [float(v) for v in counter.solution],
        }), flush=True)


if __name__ == "__main__":
    main()
