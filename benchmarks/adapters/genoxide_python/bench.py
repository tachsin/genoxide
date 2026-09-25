"""Benchmark adapter for genoxide's Python package (../../../python), whose algorithms are genoxide's
Rust, calling Python fitness functions.

Usage:
    python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
    python bench.py values <problem> <size>
    python bench.py --self-check

The first prints one JSON line per solver per seed, with the best solution (or the final front and
its solutions), see ../../README.md for the fields. The second reads one JSON solution per line
from stdin and prints its value (or its list of objectives), with the fitness functions below. The
third compares the fitness functions with problems.py at random points.

The methods are those the Python package's own docs (python/README.md, python/examples/ and the
docstrings of python/genoxide/__init__.py) present for the problem type, chosen as rule 6.2 says;
they're not the Rust adapter's, which follow the Rust docs. They run with their defaults, and a
setting without a default takes a standard value from the literature (rule 6.6). Where each comes
from, and the separate test runs: docs/benchmarks/libraries/genoxide_python.md. The fitness
functions are written as those docs write them:
- a function per genome for OneMax and N-Queens, as python/README.md's first example and
  python/examples/onemax.py and n_queens.py;
- `batch=True` with vectorized numpy functions, a generation per call, for the real-valued and
  multi-objective problems, as python/examples/rastrigin.py and zdt1.py, and what python/README.md
  presents for vectorized numpy ("one call per generation"). A batch is the same algorithm: the
  package asks for the same genomes either way ("the same seed repeats a run exactly, with a
  genome at a time, in batches or in parallel").

The rules (docs/benchmarks/rules.md), as this adapter follows them:
- each fitness function counts the genomes it evaluates itself (rule 3), and records the first
  one that reaches the target ("first_hit"): that count is the reported "evaluations", and the
  package's own `result.evaluations` must be the same (a difference is printed to stderr);
- a run ends at the target, the evaluation budget or the time cap only (rule 2.1). On convergence
  a method starts again (rule 2.2): DE and CMA-ES (IPOP) with their own restarts; the GA has no
  convergence criterion but the package's stall ("stalled": 10,000 generations without a genome to
  evaluate), after which the adapter starts it again with the seed
  `(seed + 1) * 1_000_000 + restart`, keeping the best and counting every evaluation. The stall
  needs stop conditions that all need new evaluations, so the GA's time cap is its
  `on_generation` callback, not `time`;
- every evaluated solution stays inside the bounds, by genoxide's own bound handling, and the
  fitness functions count any outside them ("outside", rule 2.4);
- the clock starts before the algorithm object is created, and stops when `run` returns, whose
  first step creates the random initial population (rule 4.1);
- one thread: `parallel` is off, and numpy's BLAS runs with 1 thread, set below (rule 4.3);
- each seed goes to the algorithm's `seed`, so a seed repeats a run exactly (rule 5.2).
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

import genoxide as gx

# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to problems.py. Each counts the genomes it evaluates in EVALUATIONS,
# the real-valued ones count the genomes outside the bounds in OUTSIDE (rule 2.4), and the
# single-objective ones record the first evaluation that reaches the target in HIT (rule 3.3).
# -------------------------------------------------------------------------------------------------

EVALUATIONS = 0
OUTSIDE = 0
# the target of the run (None outside a single-objective run), whether it's a maximum, the start
# of the run's clock, and the first hit: {"evaluations": E, "time_s": T}
HIT = {"target": None, "maximize": False, "start": 0.0, "first": None}


def count(x, low, high):
    """Counts the rows of x, and those with a gene outside [low, high]."""
    global EVALUATIONS, OUTSIDE
    EVALUATIONS += len(x)
    OUTSIDE += int(np.count_nonzero(((x < low) | (x > high)).any(axis=1)))


def record(value):
    """Records the first hit of the target, if `value`, the latest evaluation's, reaches it."""
    target = HIT["target"]
    if target is not None and HIT["first"] is None and (
            value >= target if HIT["maximize"] else value <= target):
        HIT["first"] = {"evaluations": EVALUATIONS, "time_s": time.perf_counter() - HIT["start"]}


def record_batch(values):
    """Records the first hit of the target among the values of the latest batch, in order."""
    target = HIT["target"]
    if target is None or HIT["first"] is not None:
        return
    hits = np.flatnonzero(values >= target if HIT["maximize"] else values <= target)
    if len(hits):
        HIT["first"] = {"evaluations": EVALUATIONS - len(values) + int(hits[0]) + 1,
                        "time_s": time.perf_counter() - HIT["start"]}


def onemax(bits):
    """One genome, a numpy bool array, whose sum is numpy's (`bits.sum()`, as python/README.md)."""
    global EVALUATIONS
    EVALUATIONS += 1
    value = int(bits.sum())
    record(value)
    return value


def nqueens(order):
    """The diagonal conflicts of one genome, as python/examples/n_queens.py: for each diagonal,
    its queens minus one, which is n minus the number of occupied diagonals, in both directions."""
    global EVALUATIONS
    EVALUATIONS += 1
    size = len(order)
    rows = np.arange(size)
    value = float(2 * size - len(np.unique(order + rows)) - len(np.unique(order - rows)))
    record(value)
    return value


def shift(upper, n):
    """The optimum of Rastrigin and Ackley (rule 1.4): s_i = 0.8 upper (2 ((37 i + 11) mod 101) /
    101 - 1), with `upper` the box's upper bound, computed in this order."""
    i = np.arange(n)
    return 0.8 * upper * (2 * ((37 * i + 11) % 101) / 101 - 1)


def rastrigin(x):
    count(x, -5.12, 5.12)
    y = x - shift(5.12, x.shape[1])
    values = 10 * x.shape[1] + np.sum(y**2 - 10 * np.cos(2 * np.pi * y), axis=1)
    record_batch(values)
    return values


def rosenbrock(x):
    count(x, -5.0, 10.0)
    a, b = x[:, :-1], x[:, 1:]
    values = np.sum(100 * (b - a * a) ** 2 + (1 - a) ** 2, axis=1)
    record_batch(values)
    return values


def ackley(x):
    count(x, -32.768, 32.768)
    n = x.shape[1]
    y = x - shift(32.768, n)
    squares = np.sum(y**2, axis=1) / n
    cosines = np.sum(np.cos(2 * np.pi * y), axis=1) / n
    values = -20 * np.exp(-0.2 * np.sqrt(squares)) - np.exp(cosines) + 20 + math.e
    record_batch(values)
    return values


# the real-valued problems: fitness function (a generation per call) and bounds
REAL_PROBLEMS = {
    "rastrigin": (rastrigin, -5.12, 5.12),
    "rosenbrock": (rosenbrock, -5.0, 10.0),
    "ackley": (ackley, -32.768, 32.768),
}


def zdt_g(x):
    return 1 + 9 * x[:, 1:].sum(axis=1) / (x.shape[1] - 1)


def zdt1(x):
    count(x, 0.0, 1.0)
    f1, g = x[:, 0], zdt_g(x)
    return np.column_stack([f1, g * (1 - np.sqrt(f1 / g))])


def zdt2(x):
    count(x, 0.0, 1.0)
    f1, g = x[:, 0], zdt_g(x)
    return np.column_stack([f1, g * (1 - (f1 / g) ** 2)])


def zdt3(x):
    count(x, 0.0, 1.0)
    f1, g = x[:, 0], zdt_g(x)
    return np.column_stack([f1, g * (1 - np.sqrt(f1 / g) - f1 / g * np.sin(10 * np.pi * f1))])


def dtlz(x, scale, objectives, head, last):
    """DTLZ1 and DTLZ2: objective m is scale times head(v) of each of the first M - 1 - m
    variables, times last(v) of the next one for m > 0."""
    values = []
    for m in range(objectives):
        f = scale.copy()
        for i in range(objectives - 1 - m):
            f *= head(x[:, i])
        if m > 0:
            f *= last(x[:, objectives - 1 - m])
        values.append(f)
    return np.column_stack(values)


def dtlz2(x, objectives=3):
    count(x, 0.0, 1.0)
    g = np.sum((x[:, objectives - 1:] - 0.5) ** 2, axis=1)
    return dtlz(x, 1 + g, objectives, lambda v: np.cos(v * np.pi / 2), lambda v: np.sin(v * np.pi / 2))


def dtlz1(x, objectives=3):
    count(x, 0.0, 1.0)
    tail = x[:, objectives - 1:]
    g = 100 * (tail.shape[1] + np.sum((tail - 0.5) ** 2 - np.cos(20 * np.pi * (tail - 0.5)), axis=1))
    return dtlz(x, 0.5 * (1 + g), objectives, lambda v: v, lambda v: 1 - v)


# (fitness function, variables, objectives, population size, Das-Dennis divisions)
FRONT_PROBLEMS = {
    "zdt1": (zdt1, lambda size: size, 2, 100, 99),
    "zdt2": (zdt2, lambda size: size, 2, 100, 99),
    "zdt3": (zdt3, lambda size: size, 2, 100, 99),
    # size: the number of objectives, with k = 10 (DTLZ2) and 5 (DTLZ1)
    "dtlz2": (dtlz2, lambda size: size + 9, 3, 92, 12),
    "dtlz1": (dtlz1, lambda size: size + 4, 3, 92, 12),
}

# -------------------------------------------------------------------------------------------------
# Solvers, from the Python package's docs. Each is made by a function of the seed, so a restart
# can make it again with another seed.
#
# The standard values from the literature for settings without a default (each is cited on
# docs/benchmarks/libraries/genoxide_python.md):
# - a GA's population of 100, binary tournament selection, SBX with eta 20 and polynomial
#   mutation with eta 20 at 1 / n: Deb, Pratap, Agarwal and Meyarivan, "A fast and elitist
#   multiobjective genetic algorithm: NSGA-II", IEEE TEC 2002;
# - bit-flip mutation at 1 / n per gene: Muehlenbein, "How genetic algorithms really work:
#   mutation and hillclimbing", PPSN 1992;
# - swap mutation for permutations.
# -------------------------------------------------------------------------------------------------

REAL_TARGET = 0.01
GA_POPULATION = 100
TOURNAMENT = 2
ETA = 20.0


def onemax_solvers(size, mode):
    """[(solver, make the algorithm from a seed, fitness, batch)]"""
    if mode == "matched":
        # the matched settings (benchmarks/README.md), as DEAP's eaSimple: population 300,
        # tournament 3 (with replacement), two-point crossover of each consecutive pair with
        # probability 0.5, each child mutated with probability 0.2 by a bit-flip at 1 / size per
        # gene, generational replacement without elitism; the package's own components
        return [("ga", lambda seed: gx.Ga(
            gx.Binary(size),
            population_size=300,
            select=gx.Tournament(3),
            crossover=gx.PointCrossover(2),
            crossover_rate=0.5,
            mutation=gx.BitFlip(rate=1.0 / size),
            mutation_rate=0.2,
            scheme=gx.Generational(elitism=0),
            seed=seed,
        ), onemax, False)]
    # idiomatic: the GA, the method of both OneMax examples (python/README.md's first example and
    # python/examples/onemax.py), with its defaults (crossover rate 0.9, mutation rate 1,
    # generational with an elitism of 1), UniformCrossover, the first crossover the README lists,
    # and the literature's population 100, binary tournament and bit-flip at 1 / n
    return [("ga", lambda seed: gx.Ga(
        gx.Binary(size),
        population_size=GA_POPULATION,
        select=gx.Tournament(TOURNAMENT),
        crossover=gx.UniformCrossover(),
        mutation=gx.BitFlip(rate=1.0 / size),
        seed=seed,
    ), onemax, False)]


def nqueens_solvers(size):
    # the package's example for N-Queens, python/examples/n_queens.py, is tabu search, whose tenure
    # has no default and no standard value: left out. The methods python/README.md lists first
    # for any genome, Ga and LocalSearch, with their defaults; the GA with OrderCrossover, the
    # first permutation crossover the README lists, a swap mutation, and the literature's
    # population 100 and binary tournament; local search with swap neighbors
    return [
        ("ga", lambda seed: gx.Ga(
            gx.Permutation(size),
            population_size=GA_POPULATION,
            select=gx.Tournament(TOURNAMENT),
            crossover=gx.OrderCrossover(),
            mutation=gx.SwapMutation(),
            objective="minimize",
            seed=seed,
        ), nqueens, False),
        ("local_search", lambda seed: gx.LocalSearch(
            gx.Permutation(size),
            neighbor=gx.SwapMutation(),
            objective="minimize",
            seed=seed,
        ), nqueens, False),
    ]


def real_solvers(problem, size):
    function, low, high = REAL_PROBLEMS[problem]
    genome = gx.Real((low, high), length=size)
    mutation = gx.PolynomialMutation(ETA, rate=1.0 / size)
    de = ("de", lambda seed: gx.De(genome, objective="minimize", seed=seed), function, True)
    if problem == "rosenbrock":
        # the Python docs have no example for a unimodal function and state no preference for
        # one, so the first methods python/README.md lists for real genomes (rule 6.2): Ga,
        # LocalSearch and De, with their defaults. The GA with the literature's population 100,
        # binary tournament, SBX with eta 20 and polynomial mutation with eta 20 at 1 / n; local
        # search with that polynomial mutation as its neighbor; DE with SHADE's published settings
        # and genoxide's restarts
        ga = ("ga", lambda seed: gx.Ga(
            genome,
            population_size=GA_POPULATION,
            select=gx.Tournament(TOURNAMENT),
            crossover=gx.SimulatedBinaryCrossover(ETA),
            mutation=mutation,
            objective="minimize",
            seed=seed,
        ), function, True)
        local_search = ("local_search", lambda seed: gx.LocalSearch(
            genome, neighbor=mutation, objective="minimize", seed=seed), function, True)
        return [ga, local_search, de]
    # multimodal: the package's example for Rastrigin, python/examples/rastrigin.py, runs CMA-ES
    # and DE. CMA-ES with its defaults and IPOP restarts, which the example, python/README.md's
    # first example and the Cmaes docstring ("for multimodal functions") give it; DE with its
    # defaults (the example's L-SHADE is a setting of De, `l_shade`)
    cmaes = ("cma_es", lambda seed: gx.Cmaes(genome, restarts="ipop", objective="minimize", seed=seed),
             function, True)
    return [cmaes, de]


def front_solvers(problem, size):
    """The matched settings (benchmarks/README.md), with the package's own operators: NSGA-II,
    SPEA2 and SMS-EMOA with SBX with eta 15 at 0.9 (their default rate) and polynomial mutation
    with eta 20 at 1 / n; NSGA-III with Das-Dennis directions (99 divisions with 2 objectives, 12
    with 3) and SBX with eta 30 at 1; MOEA/D with SBX with eta 20 at 1, 20 neighbors and
    neighborhood mating 0.9 (its defaults), Tchebycheff, or PBI with theta 5 with 3 objectives. No
    duplicate elimination (on by default, so turned off), and SMS-EMOA is steady-state: one child
    per generation. SBX and polynomial mutation keep the genes in [0, 1] (rule 2.4)."""
    function, variables, objectives, population, divisions = FRONT_PROBLEMS[problem]
    n = variables(size)
    genome = gx.Real((0.0, 1.0), length=n)
    minimize = ["minimize"] * objectives
    mutation = gx.PolynomialMutation(20.0, rate=1.0 / n)
    sbx = gx.SimulatedBinaryCrossover(15.0)
    directions = gx.das_dennis(objectives, divisions)
    solvers = [
        ("nsga2", lambda seed: gx.Nsga2(
            genome, objectives=minimize, population_size=population, crossover=sbx,
            mutation=mutation, eliminate_duplicates=False, seed=seed)),
        ("nsga3", lambda seed: gx.Nsga3(
            genome, objectives=minimize, reference_directions=directions,
            population_size=population, crossover=gx.SimulatedBinaryCrossover(30.0),
            mutation=mutation, eliminate_duplicates=False, seed=seed)),
        ("spea2", lambda seed: gx.Spea2(
            genome, objectives=minimize, population_size=population, crossover=sbx,
            mutation=mutation, eliminate_duplicates=False, seed=seed)),
        ("sms_emoa", lambda seed: gx.SmsEmoa(
            genome, objectives=minimize, population_size=population, crossover=sbx,
            mutation=mutation, offspring=1, eliminate_duplicates=False, seed=seed)),
        ("moead", lambda seed: gx.Moead(
            genome, objectives=minimize, weights=directions,
            decomposition=gx.Tchebycheff() if objectives == 2 else gx.Pbi(5.0),
            crossover=gx.SimulatedBinaryCrossover(20.0), mutation=mutation, seed=seed)),
    ]
    return function, solvers


# -------------------------------------------------------------------------------------------------
# Runs
# -------------------------------------------------------------------------------------------------


def compare(common, solver, seed, counted, reported):
    """The adapter's own count (rule 3) against the package's: they must agree."""
    if counted != reported:
        print(f"evaluations differ: {common['problem']} {common['size']} {solver} seed {seed}: the "
              f"adapter counted {counted}, genoxide reports {reported}", file=sys.stderr, flush=True)


def attempt_seed(seed, restart):
    """The seed of attempt `restart` of the run with `seed` (rule 2.2): the seed itself, then
    (seed + 1) * 1_000_000 + restart, so no two runs share a seed."""
    return seed if restart == 0 else (seed + 1) * 1_000_000 + restart


def solve(common, solver, seed, make, function, batch, target, max_evaluations, max_seconds):
    """Runs one solver, with restarts after a stall (rule 2.2), and prints the run."""
    global EVALUATIONS, OUTSIDE
    EVALUATIONS = OUTSIDE = 0
    maximize = common["problem"] == "onemax"
    # CMA-ES with IPOP restarts doubles its population at each restart, so its last generation
    # can be larger than the average: the run reports it, for the budget check (rule 2.3). The
    # other solvers' generations don't grow.
    generation = {"evaluated": 0, "last": 0}
    # the GA is the one solver here that can stall (every child a copy of a parent): the others
    # evaluate new genomes every generation (local search redraws a neighbor that didn't change).
    # Its time cap is this callback, so that a stall can end the run and the GA can start again
    can_stall = solver == "ga"

    def track(progress):
        generation["last"] = progress.evaluations - generation["evaluated"]
        generation["evaluated"] = progress.evaluations

    def in_time(progress):
        return time.perf_counter() - start < max_seconds

    best = solution = None
    generations = reported = restart = 0
    HIT.update(target=target, maximize=maximize, first=None)
    # the clock starts before the algorithm is made and its initial population created
    start = HIT["start"] = time.perf_counter()
    while True:
        generation["evaluated"] = 0
        budget = max_evaluations - EVALUATIONS
        if can_stall:
            limits = {"on_generation": in_time}
        else:
            limits = {"time": max(max_seconds - (time.perf_counter() - start), 0.0),
                      "on_generation": track if solver == "cma_es" else None}
        result = make(attempt_seed(seed, restart)).run(
            function, batch=batch, target=target, evaluations=budget, **limits)
        generations += result.generations
        reported += result.evaluations
        score = result.best_fitness
        if score is not None and (best is None or (score > best if maximize else score < best)):
            best, solution = score, result.best_genome
        if (result.stop_reason != "stalled" or EVALUATIONS >= max_evaluations
                or time.perf_counter() - start >= max_seconds):
            break
        restart += 1
    elapsed = time.perf_counter() - start
    HIT["target"] = None
    compare(common, solver, seed, EVALUATIONS, reported)
    extra = {}
    if solver == "cma_es":
        extra["last_generation"] = generation["last"]
    if common["problem"] in REAL_PROBLEMS:
        extra["outside"] = OUTSIDE
    if restart:
        extra["restarts"] = restart
    success = best is not None and (best >= target if maximize else best <= target)
    print(json.dumps({
        **common, "solver": solver, "seed": seed, "time_s": round(elapsed, 6),
        "generations": generations, "evaluations": EVALUATIONS, **extra,
        "first_hit": HIT["first"], "best": best, "target": target, "success": success,
        "solution": (solution.astype(int) if maximize else solution).tolist(),
    }), flush=True)


def main():
    global EVALUATIONS, OUTSIDE
    if sys.argv[1:] == ["--self-check"]:
        self_check()
        return
    if len(sys.argv) == 4 and sys.argv[1] == "values":
        values(sys.argv[2], int(sys.argv[3]))
        return
    if len(sys.argv) != 8:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    problem, size, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed_from, seed_to = int(sys.argv[4]), int(sys.argv[5])
    max_evaluations, max_seconds = int(sys.argv[6]), float(sys.argv[7])
    common = {"library": "genoxide_python", "problem": problem, "size": size, "mode": mode}

    for seed in range(seed_from, seed_to + 1):
        if problem in FRONT_PROBLEMS:
            # multi-objective runs have a budget and no target: they print the non-dominated
            # individuals of the final population (rule 7.2) and their solutions. None of these
            # algorithms has a convergence criterion (rule 2.2), and with a time limit the package
            # doesn't stall: every run ends at its budget or the time cap
            function, solvers = front_solvers(problem, size)
            for solver, make in solvers:
                EVALUATIONS = OUTSIDE = 0
                start = time.perf_counter()
                result = make(seed).run(function, batch=True, evaluations=max_evaluations, time=max_seconds)
                elapsed = time.perf_counter() - start
                compare(common, solver, seed, EVALUATIONS, result.evaluations)
                print(json.dumps({
                    **common, "solver": solver, "seed": seed, "time_s": round(elapsed, 6),
                    "generations": result.generations, "evaluations": EVALUATIONS,
                    "outside": OUTSIDE,
                    "front": result.front_objectives.tolist(),
                    "solutions": result.front_genomes.tolist(),
                }), flush=True)
            continue

        if problem == "onemax":
            solvers, target = onemax_solvers(size, mode), size
        elif problem == "nqueens":
            solvers, target = nqueens_solvers(size), 0
        elif problem in REAL_PROBLEMS:
            solvers, target = real_solvers(problem, size), REAL_TARGET
        else:
            print(f"unknown problem {problem}", file=sys.stderr)
            sys.exit(2)
        for solver, make, function, batch in solvers:
            solve(common, solver, seed, make, function, batch, target, max_evaluations, max_seconds)


def value(problem, size, solution):
    """The value of one solution (or its objectives), with the functions the runs call."""
    if problem == "onemax":
        return onemax(np.array(solution, dtype=bool))
    if problem == "nqueens":
        return nqueens(np.array(solution, dtype=np.int64))
    x = np.array([solution], dtype=np.float64)
    if problem in REAL_PROBLEMS:
        return float(REAL_PROBLEMS[problem][0](x)[0])
    function, variables, objectives, *_ = FRONT_PROBLEMS[problem]
    if problem.startswith("dtlz"):
        return function(x, size)[0].tolist()
    return function(x)[0].tolist()


def values(problem, size):
    for line in sys.stdin:
        if line.strip():
            print(json.dumps(value(problem, size, json.loads(line))), flush=True)


def self_check():
    """Compares the fitness functions with problems.py at random points and at the optimum of the
    shifted functions, and checks the outside count and the first hit of a batch: run it after
    changing them."""
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    import problems

    rng = np.random.default_rng(7)

    def close(ours, theirs, name):
        if not problems.close(ours, theirs):
            raise SystemExit(f"{name}: {ours} != {theirs}")

    for size in (100, 1000):
        for bits in rng.random((20, size)) < 0.5:
            close(value("onemax", size, bits.tolist()), problems.onemax(bits.tolist()), "onemax")
    for size in (8, 32, 64):
        for order in (rng.permutation(size) for _ in range(50)):
            close(value("nqueens", size, order.tolist()), problems.nqueens(order.tolist()), "nqueens")
    for name, (_, low, high) in REAL_PROBLEMS.items():
        for size in (10, 30):
            for x in rng.uniform(low, high, (20, size)):
                close(value(name, size, x.tolist()), problems.value(name, size, x.tolist()), name)
    for name in ("rastrigin", "ackley"):
        close(value(name, 30, problems.shift(name, 30)), 0.0, f"{name} at the optimum")
    for name, (_, variables, *_rest) in FRONT_PROBLEMS.items():
        size = 30 if name.startswith("zdt") else 3
        for x in rng.random((20, variables(size))):
            close(value(name, size, x.tolist()), problems.value(name, size, x.tolist()), name)
    # the outside count: a row with one gene past a bound counts once
    global EVALUATIONS, OUTSIDE
    OUTSIDE = 0
    rastrigin(np.array([[0.0, 5.12], [0.0, 5.13], [-6.0, 6.0]]))
    if OUTSIDE != 2:
        raise SystemExit(f"outside: {OUTSIDE} instead of 2")
    # the first hit of a batch: the first row that reaches the target, counted from the run's start
    EVALUATIONS = 5
    HIT.update(target=1.0, maximize=False, start=time.perf_counter(), first=None)
    record_batch(np.array([3.0, 0.5, 0.2]))
    if HIT["first"]["evaluations"] != 4:
        raise SystemExit(f"first hit: {HIT['first']} instead of evaluation 4")
    HIT["target"] = None
    print("the fitness functions match problems.py")


if __name__ == "__main__":
    main()
