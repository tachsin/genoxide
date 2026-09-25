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

The solvers and their settings are the Rust genoxide adapter's (../genoxide/src/main.rs), through
the Python API, except the island model, which the package doesn't have: its GA runs as one
population, with the settings of examples/rastrigin.rs. Where genoxide's docs recommend each of
them, and the separate test runs: docs/benchmarks/libraries/genoxide_python.md. The fitness
functions are written as python/README.md and python/examples/ write them:
- `batch=True` with vectorized numpy functions, called with a generation at a time: what
  python/README.md recommends for vectorized numpy ("one call per generation"), as in
  python/examples/rastrigin.py and zdt1.py; used by the GA, DE, CMA-ES, PSO and the
  multi-objective algorithms;
- a function per genome for local search and tabu search, which evaluate a few neighbors per
  step, as python/examples/n_queens.py;
- a function per genome in the matched OneMax runs, like DEAP's: a Python call per genome, which
  counts the ones with numpy's sum, as python/examples/onemax.py and README.md's first example.

The rules (docs/benchmarks/rules.md), as this adapter follows them:
- each fitness function counts the genomes it evaluates itself (rule 3): that count is the
  reported "evaluations", and the package's own `result.evaluations` must be the same (a
  difference is printed to stderr);
- a run ends at the target, the evaluation budget or the time cap only (rule 2.1). On convergence
  a method starts again (rule 2.2): DE and CMA-ES (IPOP) with their own restarts; the others have
  no convergence criterion but the package's stall ("stalled": 10,000 generations without a
  genome to evaluate), after which the adapter starts them again with the seed
  `seed * 1000 + restart`, keeping the best and counting every evaluation;
- every evaluated solution stays inside the bounds, by genoxide's own bound handling, and the
  fitness functions count any outside them ("outside", rule 2.4);
- the clock covers creating the algorithm and the whole run, whose first step creates the random
  initial population (rule 4.1);
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
# and the real-valued ones count the genomes outside the bounds in OUTSIDE (rule 2.4).
# -------------------------------------------------------------------------------------------------

EVALUATIONS = 0
OUTSIDE = 0


def count(x, low, high):
    """Counts the rows of x, and those with a gene outside [low, high]."""
    global EVALUATIONS, OUTSIDE
    EVALUATIONS += len(x)
    OUTSIDE += int(np.count_nonzero(((x < low) | (x > high)).any(axis=1)))


def onemax(bits):
    """One genome, a numpy bool array: the matched runs' function, called once per genome like
    DEAP's, whose sum is numpy's."""
    global EVALUATIONS
    EVALUATIONS += 1
    return int(bits.sum())


def onemax_batch(bits):
    """A genome per row."""
    global EVALUATIONS
    EVALUATIONS += len(bits)
    return bits.sum(axis=1)


def nqueens_batch(orders):
    """Diagonal conflicts of each row: for each diagonal, its queens minus one, which is n minus
    the number of occupied diagonals, summed over both directions."""
    global EVALUATIONS
    EVALUATIONS += len(orders)
    count, size = orders.shape
    rows = np.arange(size)
    conflicts = np.zeros(count)
    for diagonals in (orders + rows, orders - rows):
        ordered = np.sort(diagonals, axis=1)
        occupied = 1 + np.count_nonzero(np.diff(ordered, axis=1), axis=1)
        conflicts += size - occupied
    return conflicts


def nqueens(order):
    """The diagonal conflicts of one genome, for local search and tabu search, which evaluate a
    few neighbors per step: as python/examples/n_queens.py."""
    global EVALUATIONS
    EVALUATIONS += 1
    size = len(order)
    rows = np.arange(size)
    return float(2 * size - len(np.unique(order + rows)) - len(np.unique(order - rows)))


# Rastrigin and Ackley are shifted, so an optimum at the origin can't favour operators that drift
# towards 0: gene i is measured from s_i = 2 ((37 i + 11) mod 101) / 101 - 1, in [-1, 1]
SHIFT = 2 * ((37 * np.arange(1000) + 11) % 101) / 101 - 1


def rastrigin_batch(x):
    count(x, -5.12, 5.12)
    y = x - SHIFT[: x.shape[1]]
    return 10 * x.shape[1] + np.sum(y**2 - 10 * np.cos(2 * np.pi * y), axis=1)


def rosenbrock_batch(x):
    count(x, -5.0, 10.0)
    a, b = x[:, :-1], x[:, 1:]
    return np.sum(100 * (b - a * a) ** 2 + (1 - a) ** 2, axis=1)


def ackley_batch(x):
    count(x, -32.768, 32.768)
    n = x.shape[1]
    y = x - SHIFT[:n]
    squares = np.sum(y**2, axis=1) / n
    cosines = np.sum(np.cos(2 * np.pi * y), axis=1) / n
    return -20 * np.exp(-0.2 * np.sqrt(squares)) - np.exp(cosines) + 20 + math.e


# the real-valued problems: fitness function and bounds
REAL_PROBLEMS = {
    "rastrigin": (rastrigin_batch, -5.12, 5.12),
    "rosenbrock": (rosenbrock_batch, -5.0, 10.0),
    "ackley": (ackley_batch, -32.768, 32.768),
}


def zdt_g(x):
    return 1 + 9 * x[:, 1:].sum(axis=1) / (x.shape[1] - 1)


def zdt1_batch(x):
    count(x, 0.0, 1.0)
    f1, g = x[:, 0], zdt_g(x)
    return np.column_stack([f1, g * (1 - np.sqrt(f1 / g))])


def zdt2_batch(x):
    count(x, 0.0, 1.0)
    f1, g = x[:, 0], zdt_g(x)
    return np.column_stack([f1, g * (1 - (f1 / g) ** 2)])


def zdt3_batch(x):
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


def dtlz2_batch(x, objectives=3):
    count(x, 0.0, 1.0)
    g = np.sum((x[:, objectives - 1:] - 0.5) ** 2, axis=1)
    return dtlz(x, 1 + g, objectives, lambda v: np.cos(v * np.pi / 2), lambda v: np.sin(v * np.pi / 2))


def dtlz1_batch(x, objectives=3):
    count(x, 0.0, 1.0)
    tail = x[:, objectives - 1:]
    g = 100 * (tail.shape[1] + np.sum((tail - 0.5) ** 2 - np.cos(20 * np.pi * (tail - 0.5)), axis=1))
    return dtlz(x, 0.5 * (1 + g), objectives, lambda v: v, lambda v: 1 - v)


# (fitness function, variables, objectives, population size, Das-Dennis divisions)
FRONT_PROBLEMS = {
    "zdt1": (zdt1_batch, lambda size: size, 2, 100, 99),
    "zdt2": (zdt2_batch, lambda size: size, 2, 100, 99),
    "zdt3": (zdt3_batch, lambda size: size, 2, 100, 99),
    # size: the number of objectives, with k = 10 (DTLZ2) and 5 (DTLZ1)
    "dtlz2": (dtlz2_batch, lambda size: size + 9, 3, 92, 12),
    "dtlz1": (dtlz1_batch, lambda size: size + 4, 3, 92, 12),
}

# -------------------------------------------------------------------------------------------------
# Solvers: the Rust genoxide adapter's, with the same settings and citations. Each is made by a
# function of the seed, so a restart can make it again with another seed.
# -------------------------------------------------------------------------------------------------

REAL_TARGET = 0.01


def onemax_solvers(size, mode):
    """[(solver, make the algorithm from a seed, fitness, batch)]"""
    if mode == "matched":
        # the matched settings (benchmarks/README.md), as DEAP's eaSimple: population 300,
        # tournament 3, two-point crossover with probability 0.5, bit-flip with probability
        # 1 / size per gene on 20% of the children, no elitism; a Python function per genome
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
    # idiomatic: genoxide's example for OneMax, the same in python/README.md's first example,
    # examples/one_max.rs and AGENTS.md: population 100, tournament 3, uniform crossover, bit-flip
    # at 1 / length per gene; the default rates and scheme (generational, elitism 1)
    return [("ga", lambda seed: gx.Ga(
        gx.Binary(size),
        population_size=100,
        select=gx.Tournament(3),
        crossover=gx.UniformCrossover(),
        mutation=gx.BitFlip(rate=1.0 / size),
        seed=seed,
    ), onemax_batch, True)]


def nqueens_solvers(size):
    return [
        # genoxide's example for N-Queens, examples/n_queens.rs, and AGENTS.md's permutation
        # template: (20 + 20) with tournament 2, no crossover and swap mutation
        ("ga", lambda seed: gx.Ga(
            gx.Permutation(size),
            population_size=20,
            select=gx.Tournament(2),
            crossover=gx.NoCrossover(),
            mutation=gx.SwapMutation(),
            scheme=gx.MuPlusLambda(20),
            objective="minimize",
            seed=seed,
        ), nqueens_batch, True),
        # local search, which AGENTS.md prefers on permutations, as the rustdoc's example for
        # N-Queens sets it: hill climbing with swap neighbors, the best of 4 per step, the default
        # acceptance NotWorse
        ("local_search", lambda seed: gx.LocalSearch(
            gx.Permutation(size),
            neighbor=gx.SwapMutation(),
            neighbors=4,
            objective="minimize",
            seed=seed,
        ), nqueens, False),
        # tabu search, the package's example for N-Queens (python/examples/n_queens.py): swap
        # neighbors, 32 per step, tenure 20
        ("tabu_search", lambda seed: gx.LocalSearch(
            gx.Permutation(size),
            neighbor=gx.SwapMutation(),
            neighbors=32,
            acceptance=gx.Tabu(20),
            objective="minimize",
            seed=seed,
        ), nqueens, False),
    ]


def real_solvers(problem, size):
    function, low, high = REAL_PROBLEMS[problem]
    genome = gx.Real((low, high), length=size)
    # CMA-ES with its defaults and IPOP restarts: python/README.md's first example and
    # python/examples/rastrigin.py, AGENTS.md's CMA-ES template; for Rosenbrock too (rule 2.2)
    cmaes = ("cma_es", lambda seed: gx.Cmaes(genome, restarts="ipop", objective="minimize", seed=seed),
             function, True)
    # differential evolution with the package's defaults (AGENTS.md's DE template): SHADE with
    # current-to-pbest/1 and an archive, the number of genes + 10 individuals, and restarts
    de = ("de", lambda seed: gx.De(genome, objective="minimize", seed=seed), function, True)
    if problem == "rosenbrock":
        # AGENTS.md's example for this problem, its PSO template (on Rosenbrock): 40 particles,
        # the global topology
        pso = ("pso", lambda seed: gx.Pso(genome, population_size=40, objective="minimize", seed=seed),
               function, True)
        return [cmaes, de, pso]
    # the Rust adapter runs the GA as AGENTS.md's island model, which the package doesn't have:
    # here it's one population, the GA of examples/rastrigin.rs (polynomial mutation at the
    # usual rate of 1 / length, elitism 2)
    ga = ("ga", lambda seed: gx.Ga(
        genome,
        population_size=100,
        select=gx.Tournament(3),
        crossover=gx.UniformCrossover(),
        mutation=gx.PolynomialMutation(20.0, rate=1.0 / size),
        scheme=gx.Generational(elitism=2),
        objective="minimize",
        seed=seed,
    ), function, True)
    return [ga, de, cmaes]


def front_solvers(problem, size):
    """The matched settings (benchmarks/README.md): NSGA-II, SPEA2 and SMS-EMOA with SBX with eta
    15 at 0.9 (their default rate) and polynomial mutation with eta 20 at 1 / n; NSGA-III (3
    objectives only, as AGENTS.md presents it) with SBX with eta 30 at 1; MOEA/D with SBX with eta
    20 at 1, 20 neighbors and neighborhood mating 0.9 (its defaults), Tchebycheff, or PBI with
    theta 5 with 3 objectives. SBX and polynomial mutation keep the genes in [0, 1] (rule 2.4)."""
    function, variables, objectives, population, divisions = FRONT_PROBLEMS[problem]
    n = variables(size)
    genome = gx.Real((0.0, 1.0), length=n)
    minimize = ["minimize"] * objectives
    mutation = gx.PolynomialMutation(20.0, rate=1.0 / n)
    sbx = gx.SimulatedBinaryCrossover(15.0)
    solvers = [
        ("nsga2", lambda seed: gx.Nsga2(genome, objectives=minimize, population_size=population,
                                        crossover=sbx, mutation=mutation, seed=seed)),
    ]
    if objectives > 2:
        solvers.append(("nsga3", lambda seed: gx.Nsga3(
            genome, objectives=minimize, reference_directions=gx.das_dennis(objectives, divisions),
            population_size=population, crossover=gx.SimulatedBinaryCrossover(30.0),
            mutation=mutation, seed=seed)))
    solvers += [
        ("spea2", lambda seed: gx.Spea2(genome, objectives=minimize, population_size=population,
                                        crossover=sbx, mutation=mutation, seed=seed)),
        ("sms_emoa", lambda seed: gx.SmsEmoa(genome, objectives=minimize, population_size=population,
                                             crossover=sbx, mutation=mutation, seed=seed)),
        ("moead", lambda seed: gx.Moead(
            genome, objectives=minimize, weights=gx.das_dennis(objectives, divisions),
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


def solve(common, solver, seed, make, function, batch, target, max_evaluations, max_seconds):
    """Runs one solver, with restarts after a stall (rule 2.2), and prints the run."""
    global EVALUATIONS, OUTSIDE
    EVALUATIONS = OUTSIDE = 0
    maximize = common["problem"] == "onemax"
    # CMA-ES with IPOP restarts doubles its population at each restart, so its last generation
    # can be larger than the average: the run reports it, for the budget check (rule 2.3). The
    # other solvers' generations don't grow.
    generation = {"evaluated": 0, "last": 0}

    def track(progress):
        generation["last"] = progress.evaluations - generation["evaluated"]
        generation["evaluated"] = progress.evaluations

    best = solution = None
    generations = reported = restart = 0
    # the clock covers creating the algorithm and the run, which creates the initial population
    start = time.perf_counter()
    while True:
        generation["evaluated"] = 0
        left = max_seconds - (time.perf_counter() - start)
        result = make(seed if restart == 0 else seed * 1000 + restart).run(
            function, batch=batch, target=target, evaluations=max_evaluations - EVALUATIONS,
            time=max(left, 0.0), on_generation=track if solver == "cma_es" else None)
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
        "best": best, "target": target, "success": success,
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
            # algorithms has a convergence criterion (rule 2.2); a stall would end a run before its
            # budget, which run.py check reports, and happened in no test run
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
    """The value of one solution (or its objectives), with the adapter's functions."""
    if problem == "onemax":
        return int(onemax_batch(np.array([solution], dtype=bool))[0])
    if problem == "nqueens":
        return float(nqueens_batch(np.array([solution], dtype=np.int64))[0])
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
    """Compares the fitness functions with problems.py at random points, and the per-genome
    N-Queens function with the batch one: run it after changing them."""
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    import problems

    rng = np.random.default_rng(7)

    def close(ours, theirs, name):
        if not problems.close(ours, theirs):
            raise SystemExit(f"{name}: {ours} != {theirs}")

    for size in (100, 1000):
        for bits in rng.random((20, size)) < 0.5:
            reference = problems.onemax(bits.tolist())
            close(onemax(bits), reference, "onemax")
            close(value("onemax", size, bits.tolist()), reference, "onemax batch")
    for size in (8, 32, 64):
        for order in (rng.permutation(size) for _ in range(50)):
            reference = problems.nqueens(order.tolist())
            close(nqueens(order), reference, "nqueens")
            close(value("nqueens", size, order.tolist()), reference, "nqueens batch")
    for name, (_, low, high) in REAL_PROBLEMS.items():
        for size in (10, 30):
            for x in rng.uniform(low, high, (20, size)):
                close(value(name, size, x.tolist()), problems.value(name, size, x.tolist()), name)
    for name, (_, variables, *_rest) in FRONT_PROBLEMS.items():
        size = 30 if name.startswith("zdt") else 3
        for x in rng.random((20, variables(size))):
            close(value(name, size, x.tolist()), problems.value(name, size, x.tolist()), name)
    # the outside count: a row with one gene past a bound counts once
    global OUTSIDE
    OUTSIDE = 0
    rastrigin_batch(np.array([[0.0, 5.12], [0.0, 5.13], [-6.0, 6.0]]))
    if OUTSIDE != 2:
        raise SystemExit(f"outside: {OUTSIDE} instead of 2")
    print("the fitness functions match problems.py")


if __name__ == "__main__":
    main()
