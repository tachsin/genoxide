"""Benchmark adapter for genoxide's Python package (../../../python), whose algorithms are genoxide's
Rust, calling Python fitness functions.

Usage: python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
Prints one JSON line per solver per seed, see ../../README.md for the fields.

The solvers and their settings are the Rust genoxide adapter's (../genoxide/src/main.rs), through
the Python API:
- matched OneMax: a pure-Python fitness function, called with one genome at a time, like DEAP's;
- idiomatic scenarios and multi-objective scenarios: what the package's README recommends,
  `batch=True` with vectorized numpy functions, called with a generation at a time.

`python bench.py --self-check` compares the fitness functions with the DEAP adapter's at random
points (it needs DEAP, as in the benchmarks' .venv).
"""

import json
import math
import sys
import time

import numpy as np

import genoxide as gx

# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to the ones in the other adapters
# -------------------------------------------------------------------------------------------------


def onemax(bits):
    """One genome, a numpy bool array: the matched runs' pure-Python function, like DEAP's."""
    return int(bits.sum())


def onemax_batch(bits):
    """A genome per row."""
    return bits.sum(axis=1)


def nqueens_batch(orders):
    """Number of diagonal conflicts of each row: the queens on a diagonal, minus 1 if there are
    any, summed over the diagonals, which is n minus the number of occupied diagonals."""
    count, size = orders.shape
    rows = np.arange(size)
    conflicts = np.zeros(count)
    for diagonals in (orders + rows, orders - rows):
        ordered = np.sort(diagonals, axis=1)
        occupied = 1 + np.count_nonzero(np.diff(ordered, axis=1), axis=1)
        conflicts += size - occupied
    return conflicts


# Rastrigin and Ackley are shifted, so an optimum at the origin can't favour operators that drift
# towards 0: gene i is measured from s_i = 2 ((37 i + 11) mod 101) / 101 - 1, in [-1, 1]
SHIFT = 2 * ((37 * np.arange(1000) + 11) % 101) / 101 - 1


def rastrigin_batch(x):
    y = x - SHIFT[: x.shape[1]]
    return 10 * x.shape[1] + np.sum(y**2 - 10 * np.cos(2 * np.pi * y), axis=1)


def rosenbrock_batch(x):
    a, b = x[:, :-1], x[:, 1:]
    return np.sum(100 * (b - a * a) ** 2 + (1 - a) ** 2, axis=1)


def ackley_batch(x):
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
    f1, g = x[:, 0], zdt_g(x)
    return np.column_stack([f1, g * (1 - np.sqrt(f1 / g))])


def zdt2_batch(x):
    f1, g = x[:, 0], zdt_g(x)
    return np.column_stack([f1, g * (1 - (f1 / g) ** 2)])


def zdt3_batch(x):
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
    g = np.sum((x[:, objectives - 1:] - 0.5) ** 2, axis=1)
    return dtlz(x, 1 + g, objectives, lambda v: np.cos(v * np.pi / 2), lambda v: np.sin(v * np.pi / 2))


def dtlz1_batch(x, objectives=3):
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
# Solvers: the Rust genoxide adapter's, with the same settings
# -------------------------------------------------------------------------------------------------

RASTRIGIN_TARGET = 0.01


def onemax_solvers(size, mode, seed):
    """[(solver, algorithm, fitness, batch, target, success)]"""
    if mode == "matched":
        # as DEAP eaSimple: population 300, tournament 3, two-point crossover with probability
        # 0.5, bit-flip with probability 1 / size on 20% of the children, no elitism; a Python
        # function per genome
        ga = gx.Ga(
            gx.Binary(size),
            population_size=300,
            select=gx.Tournament(3),
            crossover=gx.PointCrossover(2),
            crossover_rate=0.5,
            mutation=gx.BitFlip(rate=1.0 / size),
            mutation_rate=0.2,
            scheme=gx.Generational(elitism=0),
            seed=seed,
        )
        return [("ga", ga, onemax, False, size, lambda best: best >= size)]
    # the binary template of AGENTS.md, with a numpy function per generation
    ga = gx.Ga(
        gx.Binary(size),
        population_size=100,
        select=gx.Tournament(3),
        crossover=gx.PointCrossover(2),
        mutation=gx.BitFlip(rate=1.0 / size),
        seed=seed,
    )
    return [("ga", ga, onemax_batch, True, size, lambda best: best >= size)]


def nqueens_solvers(size, seed):
    # the permutation template of AGENTS.md: (mu + lambda) with swap mutation
    ga = gx.Ga(
        gx.Permutation(size),
        population_size=20,
        select=gx.Tournament(2),
        crossover=gx.NoCrossover(),
        mutation=gx.SwapMutation(),
        scheme=gx.MuPlusLambda(20),
        objective="minimize",
        seed=seed,
    )
    # the local search template of AGENTS.md: one neighbor per step, and moves to equal neighbors
    search = gx.LocalSearch(
        gx.Permutation(size),
        neighbor=gx.SwapMutation(),
        acceptance=gx.NotWorse(),
        objective="minimize",
        seed=seed,
    )
    solved = lambda best: best == 0  # noqa: E731
    return [
        ("ga", ga, nqueens_batch, True, 0, solved),
        ("local_search", search, nqueens_batch, True, 0, solved),
    ]


def real_solvers(problem, size, seed):
    function, low, high = REAL_PROBLEMS[problem]
    genome = gx.Real((low, high), length=size)
    # the settings of examples/rastrigin.rs: polynomial mutation at the usual rate of 1 / length
    ga = gx.Ga(
        genome,
        population_size=100,
        select=gx.Tournament(3),
        crossover=gx.UniformCrossover(),
        mutation=gx.PolynomialMutation(20.0, rate=1.0 / size),
        scheme=gx.Generational(elitism=2),
        objective="minimize",
        seed=seed,
    )
    # differential evolution with its defaults: SHADE with current-to-pbest/1 and an archive, the
    # number of genes + 10 individuals, and restarts
    de = gx.De(genome, objective="minimize", seed=seed)
    # CMA-ES with its defaults and IPOP restarts, for a multimodal function
    cmaes = gx.Cmaes(genome, restarts="ipop", objective="minimize", seed=seed)
    reached = lambda best: best <= RASTRIGIN_TARGET  # noqa: E731
    return [
        (name, algorithm, function, True, RASTRIGIN_TARGET, reached)
        for name, algorithm in (("ga", ga), ("de", de), ("cma_es", cmaes))
    ]


def front_solvers(problem, size, seed):
    """The matched settings of every library: SBX with eta 15 at 0.9 and polynomial mutation with
    eta 20 at 1 / n; MOEA/D and NSGA-III with their usual SBX (eta 20 and 30 at 1)."""
    function, variables, objectives, population, divisions = FRONT_PROBLEMS[problem]
    n = variables(size)
    genome = gx.Real((0.0, 1.0), length=n)
    minimize = ["minimize"] * objectives
    mutation = gx.PolynomialMutation(20.0, rate=1.0 / n)
    sbx = gx.SimulatedBinaryCrossover(15.0)
    solvers = [
        ("nsga2", gx.Nsga2(genome, objectives=minimize, population_size=population, crossover=sbx,
                           mutation=mutation, seed=seed)),
    ]
    if objectives > 2:
        solvers.append(("nsga3", gx.Nsga3(genome, objectives=minimize,
                                          reference_directions=gx.das_dennis(objectives, divisions),
                                          population_size=population,
                                          crossover=gx.SimulatedBinaryCrossover(30.0), mutation=mutation,
                                          seed=seed)))
    solvers += [
        ("spea2", gx.Spea2(genome, objectives=minimize, population_size=population, crossover=sbx,
                           mutation=mutation, seed=seed)),
        ("sms_emoa", gx.SmsEmoa(genome, objectives=minimize, population_size=population, crossover=sbx,
                                mutation=mutation, seed=seed)),
        ("moead", gx.Moead(genome, objectives=minimize, weights=gx.das_dennis(objectives, divisions),
                           decomposition=gx.Tchebycheff() if objectives == 2 else gx.Pbi(5.0),
                           crossover=gx.SimulatedBinaryCrossover(20.0), mutation=mutation, seed=seed)),
    ]
    return function, solvers


# -------------------------------------------------------------------------------------------------
# Runs
# -------------------------------------------------------------------------------------------------


def main():
    if sys.argv[1:] == ["--self-check"]:
        self_check()
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
            # multi-objective runs have a budget and no target: they print their final front
            function, solvers = front_solvers(problem, size, seed)
            for solver, algorithm in solvers:
                start = time.perf_counter()
                result = algorithm.run(function, batch=True, evaluations=max_evaluations, time=max_seconds)
                elapsed = time.perf_counter() - start
                print(json.dumps({
                    **common, "solver": solver, "seed": seed, "time_s": round(elapsed, 6),
                    "generations": result.generations, "evaluations": result.evaluations,
                    "front": result.front_objectives.tolist(),
                }), flush=True)
            continue

        if problem == "onemax":
            solvers = onemax_solvers(size, mode, seed)
        elif problem == "nqueens":
            solvers = nqueens_solvers(size, seed)
        elif problem in REAL_PROBLEMS:
            solvers = real_solvers(problem, size, seed)
        else:
            print(f"unknown problem {problem}", file=sys.stderr)
            sys.exit(2)

        for solver, algorithm, function, batch, target, is_success in solvers:
            # times building the algorithm (the random initial population) and running it
            start = time.perf_counter()
            result = algorithm.run(function, batch=batch, target=target, evaluations=max_evaluations,
                                   time=max_seconds)
            elapsed = time.perf_counter() - start
            best = result.best_fitness
            print(json.dumps({
                **common, "solver": solver, "seed": seed, "time_s": round(elapsed, 6),
                "generations": result.generations, "evaluations": result.evaluations,
                "best": best, "target": target, "success": best is not None and bool(is_success(best)),
            }), flush=True)


def self_check():
    """Compares the fitness functions with the DEAP adapter's at random points: run it once after
    changing them."""
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "deap" / "bench.py"
    spec = importlib.util.spec_from_file_location("deap_bench", path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    rng = np.random.default_rng(7)

    def close(ours, theirs, name):
        ours, theirs = np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float)
        if not np.allclose(ours, theirs, rtol=1e-12, atol=1e-12):
            raise SystemExit(f"{name}: {ours} != {theirs}")

    for size in (100, 1000):
        bits = rng.random((20, size)) < 0.5
        close([onemax(row) for row in bits], [reference.onemax(list(row))[0] for row in bits], "onemax")
        close(onemax_batch(bits), [reference.onemax(list(row))[0] for row in bits], "onemax batch")
    for size in (8, 32, 64):
        orders = np.array([rng.permutation(size) for _ in range(50)])
        close(nqueens_batch(orders), [reference.nqueens([int(v) for v in row])[0] for row in orders], "nqueens")
    for name, (function, low, high) in REAL_PROBLEMS.items():
        for size in (10, 30):
            x = rng.uniform(low, high, (20, size))
            close(function(x), [reference.REAL_PROBLEMS[name][0](list(row))[0] for row in x], name)
            if name != "rosenbrock":
                close(function(SHIFT[:size][None, :]), [0.0], f"{name} at the shift")
    close(rosenbrock_batch(np.ones((1, 10))), [0.0], "rosenbrock at 1")
    for name, (function, variables, *_rest) in FRONT_PROBLEMS.items():
        n = variables(30 if name.startswith("zdt") else 3)
        x = rng.random((20, n))
        close(function(x), [reference.FRONT_PROBLEMS[name][0](list(row)) for row in x], name)
    print("the fitness functions match the DEAP adapter's")


if __name__ == "__main__":
    main()
