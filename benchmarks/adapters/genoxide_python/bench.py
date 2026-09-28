"""Benchmark adapter for genoxide's Python package (../../../python), whose algorithms are genoxide's
Rust, calling Python fitness functions.

Usage:
    python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
    python bench.py values <problem> <size>
    python bench.py --self-check

The first prints one JSON line per seed, with the best solution, see ../../README.md for the
fields. The second reads one JSON solution per line from stdin and prints its value, with the
fitness functions below. The third compares the fitness functions with problems.py at random
points.

The suite is matched: each problem has one method, defined in docs/benchmarks/rules.md (rule 6),
and every library runs it with its own implementation. The package runs the three, as the Rust
adapter does:
- OneMax 1000: the GA of DEAP's eaSimple (`gx.Ga`);
- Rastrigin 30: DE/rand/1/bin with F 0.5, CR 0.9 and 100 individuals (`gx.De`);
- Rosenbrock 10: CMA-ES with its defaults and no restarts (`gx.Cmaes`).
Any other problem, size or mode prints nothing. How each setting maps to the definition:
docs/benchmarks/libraries/genoxide_python.md. The fitness functions are numpy, with `batch=True`, a
generation per call, as python/README.md's Rastrigin example ("one call per generation") and its
"Fitness functions" section ("Vectorized numpy ... pays its cost per call once per generation, not
once per genome") write a fast one, and as rule 3.4 asks of a Python library with a documented
batch interface. A batch is the same algorithm: the package asks for the same genomes either way
("The same seed repeats a run exactly: one genome at a time, in batches or in parallel").

The rules (docs/benchmarks/rules.md), as this adapter follows them:
- each fitness function counts the genomes it evaluates itself, a row of a batch each (rule 3),
  and records the first one that reaches the target, in the batch's order ("first_hit"): that
  count is the reported "evaluations", and the package's own `result.evaluations` must be the
  same (a difference is printed to stderr);
- a run ends at the target, the evaluation budget or the time cap only (rule 2.1). No method has
  a convergence criterion or restarts. The GA can stall, 10 generations in a row without a genome
  to evaluate (a child identical to a parent isn't evaluated): its `on_generation` callback then
  ends the attempt and the adapter starts it again with the seed `(seed + 1) * 1_000_000 +
  restart`, keeping the best and counting every evaluation (rule 2.2). DE and CMA-ES evaluate
  every trial and sample, so they never stall, and have no callback. The time cap is the
  package's `time` stop, for every method;
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

import functools
import json
import sys
import time

import numpy as np

import genoxide as gx

# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to problems.py. Each counts the genomes it evaluates in EVALUATIONS,
# the real-valued ones count the genomes outside the bounds in OUTSIDE (rule 2.4), and each records
# the first evaluation that reaches the target in HIT (rule 3.3).
# -------------------------------------------------------------------------------------------------

EVALUATIONS = 0
OUTSIDE = 0
# the rows of the latest batch: with batch=True a call is a generation (rule 2.3)
BATCH = 0
# the target of the run (None outside a run), whether it's a maximum, the start of the run's clock,
# and the first hit: {"evaluations": E, "time_s": T}
HIT = {"target": None, "maximize": False, "start": 0.0, "first": None}


def count(x, low, high):
    """Counts the rows of x, and those with a gene outside [low, high]."""
    global EVALUATIONS, OUTSIDE, BATCH
    EVALUATIONS += len(x)
    BATCH = len(x)
    OUTSIDE += int(np.count_nonzero(((x < low) | (x > high)).any(axis=1)))


def record_batch(values):
    """Records the first hit of the target among the values of the latest batch, in order."""
    target = HIT["target"]
    if target is None or HIT["first"] is not None:
        return
    hits = np.flatnonzero(values >= target if HIT["maximize"] else values <= target)
    if len(hits):
        HIT["first"] = {"evaluations": EVALUATIONS - len(values) + int(hits[0]) + 1,
                        "time_s": time.perf_counter() - HIT["start"]}


def onemax(x):
    """A generation, a genome per row (numpy bools): the ones of each row, numpy's sum (python/
    README.md's OneMax, `bits.sum()`, over a batch)."""
    global EVALUATIONS, BATCH
    EVALUATIONS += len(x)
    BATCH = len(x)
    values = x.sum(axis=1)
    record_batch(values)
    return values


@functools.cache
def shift(upper, n):
    """The optimum of Rastrigin (rule 1.4): s_i = 0.8 upper (2 ((37 i + 11) mod 101) /
    101 - 1), with `upper` the box's upper bound, computed in this order, once per size."""
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


# the real-valued problems: fitness function (a generation per call) and bounds
REAL_PROBLEMS = {
    "rastrigin": (rastrigin, -5.12, 5.12),
    "rosenbrock": (rosenbrock, -5.0, 10.0),
}

# -------------------------------------------------------------------------------------------------
# The methods of the matched suite (docs/benchmarks/rules.md, rule 6), one per problem, each made
# by a function of the seed, so a restart after a stall can make it again with another seed
# -------------------------------------------------------------------------------------------------

REAL_TARGET = 0.01


def onemax_ga(size):
    """OneMax 1000: the GA of rule 6.2, DEAP's eaSimple: population 300, tournament 3 (with
    replacement), two-point crossover of each consecutive pair with probability 0.5, each child
    mutated with probability 0.2 by a bit-flip at 1 / size per gene, generational replacement
    without elitism; the package's own components. (solver, make, fitness, batch)"""
    return ("ga", lambda seed: gx.Ga(
        gx.Binary(size),
        population_size=300,
        select=gx.Tournament(3),
        crossover=gx.PointCrossover(2),
        crossover_rate=0.5,
        mutation=gx.BitFlip(rate=1.0 / size),
        mutation_rate=0.2,
        scheme=gx.Generational(elitism=0),
        seed=seed,
    ), onemax, True)


def rastrigin_de(size):
    """Rastrigin 30, with no target (a fixed budget): DE/rand/1/bin of rule 6.3, `gx.De` with the
    strategy "rand1", a fixed F 0.5
    and CR 0.9, 100 individuals and no restarts. A trial gene outside the box is set halfway
    between the target's gene and the bound (rule 2.4). (solver, make, fitness, batch)"""
    genome = gx.Real((-5.12, 5.12), length=size)
    return ("de", lambda seed: gx.De(
        genome,
        population_size=100,
        strategy="rand1",
        control={"f": 0.5, "cr": 0.9},
        restarts="never",
        objective="minimize",
        seed=seed,
    ), rastrigin, True)


def rosenbrock_cmaes(size):
    """Rosenbrock 10: CMA-ES of rule 6.4, `gx.Cmaes` with its defaults (4 + 3 ln n samples, an
    initial step of 0.3 of each range, a mean drawn uniformly in the box) and no restarts. A
    sample outside the box is drawn again, up to 100 times, and then clipped (rule 2.4).
    (solver, make, fitness, batch)"""
    genome = gx.Real((-5.0, 10.0), length=size)
    return ("cma_es", lambda seed: gx.Cmaes(genome, restarts="never", objective="minimize", seed=seed),
            rosenbrock, True)


# the matched suite's scenarios: (problem, size, mode) -> (method, target)
SCENARIOS = {
    ("onemax", 1000, "matched"): (onemax_ga, 1000),
    # no target: every run uses the fixed budget
    ("rastrigin", 30, "matched"): (rastrigin_de, None),
    ("rosenbrock", 10, "matched"): (rosenbrock_cmaes, REAL_TARGET),
}


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


# an attempt whose generations evaluate nothing for this many in a row has converged (rule 2.2)
STALL_GENERATIONS = 10


def solve(common, solver, seed, make, function, batch, target, max_evaluations, max_seconds):
    """Runs one solver, with restarts after a stall (rule 2.2), and prints the run."""
    global EVALUATIONS, OUTSIDE, BATCH
    EVALUATIONS = OUTSIDE = BATCH = 0
    maximize = common["problem"] == "onemax"
    # the GA is the one solver here that can stall (every child a copy of a parent): DE and CMA-ES
    # evaluate new genomes every generation. Its `on_generation` callback ends an attempt after
    # STALL_GENERATIONS generations without an evaluation: a batch function isn't called for such
    # a generation, so the callback is the only way to see them
    can_stall = solver == "ga"
    # the last generation's evaluations, for the budget check (rule 2.3): the GA's callback, called
    # after every generation (each attempt's initial population included), records the count, the
    # generation's size and the generations in a row without an evaluation; see the end of the run
    # for the other solvers
    generation = {"end": 0, "last": 0, "idle": 0}

    def not_stalled(progress):
        generation["last"] = EVALUATIONS - generation["end"]
        generation["end"] = EVALUATIONS
        generation["idle"] = 0 if generation["last"] else generation["idle"] + 1
        return generation["idle"] < STALL_GENERATIONS

    best = solution = None
    generations = reported = restart = 0
    HIT.update(target=target, maximize=maximize, first=None)
    # the clock starts before the algorithm is made and its initial population created
    start = HIT["start"] = time.perf_counter()
    while True:
        budget = max_evaluations - EVALUATIONS
        generation["idle"] = 0
        # the time cap: the package's `time` stop, the time left, checked after every generation
        seconds = max(max_seconds - (time.perf_counter() - start), 0.0)
        callback = {"on_generation": not_stalled} if can_stall else {}
        result = make(attempt_seed(seed, restart)).run(
            function, batch=batch, target=target, evaluations=budget, time=seconds, **callback)
        generations += result.generations
        reported += result.evaluations
        score = result.best_fitness
        if score is not None and (best is None or (score > best if maximize else score < best)):
            best, solution = score, result.best_genome
        if (generation["idle"] < STALL_GENERATIONS or EVALUATIONS >= max_evaluations
                or time.perf_counter() - start >= max_seconds):
            break
        restart += 1
    elapsed = time.perf_counter() - start
    HIT["target"] = None
    compare(common, solver, seed, EVALUATIONS, reported)
    if can_stall:
        last_generation = EVALUATIONS - generation["end"] or generation["last"]
    else:
        # with batch=True a call is a generation
        last_generation = BATCH
    extra = {"last_generation": last_generation}
    if common["problem"] in REAL_PROBLEMS:
        extra["outside"] = OUTSIDE
    if restart:
        extra["restarts"] = restart
    # a fixed budget (no target) is never a success
    success = (best is not None and target is not None
               and (best >= target if maximize else best <= target))
    print(json.dumps({
        **common, "solver": solver, "seed": seed, "time_s": round(elapsed, 6),
        "generations": generations, "evaluations": EVALUATIONS, **extra,
        "first_hit": HIT["first"], "best": best, "target": target, "success": success,
        "solution": (solution.astype(int) if maximize else solution).tolist(),
    }), flush=True)


def main():
    global EVALUATIONS, OUTSIDE, BATCH
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

    if (problem, size, mode) not in SCENARIOS:
        # not a scenario of the matched suite: nothing to run
        return
    method, target = SCENARIOS[problem, size, mode]
    solver, make, function, batch = method(size)
    for seed in range(seed_from, seed_to + 1):
        solve(common, solver, seed, make, function, batch, target, max_evaluations, max_seconds)


def value(problem, size, solution):
    """The value of one solution, with the functions the runs call: a batch of one row."""
    if problem == "onemax":
        return int(onemax(np.array([solution], dtype=bool))[0])
    x = np.array([solution], dtype=np.float64)
    return float(REAL_PROBLEMS[problem][0](x)[0])


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

    for size in (1000,):
        for bits in rng.random((20, size)) < 0.5:
            close(value("onemax", size, bits.tolist()), problems.onemax(bits.tolist()), "onemax")
    for name, (_, low, high) in REAL_PROBLEMS.items():
        for size in (10, 30):
            for x in rng.uniform(low, high, (20, size)):
                close(value(name, size, x.tolist()), problems.value(name, size, x.tolist()), name)
    close(value("rastrigin", 30, problems.shift("rastrigin", 30)), 0.0, "rastrigin at the optimum")
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
