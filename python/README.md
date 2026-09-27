<p align="center">
  <img src="https://raw.githubusercontent.com/tachsin/genoxide/main/assets/brand/banner.svg" alt="genoxide: evolutionary computation for Rust and Python" width="100%">
</p>

# genoxide for Python

The algorithms of [genoxide](https://github.com/tachsin/genoxide), a Rust library, with fitness functions in Python and numpy:

- genetic algorithms and local search
- differential evolution, CMA-ES and particle swarm optimization
- NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA for several objectives

The API reference: [tachsin.github.io/genoxide/api/python](https://tachsin.github.io/genoxide/api/python/)

```python
import numpy as np
import genoxide as gx

# OneMax: the genome with the most ones
ga = gx.Ga(
    gx.Binary(100),
    population_size=100,
    select=gx.Tournament(3),
    crossover=gx.UniformCrossover(),
    mutation=gx.BitFlip(rate=0.01),
    seed=42,
)
result = ga.run(lambda bits: bits.sum(), target=100, generations=1_000)
print(result.best_fitness, result.generations)

# Rastrigin with CMA-ES and IPOP restarts, a generation per call
def rastrigin(x):  # x: a genome per row
    return 10 * x.shape[1] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)

cmaes = gx.Cmaes(gx.Real((-5.12, 5.12), length=10), restarts="ipop", objective="minimize", seed=1)
result = cmaes.run(rastrigin, batch=True, target=1e-8, evaluations=500_000)
print(result.best_genome, result.best_fitness)
```

More in [examples/](https://github.com/tachsin/genoxide/tree/main/examples), each the same program in Python and Rust
(`python examples/<name>/main.py` in the repository), on the [docs site](https://tachsin.github.io/genoxide/examples/), and
played back with charts made for each problem on [tachsin.gr](https://tachsin.gr/projects/genoxide/examples):
- OneMax, a knapsack with a constraint, and N-Queens
- the travelling salesman (TSPLIB berlin52) and job shop scheduling (ft06)
- Rastrigin with CMA-ES and L-SHADE, and the pressure vessel and welded beam designs with constraints
- the gear train design, an integer problem
- CMA-ES, SHADE and PSO on twelve test functions, and Himmelblau's four minima by restarts of a local search
- ZDT1 with NSGA-II, and DTLZ2 with NSGA-III
- the constrained BNH with NSGA-II, and Kursawe's disconnected front with SPEA2 and NSGA-II
- XOR neuroevolution with CMA-ES

## Install

```sh
pip install genoxide
```

Wheels for CPython 3.10 or later, with numpy:
- Linux: x86_64 and aarch64, glibc and musl
- macOS: Apple silicon and Intel
- Windows: x64

To build from the repository, with Rust and [maturin](https://www.maturin.rs/):

```sh
cd python
pip install maturin
maturin develop --release
```

## Fitness functions

A fitness function takes a genome as a numpy array:

| Genome | Array |
|---|---|
| `Binary(length)` | `bool` |
| `Integer(bounds, length)` | `int64` |
| `Real(bounds, length)` | `float64` |
| `Permutation(length)` | `int64`, an ordering of `0 .. length - 1` |

`bounds` is one pair `(low, high)` for every gene, with `length`, or a list of pairs, one per gene.

It returns one of these:
- a number
- `None` or NaN, for a solution that can't be scored
- `(score, constraint_violation)`: a positive violation marks an infeasible solution. Infeasible solutions rank below feasible ones, and among themselves by violation (Deb's rules).

The fitness function must be deterministic: genoxide doesn't evaluate again a child identical to one of its parents.

With `batch=True`, the function takes a whole generation as a 2-D array, a genome per row. It returns:
- an array of scores, or a tuple of scores and constraint violations (arrays or `(n, 1)` columns)
- for several objectives, a 2-D array or a list with a row of objective values per genome, or a tuple of one array per objective (`return f1, f2`)

It's one call per generation, and none for a generation whose children are all copies of their parents. Vectorized numpy, a GPU or a remote service pays its cost per call once per generation, not once per genome.

With `parallel=True`, genoxide calls a non-batch function from several threads at once. It pays off when the function releases the GIL (numpy on large arrays, waiting for I/O), or on free-threaded Python.

An exception in the fitness function stops the run, and `run` raises it. So does Ctrl+C.

## Choosing an algorithm

| Problem | Genome | Algorithms, the first preferred |
|---|---|---|
| Yes / no choices (subsets) | `Binary` | `Ga` with `UniformCrossover()` or `PointCrossover(points)`, and `BitFlip` |
| An order (tours, sequencing) | `Permutation` | `LocalSearch`, which often beats a GA on permutations; `Ga` with `OrderCrossover()` (sequences) or `EdgeRecombinationCrossover()` (tours) |
| Reals in ranges | `Real` | `Cmaes`; `De`; `Ga` with `SimulatedBinaryCrossover(eta)` and `PolynomialMutation(eta)` |
| Several objectives | any | `Nsga2` for 2 or 3 objectives; `Nsga3` or `Moead` for more |

- `Cmaes` is the strongest general choice for continuous problems with up to a few hundred genes, especially when the genes interact. Its defaults need no tuning. For multimodal functions, add `restarts="ipop"` or `"bipop"`.
- `De` often needs far fewer evaluations than a GA on continuous problems.
- `Pso` with `ring=1` explores longer than the default global topology, for multimodal functions.
- For smooth problems that need precise answers, the Rust library also has an evolution strategy, `Es`.

## Algorithms

| Algorithm | Genomes | Settings |
|---|---|---|
| `Ga` | all | `population_size`, `select`, `crossover`, `mutation`, `crossover_rate` (0.9), `mutation_rate` (1), `scheme` |
| `LocalSearch` | all | `neighbor` (a mutation), `neighbors` (1), `acceptance`, `restart=(patience, kicks)` |
| `De` | real | `population_size` (100; with `l_shade`, 18 × genes, at least 4), `l_shade` (a budget of evaluations, for L-SHADE) |
| `Cmaes` | real | `population_size`, `restarts` (`"ipop"`, `"bipop"`), `initial_step` |
| `Pso` | real | `population_size` (needed), `ring` (neighbors on each side) |
| `Nsga2` | all | `objectives`, `population_size`, `crossover`, `mutation`, `crossover_rate` (0.9), `mutation_rate` (1) |
| `Nsga3` | all | `objectives`, `reference_directions`, `crossover`, `mutation`, `population_size` (the number of reference directions), `crossover_rate` (1), `mutation_rate` (1) |
| `Spea2` | all | `objectives`, `population_size` (the archive's), `crossover`, `mutation`, `crossover_rate` (0.9), `mutation_rate` (1) |
| `Moead` | all | `objectives`, `weights` (a subproblem each), `crossover`, `mutation`, `decomposition` (`Tchebycheff()`), `neighbors` (20), `neighbor_mating` (0.9), `max_replacements` (2), `crossover_rate` (1), `mutation_rate` (1) |
| `SmsEmoa` | all | `objectives`, `population_size`, `crossover`, `mutation`, `offspring` (`population_size`), `crossover_rate` (0.9), `mutation_rate` (1) |

Single-objective algorithms maximize, or minimize with `objective="minimize"`.

The multi-objective algorithms take `objectives=["minimize", "maximize", ...]`, 2 to 6 of them. Their fitness function returns a sequence of objective values. Their result is the final non-dominated front: `front_genomes`, `front_objectives` and `front_violations`.

A solution that couldn't be scored has NaN objective values and a NaN violation. A single-objective result without a valid solution has a NaN `violation`.

- `Nsga2` spreads the front by crowding distance, which works poorly beyond 2 or 3 objectives.
- `Nsga3` spreads it along reference directions instead.
- `Moead` solves a single-objective subproblem per weight vector.
- `das_dennis(objectives, divisions)` gives evenly spread directions or weights for `Nsga3` and `Moead`, a row each: 91 for 3 objectives and 12 divisions.
- `Spea2` keeps an archive of the best solutions, the non-dominated ones first, truncated by the distance to their nearest neighbors.
- `Nsga2`, `Nsga3`, `Spea2` and `SmsEmoa` drop a child that equals a member of the population or an earlier child, and breed another. `eliminate_duplicates=False` keeps copies.
- `SmsEmoa` removes the solutions that contribute the least hypervolume, from the last front that fits partly. It costs more per generation than `Nsga2`: O(N log N) per removal for 2 objectives, O(N²) for 3, O(N³) for 4 and O(N⁴) for 5. For 4 or 5 objectives, `Nsga3` or `Moead` are better choices.

Every algorithm takes a `seed`. The same seed repeats a run exactly: one genome at a time, in batches or in parallel.

Operators:
- **Selection:** `Tournament(size)`, `Rank(pressure)`, `Roulette()`, `StochasticUniversalSampling()`, `Truncation(fraction)`, `RandomSelection()`
- **Crossover:**
  - any list genome: `UniformCrossover()`, `PointCrossover(points)`, `NoCrossover()`
  - real genomes: `SimulatedBinaryCrossover(eta)`, `BlendCrossover(alpha)`, `ArithmeticCrossover()`
  - permutations: `OrderCrossover()`, `PartiallyMappedCrossover()`, `CycleCrossover()`, `EdgeRecombinationCrossover()`
- **Mutation:**
  - binary genomes: `BitFlip(rate=... | count=...)`
  - integer and real genomes: `UniformMutation(rate=... | count=...)`
  - real genomes: `GaussianMutation(sigma, rate=... | count=...)`, `PolynomialMutation(eta, rate=... | count=...)`
  - permutations: `SwapMutation(count)`, `InversionMutation()`, `InsertionMutation()`, `ScrambleMutation()`
- **Genetic algorithm schemes:** `Generational(elitism)` (the default, with 1), `SteadyState(replacements)`, `MuPlusLambda(offspring)`, `MuCommaLambda(offspring)`
- **Local search acceptance:** `NotWorse()` (the default), `Improving()`, `Annealing(initial_temperature, cooling)`, `Tabu(tenure)`
- **MOEA/D decomposition:** `Tchebycheff()` (the default), `Pbi(theta)` (penalty-based boundary intersection, `theta` 5 by default). `Pbi` spreads fronts of 3 or more objectives well.

## Test problems and indicators

`gx.problems` has classic test functions from the literature, such as `Rastrigin(dimensions)`,
`Rosenbrock(dimensions)` and `Branin()`, all minimized. Each gives its `genome`, `objective`,
`optimum` (`value`, `solutions`, `proven`) and `reference`, and is a fitness function that `run`
evaluates in Rust, with no Python call: `parallel=True` uses every core, and a seed gives the same
result as in Rust. `problem(x)` and `problem.evaluate(genomes)` run the same code.

```python
problem = gx.problems.Rastrigin(10)
de = gx.De(problem.genome, objective=problem.objective, seed=1)
result = de.run(problem, target=problem.optimum.value + 1e-8, evaluations=200_000)
print(result.best_fitness, result.evaluations, problem.reference)
```

The multi-objective problems, `Zdt1` to `Zdt6`, `Dtlz1(objectives, variables)` to `Dtlz4`,
`Schaffer1`, `Schaffer2`, `FonsecaFleming`, `Kursawe`, `Poloni`, `Viennet1` to `Viennet3` and the
constrained `Bnh`, `Srn`, `Tnk`, `Osy` and `Constr`, run with the multi-objective algorithms in the
same way. Each gives its `objectives`, for the algorithm, and `optimal_front(points)`, None where
the front isn't known; a constrained one returns `(objectives, violation)` and gives its
`constraints(x)`.

Two submodules have constrained single-objective problems, whose fitness is `(score,
violation)`: `gx.problems.cec2006` has CEC 2006's `G01()` to `G06()`, and
`gx.problems.engineering` has `WeldedBeam()`, `WeldedBeamRagsdell()`, `PressureVessel()`,
`TensionCompressionSpring()`, `SpeedReducer()`, `GearTrain()` (an `Integer` genome),
`ThreeBarTruss()`, `CantileverBeam()` and `CarSideImpact()`. `PressureVessel` and `SpeedReducer`
round their discrete genes, and `design(x)` gives the rounded design.

```python
problem = gx.problems.engineering.WeldedBeam()
de = gx.De(problem.genome, objective=problem.objective, seed=1)
result = de.run(problem, evaluations=40_000)
print(result.best_fitness, result.violation, problem.optimum.value)
```

```python
problem = gx.problems.Dtlz2(objectives=3)
nsga3 = gx.Nsga3(
    problem.genome,
    objectives=problem.objectives,
    reference_directions=gx.das_dennis(3, 12),
    crossover=gx.SimulatedBinaryCrossover(30),
    mutation=gx.PolynomialMutation(20, rate=1 / problem.dimensions),
    seed=1,
)
result = nsga3.run(problem, generations=100)
print(gx.indicators.igd(result.front_objectives, problem.optimal_front(91)))
```

`gx.indicators` measures multi-objective fronts, a point per row: `hypervolume(front,
reference_point)`, and `igd`, `igd_plus`, `gd` and `spread` against a reference front.
`hypervolume`, `igd_plus` and `spread` take `objectives` ("minimize" by default); `igd` and `gd`
measure distances, the same for either direction.

## Stopping

`run` stops at the first of its stop conditions, and needs at least one:
- `generations`
- `evaluations`
- `target`: a score at least as good (single objective)
- `time`: seconds (`math.inf` for no limit)
- `stagnation`: generations without improvement

The result has the condition that stopped it, `stop_reason`, and the `generations`, `evaluations` and `seconds` it took.

## Progress

`run(..., on_generation=callback)` calls `callback` after every generation, the initial population (generation 0) included. It runs on the thread that called `run`. It gets a read-only object:
- `Progress`: the `generation`, `evaluations`, `seconds` and `best_fitness` so far (`None` before a valid solution), the `best_genome` so far, and the `population` (a genome per row) with its `scores` and `violations`
- `MultiProgress`, for a multi-objective algorithm: the same, with `front_size` (the number of non-dominated individuals in the population) instead of `best_fitness` and the best genome, the population's `objectives` (a row per genome) instead of its scores, and the `front_objectives` and `front_violations` of its non-dominated individuals

If `callback` returns `False`, the run stops with the stop reason `"aborted"`. If it raises an exception, the run stops and `run` raises it.

```python
def report(progress):
    if progress.generation % 100 == 0:
        print(progress.generation, progress.evaluations, progress.best_fitness)

result = ga.run(lambda bits: bits.sum(), generations=1_000, on_generation=report)
```

## The Rust library

The package covers a subset of the Rust library. These parts are only in Rust:
- the evolution strategy `Es`, with self-adaptive mutation
- the island model, `Islands`
- `SteadyGa` and the asynchronous engine, for evaluations of varying duration
- memetic search in `Ga`, and initial genomes for a population
- operators of your own
- checkpoints, to save and resume a run
- observers: statistics, a hall of fame and reports
- stop conditions combined with `and`, and custom ones
- penalty functions for constraints, and the NaN policy: in Python, NaN is always an invalid solution
- advanced settings:
  - CMA-ES: a diagonal covariance matrix (sep-CMA-ES) and the initial mean
  - PSO: the inertia, the acceleration and the maximum velocity
  - DE: the strategy, the control of F and CR, the restarts, and population size reduction other than L-SHADE's
  - the rate of `UniformCrossover` and the weight of `ArithmeticCrossover`

Some names differ:

| Python | Rust |
|---|---|
| `mutation=` | `.mutate(...)` |
| `objective="minimize"` | `.minimize()`, `Objective::Minimize` |
| `Binary(length)`, `Permutation(length)` | `Binary::new(len)`, `Permutation::new(len)` |
| `BitFlip(rate=...)`, `BitFlip(count=...)` | `BitFlip::per_gene(rate)`, `BitFlip::count(count)`, and so for the other mutations |
| `PointCrossover(points)` | `PointCrossover::k_point(points)` |
| `MuPlusLambda(offspring)`, `MuCommaLambda(offspring)` | `Scheme::MuPlusLambda { lambda }`, `Scheme::MuCommaLambda { lambda }` |
| `Cmaes(restarts="ipop")` | `.restarts(cmaes::Restarts::Ipop)` |
| `Pso(ring=k)` | `.topology(pso::Topology::Ring { neighbors: k })` |
| `De(l_shade=n)` | `De::l_shade(real, n)` |
| `Pbi(theta)` | `Decomposition::Pbi { theta }` |
| `run(generations=..., time=..., ...)` | `Stop::generations(...).or(Stop::time(...))` |
| `time` (seconds) | `Stop::time(Duration)` |
| `result.seconds` | `Outcome::elapsed()` |
