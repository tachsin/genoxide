# genoxide for Python

Evolutionary computation in Rust, for Python: genetic algorithms, local search, differential evolution, CMA-ES, particle swarm optimization and NSGA-II from [genoxide](https://github.com/tachsin/genoxide), with fitness functions in Python and numpy.

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

More in [examples/](examples/): OneMax, a knapsack with a constraint, N-Queens with tabu search, Rastrigin with CMA-ES and L-SHADE, and ZDT1 with NSGA-II.

## Install

It isn't on PyPI yet. Build it from the repository with Rust and [maturin](https://www.maturin.rs/):

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
- `(score, constraint_violation)`: infeasible solutions, with a positive violation, rank below feasible ones, and among themselves by violation (Deb's rules)

The fitness function must be deterministic: genoxide doesn't evaluate a child identical to one of its parents again.

With `batch=True`, the function takes a whole generation as a 2-D array, a genome per row, and returns an array of scores, or a tuple of scores and constraint violations. It's one call per generation, so vectorized numpy, a GPU or a remote service pays its cost per call once per generation instead of once per genome.

With `parallel=True`, genoxide calls a function that isn't a batch function from several threads at once. It pays off when the function releases the GIL, e.g. in numpy on large arrays or waiting for I/O, or on free-threaded Python.

An exception in the fitness function stops the run and is raised by `run`, and so is Ctrl+C.

## Algorithms

| Algorithm | Genomes | Settings |
|---|---|---|
| `Ga` | all | `population_size`, `select`, `crossover`, `mutation`, `crossover_rate` (0.9), `mutation_rate` (1), `scheme` |
| `LocalSearch` | all | `neighbor` (a mutation), `neighbors` (1), `acceptance`, `restart=(patience, kicks)` |
| `De` | real | `population_size`, `l_shade` (a budget of evaluations, for L-SHADE) |
| `Cmaes` | real | `population_size`, `restarts` (`"ipop"`, `"bipop"`), `initial_step` |
| `Pso` | real | `population_size`, `ring` (neighbors on each side) |
| `Nsga2` | all | `objectives`, `population_size`, `crossover`, `mutation`, `crossover_rate` |

Single-objective algorithms maximize, or minimize with `objective="minimize"`. `Nsga2` takes `objectives=["minimize", "maximize", ...]`, 2 to 6 of them. Its fitness function returns a sequence of objective values, and its result is the final non-dominated front: `front_genomes`, `front_objectives` and `front_violations`.

Every algorithm takes a `seed`: the same seed repeats a run exactly, with a genome at a time, in batches or in parallel.

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

## Stopping

`run` stops at the first of its stop conditions, and needs at least one:
- `generations`
- `evaluations`
- `target`: a score at least as good (single objective)
- `time`: seconds
- `stagnation`: generations without improvement

The result says which one stopped it, in `stop_reason`, with the numbers of `generations` and `evaluations` and the `seconds` it took.
