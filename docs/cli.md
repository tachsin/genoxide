# The genoxide program

`genoxide` runs an optimization described in a TOML or JSON run file. The fitness function is any program, in any language, that reads genomes and writes their fitness.

```text
cargo install genoxide --features cli

genoxide run <file> [--resume]   run it; the result goes to stdout as JSON, progress to stderr
genoxide check <file>            check the run file without running it
genoxide fitness [<name>]        a built-in fitness program, or the list of them
genoxide --help | -h             this usage, also with no arguments
genoxide --version | -V          the version: genoxide <version>
```

`genoxide check` prints `<file>: ok` for a valid run file. It builds the algorithm but doesn't start the fitness program.

## A first run

`sphere.toml`, with a built-in test function as the fitness:

```toml
[genome]
type = "real"
length = 10
bounds = [-5.12, 5.12]

[fitness]
builtin = "sphere"
objectives = ["minimize"]

[algorithm]
type = "cmaes"
seed = 1

[stop]
target = 1e-8
evaluations = 100000
```

```text
$ genoxide run sphere.toml
generation 0: 10 evaluations, best 41.28 (generation 0), 0.004 s
{
  "evaluations": 1420,
  "fitness": 9.184869863140121e-9,
  "generations": 141,
  "genome": [0.0000399...,...],
  "seconds": 0.044,
  "stop_reason": "target",
  "violation": 0.0
}
```

## The fitness program

`fitness.command` is the program and its arguments. genoxide starts one copy per worker, in the run file's directory, and keeps them running.

- **Input:** genoxide writes one genome per line to the program's stdin, the genes separated by spaces. Bits are `0` and `1`. Integers and permutations are whole numbers. Reals are written so they read back exactly.
- **Output:** the program writes one line per genome to stdout: the objective values, then optionally a constraint violation, separated by spaces. The violation is 0 for a feasible genome, otherwise how far it is from feasible. A negative violation stops the run with an error. `inf` and `-inf` are valid values. `nan`, as an objective value or the violation, marks a genome that can't be scored.
- **Flush after each line.** genoxide waits for each answer, so output left in a buffer makes the run wait forever. Python buffers stdout when it's a pipe: use `flush=True`.
- **stderr:** anything the program writes there shows on genoxide's stderr.

A Python fitness function, minimizing the sum of squares with the first gene at least 1:

```python
import sys

for line in sys.stdin:
    x = [float(gene) for gene in line.split()]
    violation = max(0.0, 1.0 - x[0])
    print(sum(gene * gene for gene in x), violation, flush=True)
```

```toml
[fitness]
command = ["python3", "fitness.py"]
objectives = ["minimize"]
workers = 8
```

If the program exits, can't be started, or writes something else, the run stops with an error. The error names the program and the line it wrote. No checkpoint is saved after the failure.

A relative program path with a directory, like `./fitness`, is relative to the run file.

## The run file

TOML, or JSON for files ending in `.json`, with the same structure. Unknown settings are errors, so typos are caught.

### `[genome]`

| `type` | Settings | Genes |
|---|---|---|
| `"binary"` | `length` | 0 or 1 |
| `"integer"` | `bounds`, and `length` with one pair of bounds | whole numbers in their bounds |
| `"real"` | `bounds`, and `length` with one pair of bounds | reals in their bounds |
| `"permutation"` | `length` | an ordering of 0 to `length` − 1 |

`bounds` is one `[low, high]` pair for every gene (with `length`), or a list of pairs, one per gene: `bounds = [[0, 1], [-10, 10]]`.

### `[fitness]`

| Setting | Default | |
|---|---|---|
| `command` | | the program and its arguments, e.g. `["python3", "fitness.py"]` |
| `builtin` | | a built-in test function instead: `genoxide fitness` lists them |
| `objectives` | `["maximize"]` | `"maximize"` or `"minimize"`, one per objective |
| `workers` | the number of CPUs | programs evaluating at the same time, 1 to 4096 |
| `nan` | `"invalid"` | a NaN makes the genome invalid (`"invalid"`), or stops the run (`"error"`) |

Set exactly one of `command` and `builtin`.

### `[algorithm]`

`type` picks the algorithm. The other settings depend on it. `seed`, optional for every type, makes a run reproducible.

| `type` | Genomes | Required | Optional (default) |
|---|---|---|---|
| `"ga"` | any | `population_size`, `select`, `crossover`, `mutate` | `crossover_rate` (0.9), `mutation_rate` (1.0), `scheme` (generational, `elitism = 1`) |
| `"steady-ga"` | any | as `"ga"` | as `"ga"`, without `scheme` |
| `"de"` | real | | `population_size` (100), `l_shade` |
| `"cmaes"` | real | | `population_size` (4 + ⌊3 ln n⌋), `restarts` (`"never"`), `initial_step` (0.3) |
| `"pso"` | real | `population_size` | `ring` (off: the whole swarm) |
| `"local-search"` | any | `neighbor` (a mutation) | `neighbors` (1), `acceptance` (`not-worse`), `restart` (off) |
| `"nsga2"` | any | `population_size`, `crossover`, `mutate` | `crossover_rate` (0.9) |

- `population_size`: at most 2^24. At least 4 for `de`; 2 for `cmaes`, `pso` and `nsga2`; 1 for `ga` and `steady-ga`. A generational `ga` needs more than its `elitism`.
- `steady-ga` evaluates asynchronously: each worker gets a new genome as soon as it's done.
- `crossover_rate` and `mutation_rate`: 0 to 1, the probability for each pair of parents and each child. They can't both be 0. `mutation_rate` can't be 0 with the crossover `none`.
- `nsga2`: 2 to 6 objectives. It has no `mutation_rate`: every child is mutated.
- `l_shade = <evaluations>`: L-SHADE, for a run of that many evaluations, at least 1. The default population becomes max(18 × genes, 4), and shrinks linearly to 4 over the evaluations. `population_size` sets the initial size instead. The run doesn't stop at the budget by itself: set `stop.evaluations` to the same number.
- `cmaes`: n is the number of genes whose bounds differ. `restarts` is `"never"`, `"ipop"` or `"bipop"`. `initial_step` is the initial step size as a fraction of each gene's range, greater than 0 and at most 1.
- `ring = <neighbors>`: a ring topology, with that many neighbors on each side, at least 1.
- `neighbors`: the neighbors evaluated per step, 1 to 2^24.
- `restart = [patience, kicks]`: after `patience` steps without a new best (at least 1), restart from the best, changed by `kicks` random neighbor moves (1 to 2^24).

#### Operators

Operators are tables with a `type`, e.g. `crossover = { type = "uniform" }`. The settings in the tables below are required, except where a default is given. Values in examples, like `size = 3`, aren't defaults.

`select`:

| `type` | Settings |
|---|---|
| `"tournament"` | `size`: 1 to 2^24, e.g. `{ type = "tournament", size = 3 }` |
| `"rank"` | `pressure`: 1 (uniform) to 2 |
| `"roulette"` | |
| `"stochastic-universal"` | |
| `"truncation"` | `fraction`: the best part of the population to select from, greater than 0 and at most 1 |
| `"random"` | |

`crossover`:

| `type` | Genomes | Settings |
|---|---|---|
| `"uniform"`, `"one-point"`, `"two-point"` | binary, integer, real | |
| `"k-point"` | binary, integer, real | `points`: at least 1 |
| `"simulated-binary"` | real | `eta`: 0 or more, e.g. 15.0 |
| `"blend"` | real | `alpha`: 0 or more, e.g. 0.5 |
| `"arithmetic"` | real | |
| `"order"`, `"partially-mapped"`, `"cycle"`, `"edge-recombination"` | permutation | |
| `"none"` | any | |

`mutate`, and a local search's `neighbor`:

| `type` | Genomes | Settings |
|---|---|---|
| `"bit-flip"` | binary | `rate` or `count` |
| `"uniform"` | integer, real | `rate` or `count` |
| `"gaussian"` | real | `rate` or `count`; `sigma`: the standard deviation as a fraction of each gene's range, greater than 0 |
| `"polynomial"` | real | `rate` or `count`; `eta`: 0 or more, e.g. 20.0 |
| `"swap"` | permutation | `count`: the pairs of positions swapped, at least 1; 1 by default |
| `"inversion"`, `"insertion"`, `"scramble"` | permutation | none |

`rate` or `count` means exactly one of them. `rate` is the probability for each gene, greater than 0 and at most 1. `count` is exactly that many distinct genes, at least 1, e.g. `{ type = "bit-flip", count = 1 }`.

`scheme`, for `ga`:

| `type` | Settings |
|---|---|
| `"generational"` | `elitism`: the best parents that survive, less than `population_size` |
| `"steady-state"` | `replacements`: the children per generation, which replace the worst parents, 1 to `population_size` |
| `"mu-plus-lambda"` | `lambda`: the children per generation, at least 1; the best of parents and children survive |
| `"mu-comma-lambda"` | `lambda`: the children per generation, at least `population_size`; the best children survive |

`acceptance`, for `local-search`:

| `type` | Settings |
|---|---|
| `"improving"` | |
| `"not-worse"` | |
| `"annealing"` | `initial_temperature`: greater than 0; `cooling`: greater than 0 and at most 1 |
| `"tabu"` | `tenure`: at least 1 |

### `[stop]`

The run stops at the first condition met. At least one is needed.

| Setting | Stops |
|---|---|
| `generations` | after this many generations |
| `evaluations` | after this many fitness evaluations |
| `target` | at a fitness at least this good (one objective only) |
| `time` | after this long, e.g. `"500ms"`, `"1.5s"`, `"10m"`, `"2h"` |
| `stagnation` | after this many generations without improvement, at least 1 |

A duration is a number, decimals allowed, and a unit: `ms`, `s`, `m` or `min`, or `h`.

### `report`

Progress lines on stderr. Put `report` before the first table.

- `report = "1s"` (the default): one line every second
- `report = "30s"`: one line every 30 seconds, any duration as in `[stop]`
- `report = 10`: one line every 10 generations, at least 1
- `report = "off"`: none

### `[checkpoint]`

```toml
[checkpoint]
path = "run.ckpt"   # relative to the run file
every = 50          # generations, at least 1
```

genoxide saves the run every `every` generations and when it stops. `genoxide run <file> --resume` continues from the checkpoint, with exactly the results of an uninterrupted run. `--resume` needs a `[checkpoint]` table.

For `steady-ga`, that holds with one worker only. With more, the results depend on timing, and the evaluations in flight aren't in a checkpoint.

Between runs, you can change `fitness.command`, `fitness.builtin`, `fitness.workers`, `fitness.nan`, `stop`, `report` and `checkpoint`, e.g. to run longer. A change to `genome`, `fitness.objectives` or `algorithm` is an error.

## The result

On stdout, as JSON:

- `stop_reason`: see below
- `generations`, `evaluations` and `seconds`
- For one objective: `fitness`, `violation` and `genome`. `fitness` is `null` for a genome that can't be scored, and its violation is 0.
- For several objectives: `front`, the trade-offs found, without copies of a genome. Each has its `objectives`, `violation` and `genome`. `objectives` is `null` for a genome that can't be scored, and its violation is 0.
- Infinite values are the text `"inf"` or `"-inf"`, since JSON has no numbers for them.

| `stop_reason` | The run stopped |
|---|---|
| `"target"` | at `stop.target` |
| `"generations"` | at `stop.generations` |
| `"evaluations"` | at `stop.evaluations` |
| `"time"` | at `stop.time` |
| `"stagnation"` | at `stop.stagnation` |
| `"stalled"` | after 10,000 generations without a new genome to evaluate, when only `target` or `evaluations` could stop it. E.g. a GA whose children are all copies of their parents. |
| `"aborted"`, `"other"` | reserved: a run file doesn't cause them |

The exit code is 0 after a run and 1 after an error, with the error on stderr.

## Built-in fitness programs

`genoxide fitness` lists them. `genoxide fitness <name>` runs one on stdin and stdout, with the same protocol:

| Name | Genome | Scores |
|---|---|---|
| `one-max` | binary | the number of ones (maximize) |
| `sphere` | real or integer | the sum of squares (minimize) |
| `rastrigin` | real | Rastrigin's function (minimize) |
| `rosenbrock` | real | Rosenbrock's function (minimize) |
| `ackley` | real | Ackley's function (minimize) |
| `inversions` | permutation | the pairs out of order (minimize) |
| `zdt1` | real in [0, 1] | ZDT1's two objectives (minimize both) |
| `schaffer` | real | Schaffer's two objectives (minimize both) |
