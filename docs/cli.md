# The genoxide program

`genoxide` runs an optimization described in a TOML or JSON run file. The fitness function is any program, in any language, that reads genomes and writes their fitness.

```text
cargo install genoxide --features cli

genoxide run <file> [--resume]   run it; the result goes to stdout as JSON, progress to stderr
genoxide check <file>            check the run file without running it
genoxide fitness [<name>]        a built-in fitness program, or the list of them
```

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

`fitness.command` is the program and its arguments. genoxide starts one copy per worker, in the run file's directory, and keeps them running:

- It writes one genome per line to the program's stdin, the genes separated by spaces. Bits are `0` and `1`, integers and permutations are whole numbers, and reals are written so they read back exactly.
- The program writes one line per genome to stdout: the objective values, then optionally a constraint violation, separated by spaces. The violation is 0 when the genome is feasible, otherwise how far it is from feasible. `nan` marks a genome that can't be scored.
- The program must flush its output after each line: genoxide waits for each answer, so a program whose output sits in a buffer (Python's, when stdout is a pipe, without `flush=True`) makes the run wait forever. Anything it writes to stderr shows on genoxide's stderr.

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

If the program exits, can't be started, or writes something else, the run stops with an error naming the program and the line it wrote, and no checkpoint is saved after the failure. A relative program path with a directory, like `./fitness`, is relative to the run file.

## The run file

TOML, or JSON for files ending in `.json`, with the same structure. Unknown settings are errors, so typos don't go unnoticed.

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
| `workers` | the number of CPUs | programs evaluating at the same time |
| `nan` | `"invalid"` | a NaN makes the genome invalid (`"invalid"`), or stops the run (`"error"`) |

### `[algorithm]`

`type` picks the algorithm. The other settings depend on it.

| `type` | Genomes | Settings |
|---|---|---|
| `"ga"` | any | `population_size`, `select`, `crossover`, `mutate`; optional `seed`, `crossover_rate` (0.9), `mutation_rate` (1.0), `scheme` |
| `"steady-ga"` | any | as `"ga"`, without `scheme`: asynchronous evaluation, where each worker gets a new genome as soon as it's done |
| `"de"` | real | optional `population_size` (100), `seed`, or `l_shade = <evaluations>` for L-SHADE |
| `"cmaes"` | real | optional `population_size`, `seed`, `restarts` (`"never"`, `"ipop"`, `"bipop"`), `initial_step` (a fraction of each gene's range) |
| `"pso"` | real | `population_size`; optional `seed`, `ring = <neighbors>` for a ring topology |
| `"local-search"` | any | `neighbor` (a mutation); optional `seed`, `neighbors` (1), `acceptance`, `restart = [patience, kicks]` |
| `"nsga2"` | any | 2 to 6 objectives: `population_size`, `crossover`, `mutate`; optional `seed`, `crossover_rate` |

Operators are tables with a `type`, e.g. `crossover = { type = "uniform" }`; the names below are their types:

- **`select`:**
  - `{ type = "tournament", size = 3 }`
  - `{ type = "rank", pressure = 1.5 }`
  - `{ type = "roulette" }`
  - `{ type = "stochastic-universal" }`
  - `{ type = "truncation", fraction = 0.5 }`
  - `{ type = "random" }`
- **`crossover`:**
  - binary and integer: `uniform`, `one-point`, `two-point`, `{ type = "k-point", points = 3 }` or `none`
  - real: those, plus `{ type = "simulated-binary", eta = 15.0 }`, `{ type = "blend", alpha = 0.5 }` or `arithmetic`
  - permutation: `order`, `partially-mapped`, `cycle`, `edge-recombination` or `none`
- **`mutate`** (and a local search's `neighbor`) takes either `rate`, the probability for each gene, or `count`, exactly that many genes:
  - binary: `{ type = "bit-flip", rate = 0.01 }`
  - integer: `{ type = "uniform", count = 1 }`
  - real: `{ type = "polynomial", rate = 0.1, eta = 20.0 }`, `{ type = "gaussian", rate = 0.1, sigma = 0.1 }` or `uniform`
  - permutation: `{ type = "swap", count = 1 }`, `inversion`, `insertion` or `scramble`
- **`scheme`:**
  - `{ type = "generational", elitism = 1 }` (the default)
  - `{ type = "steady-state", replacements = 10 }`
  - `{ type = "mu-plus-lambda", lambda = 100 }`
  - `{ type = "mu-comma-lambda", lambda = 200 }`
- **`acceptance`:**
  - `improving`
  - `not-worse` (the default)
  - `{ type = "annealing", initial_temperature = 1.0, cooling = 0.999 }`
  - `{ type = "tabu", tenure = 20 }`

### `[stop]`

The run stops at the first condition met. At least one is needed.

| Setting | Stops |
|---|---|
| `generations` | after this many generations |
| `evaluations` | after this many fitness evaluations |
| `target` | at a fitness at least this good (one objective only) |
| `time` | after this long: `"500ms"`, `"30s"`, `"10m"`, `"2h"` |
| `stagnation` | after this many generations without improvement |

### `report`

Progress lines on stderr: `report = "1s"` (the default) prints one every second, `report = 10` every 10 generations, and `report = "off"` none. Put it before the first table.

### `[checkpoint]`

```toml
[checkpoint]
path = "run.ckpt"   # relative to the run file
every = 50          # generations
```

genoxide saves the run every `every` generations and when it stops. `genoxide run <file> --resume` continues from the checkpoint, with exactly the results of an uninterrupted run (for `steady-ga`, with one worker: with more, the results depend on timing, and the evaluations in flight aren't in a checkpoint). You can change `fitness.command`, `fitness.builtin`, `fitness.workers`, `fitness.nan`, `stop`, `report` and `checkpoint` in between, e.g. to run longer. A change to `genome`, `fitness.objectives` or `algorithm` is an error.

## The result

On stdout, as JSON:

- `stop_reason`
- `generations`, `evaluations` and `seconds`
- For one objective: `fitness` (`null` if invalid), `violation` and `genome`. Infinite values, which JSON has no numbers for, are the text `"inf"` or `"-inf"`.
- For several objectives: `front`, the trade-offs found. Each has its `objectives`, `violation` and `genome`, without copies of a genome.

The exit code is 0 after a run and 1 after an error, with the error on stderr.

## Built-in fitness programs

`genoxide fitness` lists them, and `genoxide fitness <name>` runs one on stdin and stdout, speaking the same protocol:

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
