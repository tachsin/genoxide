# Benchmarks

Evolutionary computation libraries on the same problems, with identical fitness functions and the same evaluation budgets.

| Library | Language | Version |
|---|---|---|
| [genetic_algorithm](https://crates.io/crates/genetic_algorithm) | Rust | 0.27.3 |
| [DEAP](https://github.com/DEAP/deap) | Python | 1.4.4 |
| [pymoo](https://github.com/anyoptimization/pymoo) | Python | 0.6.2 |
| [PyGAD](https://github.com/ahmedfgad/GeneticAlgorithmPython) | Python | 3.7.0 |

genoxide itself will be added as soon as it can run the scenarios.

## Running

This needs Python 3.10+ and Rust (cargo). The libraries are downloaded and built by the commands below.

```sh
cd benchmarks
python run.py setup      # .venv with the Python libraries from requirements.txt
python run.py --quick    # 4 small scenarios, 3 seeds
python run.py            # all scenarios, 10 seeds
python run.py --scenarios nqueens-64-idiomatic --seeds 20 --libraries genetic_algorithm deap
```

Each run writes the raw runs, with library versions, to `results/<timestamp>.json`, and a table to `results/latest.md`.

## Scenarios

| Problem | Representation | Sizes | Target |
|---|---|---|---|
| OneMax | binary | 100, 1000 | all ones |
| N-Queens | permutation | 32, 64 | no conflicts |
| Rastrigin | real, [-5.12, 5.12] | 10, 30 dimensions | ≤ 0.01 |

A run stops at the first of these:
- the target is reached
- the evaluation budget of the scenario is used up
- 60 seconds have passed

Time is measured inside the adapter, so interpreter startup and imports are not included. All libraries run single-threaded.

- **matched:** configurations as equal as the libraries allow. This measures framework cost.
- **idiomatic:** each library's recommended configuration, from its docs and examples. This measures what users get. A library can have several solvers here, e.g. GA, CMA-ES and DE.

The GA loops differ between libraries, so "matched" is as close as possible, not identical. For example:
- DEAP selects parents by tournament with replacement.
- genetic_algorithm selects survivors from parents plus offspring by tournament without replacement.

Evaluations to target measure search efficiency independently of the language.

## Adding a library

Write an adapter with this command line:

```
<adapter> <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
```

It prints one JSON line per solver per seed:

```json
{"library": "deap", "solver": "ga", "problem": "onemax", "size": 100, "mode": "matched", "seed": 0,
 "time_s": 0.21, "generations": 37, "evaluations": 7041, "best": 100, "target": 100, "success": true}
```

Then register it in `ADAPTERS` in `run.py`. Keep the fitness functions identical to the existing adapters.
