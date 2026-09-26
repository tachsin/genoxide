# Benchmarks

16 evolutionary computation libraries in 5 languages, genoxide included, and genoxide's Python package, on the same problems: OneMax, N-Queens, Rastrigin, Rosenbrock, Ackley, ZDT1 to 3, DTLZ1 and 2. Every library gets the same fitness functions, evaluation budgets and time cap. Matched scenarios run the same algorithm in every library; idiomatic scenarios run what each library's own docs recommend. Every setting and every bug found is documented per library, and a better way to run one is [welcome](rules.md#9-open-documentation).

| Page | What it has |
|---|---|
| [Methodology](../../benchmarks/README.md) | the libraries and solvers, the protocol, the scenarios, the matched settings, how to run it |
| [Rules](rules.md) | the rules every adapter follows, and which `run.py check` tests |
| [Notes](notes.md) | what each library can't run, the bugs found, and the rule-level choices |
| [Results](results.md) | the full numbers of the latest run |

Each library has its own page: its methods and where its docs recommend them, what it leaves out, its separate test runs and its bugs.

| Rust | Python | C++, Java, Julia |
|---|---|---|
| [genoxide](libraries/genoxide.md) | [genoxide (Python)](libraries/genoxide_python.md) | [openGA](libraries/openga.md) |
| [genetic_algorithm](libraries/genetic_algorithm.md) | [DEAP](libraries/deap.md) | [pygmo](libraries/pygmo.md) (C++ via Python) |
| [radiate](libraries/radiate.md) | [pymoo](libraries/pymoo.md) | [Jenetics](libraries/jenetics.md) |
| [moors](libraries/moors.md) | [PyGAD](libraries/pygad.md) | [jMetal](libraries/jmetal.md) |
| | [pycma](libraries/pycma.md) | [Evolutionary.jl](libraries/evolutionary_jl.md) |
| | [Nevergrad](libraries/nevergrad.md) | [Metaheuristics.jl](libraries/metaheuristics_jl.md) |
| | [SciPy](libraries/scipy.md) | |
