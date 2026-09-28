# Benchmarks

16 evolutionary computation libraries in 5 languages, genoxide included, and genoxide's Python package, on the same problems: OneMax, N-Queens, Rastrigin, Rosenbrock, Ackley, ZDT1 to 3, DTLZ1 and 2. Every library gets the same fitness functions, evaluation budgets and time caps. Matched scenarios run the same algorithm in every library; idiomatic scenarios run what each library's own docs recommend. Every setting and every bug found is documented per library, and a better way to run one is [welcome](rules.md#9-open-documentation).

[![Overall score: each library's speed to a solution over the 14 scenarios](overall.svg)](https://tachsin.gr/projects/genoxide/benchmarks)

Per scenario, the fastest library gets 100 points and a library that doesn't solve it within the time cap 0, evenly per order of magnitude of time in between; the overall score is the mean over the scenarios a library runs ([rule 8.5](rules.md#8-reporting)).

genoxide's own releases are compared on the same runs by the CPU instructions Callgrind counts: one seeded run of each method per scenario, exact whatever the machine's load ([rule 10](rules.md#10-instruction-counts-genoxides-versions), history in [genoxide-versions.json](genoxide-versions.json)).

[![genoxide's versions: the CPU instructions of the same runs](genoxide_versions.svg)](https://tachsin.gr/projects/genoxide/benchmarks#genoxide-versions)

**Interactive results, every chart and table: [tachsin.gr/projects/genoxide/benchmarks](https://tachsin.gr/projects/genoxide/benchmarks).**

| Page | What it has |
|---|---|
| [Methodology](../../benchmarks/README.md) | the libraries and solvers, the protocol, the scenarios, the matched settings, how to run it |
| [Rules](rules.md) | the rules every adapter follows, and which `run.py check` tests |
| [Notes](notes.md) | what each library can't run, the bugs found, and the rule-level choices |
| [Results](results.md) | the charts and full numbers of the latest run |

Each library has its own page: its methods and where its docs recommend them, what it leaves out, its separate test runs and its bugs.

| Rust | Python | C++, Java, Julia |
|---|---|---|
| [genoxide](libraries/genoxide.md) 0.7.0 | [genoxide (Python)](libraries/genoxide_python.md) 0.7.0 | [openGA](libraries/openga.md) 1.0.5 |
| [genetic_algorithm](libraries/genetic_algorithm.md) 0.27.4 | [DEAP](libraries/deap.md) 1.4.4 | [pygmo](libraries/pygmo.md) 2.19.8 (C++ via Python) |
| [radiate](libraries/radiate.md) 1.3.1 | [pymoo](libraries/pymoo.md) 0.6.2 | [Jenetics](libraries/jenetics.md) 9.1.0 |
| [moors](libraries/moors.md) 0.2.11 | [PyGAD](libraries/pygad.md) 3.7.0 | [jMetal](libraries/jmetal.md) 7.5 |
| | [pycma](libraries/pycma.md) 4.5.0 | [Evolutionary.jl](libraries/evolutionary_jl.md) 0.12.1 |
| | [Nevergrad](libraries/nevergrad.md) 1.0.12 | [Metaheuristics.jl](libraries/metaheuristics_jl.md) 3.5.0 |
| | [SciPy](libraries/scipy.md) 1.18.1 | |
