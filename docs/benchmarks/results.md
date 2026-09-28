# Results 20260928-183253

The matched suite: three problems, one method each, the same in every library, with its own implementation ([rule 6](rules.md#6-the-methods)).

Seeds per scenario: 10, wall time cap per run: 60 s
Linux, Intel(R) Core(TM) Ultra 7 265K, WSL pinned to cores 8, 19

- genoxide 0.9.1+5f175ad
- genoxide_python 0.9.1+5f175ad
- deap 1.4.4
- pygad 3.7.0
- pymoo 0.6.2
- radiate 1.3.2
- pycma 4.5.0
- scipy 1.18.1
- pygmo 2.19.8
- jmetal 7.5
- evolutionary_jl 0.12.1
- metaheuristics_jl 3.5.0

Invalid runs, left out of every table and chart: 0

## Charts

Interactive, with each bar's numbers and runs: [tachsin.gr/projects/genoxide/benchmarks](https://tachsin.gr/projects/genoxide/benchmarks).

![Expected time to target: a panel per problem, a bar per library](time_to_target.svg)

- [Expected evaluations to target](evaluations_to_target.svg)
- [Distance to the optimum at the end](distance_to_optimum.svg)
- [genoxide's versions](genoxide_versions.svg): the CPU instructions of the same runs in each release, genoxide only ([rule 10](rules.md#10-instruction-counts-genoxides-versions))

## Coverage

✓ ran, ✗ every run invalid, – can't run the scenario's method: why, and the bugs found in the libraries, in [notes.md](notes.md).

| Library | OneMax 1000: GA | Rastrigin 30: DE/rand/1/bin | Rosenbrock 10: CMA-ES |
|---|---|---|---|
| genoxide | ✓ | ✓ | ✓ |
| genoxide (Python) | ✓ | ✓ | ✓ |
| DEAP | ✓ | – | ✓ |
| PyGAD | ✓ | – | – |
| pymoo | – | ✓ | ✓ |
| radiate | ✓ | – | – |
| pycma | – | – | ✓ |
| SciPy | – | ✓ | – |
| pygmo | – | ✓ | ✓ |
| jMetal | – | ✓ | ✓ |
| Evolutionary.jl | ✓ | – | ✓ |
| Metaheuristics.jl | – | ✓ | – |

## Results

Expected time and evaluations to target: the expected running time (ERT), what all runs spent, up to the first hit of the target in the runs that reached it, divided by the number of runs that reached it; with fewer than 3, how many reached it. A first hit after the time cap counts as not reached. Stopped by the time cap: runs that ended at the cap, not at the target or the budget, and the median share of the budget they used. A problem without a target (Rastrigin 30, [rule 6.3](rules.md#6-the-methods)) has a table of its own: the median time of the runs that used the whole fixed budget, the error at the end of every run, and the runs the library ended early.

| Scenario | Library / method | Reached the target | Stopped by the time cap | Expected time to target | Expected evaluations to target | Median evaluations | Distance to the optimum at the end: median (best to worst) | Evaluations/s | Throughput vs DEAP |
|---|---|---|---|---|---|---|---|---|---|
| onemax-1000-matched | deap / ga | 10 of 10 | 0 | 15.52 s | 103,982 | 105,287 | 0 (0 to 0) | 6,706 | - |
| onemax-1000-matched | evolutionary_jl / ga | 10 of 10 | 0 | 25.8 ms | 116,421 | 117,151 | 0 (0 to 0) | 4,415,193 | 658× |
| onemax-1000-matched | genoxide / ga | 10 of 10 | 0 | 15.6 ms | 54,566 | 54,038 | 0 (0 to 0) | 3,506,652 | 523× |
| onemax-1000-matched | genoxide_python / ga | 10 of 10 | 0 | 144.8 ms | 54,566 | 54,038 | 0 (0 to 0) | 376,585 | 56× |
| onemax-1000-matched | pygad / ga | 10 of 10 | 0 | 6.88 s | 113,377 | 113,850 | 0 (0 to 0) | 16,495 | 2.5× |
| onemax-1000-matched | radiate / ga | 10 of 10 | 0 | 36.4 ms | 78,589 | 78,776 | 0 (0 to 0) | 2,155,658 | 321× |
| rosenbrock-10-matched | deap / cma_es | 10 of 10 | 0 | 62.2 ms | 5,846 | 5,165 | 0.009033 (0.006805 to 0.009883) | 93,971 | - |
| rosenbrock-10-matched | evolutionary_jl / cma_es | 0 of 10, 10 ended early | 0 | 0/10 reached | 0/10 reached | 98,506 | 98.12 (7.439 to 608.6) | 757,708 | 8.1× |
| rosenbrock-10-matched | genoxide / cma_es | 9 of 10 | 0 | 29.2 ms | 61,477 | 5,945 | 0.009444 (0.007279 to 3.987) | 2,107,360 | 22× |
| rosenbrock-10-matched | genoxide_python / cma_es | 9 of 10 | 0 | 77.4 ms | 61,477 | 5,945 | 0.009444 (0.007279 to 3.987) | 793,844 | 8.4× |
| rosenbrock-10-matched | jmetal / cma_es | 0 of 10, 10 ended early | 0 | 0/10 reached | 0/10 reached | 204,465 | 9.588 (7.478 to 232.2) | 719,339 | 7.7× |
| rosenbrock-10-matched | pycma / cma_es | 9 of 10 | 0 | 1.86 s | 60,966 | 5,490 | 0.009531 (0.008867 to 3.987) | 32,724 | 0.3× |
| rosenbrock-10-matched | pygmo / cma_es | 8 of 10 | 0 | 236.6 ms | 129,910 | 5,155 | 0.0092 (0.007793 to 3.987) | 548,967 | 5.8× |
| rosenbrock-10-matched | pymoo / cma_es | 9 of 10, 1 ended early | 0 | 441.7 ms | 8,571 | 5,206 | 0.008609 (0.005965 to 3.987) | 19,393 | 0.2× |

| Scenario (fixed budget) | Library / method | Runs | Ended early by the library | Stopped by the time cap | Median time for the budget | Median evaluations | Error at the end: median (best to worst) | Evaluations/s |
|---|---|---|---|---|---|---|---|---|
| rastrigin-30-matched | genoxide / de | 10 | 0 | 0 | 148.8 ms | 300,000 | 124.4 (85.26 to 158.9) | 2,015,704 |
| rastrigin-30-matched | genoxide_python / de | 10 | 0 | 0 | 180.2 ms | 300,000 | 124.4 (85.26 to 158.9) | 1,656,991 |
| rastrigin-30-matched | jmetal / de | 10 | 0 | 0 | 950.0 ms | 300,000 | 129 (100.7 to 152.4) | 313,585 |
| rastrigin-30-matched | metaheuristics_jl / de | 10 | 0 | 0 | 231.2 ms | 300,000 | 134.3 (84.94 to 176.2) | 1,278,759 |
| rastrigin-30-matched | pygmo / de | 10 | 0 | 0 | 2.44 s | 300,000 | 141.4 (87.5 to 182.2) | 123,046 |
| rastrigin-30-matched | pymoo / de | 10 | 0 | 0 | 5.32 s | 300,000 | 69.55 (59.68 to 96.84) | 56,306 |
| rastrigin-30-matched | scipy / de | 10 | 0 | 0 | 1.15 s | 300,000 | 139 (92.96 to 169.7) | 260,963 |
