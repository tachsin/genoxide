# Results 20260926-190143

The matched suite: three problems, one method each, the same in every library, with its own implementation ([rule 6](rules.md#6-the-methods)).

Seeds per scenario: 10, wall time cap per run: 60 s
Linux, Intel(R) Core(TM) Ultra 7 265K, WSL pinned to cores 8, 19

- genoxide 0.7.0+8dd8436
- genoxide_python 0.7.0+8dd8436
- deap 1.4.4
- pygad 3.7.0
- radiate 1.3.1
- evolutionary_jl 0.12.0

Invalid runs, left out of every table and chart: 0

This run predates the matched suite: it used the rules and adapters at 8dd8436, when the suite ran many methods per library. Its OneMax 1000 runs are the suite's GA ([rule 6.2](rules.md#6-the-methods)), so they're published here, with the versions they measured; its other runs, of methods and problems no longer in the suite, aren't. Rastrigin 30 (DE/rand/1/bin, a fixed budget of 300,000 evaluations) and Rosenbrock 10 (CMA-ES) await the next timed run.

## Charts

Interactive, with each bar's numbers and runs: [tachsin.gr/projects/genoxide/benchmarks](https://tachsin.gr/projects/genoxide/benchmarks).

![Expected time to target: a panel per problem, a bar per library](time_to_target.svg)

- [Expected evaluations to target](evaluations_to_target.svg)
- [Distance to the optimum at the end](distance_to_optimum.svg)
- [genoxide's versions](genoxide_versions.svg): the CPU instructions of the same runs in each release, genoxide only ([rule 10](rules.md#10-instruction-counts-genoxides-versions))

Awaiting the next run: Rastrigin 30: DE/rand/1/bin, Rosenbrock 10: CMA-ES. This run has no runs of them.

## Coverage

✓ ran, ✗ every run invalid, – can't run the scenario's method: why, and the bugs found in the libraries, in [notes.md](notes.md).

| Library | OneMax 1000: GA | Rastrigin 30: DE/rand/1/bin | Rosenbrock 10: CMA-ES |
|---|---|---|---|
| genoxide | ✓ |  |  |
| genoxide (Python) | ✓ |  |  |
| DEAP | ✓ |  |  |
| PyGAD | ✓ |  |  |
| radiate | ✓ |  |  |
| Evolutionary.jl | ✓ |  |  |

## Results

Expected time and evaluations to target: the expected running time (ERT), what all runs spent, up to the first hit of the target in the runs that reached it, divided by the number of runs that reached it; with fewer than 3, how many reached it. A first hit after the time cap counts as not reached. Stopped by the time cap: runs that ended at the cap, not at the target or the budget, and the median share of the budget they used. A problem without a target (Rastrigin 30, [rule 6.3](rules.md#6-the-methods)) has a table of its own: the median time of the runs that used the whole fixed budget, the error at the end of every run, and the runs the library ended early.

| Scenario | Library / method | Reached the target | Stopped by the time cap | Expected time to target | Expected evaluations to target | Median evaluations | Distance to the optimum at the end: median (best to worst) | Evaluations/s | Throughput vs DEAP |
|---|---|---|---|---|---|---|---|---|---|
| onemax-1000-matched | deap / ga | 10 of 10 | 0 | 15.31 s | 103,982 | 105,287 | 0 (0 to 0) | 6,799 | - |
| onemax-1000-matched | evolutionary_jl / ga | 10 of 10 | 0 | 24.6 ms | 116,421 | 117,151 | 0 (0 to 0) | 4,735,472 | 697× |
| onemax-1000-matched | genoxide / ga | 10 of 10 | 0 | 15.2 ms | 54,566 | 54,038 | 0 (0 to 0) | 3,577,532 | 526× |
| onemax-1000-matched | genoxide_python / ga | 10 of 10 | 0 | 80.1 ms | 54,566 | 54,038 | 0 (0 to 0) | 680,926 | 100× |
| onemax-1000-matched | pygad / ga | 10 of 10 | 0 | 7.07 s | 113,377 | 113,850 | 0 (0 to 0) | 16,057 | 2.4× |
| onemax-1000-matched | radiate / ga | 10 of 10 | 0 | 284.5 ms | 78,858 | 79,288 | 0 (0 to 0) | 277,632 | 41× |
