# Results 20260926-190143

Seeds per scenario: 10, wall time cap per run: 60 s
Linux, Intel(R) Core(TM) Ultra 7 265K, WSL pinned to cores 8, 19

- genoxide 0.7.0+8dd8436
- genoxide_python 0.7.0+8dd8436
- genetic_algorithm 0.27.3
- deap 1.4.4
- pygad 3.7.0
- pymoo 0.6.2
- radiate 1.3.1
- moors 0.2.11
- pycma 4.5.0
- nevergrad 1.0.12
- scipy 1.18.1
- pygmo 2.19.8
- jenetics 9.1.0
- jmetal 7.5
- evolutionary_jl 0.12.0
- metaheuristics_jl 3.5.0
- openga 1.0.5+f9b15e7

Invalid runs, left out of every table and chart: 3

- jenetics / ga, nqueens-64-idiomatic, seed 9: ga seed 9: 1000393 evaluations, over the budget of 1000000 by more than a generation (253)
- jenetics / ga, rastrigin-30-idiomatic, seed 8: ga seed 8: 2000388 evaluations, over the budget of 2000000 by more than a generation (193)
- radiate / ga_intermediate, ackley-30-idiomatic, seed 5: ga_intermediate seed 5: 1000316 evaluations, over the budget of 1000000 by more than a generation (310)

The check was too strict for these three: each ends within its last generation, which a later fix of the check allows (#172).

This run used the rules and adapters at 8dd8436. The changes merged since, in #172 to #174, apply from the next run: a 600 s cap for multi-objective scenarios, restarts of stalled attempts, CMA in place of Nevergrad's Hammersley search, and the methods of genoxide (Python) aligned with the Rust ones.

## Charts

Interactive, with each bar's numbers: [tachsin.gr/projects/genoxide/benchmarks](https://tachsin.gr/projects/genoxide/benchmarks).

![Overall score: each library's speed to a solution over the 14 scenarios](overall.svg)

- [Time to target: each library's fastest method](summary.svg)
- [Expected time to target](time_to_target.svg), every method
- [Expected evaluations to target](evaluations_to_target.svg), every method
- [Distance to the optimum at the end](distance_to_optimum.svg)
- [Hypervolume of the multi-objective fronts](hypervolume.svg)
- [Time of the multi-objective runs](front_time.svg)
- [genoxide's versions](genoxide_versions.svg): the CPU instructions of the same runs in each release, genoxide only ([rule 10](rules.md#10-instruction-counts-genoxides-versions))

## Coverage

✓ ran, ✗ every run invalid, – can't run the scenario: why, and the bugs found in the libraries, in [notes.md](notes.md).

| Library | OneMax 100 (matched) | OneMax 1000 (matched) | OneMax 100 (idiomatic) | N-Queens 32 (idiomatic) | N-Queens 64 (idiomatic) | Rastrigin 10 (idiomatic) | Rastrigin 30 (idiomatic) | Rosenbrock 10 (idiomatic) | Ackley 30 (idiomatic) | ZDT1, 30 variables | ZDT2, 30 variables | ZDT3, 30 variables | DTLZ2, 3 objectives | DTLZ1, 3 objectives |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| genoxide | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| genoxide (Python) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| genetic_algorithm | – | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – | – | – | – |
| DEAP | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| PyGAD | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| pymoo | – | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| radiate | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| moors | – | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – | – | – | – |
| pycma | – | – | – | – | – | ✓ | ✓ | ✓ | ✓ | – | – | – | – | – |
| Nevergrad | – | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – | – | – | – |
| SciPy | – | – | – | – | – | ✓ | ✓ | ✓ | ✓ | – | – | – | – | – |
| pygmo | – | – | ✓ | – | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Jenetics | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – | – | – | – |
| jMetal | – | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Evolutionary.jl | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Metaheuristics.jl | – | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| openGA | – | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – | – | – | – |

## Single-objective

Expected time and evaluations to target: the expected running time (ERT), what all runs spent, up to the first hit of the target in the runs that reached it, divided by the number of runs that reached it; with fewer than 3, how many reached it. A first hit after the time cap counts as not reached. Stopped by the time cap: runs that ended at the cap, not at the target or the budget, and the median share of the budget they used.

| Scenario | Library / solver | Reached the target | Stopped by the time cap | Expected time to target | Expected evaluations to target | Median evaluations | Distance to the optimum at the end: median (best to worst) | Evaluations/s | Throughput vs DEAP GA |
|---|---|---|---|---|---|---|---|---|---|
| ackley-30-idiomatic | deap / cma_es | 10 of 10 | 0 | 66.4 ms | 3,292 | 3,297 | 0.009134 (0.008292 to 0.009873) | 49,634 | - |
| ackley-30-idiomatic | deap / de | 10 of 10 | 0 | 8.74 s | 239,272 | 240,000 | 0.009651 (0.008802 to 0.00996) | 27,388 | - |
| ackley-30-idiomatic | evolutionary_jl / cma_es | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,012 | 19.65 (19.38 to 19.75) | 345,688 | - |
| ackley-30-idiomatic | evolutionary_jl / de | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,032 | 1.395 (0.0399 to 3.426) | 883,929 | - |
| ackley-30-idiomatic | evolutionary_jl / es | 10 of 10 | 0 | 56.5 ms | 16,598 | 13,466 | 0.008956 (0.008019 to 0.009201) | 294,378 | - |
| ackley-30-idiomatic | genetic_algorithm / evolve | 10 of 10 | 0 | 23.8 ms | 83,871 | 50,336 | 0.00973 (0.009155 to 0.009932) | 3,526,221 | - |
| ackley-30-idiomatic | genoxide / cma_es | 10 of 10 | 0 | 15.9 ms | 3,141 | 3,150 | 0.009446 (0.008716 to 0.009948) | 197,854 | - |
| ackley-30-idiomatic | genoxide / de | 10 of 10 | 0 | 17.2 ms | 19,078 | 19,100 | 0.009189 (0.008253 to 0.009884) | 1,113,479 | - |
| ackley-30-idiomatic | genoxide / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,056 | 0.02371 (0.01815 to 0.03297) | 894,147 | - |
| ackley-30-idiomatic | genoxide_python / cma_es | 10 of 10 | 0 | 18.6 ms | 3,141 | 3,150 | 0.009446 (0.008716 to 0.009948) | 169,080 | - |
| ackley-30-idiomatic | genoxide_python / de | 10 of 10 | 0 | 18.3 ms | 19,078 | 19,100 | 0.009189 (0.008253 to 0.009884) | 1,041,219 | - |
| ackley-30-idiomatic | jenetics / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,087 | 11.75 (11.29 to 12.38) | 947,521 | - |
| ackley-30-idiomatic | jmetal / cma_es | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,000 | 19.86 (19.54 to 19.94) | 113,898 | - |
| ackley-30-idiomatic | jmetal / de | 10 of 10 | 0 | 190.0 ms | 56,124 | 56,023 | 0.009715 (0.008581 to 0.009891) | 295,311 | - |
| ackley-30-idiomatic | jmetal / ga | 10 of 10 | 0 | 868.3 ms | 147,867 | 145,676 | 0.009596 (0.00907 to 0.009972) | 170,296 | - |
| ackley-30-idiomatic | metaheuristics_jl / de | 10 of 10 | 0 | 423.8 ms | 471,458 | 470,850 | 0.009783 (0.0092 to 0.009995) | 1,112,727 | - |
| ackley-30-idiomatic | metaheuristics_jl / eca | 9 of 10 | 0 | 323.8 ms | 247,643 | 69,195 | 0.009749 (0.008992 to 1.155) | 764,796 | - |
| ackley-30-idiomatic | metaheuristics_jl / pso | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,200 | 3.996 (3.009 to 13.23) | 700,252 | - |
| ackley-30-idiomatic | moors / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,200 | 14.96 (14.02 to 15.85) | 589,341 | - |
| ackley-30-idiomatic | nevergrad / ngiohtuned | 6 of 10 | 4 (at 7% of the budget) | 97.18 s | 114,571 | 69,419 | 0.005321 (0.001865 to 0.01309) | 1,179 | - |
| ackley-30-idiomatic | nevergrad / one_plus_one | 0 of 3 | 3 (at 59% of the budget) | 0/3 reached | 0/3 reached | 597,916 | 18.99 (18.59 to 19.01) | 9,998 | - |
| ackley-30-idiomatic | nevergrad / scr_hammersley | 0 of 3 | 3 (at 14% of the budget) | 0/3 reached | 0/3 reached | 143,440 | 19.94 (19.69 to 20.22) | 2,390 | - |
| ackley-30-idiomatic | openga / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,004,000 | 9.39 (7.564 to 11.03) | 156,109 | - |
| ackley-30-idiomatic | openga / ga_assist | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,020 | 13.54 (13.04 to 14.12) | 966,461 | - |
| ackley-30-idiomatic | pycma / bipop_cma_es | 10 of 10 | 0 | 158.2 ms | 3,638 | 3,633 | 0.009543 (0.008723 to 0.009995) | 23,053 | - |
| ackley-30-idiomatic | pycma / ipop_cma_es | 10 of 10 | 0 | 159.8 ms | 3,638 | 3,633 | 0.009543 (0.008723 to 0.009995) | 22,815 | - |
| ackley-30-idiomatic | pygad / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,019 | 19.65 (16.69 to 19.95) | 34,224 | - |
| ackley-30-idiomatic | pygmo / cma_es | 10 of 10 | 0 | 24.8 ms | 3,896 | 3,870 | 0.009635 (0.008864 to 0.009983) | 157,393 | - |
| ackley-30-idiomatic | pygmo / sade | 10 of 10 | 0 | 137.4 ms | 12,294 | 12,248 | 0.009557 (0.008561 to 0.0099) | 89,413 | - |
| ackley-30-idiomatic | pygmo / simulated_annealing | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,000 | 0.3373 (0.294 to 0.3875) | 91,575 | - |
| ackley-30-idiomatic | pymoo / cma_es | 10 of 10 | 0 | 186.2 ms | 3,501 | 3,429 | 0.009455 (0.008173 to 0.00992) | 18,780 | - |
| ackley-30-idiomatic | pymoo / de | 10 of 10 | 0 | 937.1 ms | 51,898 | 52,000 | 0.009733 (0.008993 to 0.00997) | 55,359 | - |
| ackley-30-idiomatic | pymoo / es | 10 of 10 | 0 | 2.36 s | 84,259 | 84,529 | 0.009548 (0.008713 to 0.009872) | 35,757 | - |
| ackley-30-idiomatic | radiate / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,184 | 0.04399 (0.03503 to 0.05937) | 1,278,526 | - |
| ackley-30-idiomatic | radiate / ga_blend | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,135 | 20.64 (20.59 to 20.65) | 1,291,561 | - |
| ackley-30-idiomatic | radiate / ga_intermediate | 2 of 9 | 0 | 2/9 reached | 2/9 reached | 1,000,218 | 0.01324 (0.007582 to 0.02705) | 1,352,139 | - |
| ackley-30-idiomatic | scipy / de | 10 of 10 | 0 | 2.25 s | 120,178 | 113,866 | 0.009589 (0.008985 to 0.009939) | 53,461 | - |
| ackley-30-idiomatic | scipy / direct | 10 of 10 | 0 | 5.83 s | 652,801 | 652,801 | 0.009918 (0.009918 to 0.009918) | 111,787 | - |
| ackley-30-idiomatic | scipy / dual_annealing | 10 of 10 | 0 | 319.9 ms | 26,068 | 25,854 | 0.005814 (0.00299 to 0.009744) | 81,494 | - |
| nqueens-32-idiomatic | deap / ga | 10 of 10 | 0 | 768.6 ms | 56,534 | 39,486 | 0 (0 to 0) | 73,637 | - |
| nqueens-32-idiomatic | evolutionary_jl / es | 1 of 10 | 0 | 1/10 reached | 1/10 reached | 500,036 | 1 (0 to 1) | 2,835,696 | 39× |
| nqueens-32-idiomatic | evolutionary_jl / ga | 5 of 10 | 0 | 255.3 ms | 639,012 | 467,686 | 0.5 (0 to 1) | 2,503,347 | 34× |
| nqueens-32-idiomatic | genetic_algorithm / hill_climb | 10 of 10 | 0 | 371 µs | 1,678 | 1,494 | 0 (0 to 0) | 4,509,135 | 61× |
| nqueens-32-idiomatic | genoxide / ga | 4 of 10 | 0 | 307.5 ms | 966,931 | 500,034 | 1 (0 to 2) | 3,144,463 | 43× |
| nqueens-32-idiomatic | genoxide / local_search | 10 of 10 | 0 | 246 µs | 1,140 | 1,136 | 0 (0 to 0) | 4,606,869 | 63× |
| nqueens-32-idiomatic | genoxide_python / ga | 4 of 10 | 0 | 4.96 s | 966,931 | 500,034 | 1 (0 to 2) | 194,997 | 2.6× |
| nqueens-32-idiomatic | genoxide_python / local_search | 10 of 10 | 0 | 6.3 ms | 1,140 | 1,136 | 0 (0 to 0) | 180,172 | 2.4× |
| nqueens-32-idiomatic | jenetics / ga | 8 of 10 | 0 | 558.8 ms | 326,094 | 251,306 | 0 (0 to 1) | 583,707 | 7.9× |
| nqueens-32-idiomatic | jmetal / ga | 5 of 10 | 0 | 2.00 s | 528,519 | 267,893 | 0.5 (0 to 3) | 263,826 | 3.6× |
| nqueens-32-idiomatic | metaheuristics_jl / brkga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,035 | 4 (4 to 5) | 1,193,499 | 16× |
| nqueens-32-idiomatic | metaheuristics_jl / ga | 5 of 10 | 0 | 983.4 ms | 682,686 | 376,400 | 0.5 (0 to 1) | 694,250 | 9.4× |
| nqueens-32-idiomatic | moors / ga | 10 of 10 | 0 | 135.7 ms | 78,262 | 72,800 | 0 (0 to 0) | 577,451 | 7.8× |
| nqueens-32-idiomatic | nevergrad / genetic_de | 0 of 3 | 3 (at 91% of the budget) | 0/3 reached | 0/3 reached | 457,975 | 2 (1 to 2) | 7,646 | 0.1× |
| nqueens-32-idiomatic | nevergrad / ngiohtuned | 0 of 3 | 3 (at 43% of the budget) | 0/3 reached | 0/3 reached | 215,804 | 2 (1 to 2) | 3,594 | 0.0× |
| nqueens-32-idiomatic | nevergrad / rotated_two_points_de | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,000 | 3.5 (3 to 5) | 9,913 | 0.1× |
| nqueens-32-idiomatic | openga / ga | 2 of 10 | 0 | 2/10 reached | 2/10 reached | 500,050 | 1 (0 to 1) | 1,421,584 | 19× |
| nqueens-32-idiomatic | pygad / ga | 0 of 3 | 3 (at <1% of the budget) | 0/3 reached | 0/3 reached | 184 | 7 (7 to 7) | 3 | 0.0× |
| nqueens-32-idiomatic | pymoo / brkga | 4 of 10 | 0 | 40.23 s | 1,094,086 | 500,000 | 1 (0 to 2) | 27,199 | 0.4× |
| nqueens-32-idiomatic | pymoo / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,000 | 1 (1 to 2) | 25,602 | 0.3× |
| nqueens-32-idiomatic | radiate / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,062 | 5 (3 to 5) | 731,565 | 9.9× |
| nqueens-32-idiomatic | radiate / ga_pmx | 2 of 10 | 0 | 2/10 reached | 2/10 reached | 500,034 | 4 (0 to 5) | 559,013 | 7.6× |
| nqueens-64-idiomatic | deap / ga | 10 of 10 | 0 | 2.40 s | 109,295 | 63,456 | 0 (0 to 0) | 45,547 | - |
| nqueens-64-idiomatic | evolutionary_jl / es | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,048 | 7 (5 to 8) | 1,779,199 | 39× |
| nqueens-64-idiomatic | evolutionary_jl / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,054 | 5.5 (3 to 7) | 1,244,521 | 27× |
| nqueens-64-idiomatic | genetic_algorithm / hill_climb | 10 of 10 | 0 | 1.2 ms | 4,139 | 3,728 | 0 (0 to 0) | 3,531,615 | 78× |
| nqueens-64-idiomatic | genoxide / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,094 | 2 (1 to 3) | 2,049,627 | 45× |
| nqueens-64-idiomatic | genoxide / local_search | 10 of 10 | 0 | 665 µs | 2,850 | 2,455 | 0 (0 to 0) | 4,279,429 | 94× |
| nqueens-64-idiomatic | genoxide_python / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,094 | 2 (1 to 3) | 132,449 | 2.9× |
| nqueens-64-idiomatic | genoxide_python / local_search | 10 of 10 | 0 | 20.6 ms | 2,850 | 2,455 | 0 (0 to 0) | 138,326 | 3.0× |
| nqueens-64-idiomatic | jenetics / ga | 0 of 9 | 0 | 0/9 reached | 0/9 reached | 1,000,133 | 3 (1 to 4) | 349,230 | 7.7× |
| nqueens-64-idiomatic | jmetal / ga | 2 of 10 | 0 | 2/10 reached | 2/10 reached | 1,000,000 | 1 (0 to 2) | 251,045 | 5.5× |
| nqueens-64-idiomatic | metaheuristics_jl / brkga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,030 | 14.5 (13 to 16) | 634,892 | 14× |
| nqueens-64-idiomatic | metaheuristics_jl / ga | 1 of 10 | 0 | 1/10 reached | 1/10 reached | 1,000,000 | 3.5 (0 to 4) | 474,690 | 10× |
| nqueens-64-idiomatic | moors / ga | 10 of 10 | 0 | 873.6 ms | 238,904 | 202,800 | 0 (0 to 0) | 273,576 | 6.0× |
| nqueens-64-idiomatic | nevergrad / genetic_de | 0 of 3 | 3 (at 45% of the budget) | 0/3 reached | 0/3 reached | 450,355 | 11 (10 to 11) | 7,516 | 0.2× |
| nqueens-64-idiomatic | nevergrad / ngiohtuned | 0 of 3 | 3 (at 23% of the budget) | 0/3 reached | 0/3 reached | 232,112 | 16 (15 to 16) | 3,881 | 0.1× |
| nqueens-64-idiomatic | nevergrad / rotated_two_points_de | 0 of 3 | 3 (at 58% of the budget) | 0/3 reached | 0/3 reached | 584,507 | 16 (15 to 17) | 9,733 | 0.2× |
| nqueens-64-idiomatic | openga / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,070 | 3.5 (2 to 5) | 1,112,040 | 24× |
| nqueens-64-idiomatic | pygad / ga | 0 of 3 | 3 (at <1% of the budget) | 0/3 reached | 0/3 reached | 216 | 18 (15 to 18) | 3 | 0.0× |
| nqueens-64-idiomatic | pymoo / brkga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,000 | 5 (2 to 6) | 22,129 | 0.5× |
| nqueens-64-idiomatic | pymoo / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,000 | 8.5 (7 to 9) | 23,964 | 0.5× |
| nqueens-64-idiomatic | radiate / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,048 | 16 (13 to 16) | 411,653 | 9.0× |
| nqueens-64-idiomatic | radiate / ga_pmx | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 1,000,080 | 15 (13 to 17) | 310,433 | 6.8× |
| onemax-100-idiomatic | deap / ga | 10 of 10 | 0 | 168.4 ms | 8,222 | 6,420 | 0 (0 to 0) | 49,473 | - |
| onemax-100-idiomatic | evolutionary_jl / ga | 10 of 10 | 0 | 1.8 ms | 19,969 | 14,754 | 0 (0 to 0) | 11,416,058 | 231× |
| onemax-100-idiomatic | genetic_algorithm / evolve | 10 of 10 | 0 | 444 µs | 1,920 | 1,952 | 0 (0 to 0) | 4,385,769 | 89× |
| onemax-100-idiomatic | genoxide / ga | 10 of 10 | 0 | 397 µs | 3,550 | 3,545 | 0 (0 to 0) | 8,939,070 | 181× |
| onemax-100-idiomatic | genoxide_python / ga | 10 of 10 | 0 | 3.1 ms | 3,550 | 3,545 | 0 (0 to 0) | 1,149,399 | 23× |
| onemax-100-idiomatic | jenetics / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 200,014 | 9.5 (6 to 13) | 522,417 | 11× |
| onemax-100-idiomatic | jmetal / es | 10 of 10 | 0 | 5.5 ms | 1,301 | 1,243 | 0 (0 to 0) | 235,269 | 4.8× |
| onemax-100-idiomatic | jmetal / ga | 10 of 10 | 0 | 41.0 ms | 4,159 | 4,224 | 0 (0 to 0) | 101,466 | 2.1× |
| onemax-100-idiomatic | metaheuristics_jl / ga | 10 of 10 | 0 | 1.6 ms | 1,906 | 2,000 | 0 (0 to 0) | 1,193,666 | 24× |
| onemax-100-idiomatic | moors / ga | 10 of 10 | 0 | 57.0 ms | 11,603 | 11,584 | 0 (0 to 0) | 204,000 | 4.1× |
| onemax-100-idiomatic | nevergrad / discrete_one_plus_one | 10 of 10 | 0 | 1.77 s | 3,031 | 2,876 | 0 (0 to 0) | 1,712 | 0.0× |
| onemax-100-idiomatic | nevergrad / ngiohtuned | 10 of 10 | 0 | 2.62 s | 3,347 | 3,096 | 0 (0 to 0) | 1,280 | 0.0× |
| onemax-100-idiomatic | nevergrad / portfolio_discrete_one_plus_one | 0 of 3 | 3 (at 31% of the budget) | 0/3 reached | 0/3 reached | 62,419 | 4 (4 to 7) | 1,040 | 0.0× |
| onemax-100-idiomatic | openga / ga | 10 of 10 | 0 | 26.4 ms | 18,369 | 10,760 | 0 (0 to 0) | 695,317 | 14× |
| onemax-100-idiomatic | pygad / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 200,014 | 10.5 (8 to 11) | 32,463 | 0.7× |
| onemax-100-idiomatic | pygmo / ga | 10 of 10 | 0 | 10.8 ms | 1,690 | 1,596 | 0 (0 to 0) | 156,141 | 3.2× |
| onemax-100-idiomatic | pygmo / gaco | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 200,000 | 24.5 (23 to 27) | 91,566 | 1.9× |
| onemax-100-idiomatic | pygmo / ihs | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 200,000 | 8.5 (7 to 10) | 126,328 | 2.6× |
| onemax-100-idiomatic | pymoo / ga | 10 of 10 | 0 | 150.7 ms | 5,675 | 5,900 | 0 (0 to 0) | 37,900 | 0.8× |
| onemax-100-idiomatic | radiate / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 200,018 | 18 (16 to 19) | 1,080,166 | 22× |
| onemax-100-idiomatic | radiate / ga_multipoint | 10 of 10 | 0 | 8.3 ms | 11,728 | 11,851 | 0 (0 to 0) | 1,418,794 | 29× |
| onemax-100-idiomatic | radiate / ga_uniform | 10 of 10 | 0 | 9.6 ms | 9,410 | 9,081 | 0 (0 to 0) | 980,435 | 20× |
| onemax-100-matched | deap / ga | 10 of 10 | 0 | 132.3 ms | 6,486 | 6,773 | 0 (0 to 0) | 49,528 | - |
| onemax-100-matched | evolutionary_jl / ga | 10 of 10 | 0 | 676 µs | 9,103 | 9,601 | 0 (0 to 0) | 13,597,705 | 275× |
| onemax-100-matched | genoxide / ga | 10 of 10 | 0 | 1000 µs | 4,859 | 4,938 | 0 (0 to 0) | 4,849,148 | 98× |
| onemax-100-matched | genoxide_python / ga | 10 of 10 | 0 | 4.7 ms | 4,859 | 4,938 | 0 (0 to 0) | 1,040,996 | 21× |
| onemax-100-matched | jenetics / ga | 7 of 10 | 0 | 312.0 ms | 194,118 | 177,885 | 0 (0 to 2) | 622,383 | 13× |
| onemax-100-matched | pygad / ga | 10 of 10 | 0 | 153.3 ms | 6,890 | 7,200 | 0 (0 to 0) | 46,092 | 0.9× |
| onemax-100-matched | radiate / ga | 10 of 10 | 0 | 3.7 ms | 5,495 | 5,648 | 0 (0 to 0) | 1,524,267 | 31× |
| onemax-1000-matched | deap / ga | 10 of 10 | 0 | 15.31 s | 103,982 | 105,287 | 0 (0 to 0) | 6,799 | - |
| onemax-1000-matched | evolutionary_jl / ga | 10 of 10 | 0 | 24.6 ms | 116,421 | 117,151 | 0 (0 to 0) | 4,735,472 | 697× |
| onemax-1000-matched | genoxide / ga | 10 of 10 | 0 | 15.2 ms | 54,566 | 54,038 | 0 (0 to 0) | 3,577,532 | 526× |
| onemax-1000-matched | genoxide_python / ga | 10 of 10 | 0 | 80.1 ms | 54,566 | 54,038 | 0 (0 to 0) | 680,926 | 100× |
| onemax-1000-matched | jenetics / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,036 | 205 (193 to 225) | 113,959 | 17× |
| onemax-1000-matched | pygad / ga | 10 of 10 | 0 | 7.07 s | 113,377 | 113,850 | 0 (0 to 0) | 16,057 | 2.4× |
| onemax-1000-matched | radiate / ga | 10 of 10 | 0 | 284.5 ms | 78,858 | 79,288 | 0 (0 to 0) | 277,632 | 41× |
| rastrigin-10-idiomatic | deap / cma_es | 10 of 10 | 0 | 1.25 s | 139,287 | 132,932 | 0.009003 (0.004843 to 0.00972) | 111,152 | - |
| rastrigin-10-idiomatic | deap / de | 10 of 10 | 0 | 515.9 ms | 23,469 | 23,750 | 0.008844 (0.00547 to 0.009819) | 45,573 | - |
| rastrigin-10-idiomatic | evolutionary_jl / cma_es | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,028 | 40.79 (30.84 to 57.71) | 1,359,371 | - |
| rastrigin-10-idiomatic | evolutionary_jl / de | 10 of 10 | 0 | 69.2 ms | 144,845 | 114,055 | 0.008902 (0.007559 to 0.009571) | 2,094,356 | - |
| rastrigin-10-idiomatic | evolutionary_jl / es | 5 of 10 | 0 | 1.76 s | 829,226 | 480,812 | 0.5021 (0.003422 to 12.67) | 471,293 | - |
| rastrigin-10-idiomatic | genetic_algorithm / evolve | 10 of 10 | 0 | 10.4 ms | 61,146 | 39,136 | 0.009687 (0.008601 to 0.009987) | 5,896,109 | - |
| rastrigin-10-idiomatic | genoxide / cma_es | 10 of 10 | 0 | 52.9 ms | 78,444 | 74,930 | 0.007188 (0.004192 to 0.008683) | 1,483,507 | - |
| rastrigin-10-idiomatic | genoxide / de | 10 of 10 | 0 | 16.2 ms | 39,915 | 40,050 | 0.008834 (0.00476 to 0.009991) | 2,466,873 | - |
| rastrigin-10-idiomatic | genoxide / ga | 10 of 10 | 0 | 38.7 ms | 79,226 | 83,200 | 0.008855 (0.002279 to 0.00993) | 2,044,985 | - |
| rastrigin-10-idiomatic | genoxide_python / cma_es | 10 of 10 | 0 | 76.7 ms | 98,576 | 79,600 | 0.006346 (0.003576 to 0.009718) | 1,286,508 | - |
| rastrigin-10-idiomatic | genoxide_python / de | 10 of 10 | 0 | 20.8 ms | 39,915 | 40,050 | 0.008834 (0.00476 to 0.009991) | 1,917,689 | - |
| rastrigin-10-idiomatic | jenetics / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,028 | 4.847 (3.389 to 7.294) | 1,320,156 | - |
| rastrigin-10-idiomatic | jmetal / cma_es | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,000 | 40.79 (2.042 to 113.9) | 733,477 | - |
| rastrigin-10-idiomatic | jmetal / de | 10 of 10 | 0 | 118.6 ms | 83,751 | 83,558 | 0.00898 (0.006879 to 0.009789) | 705,767 | - |
| rastrigin-10-idiomatic | jmetal / ga | 10 of 10 | 0 | 77.7 ms | 14,456 | 15,245 | 0.008938 (0.005235 to 0.009943) | 185,915 | - |
| rastrigin-10-idiomatic | metaheuristics_jl / de | 10 of 10 | 0 | 67.1 ms | 147,695 | 142,900 | 0.00894 (0.007175 to 0.00959) | 2,202,287 | - |
| rastrigin-10-idiomatic | metaheuristics_jl / eca | 5 of 10 | 0 | 597.7 ms | 692,623 | 462,105 | 0.5024 (0.003886 to 2.985) | 1,158,752 | - |
| rastrigin-10-idiomatic | metaheuristics_jl / pso | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,000 | 11.94 (3.982 to 19.9) | 1,081,759 | - |
| rastrigin-10-idiomatic | moors / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,200 | 6.965 (3.98 to 10.94) | 699,113 | - |
| rastrigin-10-idiomatic | nevergrad / ngiohtuned | 0 of 3 | 3 (at 4% of the budget) | 0/3 reached | 0/3 reached | 21,182 | 0.995 (0.995 to 2.985) | 350 | - |
| rastrigin-10-idiomatic | nevergrad / one_plus_one | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,000 | 68.15 (24.87 to 105.5) | 10,188 | - |
| rastrigin-10-idiomatic | nevergrad / scr_hammersley | 0 of 3 | 3 (at 35% of the budget) | 0/3 reached | 0/3 reached | 178,967 | 56.23 (55.73 to 58.09) | 2,982 | - |
| rastrigin-10-idiomatic | openga / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,000 | 4.48 (1.993 to 5.973) | 167,279 | - |
| rastrigin-10-idiomatic | openga / ga_assist | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,080 | 3.98 (2.985 to 5.97) | 1,498,768 | - |
| rastrigin-10-idiomatic | pycma / bipop_cma_es | 9 of 10 | 0 | 5.58 s | 285,763 | 269,496 | 0.007134 (0.003581 to 0.995) | 51,297 | - |
| rastrigin-10-idiomatic | pycma / ipop_cma_es | 10 of 10 | 0 | 2.68 s | 141,146 | 172,400 | 0.006497 (0.003437 to 0.009509) | 52,837 | - |
| rastrigin-10-idiomatic | pygad / ga | 10 of 10 | 0 | 900.6 ms | 51,882 | 51,003 | 0.008935 (0.005236 to 0.009966) | 57,622 | - |
| rastrigin-10-idiomatic | pygmo / cma_es | 3 of 10 | 0 | 1.32 s | 1,457,514 | 500,000 | 0.995 (0.006246 to 0.995) | 1,104,563 | - |
| rastrigin-10-idiomatic | pygmo / sade | 10 of 10 | 0 | 46.9 ms | 5,617 | 4,973 | 0.009094 (0.006238 to 0.00997) | 119,612 | - |
| rastrigin-10-idiomatic | pygmo / simulated_annealing | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,000 | 1.354 (0.07108 to 3.035) | 122,250 | - |
| rastrigin-10-idiomatic | pymoo / cma_es | 9 of 10 | 0 | 5.70 s | 275,508 | 290,171 | 0.007154 (0.004838 to 5.026) | 48,275 | - |
| rastrigin-10-idiomatic | pymoo / de | 10 of 10 | 0 | 530.2 ms | 32,204 | 32,400 | 0.007958 (0.004392 to 0.009759) | 60,740 | - |
| rastrigin-10-idiomatic | pymoo / es | 1 of 10 | 0 | 1/10 reached | 1/10 reached | 500,000 | 1.99 (0.009473 to 9.95) | 75,641 | - |
| rastrigin-10-idiomatic | radiate / ga | 10 of 10 | 0 | 155.7 ms | 316,620 | 315,079 | 0.00769 (0.001089 to 0.00961) | 2,033,650 | - |
| rastrigin-10-idiomatic | radiate / ga_blend | 1 of 10 | 0 | 1/10 reached | 1/10 reached | 500,099 | 37.09 (0.006455 to 63.99) | 1,777,851 | - |
| rastrigin-10-idiomatic | radiate / ga_intermediate | 10 of 10 | 0 | 48.5 ms | 96,910 | 64,385 | 0.006583 (0.00161 to 0.009609) | 1,999,866 | - |
| rastrigin-10-idiomatic | scipy / de | 10 of 10 | 0 | 1.64 s | 101,886 | 90,210 | 0.008911 (0.006189 to 0.009918) | 61,986 | - |
| rastrigin-10-idiomatic | scipy / direct | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,000 | 17.23 (17.23 to 17.23) | 106,165 | - |
| rastrigin-10-idiomatic | scipy / dual_annealing | 10 of 10 | 0 | 59.4 ms | 4,416 | 4,060 | 4.599e-05 (4.812e-07 to 0.005001) | 74,352 | - |
| rastrigin-30-idiomatic | deap / cma_es | 10 of 10 | 0 | 12.33 s | 833,046 | 904,285 | 0.008968 (0.005883 to 0.009945) | 67,547 | - |
| rastrigin-30-idiomatic | deap / de | 10 of 10 | 0 | 8.68 s | 236,934 | 237,300 | 0.009189 (0.006636 to 0.009929) | 27,316 | - |
| rastrigin-30-idiomatic | evolutionary_jl / cma_es | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,019 | 256.1 (192.2 to 309.1) | 346,493 | - |
| rastrigin-30-idiomatic | evolutionary_jl / de | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,048 | 142.2 (122.5 to 149.7) | 848,853 | - |
| rastrigin-30-idiomatic | evolutionary_jl / es | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,016 | 10.24 (3.98 to 16.05) | 293,732 | - |
| rastrigin-30-idiomatic | genetic_algorithm / evolve | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,022 | 2.985 (0.9954 to 3.98) | 3,868,276 | - |
| rastrigin-30-idiomatic | genoxide / cma_es | 10 of 10 | 0 | 2.79 s | 726,600 | 721,476 | 0.008234 (0.007243 to 0.009791) | 260,302 | - |
| rastrigin-30-idiomatic | genoxide / de | 10 of 10 | 0 | 68.7 ms | 101,218 | 101,300 | 0.009423 (0.008014 to 0.00999) | 1,473,447 | - |
| rastrigin-30-idiomatic | genoxide / ga | 10 of 10 | 0 | 814.1 ms | 819,917 | 794,520 | 0.009762 (0.008671 to 0.009992) | 1,007,148 | - |
| rastrigin-30-idiomatic | genoxide_python / cma_es | 10 of 10 | 0 | 2.69 s | 679,196 | 535,934 | 0.009231 (0.007402 to 0.009714) | 253,062 | - |
| rastrigin-30-idiomatic | genoxide_python / de | 10 of 10 | 0 | 76.9 ms | 101,218 | 101,300 | 0.009423 (0.008015 to 0.00999) | 1,315,730 | - |
| rastrigin-30-idiomatic | jenetics / ga | 0 of 9 | 0 | 0/9 reached | 0/9 reached | 2,000,098 | 121.9 (109.6 to 132) | 991,144 | - |
| rastrigin-30-idiomatic | jmetal / cma_es | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,000 | 283.1 (235.8 to 505.4) | 112,326 | - |
| rastrigin-30-idiomatic | jmetal / de | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,000 | 59.73 (52.54 to 65.95) | 306,050 | - |
| rastrigin-30-idiomatic | jmetal / ga | 10 of 10 | 0 | 400.7 ms | 62,212 | 58,324 | 0.009697 (0.008194 to 0.01) | 155,255 | - |
| rastrigin-30-idiomatic | metaheuristics_jl / de | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,100 | 95.08 (77.79 to 105.5) | 916,955 | - |
| rastrigin-30-idiomatic | metaheuristics_jl / eca | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,040 | 9.95 (4.975 to 15.92) | 669,185 | - |
| rastrigin-30-idiomatic | metaheuristics_jl / pso | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,100 | 92.03 (45.77 to 101.5) | 725,745 | - |
| rastrigin-30-idiomatic | moors / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,200 | 55.72 (45.77 to 81.59) | 595,534 | - |
| rastrigin-30-idiomatic | nevergrad / ngiohtuned | 0 of 3 | 3 (at 3% of the budget) | 0/3 reached | 0/3 reached | 63,779 | 142.3 (141.3 to 158.2) | 1,058 | - |
| rastrigin-30-idiomatic | nevergrad / one_plus_one | 0 of 3 | 3 (at 30% of the budget) | 0/3 reached | 0/3 reached | 610,290 | 162.2 (161.2 to 183.1) | 10,188 | - |
| rastrigin-30-idiomatic | nevergrad / scr_hammersley | 0 of 3 | 3 (at 7% of the budget) | 0/3 reached | 0/3 reached | 144,786 | 369.8 (328 to 381.6) | 2,413 | - |
| rastrigin-30-idiomatic | openga / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,001,000 | 43.28 (32.84 to 57.33) | 155,193 | - |
| rastrigin-30-idiomatic | openga / ga_assist | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,060 | 46.27 (38.81 to 54.72) | 994,266 | - |
| rastrigin-30-idiomatic | pycma / bipop_cma_es | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,000 | 7.462 (4.975 to 9.95) | 40,324 | - |
| rastrigin-30-idiomatic | pycma / ipop_cma_es | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,000 | 8.457 (5.97 to 9.95) | 41,149 | - |
| rastrigin-30-idiomatic | pygad / ga | 10 of 10 | 0 | 12.66 s | 450,864 | 452,074 | 0.009717 (0.009424 to 0.009998) | 35,624 | - |
| rastrigin-30-idiomatic | pygmo / cma_es | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,000 | 1.99 (1.99 to 3.98) | 267,889 | - |
| rastrigin-30-idiomatic | pygmo / sade | 10 of 10 | 0 | 146.6 ms | 17,346 | 15,054 | 0.009559 (0.006603 to 0.009979) | 118,248 | - |
| rastrigin-30-idiomatic | pygmo / simulated_annealing | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,000 | 14 (11.45 to 16.57) | 121,778 | - |
| rastrigin-30-idiomatic | pymoo / cma_es | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,000 | 7.489 (5.97 to 9.95) | 36,048 | - |
| rastrigin-30-idiomatic | pymoo / de | 10 of 10 | 0 | 14.45 s | 791,070 | 773,800 | 0.009295 (0.007955 to 0.009846) | 54,743 | - |
| rastrigin-30-idiomatic | pymoo / es | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,000 | 191.8 (181.4 to 200.9) | 34,755 | - |
| rastrigin-30-idiomatic | radiate / ga | 10 of 10 | 0 | 593.1 ms | 765,066 | 741,760 | 0.009455 (0.008694 to 0.009999) | 1,290,013 | - |
| rastrigin-30-idiomatic | radiate / ga_blend | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,184 | 422 (385.2 to 425.2) | 1,236,797 | - |
| rastrigin-30-idiomatic | radiate / ga_intermediate | 10 of 10 | 0 | 389.5 ms | 512,300 | 486,796 | 0.008434 (0.005892 to 0.00986) | 1,315,178 | - |
| rastrigin-30-idiomatic | scipy / de | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,000 | 71.3 (66.05 to 78.83) | 53,471 | - |
| rastrigin-30-idiomatic | scipy / direct | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 2,000,000 | 76.62 (76.62 to 76.62) | 55,976 | - |
| rastrigin-30-idiomatic | scipy / dual_annealing | 10 of 10 | 0 | 250.4 ms | 20,545 | 19,170 | 4.709e-05 (1.236e-08 to 0.004273) | 82,048 | - |
| rosenbrock-10-idiomatic | deap / cma_es | 10 of 10 | 0 | 82.0 ms | 6,097 | 5,595 | 0.008909 (0.006812 to 0.009785) | 74,420 | - |
| rosenbrock-10-idiomatic | deap / de | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,000 | 0.7212 (0.2694 to 0.9105) | 106,337 | - |
| rosenbrock-10-idiomatic | evolutionary_jl / cma_es | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,013 | 0.9038 (0.03125 to 1.394) | 956,580 | - |
| rosenbrock-10-idiomatic | evolutionary_jl / de | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,054 | 2.396 (0.1175 to 5.311) | 2,858,228 | - |
| rosenbrock-10-idiomatic | evolutionary_jl / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,015 | 7.404 (2.626 to 10.11) | 3,207,947 | - |
| rosenbrock-10-idiomatic | genetic_algorithm / evolve | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,028 | 0.1094 (0.04125 to 3.706) | 7,457,551 | - |
| rosenbrock-10-idiomatic | genetic_algorithm / hill_climb | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,003 | 0.01984 (0.01699 to 0.02296) | 32,411,297 | - |
| rosenbrock-10-idiomatic | genoxide / cma_es | 10 of 10 | 0 | 3.9 ms | 6,980 | 5,945 | 0.009263 (0.007279 to 0.009831) | 1,774,194 | - |
| rosenbrock-10-idiomatic | genoxide / de | 10 of 10 | 0 | 11.0 ms | 38,684 | 38,700 | 0.008137 (0.004516 to 0.009652) | 3,530,034 | - |
| rosenbrock-10-idiomatic | genoxide / es | 1 of 10 | 0 | 1/10 reached | 1/10 reached | 500,015 | 0.02042 (0.009992 to 0.4723) | 567,920 | - |
| rosenbrock-10-idiomatic | genoxide_python / de | 10 of 10 | 0 | 16.7 ms | 38,684 | 38,700 | 0.008137 (0.004516 to 0.009652) | 2,311,635 | - |
| rosenbrock-10-idiomatic | genoxide_python / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,036 | 5.691 (0.5505 to 6.667) | 1,890,865 | - |
| rosenbrock-10-idiomatic | genoxide_python / local_search | 1 of 10 | 0 | 1/10 reached | 1/10 reached | 500,000 | 2.874 (0.008596 to 4.718) | 163,216 | - |
| rosenbrock-10-idiomatic | jenetics / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,092 | 7.318 (1.681 to 11.62) | 1,370,714 | - |
| rosenbrock-10-idiomatic | jmetal / cma_es | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,000 | 7.016 (1.651 to 30.19) | 778,304 | - |
| rosenbrock-10-idiomatic | jmetal / de | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,000 | 0.1329 (0.09817 to 0.2299) | 863,877 | - |
| rosenbrock-10-idiomatic | jmetal / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,000 | 2.885 (0.06771 to 5.109) | 224,832 | - |
| rosenbrock-10-idiomatic | metaheuristics_jl / de | 10 of 10 | 0 | 48.2 ms | 196,792 | 194,000 | 0.009268 (0.006677 to 0.009971) | 4,083,970 | - |
| rosenbrock-10-idiomatic | metaheuristics_jl / eca | 10 of 10 | 0 | 15.1 ms | 18,038 | 18,375 | 0.007773 (0.006137 to 0.009695) | 1,193,106 | - |
| rosenbrock-10-idiomatic | metaheuristics_jl / pso | 10 of 10 | 0 | 68.7 ms | 86,545 | 91,200 | 0.008821 (0.008544 to 0.009887) | 1,260,772 | - |
| rosenbrock-10-idiomatic | moors / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,200 | 3.104 (0.02366 to 10.43) | 861,682 | - |
| rosenbrock-10-idiomatic | nevergrad / cma_es | 10 of 10 | 0 | 7.61 s | 22,059 | 23,648 | 0.009974 (0.009013 to 0.01) | 2,897 | - |
| rosenbrock-10-idiomatic | nevergrad / ngiohtuned | 0 of 3 | 3 (at 3% of the budget) | 0/3 reached | 0/3 reached | 18,468 | 0.8187 (0.4717 to 0.963) | 307 | - |
| rosenbrock-10-idiomatic | nevergrad / one_plus_one | 10 of 10 | 0 | 10.77 s | 108,449 | 121,346 | 0.009999 (0.008721 to 0.01) | 10,071 | - |
| rosenbrock-10-idiomatic | openga / ga_assist | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,000 | 2.584 (0.02888 to 3.469) | 1,639,160 | - |
| rosenbrock-10-idiomatic | pycma / cma_es | 10 of 10 | 0 | 147.9 ms | 4,128 | 4,440 | 0.009033 (0.006482 to 0.009852) | 27,954 | - |
| rosenbrock-10-idiomatic | pycma / lq_cma_es | 10 of 10 | 0 | 1.41 s | 1,353 | 1,134 | 0.009604 (0.008445 to 0.009996) | 962 | - |
| rosenbrock-10-idiomatic | pygad / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,032 | 6.995 (6.945 to 7.155) | 69,932 | - |
| rosenbrock-10-idiomatic | pygmo / cma_es | 10 of 10 | 0 | 10.0 ms | 5,164 | 5,250 | 0.008371 (0.005222 to 0.009813) | 512,274 | - |
| rosenbrock-10-idiomatic | pygmo / sade | 10 of 10 | 0 | 254.5 ms | 31,643 | 33,560 | 0.009786 (0.008697 to 0.009967) | 124,301 | - |
| rosenbrock-10-idiomatic | pygmo / xnes | 10 of 10 | 0 | 434.6 ms | 50,769 | 31,992 | 0.009571 (0.008637 to 0.009923) | 116,808 | - |
| rosenbrock-10-idiomatic | pymoo / cma_es | 10 of 10 | 0 | 227.4 ms | 4,894 | 4,175 | 0.008891 (0.004674 to 0.009685) | 21,506 | - |
| rosenbrock-10-idiomatic | pymoo / nelder_mead | 9 of 10 | 0 | 1.34 s | 57,923 | 2,290 | 0.009298 (0.006964 to 3.987) | 43,211 | - |
| rosenbrock-10-idiomatic | radiate / ga | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,036 | 6.875 (0.351 to 7.213) | 2,456,051 | - |
| rosenbrock-10-idiomatic | radiate / ga_blend | 0 of 10 | 0 | 0/10 reached | 0/10 reached | 500,036 | 0.1329 (0.0349 to 5.421) | 2,409,083 | - |
| rosenbrock-10-idiomatic | radiate / ga_intermediate | 1 of 10 | 0 | 1/10 reached | 1/10 reached | 500,045 | 0.1744 (0.009722 to 2.089) | 2,392,888 | - |
| rosenbrock-10-idiomatic | scipy / de | 10 of 10 | 0 | 663.6 ms | 40,564 | 37,336 | 0.008751 (0.006635 to 0.009871) | 61,122 | - |
| rosenbrock-10-idiomatic | scipy / lbfgsb | 10 of 10 | 0 | 9.5 ms | 763 | 820 | 0.005439 (0.003749 to 0.009126) | 80,308 | - |
| rosenbrock-10-idiomatic | scipy / nelder_mead | 10 of 10 | 0 | 360.2 ms | 31,989 | 19,185 | 0.009736 (0.007746 to 0.009942) | 88,805 | - |

## Multi-objective

| Scenario | Library / solver | Runs | Stopped by the time cap | Median hypervolume | Range | Median time | Median evaluations | Evaluations/s |
|---|---|---|---|---|---|---|---|---|
| dtlz1-3-matched | deap / nsga2 | 10 | 0 | 0.1357 | 0.1326 to 0.1367 | 2.21 s | 40,020 | 18,140 |
| dtlz1-3-matched | deap / nsga3 | 10 | 0 | 0.1394 | 0.1344 to 0.1399 | 956.2 ms | 40,020 | 41,921 |
| dtlz1-3-matched | evolutionary_jl / nsga2 | 10 | 0 | 0.0000 | 0.0000 to 0.0000 | 218.0 ms | 40,020 | 174,136 |
| dtlz1-3-matched | genoxide / moead | 10 | 0 | 0.1398 | 0.1391 to 0.1399 | 38.9 ms | 40,040 | 1,025,758 |
| dtlz1-3-matched | genoxide / nsga2 | 10 | 0 | 0.1353 | 0.0206 to 0.1369 | 34.4 ms | 40,035 | 1,158,939 |
| dtlz1-3-matched | genoxide / nsga3 | 10 | 0 | 0.1398 | 0.1390 to 0.1400 | 57.0 ms | 40,040 | 705,503 |
| dtlz1-3-matched | genoxide / sms_emoa | 10 | 0 | 0.1401 | 0.1401 to 0.1402 | 1.06 s | 40,000 | 37,745 |
| dtlz1-3-matched | genoxide / spea2 | 10 | 0 | 0.1392 | 0.1385 to 0.1398 | 329.8 ms | 40,031 | 121,341 |
| dtlz1-3-matched | genoxide_python / moead | 10 | 0 | 0.1398 | 0.1391 to 0.1399 | 46.8 ms | 40,040 | 851,649 |
| dtlz1-3-matched | genoxide_python / nsga2 | 10 | 0 | 0.1353 | 0.0206 to 0.1369 | 42.6 ms | 40,035 | 925,843 |
| dtlz1-3-matched | genoxide_python / nsga3 | 10 | 0 | 0.1398 | 0.1390 to 0.1400 | 66.8 ms | 40,040 | 600,595 |
| dtlz1-3-matched | genoxide_python / sms_emoa | 10 | 0 | 0.1401 | 0.1401 to 0.1402 | 1.62 s | 40,000 | 24,712 |
| dtlz1-3-matched | genoxide_python / spea2 | 10 | 0 | 0.1392 | 0.1385 to 0.1398 | 343.6 ms | 40,031 | 117,012 |
| dtlz1-3-matched | jmetal / moead | 10 | 0 | 0.1397 | 0.1391 to 0.1400 | 231.6 ms | 40,000 | 170,149 |
| dtlz1-3-matched | jmetal / nsga2 | 10 | 0 | 0.1358 | 0.0445 to 0.1367 | 309.4 ms | 40,020 | 128,577 |
| dtlz1-3-matched | jmetal / nsga3 | 10 | 0 | 0.1397 | 0.1378 to 0.1400 | 312.0 ms | 40,020 | 127,159 |
| dtlz1-3-matched | jmetal / sms_emoa | 10 | 0 | 0.1401 | 0.1396 to 0.1401 | 10.86 s | 40,000 | 3,683 |
| dtlz1-3-matched | jmetal / spea2 | 10 | 0 | 0.1394 | 0.1388 to 0.1396 | 3.94 s | 40,020 | 10,139 |
| dtlz1-3-matched | metaheuristics_jl / nsga2 | 10 | 0 | 0.1194 | 0.0971 to 0.1248 | 134.4 ms | 40,020 | 297,926 |
| dtlz1-3-matched | metaheuristics_jl / nsga3 | 10 | 0 | 0.1229 | 0.1110 to 0.1385 | 384.6 ms | 40,020 | 104,006 |
| dtlz1-3-matched | metaheuristics_jl / sms_emoa | 3 | 3 (at 45% of the budget) | 0.1313 | 0.0271 to 0.1378 | 60.09 s | 18,308 | 313 |
| dtlz1-3-matched | metaheuristics_jl / spea2 | 10 | 0 | 0.1378 | 0.1316 to 0.1389 | 660.0 ms | 40,020 | 59,633 |
| dtlz1-3-matched | pygad / nsga2 | 3 | 3 (at 89% of the budget) | 0.0000 | 0.0000 to 0.0000 | 60.17 s | 35,796 | 594 |
| dtlz1-3-matched | pygad / nsga3 | 10 | 0 | 0.0000 | 0.0000 to 0.0000 | 44.76 s | 40,020 | 891 |
| dtlz1-3-matched | pygmo / nsga2 | 10 | 0 | 0.1360 | 0.1342 to 0.1373 | 131.2 ms | 40,020 | 306,801 |
| dtlz1-3-matched | pymoo / moead | 10 | 0 | 0.1396 | 0.1393 to 0.1400 | 10.45 s | 40,040 | 3,831 |
| dtlz1-3-matched | pymoo / nsga2 | 10 | 0 | 0.1357 | 0.1336 to 0.1365 | 529.1 ms | 40,020 | 75,414 |
| dtlz1-3-matched | pymoo / nsga3 | 10 | 0 | 0.1399 | 0.1394 to 0.1400 | 751.6 ms | 40,020 | 53,146 |
| dtlz1-3-matched | pymoo / sms_emoa | 10 | 0 | 0.1401 | 0.1399 to 0.1402 | 20.22 s | 40,000 | 1,975 |
| dtlz1-3-matched | pymoo / spea2 | 10 | 0 | 0.1396 | 0.1390 to 0.1398 | 3.43 s | 40,020 | 11,815 |
| dtlz1-3-matched | radiate / nsga2 | 10 | 0 | 0.0000 | 0.0000 to 0.0000 | 126.1 ms | 40,054 | 319,227 |
| dtlz1-3-matched | radiate / nsga3 | 10 | 0 | 0.0000 | 0.0000 to 0.0000 | 105.0 ms | 40,076 | 380,176 |
| dtlz2-3-matched | deap / nsga2 | 10 | 0 | 0.6891 | 0.6852 to 0.6986 | 1.32 s | 25,024 | 18,988 |
| dtlz2-3-matched | deap / nsga3 | 10 | 0 | 0.7435 | 0.7418 to 0.7439 | 656.8 ms | 25,024 | 38,034 |
| dtlz2-3-matched | evolutionary_jl / nsga2 | 10 | 0 | 0.2931 | 0.2302 to 0.4128 | 251.7 ms | 25,024 | 99,013 |
| dtlz2-3-matched | genoxide / moead | 10 | 0 | 0.7441 | 0.7439 to 0.7442 | 28.8 ms | 25,014 | 865,482 |
| dtlz2-3-matched | genoxide / nsga2 | 10 | 0 | 0.6984 | 0.6888 to 0.7080 | 26.1 ms | 25,036 | 960,298 |
| dtlz2-3-matched | genoxide / nsga3 | 10 | 0 | 0.7442 | 0.7428 to 0.7444 | 42.0 ms | 25,024 | 594,923 |
| dtlz2-3-matched | genoxide / sms_emoa | 10 | 0 | 0.7557 | 0.7554 to 0.7557 | 866.6 ms | 25,000 | 28,931 |
| dtlz2-3-matched | genoxide / spea2 | 10 | 0 | 0.7300 | 0.7272 to 0.7316 | 280.8 ms | 25,031 | 89,021 |
| dtlz2-3-matched | genoxide_python / moead | 10 | 0 | 0.7441 | 0.7439 to 0.7442 | 35.0 ms | 25,014 | 703,291 |
| dtlz2-3-matched | genoxide_python / nsga2 | 10 | 0 | 0.6984 | 0.6888 to 0.7080 | 31.9 ms | 25,036 | 780,685 |
| dtlz2-3-matched | genoxide_python / nsga3 | 10 | 0 | 0.7442 | 0.7428 to 0.7444 | 47.9 ms | 25,024 | 514,383 |
| dtlz2-3-matched | genoxide_python / sms_emoa | 10 | 0 | 0.7557 | 0.7554 to 0.7557 | 1.27 s | 25,000 | 19,668 |
| dtlz2-3-matched | genoxide_python / spea2 | 10 | 0 | 0.7300 | 0.7272 to 0.7316 | 279.9 ms | 25,031 | 89,379 |
| dtlz2-3-matched | jmetal / moead | 10 | 0 | 0.7443 | 0.7441 to 0.7445 | 168.5 ms | 25,000 | 146,732 |
| dtlz2-3-matched | jmetal / nsga2 | 10 | 0 | 0.6983 | 0.6914 to 0.7126 | 201.6 ms | 25,024 | 123,917 |
| dtlz2-3-matched | jmetal / nsga3 | 10 | 0 | 0.7439 | 0.7428 to 0.7445 | 200.8 ms | 25,024 | 116,834 |
| dtlz2-3-matched | jmetal / sms_emoa | 10 | 0 | 0.7556 | 0.7556 to 0.7557 | 7.56 s | 25,000 | 3,273 |
| dtlz2-3-matched | jmetal / spea2 | 10 | 0 | 0.7302 | 0.7246 to 0.7340 | 2.93 s | 25,024 | 8,554 |
| dtlz2-3-matched | metaheuristics_jl / nsga2 | 10 | 0 | 0.6107 | 0.5279 to 0.6479 | 79.8 ms | 25,116 | 311,412 |
| dtlz2-3-matched | metaheuristics_jl / nsga3 | 10 | 0 | 0.7384 | 0.7366 to 0.7400 | 272.7 ms | 25,024 | 91,609 |
| dtlz2-3-matched | metaheuristics_jl / sms_emoa | 3 | 3 (at 35% of the budget) | 0.7363 | 0.7332 to 0.7380 | 60.20 s | 8,924 | 147 |
| dtlz2-3-matched | metaheuristics_jl / spea2 | 10 | 0 | 0.7355 | 0.7275 to 0.7380 | 1.38 s | 25,116 | 18,158 |
| dtlz2-3-matched | pygad / nsga2 | 10 | 0 | 0.0640 | 0.0133 to 0.0983 | 47.01 s | 25,049 | 533 |
| dtlz2-3-matched | pygad / nsga3 | 10 | 0 | 0.1252 | 0.0535 to 0.1867 | 30.65 s | 25,024 | 816 |
| dtlz2-3-matched | pygmo / nsga2 | 10 | 0 | 0.6994 | 0.6793 to 0.7111 | 75.1 ms | 25,024 | 332,639 |
| dtlz2-3-matched | pymoo / moead | 10 | 0 | 0.7443 | 0.7440 to 0.7445 | 6.75 s | 25,025 | 3,730 |
| dtlz2-3-matched | pymoo / nsga2 | 10 | 0 | 0.6983 | 0.6891 to 0.7100 | 352.6 ms | 25,024 | 70,833 |
| dtlz2-3-matched | pymoo / nsga3 | 10 | 0 | 0.7444 | 0.7440 to 0.7447 | 498.3 ms | 25,024 | 50,424 |
| dtlz2-3-matched | pymoo / sms_emoa | 10 | 0 | 0.7556 | 0.7553 to 0.7557 | 13.55 s | 25,000 | 1,857 |
| dtlz2-3-matched | pymoo / spea2 | 10 | 0 | 0.7328 | 0.7282 to 0.7343 | 4.20 s | 25,024 | 5,937 |
| dtlz2-3-matched | radiate / nsga2 | 10 | 0 | 0.0838 | 0.0390 to 0.1672 | 72.7 ms | 25,020 | 339,843 |
| dtlz2-3-matched | radiate / nsga3 | 10 | 0 | 0.0456 | 0.0156 to 0.1407 | 56.8 ms | 25,024 | 444,143 |
| zdt1-30-matched | deap / nsga2 | 10 | 0 | 0.8694 | 0.8689 to 0.8698 | 1.48 s | 25,000 | 16,850 |
| zdt1-30-matched | deap / nsga3 | 10 | 0 | 0.8706 | 0.8702 to 0.8706 | 589.2 ms | 25,000 | 42,533 |
| zdt1-30-matched | evolutionary_jl / nsga2 | 10 | 0 | 0.0000 | 0.0000 to 0.0000 | 224.1 ms | 25,000 | 110,759 |
| zdt1-30-matched | genoxide / moead | 10 | 0 | 0.8684 | 0.8676 to 0.8689 | 42.9 ms | 25,045 | 576,278 |
| zdt1-30-matched | genoxide / nsga2 | 10 | 0 | 0.8694 | 0.8690 to 0.8697 | 26.1 ms | 25,020 | 958,291 |
| zdt1-30-matched | genoxide / nsga3 | 10 | 0 | 0.8702 | 0.8698 to 0.8706 | 36.5 ms | 25,023 | 683,780 |
| zdt1-30-matched | genoxide / sms_emoa | 10 | 0 | 0.8719 | 0.8719 to 0.8720 | 404.5 ms | 25,000 | 61,944 |
| zdt1-30-matched | genoxide / spea2 | 10 | 0 | 0.8703 | 0.8699 to 0.8707 | 209.3 ms | 25,060 | 119,567 |
| zdt1-30-matched | genoxide_python / moead | 10 | 0 | 0.8684 | 0.8662 to 0.8697 | 47.0 ms | 25,044 | 532,201 |
| zdt1-30-matched | genoxide_python / nsga2 | 10 | 0 | 0.8694 | 0.8690 to 0.8697 | 29.4 ms | 25,020 | 832,086 |
| zdt1-30-matched | genoxide_python / nsga3 | 10 | 0 | 0.8702 | 0.8698 to 0.8706 | 40.9 ms | 25,023 | 610,892 |
| zdt1-30-matched | genoxide_python / sms_emoa | 10 | 0 | 0.8719 | 0.8719 to 0.8720 | 586.3 ms | 25,000 | 42,577 |
| zdt1-30-matched | genoxide_python / spea2 | 10 | 0 | 0.8703 | 0.8699 to 0.8707 | 206.1 ms | 25,060 | 121,524 |
| zdt1-30-matched | jmetal / moead | 10 | 0 | 0.8706 | 0.8702 to 0.8709 | 180.9 ms | 25,000 | 131,802 |
| zdt1-30-matched | jmetal / nsga2 | 10 | 0 | 0.8693 | 0.8689 to 0.8697 | 217.2 ms | 25,000 | 114,822 |
| zdt1-30-matched | jmetal / nsga3 | 10 | 0 | 0.8705 | 0.8699 to 0.8707 | 223.0 ms | 25,000 | 105,409 |
| zdt1-30-matched | jmetal / sms_emoa | 10 | 0 | 0.8719 | 0.8718 to 0.8719 | 181.2 ms | 25,000 | 135,064 |
| zdt1-30-matched | jmetal / spea2 | 10 | 0 | 0.8696 | 0.8689 to 0.8701 | 2.63 s | 25,000 | 9,486 |
| zdt1-30-matched | metaheuristics_jl / nsga2 | 10 | 0 | 0.8663 | 0.8648 to 0.8672 | 90.8 ms | 25,100 | 271,892 |
| zdt1-30-matched | metaheuristics_jl / nsga3 | 10 | 0 | 0.8676 | 0.8664 to 0.8681 | 230.0 ms | 25,000 | 107,675 |
| zdt1-30-matched | metaheuristics_jl / sms_emoa | 10 | 0 | 0.8719 | 0.8719 to 0.8720 | 1.08 s | 25,000 | 23,030 |
| zdt1-30-matched | metaheuristics_jl / spea2 | 10 | 0 | 0.8659 | 0.8645 to 0.8666 | 625.3 ms | 25,100 | 39,542 |
| zdt1-30-matched | pygad / nsga2 | 10 | 0 | 0.6866 | 0.6293 to 0.7077 | 53.16 s | 25,042 | 471 |
| zdt1-30-matched | pygad / nsga3 | 10 | 0 | 0.7927 | 0.7390 to 0.8136 | 33.67 s | 25,000 | 743 |
| zdt1-30-matched | pygmo / nsga2 | 10 | 0 | 0.8696 | 0.8691 to 0.8701 | 70.7 ms | 25,000 | 353,895 |
| zdt1-30-matched | pymoo / moead | 10 | 0 | 0.8699 | 0.8672 to 0.8707 | 6.09 s | 25,000 | 4,102 |
| zdt1-30-matched | pymoo / nsga2 | 10 | 0 | 0.8695 | 0.8688 to 0.8699 | 353.8 ms | 25,000 | 70,245 |
| zdt1-30-matched | pymoo / nsga3 | 10 | 0 | 0.8705 | 0.8702 to 0.8709 | 484.7 ms | 25,000 | 51,294 |
| zdt1-30-matched | pymoo / sms_emoa | 10 | 0 | 0.8719 | 0.8718 to 0.8719 | 12.73 s | 25,000 | 1,964 |
| zdt1-30-matched | pymoo / spea2 | 10 | 0 | 0.8703 | 0.8700 to 0.8705 | 2.01 s | 25,000 | 12,429 |
| zdt1-30-matched | radiate / nsga2 | 10 | 0 | 0.8059 | 0.7606 to 0.8382 | 90.6 ms | 25,032 | 276,562 |
| zdt1-30-matched | radiate / nsga3 | 10 | 0 | 0.7767 | 0.7226 to 0.8348 | 70.1 ms | 25,000 | 353,850 |
| zdt2-30-matched | deap / nsga2 | 10 | 0 | 0.5361 | 0.5355 to 0.5365 | 1.51 s | 25,000 | 16,579 |
| zdt2-30-matched | deap / nsga3 | 10 | 0 | 0.5367 | 0.5360 to 0.5374 | 575.8 ms | 25,000 | 43,254 |
| zdt2-30-matched | evolutionary_jl / nsga2 | 10 | 0 | 0.0000 | 0.0000 to 0.0000 | 305.0 ms | 25,000 | 82,331 |
| zdt2-30-matched | genoxide / moead | 10 | 0 | 0.5352 | 0.5343 to 0.5359 | 43.2 ms | 25,078 | 575,454 |
| zdt2-30-matched | genoxide / nsga2 | 10 | 0 | 0.5363 | 0.5358 to 0.5367 | 26.1 ms | 25,044 | 949,693 |
| zdt2-30-matched | genoxide / nsga3 | 10 | 0 | 0.5365 | 0.5360 to 0.5370 | 35.8 ms | 25,022 | 696,919 |
| zdt2-30-matched | genoxide / sms_emoa | 10 | 0 | 0.5386 | 0.5385 to 0.5387 | 374.4 ms | 25,000 | 66,489 |
| zdt2-30-matched | genoxide / spea2 | 10 | 0 | 0.5365 | 0.5361 to 0.5373 | 203.9 ms | 25,034 | 123,443 |
| zdt2-30-matched | genoxide_python / moead | 10 | 0 | 0.5351 | 0.5307 to 0.5366 | 46.7 ms | 25,070 | 536,699 |
| zdt2-30-matched | genoxide_python / nsga2 | 10 | 0 | 0.5363 | 0.5358 to 0.5367 | 29.3 ms | 25,044 | 858,519 |
| zdt2-30-matched | genoxide_python / nsga3 | 10 | 0 | 0.5365 | 0.5360 to 0.5370 | 39.1 ms | 25,022 | 632,537 |
| zdt2-30-matched | genoxide_python / sms_emoa | 10 | 0 | 0.5386 | 0.5385 to 0.5387 | 556.9 ms | 25,000 | 44,824 |
| zdt2-30-matched | genoxide_python / spea2 | 10 | 0 | 0.5365 | 0.5361 to 0.5373 | 200.2 ms | 25,034 | 124,933 |
| zdt2-30-matched | jmetal / moead | 10 | 0 | 0.5374 | 0.5371 to 0.5379 | 180.7 ms | 25,000 | 131,604 |
| zdt2-30-matched | jmetal / nsga2 | 10 | 0 | 0.5359 | 0.5354 to 0.5364 | 221.4 ms | 25,000 | 108,691 |
| zdt2-30-matched | jmetal / nsga3 | 10 | 0 | 0.5368 | 0.5364 to 0.5375 | 229.8 ms | 25,000 | 104,039 |
| zdt2-30-matched | jmetal / sms_emoa | 10 | 0 | 0.5385 | 0.5382 to 0.5386 | 174.4 ms | 25,000 | 139,634 |
| zdt2-30-matched | jmetal / spea2 | 10 | 0 | 0.5361 | 0.5352 to 0.5367 | 2.60 s | 25,000 | 9,584 |
| zdt2-30-matched | metaheuristics_jl / nsga2 | 10 | 0 | 0.5334 | 0.5326 to 0.5348 | 100.5 ms | 25,100 | 236,274 |
| zdt2-30-matched | metaheuristics_jl / nsga3 | 10 | 0 | 0.5342 | 0.5330 to 0.5363 | 232.9 ms | 25,000 | 105,931 |
| zdt2-30-matched | metaheuristics_jl / sms_emoa | 10 | 0 | 0.5386 | 0.5386 to 0.5387 | 1.16 s | 25,000 | 21,618 |
| zdt2-30-matched | metaheuristics_jl / spea2 | 10 | 0 | 0.5304 | 0.5294 to 0.5320 | 588.4 ms | 25,100 | 42,347 |
| zdt2-30-matched | pygad / nsga2 | 10 | 0 | 0.2569 | 0.2203 to 0.3174 | 51.30 s | 25,034 | 488 |
| zdt2-30-matched | pygad / nsga3 | 10 | 0 | 0.3591 | 0.2957 to 0.4968 | 32.84 s | 25,000 | 761 |
| zdt2-30-matched | pygmo / nsga2 | 10 | 0 | 0.5363 | 0.5361 to 0.5365 | 81.4 ms | 25,000 | 300,847 |
| zdt2-30-matched | pymoo / moead | 10 | 0 | 0.5375 | 0.5356 to 0.5382 | 6.30 s | 25,000 | 3,749 |
| zdt2-30-matched | pymoo / nsga2 | 10 | 0 | 0.5358 | 0.5353 to 0.5363 | 371.2 ms | 25,000 | 65,305 |
| zdt2-30-matched | pymoo / nsga3 | 10 | 0 | 0.5367 | 0.5346 to 0.5371 | 492.9 ms | 25,000 | 48,622 |
| zdt2-30-matched | pymoo / sms_emoa | 10 | 0 | 0.5385 | 0.5379 to 0.5387 | 13.30 s | 25,000 | 1,850 |
| zdt2-30-matched | pymoo / spea2 | 10 | 0 | 0.5367 | 0.5363 to 0.5374 | 1.92 s | 25,000 | 12,723 |
| zdt2-30-matched | radiate / nsga2 | 10 | 0 | 0.4383 | 0.3803 to 0.4528 | 95.7 ms | 25,032 | 262,682 |
| zdt2-30-matched | radiate / nsga3 | 10 | 0 | 0.4069 | 0.3585 to 0.4448 | 75.6 ms | 25,000 | 331,597 |
| zdt3-30-matched | deap / nsga2 | 10 | 0 | 1.3273 | 1.2446 to 1.3277 | 1.51 s | 25,000 | 16,464 |
| zdt3-30-matched | deap / nsga3 | 10 | 0 | 1.3254 | 1.2434 to 1.3268 | 593.6 ms | 25,000 | 41,617 |
| zdt3-30-matched | evolutionary_jl / nsga2 | 10 | 0 | 0.0000 | 0.0000 to 0.0000 | 225.3 ms | 25,000 | 111,173 |
| zdt3-30-matched | genoxide / moead | 10 | 0 | 1.3216 | 1.3196 to 1.3226 | 43.9 ms | 25,034 | 567,269 |
| zdt3-30-matched | genoxide / nsga2 | 10 | 0 | 1.3273 | 1.2447 to 1.3277 | 26.7 ms | 25,062 | 919,739 |
| zdt3-30-matched | genoxide / nsga3 | 10 | 0 | 1.3253 | 1.3240 to 1.3262 | 38.1 ms | 25,013 | 657,205 |
| zdt3-30-matched | genoxide / sms_emoa | 10 | 0 | 1.3292 | 1.2457 to 1.3293 | 413.9 ms | 25,000 | 60,557 |
| zdt3-30-matched | genoxide / spea2 | 10 | 0 | 1.3272 | 1.3269 to 1.3277 | 207.1 ms | 25,050 | 121,068 |
| zdt3-30-matched | genoxide_python / moead | 10 | 0 | 1.3222 | 1.3208 to 1.3236 | 48.4 ms | 25,035 | 520,102 |
| zdt3-30-matched | genoxide_python / nsga2 | 10 | 0 | 1.3273 | 1.2447 to 1.3277 | 30.4 ms | 25,062 | 812,049 |
| zdt3-30-matched | genoxide_python / nsga3 | 10 | 0 | 1.3253 | 1.3240 to 1.3262 | 42.1 ms | 25,013 | 594,681 |
| zdt3-30-matched | genoxide_python / sms_emoa | 10 | 0 | 1.3292 | 1.2457 to 1.3293 | 630.7 ms | 25,000 | 39,721 |
| zdt3-30-matched | genoxide_python / spea2 | 10 | 0 | 1.3272 | 1.3269 to 1.3277 | 205.2 ms | 25,050 | 121,995 |
| zdt3-30-matched | jmetal / moead | 10 | 0 | 1.3253 | 1.3251 to 1.3258 | 177.1 ms | 25,000 | 135,085 |
| zdt3-30-matched | jmetal / nsga2 | 10 | 0 | 1.3274 | 1.3268 to 1.3277 | 218.2 ms | 25,000 | 114,857 |
| zdt3-30-matched | jmetal / nsga3 | 10 | 0 | 1.3256 | 1.3248 to 1.3266 | 222.9 ms | 25,000 | 107,746 |
| zdt3-30-matched | jmetal / sms_emoa | 10 | 0 | 1.3292 | 1.3290 to 1.3293 | 172.8 ms | 25,000 | 140,118 |
| zdt3-30-matched | jmetal / spea2 | 10 | 0 | 1.3256 | 1.2419 to 1.3264 | 2.63 s | 25,000 | 9,493 |
| zdt3-30-matched | metaheuristics_jl / nsga2 | 10 | 0 | 1.3229 | 1.3226 to 1.3253 | 90.6 ms | 25,100 | 273,975 |
| zdt3-30-matched | metaheuristics_jl / nsga3 | 10 | 0 | 1.3222 | 1.3197 to 1.3234 | 236.7 ms | 25,000 | 105,345 |
| zdt3-30-matched | metaheuristics_jl / sms_emoa | 10 | 0 | 1.3291 | 1.3289 to 1.3293 | 1.05 s | 25,000 | 23,682 |
| zdt3-30-matched | metaheuristics_jl / spea2 | 10 | 0 | 1.3155 | 1.3112 to 1.3175 | 604.2 ms | 25,100 | 41,772 |
| zdt3-30-matched | pygad / nsga2 | 10 | 0 | 0.8424 | 0.8307 to 0.9115 | 52.48 s | 25,041 | 476 |
| zdt3-30-matched | pygad / nsga3 | 10 | 0 | 0.9897 | 0.8418 to 1.0888 | 33.45 s | 25,000 | 746 |
| zdt3-30-matched | pygmo / nsga2 | 10 | 0 | 1.3274 | 1.3272 to 1.3279 | 71.8 ms | 25,000 | 346,280 |
| zdt3-30-matched | pymoo / moead | 10 | 0 | 1.3239 | 1.3220 to 1.3251 | 6.06 s | 25,000 | 4,131 |
| zdt3-30-matched | pymoo / nsga2 | 10 | 0 | 1.3274 | 1.3269 to 1.3277 | 352.3 ms | 25,000 | 71,157 |
| zdt3-30-matched | pymoo / nsga3 | 10 | 0 | 1.3252 | 1.3248 to 1.3258 | 487.1 ms | 25,000 | 51,386 |
| zdt3-30-matched | pymoo / sms_emoa | 10 | 0 | 1.3292 | 1.3291 to 1.3293 | 12.62 s | 25,000 | 1,980 |
| zdt3-30-matched | pymoo / spea2 | 10 | 0 | 1.3275 | 1.3267 to 1.3281 | 1.78 s | 25,000 | 13,994 |
| zdt3-30-matched | radiate / nsga2 | 10 | 0 | 1.2068 | 1.1087 to 1.2792 | 88.6 ms | 25,028 | 282,900 |
| zdt3-30-matched | radiate / nsga3 | 10 | 0 | 1.1919 | 1.0397 to 1.2504 | 69.7 ms | 25,000 | 359,447 |
