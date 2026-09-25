# Results 20260925-092215

Seeds per scenario: 10, wall time cap per run: 60.0 s
Linux, Intel(R) Core(TM) Ultra 7 265K

- genoxide 0.6.0+3fdd8a1
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
- genoxide_python 0.6.0+3fdd8a1

## Coverage

✓ ran, – can't run the scenario: why, and the bugs found in the libraries, in [notes.md](notes.md).

| Library | OneMax 100 (matched) | OneMax 1000 (matched) | OneMax 100 (idiomatic) | N-Queens 32 (idiomatic) | N-Queens 64 (idiomatic) | Rastrigin 10 (idiomatic) | Rastrigin 30 (idiomatic) | Rosenbrock 10 (idiomatic) | Ackley 30 (idiomatic) | ZDT1, 30 variables | ZDT2, 30 variables | ZDT3, 30 variables | DTLZ2, 3 objectives | DTLZ1, 3 objectives |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| genoxide | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| genoxide (Python) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| genetic_algorithm | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | – | – | – | – | – |
| DEAP | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| PyGAD | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| pymoo | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| radiate | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| moors | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| pycma | – | – | – | – | – | ✓ | ✓ | ✓ | ✓ | – | – | – | – | – |
| Nevergrad | – | – | ✓ | – | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SciPy | – | – | – | – | – | ✓ | ✓ | ✓ | ✓ | – | – | – | – | – |
| pygmo | ✓ | ✓ | ✓ | – | – | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Jenetics | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| jMetal | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Evolutionary.jl | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Metaheuristics.jl | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| openGA | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Single-objective

| Scenario | Library / solver | Success | Median time to target | Median evaluations to target | Median evaluations | Median best | Evaluations/s | Throughput vs DEAP GA |
|---|---|---|---|---|---|---|---|---|
| ackley-30-idiomatic | deap / cma_es | 100% (10) | 698.8 ms | 82,200 | 82,200 | 0.009696 | 117,154 | - |
| ackley-30-idiomatic | deap / ga | 0% (10) | - | - | 1,000,000 | 0.2235 | 70,174 | - |
| ackley-30-idiomatic | evolutionary_jl / cma_es | 0% (10) | - | - | 583,451 | 19.46 | 358,767 | 5.1× |
| ackley-30-idiomatic | evolutionary_jl / de | 100% (10) | 142.6 ms | 156,551 | 156,551 | 0.009673 | 1,081,226 | 15× |
| ackley-30-idiomatic | evolutionary_jl / es | 100% (10) | 30.4 ms | 9,716 | 9,716 | 0.009161 | 314,841 | 4.5× |
| ackley-30-idiomatic | evolutionary_jl / ga | 0% (10) | - | - | 1,000,011 | 7.871 | 1,439,255 | 21× |
| ackley-30-idiomatic | genetic_algorithm / evolve | 100% (10) | 65.2 ms | 147,790 | 147,790 | 0.009473 | 2,265,442 | 32× |
| ackley-30-idiomatic | genoxide / cma_es | 100% (10) | 14.2 ms | 3,157 | 3,157 | 0.009242 | 217,456 | 3.1× |
| ackley-30-idiomatic | genoxide / de | 100% (10) | 6.3 ms | 9,360 | 9,360 | 0.009509 | 1,453,165 | 21× |
| ackley-30-idiomatic | genoxide / ga | 100% (10) | 111.0 ms | 254,537 | 254,537 | 0.009447 | 2,289,650 | 33× |
| ackley-30-idiomatic | genoxide_python / cma_es | 100% (10) | 15.8 ms | 3,157 | 3,157 | 0.009242 | 201,987 | 2.9× |
| ackley-30-idiomatic | genoxide_python / de | 100% (10) | 8.1 ms | 9,360 | 9,360 | 0.009509 | 1,145,853 | 16× |
| ackley-30-idiomatic | genoxide_python / ga | 100% (10) | 138.0 ms | 254,537 | 254,537 | 0.009447 | 1,863,499 | 27× |
| ackley-30-idiomatic | jenetics / ga | 0% (10) | - | - | 1,000,057 | 2.259 | 977,332 | 14× |
| ackley-30-idiomatic | jmetal / cma_es | 0% (10) | - | - | 1,000,000 | 19.6 | 117,285 | 1.7× |
| ackley-30-idiomatic | jmetal / de | 100% (10) | 184.8 ms | 57,338 | 57,338 | 0.009534 | 304,417 | 4.3× |
| ackley-30-idiomatic | jmetal / ga | 100% (10) | 838.1 ms | 146,981 | 146,981 | 0.00954 | 172,612 | 2.5× |
| ackley-30-idiomatic | jmetal / pso | 0% (10) | - | - | 1,000,000 | 3.191 | 295,472 | 4.2× |
| ackley-30-idiomatic | metaheuristics_jl / de | 100% (10) | 420.4 ms | 498,450 | 498,450 | 0.009499 | 1,173,087 | 17× |
| ackley-30-idiomatic | metaheuristics_jl / eca | 100% (10) | 93.5 ms | 66,465 | 66,465 | 0.009478 | 712,451 | 10× |
| ackley-30-idiomatic | metaheuristics_jl / ga | 100% (10) | 314.7 ms | 147,900 | 147,900 | 0.009554 | 466,773 | 6.7× |
| ackley-30-idiomatic | metaheuristics_jl / pso | 0% (10) | - | - | 1,000,200 | 2.171 | 784,818 | 11× |
| ackley-30-idiomatic | moors / ga | 0% (10) | - | - | 1,000,200 | 7.898 | 589,594 | 8.4× |
| ackley-30-idiomatic | nevergrad / cma_es | 100% (10) | 15.04 s | 44,906 | 44,906 | 0.009697 | 2,993 | 0.0× |
| ackley-30-idiomatic | nevergrad / de | 0% (10) | - | - | 498,254 | 2.325 | 8,285 | 0.1× |
| ackley-30-idiomatic | nevergrad / ngopt | 0% (10) | - | - | 7,001 | 2.562 | 117 | 0.0× |
| ackley-30-idiomatic | nevergrad / pso | 0% (10) | - | - | 487,544 | 17.45 | 8,116 | 0.1× |
| ackley-30-idiomatic | openga / ga | 0% (10) | - | - | 1,004,000 | 3.467 | 158,570 | 2.3× |
| ackley-30-idiomatic | pycma / cma_es | 100% (10) | 150.0 ms | 3,361 | 3,361 | 0.007707 | 22,711 | 0.3× |
| ackley-30-idiomatic | pygad / ga | 0% (10) | - | - | 1,000,000 | 9.705 | 54,392 | 0.8× |
| ackley-30-idiomatic | pygmo / cma_es | 100% (10) | 35.4 ms | 3,890 | 3,890 | 0.009408 | 110,472 | 1.6× |
| ackley-30-idiomatic | pygmo / ga | 90% (10) | 3.72 s | 813,900 | 837,240 | 0.009776 | 218,584 | 3.1× |
| ackley-30-idiomatic | pygmo / pso | 100% (10) | 55.2 ms | 13,760 | 13,760 | 0.009746 | 250,216 | 3.6× |
| ackley-30-idiomatic | pygmo / sade | 100% (10) | 56.4 ms | 11,950 | 11,950 | 0.00959 | 210,839 | 3.0× |
| ackley-30-idiomatic | pymoo / cma_es | 100% (10) | 173.3 ms | 3,249 | 3,249 | 0.009733 | 18,531 | 0.3× |
| ackley-30-idiomatic | pymoo / de | 100% (10) | 641.7 ms | 26,700 | 26,700 | 0.009817 | 41,598 | 0.6× |
| ackley-30-idiomatic | pymoo / ga | 100% (10) | 3.57 s | 132,400 | 132,400 | 0.009826 | 37,027 | 0.5× |
| ackley-30-idiomatic | radiate / ga | 10% (10) | 782.3 ms | 952,030 | 1,000,122 | 0.01296 | 1,230,379 | 18× |
| ackley-30-idiomatic | scipy / de | 100% (10) | 2.15 s | 117,185 | 117,185 | 1.822e-08 | 54,793 | 0.8× |
| nqueens-32-idiomatic | deap / ga | 100% (10) | 523.5 ms | 39,486 | 39,486 | 0 | 74,604 | - |
| nqueens-32-idiomatic | evolutionary_jl / es | 20% (10) | 3.0 ms | 9,521 | 500,021 | 1 | 3,038,733 | 41× |
| nqueens-32-idiomatic | evolutionary_jl / ga | 10% (10) | 905 µs | 2,501 | 500,001 | 1 | 2,828,475 | 38× |
| nqueens-32-idiomatic | genetic_algorithm / evolve | 100% (10) | 1.1 ms | 4,200 | 4,200 | 0 | 3,409,365 | 46× |
| nqueens-32-idiomatic | genetic_algorithm / hill_climb | 100% (10) | 314 µs | 1,494 | 1,494 | 0 | 4,709,035 | 63× |
| nqueens-32-idiomatic | genoxide / ga | 100% (10) | 590 µs | 2,870 | 2,870 | 0 | 4,792,110 | 64× |
| nqueens-32-idiomatic | genoxide / local_search | 100% (10) | 185 µs | 1,136 | 1,136 | 0 | 5,947,835 | 80× |
| nqueens-32-idiomatic | genoxide_python / ga | 100% (10) | 2.7 ms | 2,870 | 2,870 | 0 | 1,075,766 | 14× |
| nqueens-32-idiomatic | genoxide_python / local_search | 100% (10) | 5.8 ms | 1,136 | 1,136 | 0 | 185,673 | 2.5× |
| nqueens-32-idiomatic | jenetics / ga | 80% (10) | 118.6 ms | 72,176 | 81,520 | 0 | 654,768 | 8.8× |
| nqueens-32-idiomatic | jmetal / ga | 60% (10) | 132.6 ms | 32,806 | 103,888 | 0 | 243,594 | 3.3× |
| nqueens-32-idiomatic | metaheuristics_jl / ga | 10% (10) | 41.6 ms | 33,500 | 500,000 | 1 | 865,291 | 12× |
| nqueens-32-idiomatic | moors / ga | 100% (10) | 125.4 ms | 72,800 | 72,800 | 0 | 560,077 | 7.5× |
| nqueens-32-idiomatic | openga / ga | 30% (10) | 10.5 ms | 14,760 | 500,000 | 1 | 1,332,328 | 18× |
| nqueens-32-idiomatic | pygad / ga | 100% (10) | 3.55 s | 87,486 | 87,486 | 0 | 24,708 | 0.3× |
| nqueens-32-idiomatic | pymoo / ga | 20% (10) | 783.6 ms | 19,390 | 500,000 | 1 | 25,030 | 0.3× |
| nqueens-32-idiomatic | radiate / ga | 0% (10) | - | - | 500,062 | 5 | 721,413 | 9.7× |
| nqueens-64-idiomatic | deap / ga | 100% (10) | 1.38 s | 63,456 | 63,456 | 0 | 45,756 | - |
| nqueens-64-idiomatic | evolutionary_jl / es | 0% (10) | - | - | 1,000,021 | 2.5 | 1,907,580 | 42× |
| nqueens-64-idiomatic | evolutionary_jl / ga | 0% (10) | - | - | 1,000,001 | 3.5 | 1,362,783 | 30× |
| nqueens-64-idiomatic | genetic_algorithm / evolve | 100% (10) | 5.0 ms | 14,874 | 14,874 | 0 | 2,996,690 | 65× |
| nqueens-64-idiomatic | genetic_algorithm / hill_climb | 100% (10) | 988 µs | 3,728 | 3,728 | 0 | 3,792,797 | 83× |
| nqueens-64-idiomatic | genoxide / ga | 100% (10) | 1.9 ms | 6,048 | 6,048 | 0 | 2,738,585 | 60× |
| nqueens-64-idiomatic | genoxide / local_search | 100% (10) | 558 µs | 2,455 | 2,455 | 0 | 3,893,579 | 85× |
| nqueens-64-idiomatic | genoxide_python / ga | 100% (10) | 7.3 ms | 6,048 | 6,048 | 0 | 868,481 | 19× |
| nqueens-64-idiomatic | genoxide_python / local_search | 100% (10) | 19.9 ms | 2,455 | 2,455 | 0 | 125,504 | 2.7× |
| nqueens-64-idiomatic | jenetics / ga | 60% (10) | 2.01 s | 593,590 | 786,114 | 0 | 297,698 | 6.5× |
| nqueens-64-idiomatic | jmetal / ga | 20% (10) | 856.8 ms | 200,740 | 1,000,000 | 1 | 232,190 | 5.1× |
| nqueens-64-idiomatic | metaheuristics_jl / ga | 0% (10) | - | - | 1,000,000 | 4.5 | 549,845 | 12× |
| nqueens-64-idiomatic | moors / ga | 100% (10) | 731.7 ms | 202,800 | 202,800 | 0 | 277,497 | 6.1× |
| nqueens-64-idiomatic | openga / ga | 30% (10) | 25.5 ms | 28,760 | 1,000,080 | 1 | 1,097,632 | 24× |
| nqueens-64-idiomatic | pygad / ga | 100% (10) | 32.82 s | 516,860 | 516,860 | 0 | 15,673 | 0.3× |
| nqueens-64-idiomatic | pymoo / ga | 20% (10) | 8.80 s | 194,320 | 1,000,000 | 2.5 | 21,891 | 0.5× |
| nqueens-64-idiomatic | radiate / ga | 0% (10) | - | - | 1,000,048 | 16 | 413,782 | 9.0× |
| onemax-100-idiomatic | deap / ga | 100% (10) | 136.6 ms | 6,420 | 6,420 | 100 | 49,339 | - |
| onemax-100-idiomatic | evolutionary_jl / ga | 100% (10) | 418 µs | 5,401 | 5,401 | 100 | 2,036,715 | 41× |
| onemax-100-idiomatic | genetic_algorithm / evolve | 100% (10) | 450 µs | 1,952 | 1,952 | 100 | 4,329,578 | 88× |
| onemax-100-idiomatic | genoxide / ga | 100% (10) | 478 µs | 3,183 | 3,183 | 100 | 6,682,096 | 135× |
| onemax-100-idiomatic | genoxide_python / ga | 100% (10) | 1.1 ms | 3,183 | 3,183 | 100 | 3,004,401 | 61× |
| onemax-100-idiomatic | jenetics / ga | 0% (10) | - | - | 200,014 | 90.5 | 578,988 | 12× |
| onemax-100-idiomatic | jmetal / ga | 100% (10) | 25.9 ms | 4,224 | 4,224 | 100 | 157,119 | 3.2× |
| onemax-100-idiomatic | metaheuristics_jl / ga | 100% (10) | 820 µs | 2,000 | 2,000 | 100 | 898,876 | 18× |
| onemax-100-idiomatic | moors / ga | 100% (10) | 56.7 ms | 11,584 | 11,584 | 100 | 208,140 | 4.2× |
| onemax-100-idiomatic | nevergrad / discrete_one_plus_one | 100% (10) | 1.62 s | 2,876 | 2,876 | 100 | 1,801 | 0.0× |
| onemax-100-idiomatic | nevergrad / ngopt | 100% (10) | 1.18 s | 1,724 | 1,724 | 100 | 1,451 | 0.0× |
| onemax-100-idiomatic | openga / ga | 100% (10) | 10.5 ms | 7,270 | 7,270 | 100 | 679,395 | 14× |
| onemax-100-idiomatic | pygad / ga | 0% (10) | - | - | 200,066 | 94.5 | 28,589 | 0.6× |
| onemax-100-idiomatic | pygmo / ga | 100% (10) | 4.9 ms | 1,610 | 1,610 | 100 | 335,998 | 6.8× |
| onemax-100-idiomatic | pymoo / ga | 100% (10) | 170.1 ms | 5,900 | 5,900 | 100 | 34,127 | 0.7× |
| onemax-100-idiomatic | radiate / ga | 0% (10) | - | - | 200,019 | 95 | 1,094,077 | 22× |
| onemax-100-matched | deap / ga | 100% (10) | 135.5 ms | 6,773 | 6,773 | 100 | 49,081 | - |
| onemax-100-matched | evolutionary_jl / ga | 100% (10) | 852 µs | 9,601 | 9,601 | 100 | 2,815,559 | 57× |
| onemax-100-matched | genetic_algorithm / evolve | 100% (10) | 2.1 ms | 9,702 | 9,702 | 100 | 4,315,976 | 88× |
| onemax-100-matched | genoxide / ga | 100% (10) | 1.0 ms | 4,938 | 4,938 | 100 | 4,733,804 | 96× |
| onemax-100-matched | genoxide_python / ga | 100% (10) | 4.7 ms | 4,938 | 4,938 | 100 | 1,050,985 | 21× |
| onemax-100-matched | jenetics / ga | 100% (10) | 15.0 ms | 6,005 | 6,005 | 100 | 389,107 | 7.9× |
| onemax-100-matched | jmetal / ga | 100% (10) | 197.8 ms | 11,438 | 11,438 | 100 | 57,953 | 1.2× |
| onemax-100-matched | metaheuristics_jl / ga | 100% (10) | 3.4 ms | 10,950 | 10,950 | 100 | 1,562,342 | 32× |
| onemax-100-matched | moors / ga | 100% (10) | 8.9 ms | 21,300 | 21,300 | 100 | 2,301,947 | 47× |
| onemax-100-matched | openga / ga | 100% (10) | 6.8 ms | 11,550 | 11,550 | 100 | 1,671,381 | 34× |
| onemax-100-matched | pygad / ga | 100% (10) | 134.4 ms | 6,900 | 6,900 | 100 | 52,064 | 1.1× |
| onemax-100-matched | pygmo / ga | 100% (10) | 21.1 ms | 10,800 | 10,800 | 100 | 510,689 | 10× |
| onemax-100-matched | pymoo / ga | 100% (10) | 182.7 ms | 10,350 | 10,350 | 100 | 56,037 | 1.1× |
| onemax-100-matched | radiate / ga | 100% (10) | 3.9 ms | 5,648 | 5,648 | 100 | 1,364,524 | 28× |
| onemax-1000-matched | deap / ga | 100% (10) | 15.82 s | 105,287 | 105,287 | 1,000 | 6,681 | - |
| onemax-1000-matched | evolutionary_jl / ga | 100% (10) | 58.6 ms | 143,251 | 143,251 | 1,000 | 2,415,380 | 362× |
| onemax-1000-matched | genetic_algorithm / evolve | 100% (10) | 131.9 ms | 110,526 | 110,526 | 1,000 | 844,191 | 126× |
| onemax-1000-matched | genoxide / ga | 100% (10) | 15.4 ms | 54,038 | 54,038 | 1,000 | 3,395,068 | 508× |
| onemax-1000-matched | genoxide_python / ga | 100% (10) | 81.2 ms | 54,038 | 54,038 | 1,000 | 666,579 | 100× |
| onemax-1000-matched | jenetics / ga | 100% (10) | 819.9 ms | 89,270 | 89,270 | 1,000 | 108,231 | 16× |
| onemax-1000-matched | jmetal / ga | 100% (10) | 5.09 s | 162,138 | 162,138 | 1,000 | 31,683 | 4.7× |
| onemax-1000-matched | metaheuristics_jl / ga | 100% (10) | 276.3 ms | 170,250 | 170,250 | 1,000 | 599,502 | 90× |
| onemax-1000-matched | moors / ga | 100% (10) | 939.7 ms | 337,200 | 337,200 | 1,000 | 329,021 | 49× |
| onemax-1000-matched | openga / ga | 100% (10) | 418.3 ms | 173,100 | 173,100 | 1,000 | 410,338 | 61× |
| onemax-1000-matched | pygad / ga | 100% (10) | 11.70 s | 147,150 | 147,150 | 1,000 | 12,623 | 1.9× |
| onemax-1000-matched | pygmo / ga | 100% (10) | 2.45 s | 154,050 | 154,050 | 1,000 | 63,293 | 9.5× |
| onemax-1000-matched | pymoo / ga | 100% (10) | 7.88 s | 172,200 | 172,200 | 1,000 | 21,785 | 3.3× |
| onemax-1000-matched | radiate / ga | 100% (10) | 294.5 ms | 79,288 | 79,288 | 1,000 | 266,572 | 40× |
| rastrigin-10-idiomatic | deap / cma_es | 70% (10) | 67.4 ms | 16,200 | 16,900 | 0.008381 | 251,363 | - |
| rastrigin-10-idiomatic | deap / ga | 0% (10) | - | - | 500,000 | 0.05966 | 133,498 | - |
| rastrigin-10-idiomatic | evolutionary_jl / cma_es | 0% (10) | - | - | 500,001 | 72.13 | 1,521,428 | 11× |
| rastrigin-10-idiomatic | evolutionary_jl / de | 40% (10) | 15.7 ms | 33,601 | 500,001 | 0.995 | 2,392,119 | 18× |
| rastrigin-10-idiomatic | evolutionary_jl / es | 30% (10) | 97.7 ms | 46,316 | 500,016 | 1.404 | 521,701 | 3.9× |
| rastrigin-10-idiomatic | evolutionary_jl / ga | 0% (10) | - | - | 500,061 | 7.96 | 3,480,780 | 26× |
| rastrigin-10-idiomatic | genetic_algorithm / evolve | 100% (10) | 9.6 ms | 44,408 | 44,408 | 0.009043 | 4,516,775 | 34× |
| rastrigin-10-idiomatic | genoxide / cma_es | 100% (10) | 43.1 ms | 79,700 | 79,700 | 0.007527 | 1,788,884 | 13× |
| rastrigin-10-idiomatic | genoxide / de | 100% (10) | 1.4 ms | 4,380 | 4,380 | 0.007977 | 3,142,612 | 24× |
| rastrigin-10-idiomatic | genoxide / ga | 100% (10) | 6.9 ms | 24,782 | 24,782 | 0.007142 | 3,376,664 | 25× |
| rastrigin-10-idiomatic | genoxide_python / cma_es | 100% (10) | 54.5 ms | 77,585 | 77,585 | 0.007961 | 1,450,077 | 11× |
| rastrigin-10-idiomatic | genoxide_python / de | 100% (10) | 2.4 ms | 4,390 | 4,390 | 0.007947 | 1,888,865 | 14× |
| rastrigin-10-idiomatic | genoxide_python / ga | 100% (10) | 8.8 ms | 24,782 | 24,782 | 0.007142 | 2,774,210 | 21× |
| rastrigin-10-idiomatic | jenetics / ga | 80% (10) | 228.5 ms | 352,902 | 406,752 | 0.007488 | 1,455,959 | 11× |
| rastrigin-10-idiomatic | jmetal / cma_es | 0% (10) | - | - | 203,340 | 102.2 | 739,663 | 5.5× |
| rastrigin-10-idiomatic | jmetal / de | 100% (10) | 110.7 ms | 85,012 | 85,012 | 0.007706 | 737,438 | 5.5× |
| rastrigin-10-idiomatic | jmetal / ga | 100% (10) | 74.5 ms | 14,732 | 14,732 | 0.008786 | 198,726 | 1.5× |
| rastrigin-10-idiomatic | jmetal / pso | 0% (10) | - | - | 500,000 | 9.452 | 785,910 | 5.9× |
| rastrigin-10-idiomatic | metaheuristics_jl / de | 100% (10) | 70.7 ms | 173,650 | 173,650 | 0.009076 | 2,474,019 | 19× |
| rastrigin-10-idiomatic | metaheuristics_jl / eca | 10% (10) | 33.3 ms | 44,310 | 500,010 | 2.985 | 1,397,063 | 10× |
| rastrigin-10-idiomatic | metaheuristics_jl / ga | 100% (10) | 16.5 ms | 12,150 | 12,150 | 0.009352 | 765,594 | 5.7× |
| rastrigin-10-idiomatic | metaheuristics_jl / pso | 0% (10) | - | - | 500,000 | 10.94 | 1,167,054 | 8.7× |
| rastrigin-10-idiomatic | moors / ga | 0% (10) | - | - | 500,200 | 2.487 | 727,110 | 5.4× |
| rastrigin-10-idiomatic | nevergrad / cma_es | 0% (10) | - | - | 178,648 | 1.99 | 2,973 | 0.0× |
| rastrigin-10-idiomatic | nevergrad / de | 0% (10) | - | - | 500,000 | 2.487 | 8,473 | 0.1× |
| rastrigin-10-idiomatic | nevergrad / ngopt | 0% (10) | - | - | 4,658 | 9.534 | 78 | 0.0× |
| rastrigin-10-idiomatic | nevergrad / pso | 0% (10) | - | - | 490,788 | 29.11 | 8,142 | 0.1× |
| rastrigin-10-idiomatic | openga / ga | 0% (10) | - | - | 500,000 | 3.485 | 166,385 | 1.2× |
| rastrigin-10-idiomatic | pycma / cma_es | 100% (10) | 1.35 s | 74,113 | 74,113 | 0.000541 | 50,323 | 0.4× |
| rastrigin-10-idiomatic | pygad / ga | 0% (10) | - | - | 500,050 | 0.1598 | 108,955 | 0.8× |
| rastrigin-10-idiomatic | pygmo / cma_es | 0% (10) | - | - | 3,790 | 4.975 | 373,878 | 2.8× |
| rastrigin-10-idiomatic | pygmo / ga | 100% (10) | 184.3 ms | 85,770 | 85,770 | 0.008249 | 465,341 | 3.5× |
| rastrigin-10-idiomatic | pygmo / pso | 20% (10) | 86.0 ms | 50,090 | 500,000 | 1.99 | 560,223 | 4.2× |
| rastrigin-10-idiomatic | pygmo / sade | 100% (10) | 11.2 ms | 4,780 | 4,780 | 0.007378 | 423,910 | 3.2× |
| rastrigin-10-idiomatic | pymoo / cma_es | 100% (10) | 1.61 s | 62,769 | 62,769 | 0.007352 | 39,298 | 0.3× |
| rastrigin-10-idiomatic | pymoo / de | 100% (10) | 269.1 ms | 14,350 | 14,350 | 0.008325 | 52,774 | 0.4× |
| rastrigin-10-idiomatic | pymoo / ga | 100% (10) | 252.9 ms | 11,900 | 11,900 | 0.008134 | 47,944 | 0.4× |
| rastrigin-10-idiomatic | radiate / ga | 100% (10) | 76.2 ms | 149,735 | 149,735 | 0.008089 | 1,984,785 | 15× |
| rastrigin-10-idiomatic | scipy / de | 80% (10) | 1.25 s | 96,505 | 97,030 | 2.842e-14 | 76,622 | 0.6× |
| rastrigin-30-idiomatic | deap / cma_es | 80% (10) | 1.25 s | 129,300 | 134,700 | 0.009199 | 115,494 | - |
| rastrigin-30-idiomatic | deap / ga | 0% (10) | - | - | 2,000,000 | 0.3558 | 69,819 | - |
| rastrigin-30-idiomatic | evolutionary_jl / cma_es | 0% (10) | - | - | 575,701 | 223.3 | 343,765 | 4.9× |
| rastrigin-30-idiomatic | evolutionary_jl / de | 0% (10) | - | - | 2,000,001 | 21.77 | 1,229,152 | 18× |
| rastrigin-30-idiomatic | evolutionary_jl / es | 0% (10) | - | - | 2,000,016 | 11.44 | 313,603 | 4.5× |
| rastrigin-30-idiomatic | evolutionary_jl / ga | 0% (10) | - | - | 2,000,021 | 27.87 | 1,494,863 | 21× |
| rastrigin-30-idiomatic | genetic_algorithm / evolve | 0% (10) | - | - | 202,786 | 2.487 | 2,721,242 | 39× |
| rastrigin-30-idiomatic | genoxide / cma_es | 100% (10) | 1.78 s | 519,288 | 519,288 | 0.008782 | 287,806 | 4.1× |
| rastrigin-30-idiomatic | genoxide / de | 100% (10) | 15.5 ms | 28,280 | 28,280 | 0.009415 | 1,792,364 | 26× |
| rastrigin-30-idiomatic | genoxide / ga | 100% (10) | 40.6 ms | 99,683 | 99,683 | 0.009631 | 2,481,749 | 36× |
| rastrigin-30-idiomatic | genoxide_python / cma_es | 100% (10) | 1.86 s | 542,024 | 542,024 | 0.008846 | 291,156 | 4.2× |
| rastrigin-30-idiomatic | genoxide_python / de | 100% (10) | 18.3 ms | 28,280 | 28,280 | 0.00932 | 1,525,643 | 22× |
| rastrigin-30-idiomatic | genoxide_python / ga | 100% (10) | 46.9 ms | 99,683 | 99,683 | 0.009631 | 2,093,077 | 30× |
| rastrigin-30-idiomatic | jenetics / ga | 0% (10) | - | - | 2,000,124 | 19.52 | 975,204 | 14× |
| rastrigin-30-idiomatic | jmetal / cma_es | 0% (10) | - | - | 1,447,515 | 264.2 | 111,195 | 1.6× |
| rastrigin-30-idiomatic | jmetal / de | 0% (10) | - | - | 2,000,000 | 68.81 | 309,131 | 4.4× |
| rastrigin-30-idiomatic | jmetal / ga | 100% (10) | 360.3 ms | 56,602 | 56,602 | 0.008953 | 157,711 | 2.3× |
| rastrigin-30-idiomatic | jmetal / pso | 0% (10) | - | - | 2,000,000 | 42.29 | 296,909 | 4.3× |
| rastrigin-30-idiomatic | metaheuristics_jl / de | 0% (10) | - | - | 2,000,100 | 113.5 | 973,908 | 14× |
| rastrigin-30-idiomatic | metaheuristics_jl / eca | 0% (10) | - | - | 2,000,040 | 7.462 | 807,085 | 12× |
| rastrigin-30-idiomatic | metaheuristics_jl / ga | 100% (10) | 126.7 ms | 57,450 | 57,450 | 0.009439 | 460,598 | 6.6× |
| rastrigin-30-idiomatic | metaheuristics_jl / pso | 0% (10) | - | - | 2,000,100 | 35.82 | 769,703 | 11× |
| rastrigin-30-idiomatic | moors / ga | 0% (10) | - | - | 2,000,200 | 16.91 | 615,782 | 8.8× |
| rastrigin-30-idiomatic | nevergrad / cma_es | 0% (10) | - | - | 182,540 | 11.44 | 3,041 | 0.0× |
| rastrigin-30-idiomatic | nevergrad / de | 0% (10) | - | - | 500,168 | 14.89 | 8,339 | 0.1× |
| rastrigin-30-idiomatic | nevergrad / ngopt | 0% (10) | - | - | 5,084 | 105.1 | 85 | 0.0× |
| rastrigin-30-idiomatic | nevergrad / pso | 0% (10) | - | - | 489,378 | 297 | 8,149 | 0.1× |
| rastrigin-30-idiomatic | openga / ga | 0% (10) | - | - | 2,005,000 | 31.84 | 158,442 | 2.3× |
| rastrigin-30-idiomatic | pycma / cma_es | 100% (10) | 18.37 s | 782,502 | 782,502 | 0.0001442 | 42,999 | 0.6× |
| rastrigin-30-idiomatic | pygad / ga | 0% (10) | - | - | 2,000,098 | 53.33 | 54,486 | 0.8× |
| rastrigin-30-idiomatic | pygmo / cma_es | 0% (10) | - | - | 7,700 | 36.81 | 113,398 | 1.6× |
| rastrigin-30-idiomatic | pygmo / ga | 100% (10) | 896.0 ms | 214,640 | 214,640 | 0.009409 | 239,164 | 3.4× |
| rastrigin-30-idiomatic | pygmo / pso | 0% (10) | - | - | 2,000,000 | 58.7 | 231,403 | 3.3× |
| rastrigin-30-idiomatic | pygmo / sade | 100% (10) | 70.0 ms | 15,570 | 15,570 | 0.008959 | 221,959 | 3.2× |
| rastrigin-30-idiomatic | pymoo / cma_es | 100% (10) | 18.02 s | 670,293 | 670,293 | 0.009177 | 37,096 | 0.5× |
| rastrigin-30-idiomatic | pymoo / de | 100% (10) | 2.93 s | 123,100 | 123,100 | 0.005351 | 41,654 | 0.6× |
| rastrigin-30-idiomatic | pymoo / ga | 100% (10) | 1.65 s | 62,050 | 62,050 | 0.009649 | 37,365 | 0.5× |
| rastrigin-30-idiomatic | radiate / ga | 100% (10) | 338.3 ms | 426,978 | 426,978 | 0.009435 | 1,260,217 | 18× |
| rastrigin-30-idiomatic | scipy / de | 0% (10) | - | - | 450,822 | 51.24 | 53,607 | 0.8× |
| rosenbrock-10-idiomatic | deap / cma_es | 100% (10) | 107.2 ms | 25,700 | 25,700 | 0.008626 | 240,722 | - |
| rosenbrock-10-idiomatic | deap / ga | 0% (10) | - | - | 500,000 | 1.926 | 136,425 | - |
| rosenbrock-10-idiomatic | evolutionary_jl / cma_es | 70% (10) | 63.9 ms | 104,901 | 117,401 | 0.009977 | 1,757,996 | 13× |
| rosenbrock-10-idiomatic | evolutionary_jl / de | 100% (10) | 28.7 ms | 89,401 | 89,401 | 0.009321 | 3,064,829 | 22× |
| rosenbrock-10-idiomatic | evolutionary_jl / es | 0% (10) | - | - | 500,016 | 0.07397 | 1,697,636 | 12× |
| rosenbrock-10-idiomatic | evolutionary_jl / ga | 0% (10) | - | - | 500,061 | 4.725 | 4,875,742 | 36× |
| rosenbrock-10-idiomatic | genetic_algorithm / evolve | 10% (10) | 34.4 ms | 249,699 | 500,038 | 0.08916 | 7,409,460 | 54× |
| rosenbrock-10-idiomatic | genoxide / cma_es | 100% (10) | 3.3 ms | 5,945 | 5,945 | 0.009263 | 1,867,797 | 14× |
| rosenbrock-10-idiomatic | genoxide / de | 100% (10) | 1.6 ms | 7,010 | 7,010 | 0.009055 | 4,266,180 | 31× |
| rosenbrock-10-idiomatic | genoxide / ga | 0% (10) | - | - | 500,034 | 1.761 | 4,646,268 | 34× |
| rosenbrock-10-idiomatic | genoxide_python / cma_es | 100% (10) | 6.3 ms | 5,945 | 5,945 | 0.009263 | 979,746 | 7.2× |
| rosenbrock-10-idiomatic | genoxide_python / de | 100% (10) | 3.8 ms | 7,200 | 7,200 | 0.009283 | 1,721,092 | 13× |
| rosenbrock-10-idiomatic | genoxide_python / ga | 0% (10) | - | - | 500,034 | 1.761 | 3,127,561 | 23× |
| rosenbrock-10-idiomatic | jenetics / ga | 0% (10) | - | - | 500,110 | 1.065 | 1,514,035 | 11× |
| rosenbrock-10-idiomatic | jmetal / cma_es | 0% (10) | - | - | 204,685 | 10.2 | 764,164 | 5.6× |
| rosenbrock-10-idiomatic | jmetal / de | 0% (10) | - | - | 500,000 | 0.1329 | 864,303 | 6.3× |
| rosenbrock-10-idiomatic | jmetal / ga | 0% (10) | - | - | 500,000 | 2.885 | 225,525 | 1.7× |
| rosenbrock-10-idiomatic | jmetal / pso | 10% (10) | 606.3 ms | 482,129 | 500,000 | 0.02349 | 781,772 | 5.7× |
| rosenbrock-10-idiomatic | metaheuristics_jl / de | 100% (10) | 42.2 ms | 194,000 | 194,000 | 0.009268 | 4,564,840 | 33× |
| rosenbrock-10-idiomatic | metaheuristics_jl / eca | 100% (10) | 12.4 ms | 18,375 | 18,375 | 0.007773 | 1,397,100 | 10× |
| rosenbrock-10-idiomatic | metaheuristics_jl / ga | 0% (10) | - | - | 500,000 | 5.182 | 834,437 | 6.1× |
| rosenbrock-10-idiomatic | metaheuristics_jl / pso | 100% (10) | 67.5 ms | 91,200 | 91,200 | 0.008821 | 1,355,924 | 9.9× |
| rosenbrock-10-idiomatic | moors / ga | 0% (10) | - | - | 500,200 | 3.104 | 869,419 | 6.4× |
| rosenbrock-10-idiomatic | nevergrad / cma_es | 100% (10) | 7.97 s | 23,648 | 23,648 | 0.009974 | 2,958 | 0.0× |
| rosenbrock-10-idiomatic | nevergrad / de | 10% (10) | 4.83 s | 41,189 | 500,000 | 5.801 | 8,575 | 0.1× |
| rosenbrock-10-idiomatic | nevergrad / ngopt | 0% (10) | - | - | 4,748 | 137 | 79 | 0.0× |
| rosenbrock-10-idiomatic | nevergrad / pso | 0% (10) | - | - | 500,000 | 13,027 | 8,428 | 0.1× |
| rosenbrock-10-idiomatic | openga / ga | 0% (10) | - | - | 500,000 | 7.367 | 178,115 | 1.3× |
| rosenbrock-10-idiomatic | pycma / cma_es | 100% (10) | 174.0 ms | 4,491 | 4,491 | 0.008701 | 28,794 | 0.2× |
| rosenbrock-10-idiomatic | pygad / ga | 0% (10) | - | - | 500,050 | 9.788 | 114,113 | 0.8× |
| rosenbrock-10-idiomatic | pygmo / cma_es | 100% (10) | 16.2 ms | 6,530 | 6,530 | 0.008506 | 396,470 | 2.9× |
| rosenbrock-10-idiomatic | pygmo / ga | 0% (10) | - | - | 500,000 | 2.935 | 479,949 | 3.5× |
| rosenbrock-10-idiomatic | pygmo / pso | 90% (10) | 158.5 ms | 102,480 | 106,970 | 0.009983 | 646,844 | 4.7× |
| rosenbrock-10-idiomatic | pygmo / sade | 100% (10) | 74.1 ms | 33,570 | 33,570 | 0.009786 | 459,266 | 3.4× |
| rosenbrock-10-idiomatic | pymoo / cma_es | 100% (10) | 228.4 ms | 4,676 | 4,676 | 0.00912 | 21,156 | 0.2× |
| rosenbrock-10-idiomatic | pymoo / de | 10% (10) | 3.51 s | 187,600 | 500,000 | 2.875 | 53,225 | 0.4× |
| rosenbrock-10-idiomatic | pymoo / ga | 0% (10) | - | - | 500,000 | 3.969 | 44,617 | 0.3× |
| rosenbrock-10-idiomatic | radiate / ga | 0% (10) | - | - | 500,036 | 6.875 | 2,476,021 | 18× |
| rosenbrock-10-idiomatic | scipy / de | 90% (10) | 505.8 ms | 37,508 | 37,502 | 1.694e-10 | 73,717 | 0.5× |

## Multi-objective

| Scenario | Library / solver | Median hypervolume | Range | Median time | Median evaluations | Evaluations/s |
|---|---|---|---|---|---|---|
| dtlz1-3-matched | deap / nsga2 | 1.3003 | 1.2972 to 1.3013 | 2.13 s | 40,020 | 18,772 |
| dtlz1-3-matched | deap / nsga3 | 1.3039 | 1.3000 to 1.3045 | 922.5 ms | 40,020 | 43,486 |
| dtlz1-3-matched | evolutionary_jl / nsga2 | 0.0000 | 0.0000 to 0.0000 | 215.8 ms | 40,020 | 184,483 |
| dtlz1-3-matched | genoxide / moead | 1.3043 | 1.3030 to 1.3045 | 37.6 ms | 40,040 | 1,043,001 |
| dtlz1-3-matched | genoxide / nsga2 | 1.3002 | 1.2956 to 1.3013 | 33.8 ms | 40,020 | 1,152,836 |
| dtlz1-3-matched | genoxide / nsga3 | 1.3044 | 1.3035 to 1.3046 | 55.1 ms | 40,020 | 708,431 |
| dtlz1-3-matched | genoxide / sms_emoa | 1.3046 | 1.3045 to 1.3047 | 217.5 ms | 40,020 | 187,126 |
| dtlz1-3-matched | genoxide / spea2 | 1.3034 | 1.2997 to 1.3035 | 301.5 ms | 40,020 | 132,054 |
| dtlz1-3-matched | genoxide_python / moead | 1.3043 | 1.3030 to 1.3045 | 43.7 ms | 40,040 | 910,114 |
| dtlz1-3-matched | genoxide_python / nsga2 | 1.3002 | 1.2956 to 1.3013 | 40.1 ms | 40,020 | 996,444 |
| dtlz1-3-matched | genoxide_python / nsga3 | 1.3044 | 1.3035 to 1.3046 | 63.2 ms | 40,020 | 626,858 |
| dtlz1-3-matched | genoxide_python / sms_emoa | 1.3046 | 1.3045 to 1.3047 | 238.4 ms | 40,020 | 169,064 |
| dtlz1-3-matched | genoxide_python / spea2 | 1.3034 | 1.2997 to 1.3035 | 310.7 ms | 40,020 | 128,489 |
| dtlz1-3-matched | jenetics / moea | 0.0000 | 0.0000 to 0.0000 | 288.0 ms | 40,016 | 137,172 |
| dtlz1-3-matched | jenetics / nsga2 | 0.0000 | 0.0000 to 0.0000 | 343.5 ms | 40,019 | 116,095 |
| dtlz1-3-matched | jmetal / moead | 1.3044 | 1.3037 to 1.3046 | 227.5 ms | 40,000 | 175,124 |
| dtlz1-3-matched | jmetal / nsga2 | 1.3003 | 1.1667 to 1.3013 | 301.3 ms | 40,020 | 133,328 |
| dtlz1-3-matched | jmetal / nsga3 | 1.3043 | 1.3023 to 1.3046 | 308.0 ms | 40,020 | 129,932 |
| dtlz1-3-matched | jmetal / smpso | 1.2978 | 1.2950 to 1.2996 | 254.3 ms | 40,020 | 159,993 |
| dtlz1-3-matched | jmetal / sms_emoa | 1.3047 | 1.3042 to 1.3048 | 10.98 s | 40,000 | 3,634 |
| dtlz1-3-matched | jmetal / spea2 | 1.3039 | 1.3032 to 1.3042 | 3.23 s | 40,020 | 12,397 |
| dtlz1-3-matched | metaheuristics_jl / moead | 1.2320 | 0.8376 to 1.2929 | 83.9 ms | 40,040 | 462,086 |
| dtlz1-3-matched | metaheuristics_jl / nsga2 | 1.2646 | 1.1927 to 1.2822 | 128.9 ms | 40,020 | 307,336 |
| dtlz1-3-matched | metaheuristics_jl / nsga3 | 1.2864 | 1.2756 to 1.3024 | 367.8 ms | 40,020 | 108,619 |
| dtlz1-3-matched | metaheuristics_jl / sms_emoa | 1.3011 | 1.0989 to 1.3022 | 60.32 s | 20,056 | 327 |
| dtlz1-3-matched | metaheuristics_jl / spea2 | 1.3015 | 1.2959 to 1.3035 | 661.9 ms | 40,020 | 60,647 |
| dtlz1-3-matched | moors / age_moea | 0.0000 | 0.0000 to 0.0000 | 8.6 ms | 6,072 | 398,291 |
| dtlz1-3-matched | moors / ibea | 0.0000 | 0.0000 to 0.4721 | 133.2 ms | 40,020 | 301,476 |
| dtlz1-3-matched | moors / nsga2 | 0.3384 | 0.0000 to 1.1973 | 38.3 ms | 40,020 | 1,033,065 |
| dtlz1-3-matched | moors / nsga3 | 1.1727 | 0.2367 to 1.2826 | 50.1 ms | 40,020 | 770,180 |
| dtlz1-3-matched | moors / revea | 0.5546 | 0.0000 to 1.2204 | 418.2 ms | 39,560 | 94,448 |
| dtlz1-3-matched | moors / spea2 | 0.1039 | 0.0000 to 0.9493 | 144.4 ms | 40,020 | 276,174 |
| dtlz1-3-matched | nevergrad / de | 0.0000 | 0.0000 to 0.0000 | 19.38 s | 40,000 | 1,996 |
| dtlz1-3-matched | openga / nsga3 | 1.2864 | 1.1616 to 1.2948 | 347.8 ms | 40,020 | 114,994 |
| dtlz1-3-matched | pygad / nsga2 | 1.2985 | 1.0817 to 1.3000 | 60.22 s | 34,954 | 580 |
| dtlz1-3-matched | pygmo / moead | 1.3008 | 0.0000 to 1.3030 | 98.6 ms | 40,040 | 407,211 |
| dtlz1-3-matched | pygmo / nsga2 | 1.3004 | 1.2930 to 1.3019 | 162.9 ms | 40,020 | 244,845 |
| dtlz1-3-matched | pygmo / nspso | 0.0000 | 0.0000 to 0.0000 | 238.2 ms | 40,020 | 167,304 |
| dtlz1-3-matched | pymoo / moead | 1.3041 | 1.3038 to 1.3045 | 9.98 s | 40,040 | 4,014 |
| dtlz1-3-matched | pymoo / nsga2 | 1.3005 | 1.2947 to 1.3015 | 867.2 ms | 40,020 | 45,862 |
| dtlz1-3-matched | pymoo / nsga3 | 1.3046 | 1.3033 to 1.3046 | 985.0 ms | 40,020 | 40,612 |
| dtlz1-3-matched | pymoo / sms_emoa | 1.3046 | 1.3045 to 1.3047 | 1.21 s | 40,020 | 33,138 |
| dtlz1-3-matched | pymoo / spea2 | 1.3042 | 1.3038 to 1.3045 | 3.40 s | 40,020 | 11,821 |
| dtlz1-3-matched | radiate / nsga2 | 0.0000 | 0.0000 to 0.0000 | 123.1 ms | 40,054 | 327,882 |
| dtlz1-3-matched | radiate / nsga3 | 0.0000 | 0.0000 to 0.0000 | 100.0 ms | 40,076 | 393,376 |
| dtlz2-3-matched | deap / nsga2 | 0.6891 | 0.6852 to 0.6986 | 1.28 s | 25,024 | 19,516 |
| dtlz2-3-matched | deap / nsga3 | 0.7435 | 0.7413 to 0.7441 | 629.2 ms | 25,024 | 39,703 |
| dtlz2-3-matched | evolutionary_jl / nsga2 | 0.3144 | 0.2609 to 0.3773 | 242.6 ms | 25,024 | 102,470 |
| dtlz2-3-matched | genoxide / moead | 0.7441 | 0.7439 to 0.7442 | 28.4 ms | 25,014 | 879,987 |
| dtlz2-3-matched | genoxide / nsga2 | 0.6960 | 0.6832 to 0.7061 | 25.2 ms | 25,024 | 965,302 |
| dtlz2-3-matched | genoxide / nsga3 | 0.7442 | 0.7434 to 0.7444 | 41.4 ms | 25,024 | 593,660 |
| dtlz2-3-matched | genoxide / sms_emoa | 0.7543 | 0.7541 to 0.7545 | 334.5 ms | 25,024 | 75,061 |
| dtlz2-3-matched | genoxide / spea2 | 0.7283 | 0.7230 to 0.7323 | 264.6 ms | 25,024 | 94,122 |
| dtlz2-3-matched | genoxide_python / moead | 0.7441 | 0.7439 to 0.7442 | 33.1 ms | 25,014 | 751,234 |
| dtlz2-3-matched | genoxide_python / nsga2 | 0.6960 | 0.6832 to 0.7061 | 30.3 ms | 25,024 | 827,861 |
| dtlz2-3-matched | genoxide_python / nsga3 | 0.7442 | 0.7434 to 0.7444 | 46.7 ms | 25,024 | 528,844 |
| dtlz2-3-matched | genoxide_python / sms_emoa | 0.7543 | 0.7541 to 0.7545 | 353.7 ms | 25,024 | 70,244 |
| dtlz2-3-matched | genoxide_python / spea2 | 0.7283 | 0.7230 to 0.7323 | 263.5 ms | 25,024 | 95,184 |
| dtlz2-3-matched | jenetics / moea | 0.4058 | 0.3590 to 0.4338 | 212.6 ms | 25,013 | 116,647 |
| dtlz2-3-matched | jenetics / nsga2 | 0.5124 | 0.4619 to 0.5482 | 211.8 ms | 25,064 | 116,626 |
| dtlz2-3-matched | jmetal / moead | 0.7443 | 0.7441 to 0.7445 | 157.0 ms | 25,000 | 159,151 |
| dtlz2-3-matched | jmetal / nsga2 | 0.6983 | 0.6914 to 0.7126 | 189.4 ms | 25,024 | 131,834 |
| dtlz2-3-matched | jmetal / nsga3 | 0.7439 | 0.7428 to 0.7445 | 188.0 ms | 25,024 | 132,023 |
| dtlz2-3-matched | jmetal / smpso | 0.6747 | 0.6623 to 0.6857 | 326.8 ms | 25,024 | 77,218 |
| dtlz2-3-matched | jmetal / sms_emoa | 0.7556 | 0.7556 to 0.7557 | 7.21 s | 25,000 | 3,446 |
| dtlz2-3-matched | jmetal / spea2 | 0.7302 | 0.7246 to 0.7340 | 2.43 s | 25,024 | 10,311 |
| dtlz2-3-matched | metaheuristics_jl / moead | 0.6434 | 0.6310 to 0.6629 | 54.3 ms | 25,025 | 436,131 |
| dtlz2-3-matched | metaheuristics_jl / nsga2 | 0.6107 | 0.5279 to 0.6479 | 74.9 ms | 25,116 | 334,956 |
| dtlz2-3-matched | metaheuristics_jl / nsga3 | 0.7384 | 0.7366 to 0.7400 | 262.5 ms | 25,024 | 94,790 |
| dtlz2-3-matched | metaheuristics_jl / sms_emoa | 0.7364 | 0.7340 to 0.7394 | 60.38 s | 9,016 | 150 |
| dtlz2-3-matched | metaheuristics_jl / spea2 | 0.7355 | 0.7275 to 0.7380 | 1.36 s | 25,116 | 18,581 |
| dtlz2-3-matched | moors / age_moea | 0.7165 | 0.6750 to 0.7190 | 604.3 ms | 25,116 | 41,308 |
| dtlz2-3-matched | moors / ibea | 0.7480 | 0.7471 to 0.7488 | 74.9 ms | 25,116 | 333,294 |
| dtlz2-3-matched | moors / nsga2 | 0.6731 | 0.6439 to 0.6790 | 21.2 ms | 25,116 | 1,179,315 |
| dtlz2-3-matched | moors / nsga3 | 0.7201 | 0.7036 to 0.7269 | 30.1 ms | 25,116 | 802,295 |
| dtlz2-3-matched | moors / revea | 0.6831 | 0.6517 to 0.6987 | 261.5 ms | 24,617 | 94,101 |
| dtlz2-3-matched | moors / spea2 | 0.4757 | 0.3963 to 0.5094 | 88.9 ms | 25,116 | 276,163 |
| dtlz2-3-matched | nevergrad / de | 0.6768 | 0.6581 to 0.6932 | 60.02 s | 12,057 | 204 |
| dtlz2-3-matched | openga / nsga3 | 0.6942 | 0.6868 to 0.7084 | 209.9 ms | 25,024 | 118,438 |
| dtlz2-3-matched | pygad / nsga2 | 0.6884 | 0.6859 to 0.7036 | 53.06 s | 25,032 | 472 |
| dtlz2-3-matched | pygmo / moead | 0.6588 | 0.6388 to 0.6705 | 64.5 ms | 25,025 | 381,439 |
| dtlz2-3-matched | pygmo / nsga2 | 0.6994 | 0.6793 to 0.7111 | 102.0 ms | 25,024 | 246,257 |
| dtlz2-3-matched | pygmo / nspso | 0.4620 | 0.4101 to 0.4850 | 150.5 ms | 25,024 | 167,335 |
| dtlz2-3-matched | pymoo / moead | 0.7443 | 0.7440 to 0.7445 | 6.32 s | 25,025 | 3,956 |
| dtlz2-3-matched | pymoo / nsga2 | 0.6970 | 0.6877 to 0.7102 | 573.1 ms | 25,024 | 43,578 |
| dtlz2-3-matched | pymoo / nsga3 | 0.7443 | 0.7439 to 0.7445 | 632.1 ms | 25,024 | 39,401 |
| dtlz2-3-matched | pymoo / sms_emoa | 0.7545 | 0.7542 to 0.7547 | 908.1 ms | 25,024 | 27,598 |
| dtlz2-3-matched | pymoo / spea2 | 0.7321 | 0.7275 to 0.7344 | 4.33 s | 25,024 | 5,780 |
| dtlz2-3-matched | radiate / nsga2 | 0.0838 | 0.0390 to 0.1672 | 69.7 ms | 25,020 | 358,680 |
| dtlz2-3-matched | radiate / nsga3 | 0.0456 | 0.0156 to 0.1407 | 52.5 ms | 25,024 | 475,257 |
| zdt1-30-matched | deap / nsga2 | 0.8694 | 0.8689 to 0.8698 | 1.44 s | 25,000 | 17,333 |
| zdt1-30-matched | evolutionary_jl / nsga2 | 0.0000 | 0.0000 to 0.0000 | 222.6 ms | 25,000 | 110,634 |
| zdt1-30-matched | genoxide / moead | 0.8684 | 0.8676 to 0.8689 | 41.3 ms | 25,045 | 594,301 |
| zdt1-30-matched | genoxide / nsga2 | 0.8696 | 0.8690 to 0.8702 | 25.4 ms | 25,000 | 955,946 |
| zdt1-30-matched | genoxide / sms_emoa | 0.8715 | 0.8714 to 0.8717 | 51.0 ms | 25,000 | 485,116 |
| zdt1-30-matched | genoxide / spea2 | 0.8702 | 0.8694 to 0.8706 | 193.8 ms | 25,000 | 127,668 |
| zdt1-30-matched | genoxide_python / moead | 0.8684 | 0.8662 to 0.8697 | 43.9 ms | 25,044 | 563,528 |
| zdt1-30-matched | genoxide_python / nsga2 | 0.8696 | 0.8690 to 0.8702 | 28.0 ms | 25,000 | 888,431 |
| zdt1-30-matched | genoxide_python / sms_emoa | 0.8715 | 0.8714 to 0.8717 | 59.4 ms | 25,000 | 419,517 |
| zdt1-30-matched | genoxide_python / spea2 | 0.8702 | 0.8694 to 0.8706 | 191.8 ms | 25,000 | 130,589 |
| zdt1-30-matched | jenetics / moea | 0.0000 | 0.0000 to 0.0000 | 146.8 ms | 25,012 | 169,806 |
| zdt1-30-matched | jenetics / nsga2 | 0.8684 | 0.8680 to 0.8691 | 196.6 ms | 25,062 | 125,347 |
| zdt1-30-matched | jmetal / moead | 0.8706 | 0.8702 to 0.8709 | 163.6 ms | 25,000 | 151,038 |
| zdt1-30-matched | jmetal / nsga2 | 0.8693 | 0.8689 to 0.8697 | 206.8 ms | 25,000 | 120,618 |
| zdt1-30-matched | jmetal / nsga3 | 0.8705 | 0.8699 to 0.8707 | 210.4 ms | 25,000 | 118,163 |
| zdt1-30-matched | jmetal / smpso | 0.8718 | 0.8716 to 0.8719 | 177.8 ms | 25,000 | 144,137 |
| zdt1-30-matched | jmetal / sms_emoa | 0.8719 | 0.8718 to 0.8719 | 163.8 ms | 25,000 | 150,852 |
| zdt1-30-matched | jmetal / spea2 | 0.8696 | 0.8689 to 0.8701 | 2.13 s | 25,000 | 11,795 |
| zdt1-30-matched | metaheuristics_jl / moead | 0.6430 | 0.5990 to 0.6829 | 51.3 ms | 25,000 | 461,627 |
| zdt1-30-matched | metaheuristics_jl / nsga2 | 0.8663 | 0.8648 to 0.8672 | 86.6 ms | 25,100 | 285,774 |
| zdt1-30-matched | metaheuristics_jl / nsga3 | 0.8676 | 0.8664 to 0.8681 | 226.9 ms | 25,000 | 109,410 |
| zdt1-30-matched | metaheuristics_jl / sms_emoa | 0.8719 | 0.8719 to 0.8720 | 1.05 s | 25,000 | 23,738 |
| zdt1-30-matched | metaheuristics_jl / spea2 | 0.8659 | 0.8645 to 0.8666 | 671.0 ms | 25,100 | 37,599 |
| zdt1-30-matched | moors / age_moea | 0.8600 | 0.8582 to 0.8619 | 543.7 ms | 25,100 | 46,337 |
| zdt1-30-matched | moors / ibea | 0.8689 | 0.7681 to 0.8714 | 85.1 ms | 25,100 | 292,441 |
| zdt1-30-matched | moors / nsga2 | 0.8670 | 0.8656 to 0.8686 | 22.9 ms | 25,100 | 1,045,916 |
| zdt1-30-matched | moors / nsga3 | 0.8556 | 0.7942 to 0.8639 | 30.2 ms | 25,100 | 801,318 |
| zdt1-30-matched | moors / revea | 0.8582 | 0.8137 to 0.8668 | 282.4 ms | 24,136 | 85,318 |
| zdt1-30-matched | moors / spea2 | 0.7184 | 0.6374 to 0.7921 | 97.8 ms | 25,100 | 259,221 |
| zdt1-30-matched | nevergrad / de | 0.7611 | 0.7211 to 0.7773 | 16.24 s | 25,000 | 1,545 |
| zdt1-30-matched | openga / nsga3 | 0.8631 | 0.8545 to 0.8651 | 207.4 ms | 25,000 | 119,244 |
| zdt1-30-matched | pygad / nsga2 | 0.8678 | 0.8673 to 0.8685 | 49.35 s | 25,062 | 507 |
| zdt1-30-matched | pygmo / moead | 0.8444 | 0.8311 to 0.8602 | 46.3 ms | 25,000 | 536,504 |
| zdt1-30-matched | pygmo / nsga2 | 0.8696 | 0.8691 to 0.8701 | 80.4 ms | 25,000 | 307,828 |
| zdt1-30-matched | pygmo / nspso | 0.8618 | 0.8566 to 0.8633 | 103.2 ms | 25,000 | 240,803 |
| zdt1-30-matched | pymoo / moead | 0.8699 | 0.8672 to 0.8707 | 5.74 s | 25,000 | 4,359 |
| zdt1-30-matched | pymoo / nsga2 | 0.8697 | 0.8693 to 0.8699 | 549.0 ms | 25,000 | 45,527 |
| zdt1-30-matched | pymoo / sms_emoa | 0.8716 | 0.8715 to 0.8718 | 692.4 ms | 25,000 | 36,048 |
| zdt1-30-matched | pymoo / spea2 | 0.8706 | 0.8703 to 0.8708 | 2.08 s | 25,000 | 12,005 |
| zdt1-30-matched | radiate / nsga2 | 0.8059 | 0.7606 to 0.8382 | 85.0 ms | 25,032 | 286,911 |
| zdt2-30-matched | deap / nsga2 | 0.5361 | 0.5355 to 0.5365 | 1.48 s | 25,000 | 16,831 |
| zdt2-30-matched | evolutionary_jl / nsga2 | 0.0000 | 0.0000 to 0.0000 | 294.9 ms | 25,000 | 84,076 |
| zdt2-30-matched | genoxide / moead | 0.5352 | 0.5343 to 0.5359 | 41.2 ms | 25,078 | 601,202 |
| zdt2-30-matched | genoxide / nsga2 | 0.5361 | 0.5355 to 0.5366 | 25.2 ms | 25,000 | 969,736 |
| zdt2-30-matched | genoxide / sms_emoa | 0.5380 | 0.5378 to 0.5382 | 45.8 ms | 25,000 | 540,722 |
| zdt2-30-matched | genoxide / spea2 | 0.5366 | 0.5360 to 0.5371 | 186.6 ms | 25,000 | 133,316 |
| zdt2-30-matched | genoxide_python / moead | 0.5351 | 0.5307 to 0.5366 | 43.7 ms | 25,070 | 553,167 |
| zdt2-30-matched | genoxide_python / nsga2 | 0.5361 | 0.5355 to 0.5366 | 27.8 ms | 25,000 | 896,257 |
| zdt2-30-matched | genoxide_python / sms_emoa | 0.5380 | 0.5378 to 0.5382 | 52.9 ms | 25,000 | 480,354 |
| zdt2-30-matched | genoxide_python / spea2 | 0.5366 | 0.5360 to 0.5371 | 185.6 ms | 25,000 | 133,976 |
| zdt2-30-matched | jenetics / moea | 0.0000 | 0.0000 to 0.0000 | 147.5 ms | 25,014 | 169,935 |
| zdt2-30-matched | jenetics / nsga2 | 0.5348 | 0.5271 to 0.5356 | 205.4 ms | 25,038 | 121,097 |
| zdt2-30-matched | jmetal / moead | 0.5374 | 0.5371 to 0.5379 | 165.0 ms | 25,000 | 152,665 |
| zdt2-30-matched | jmetal / nsga2 | 0.5359 | 0.5354 to 0.5364 | 211.4 ms | 25,000 | 117,030 |
| zdt2-30-matched | jmetal / nsga3 | 0.5368 | 0.5364 to 0.5375 | 217.6 ms | 25,000 | 114,249 |
| zdt2-30-matched | jmetal / smpso | 0.5386 | 0.5385 to 0.5386 | 205.0 ms | 25,000 | 121,863 |
| zdt2-30-matched | jmetal / sms_emoa | 0.5385 | 0.5382 to 0.5386 | 161.0 ms | 25,000 | 155,197 |
| zdt2-30-matched | jmetal / spea2 | 0.5361 | 0.5352 to 0.5367 | 2.46 s | 25,000 | 10,175 |
| zdt2-30-matched | metaheuristics_jl / moead | 0.2466 | 0.2100 to 0.2737 | 48.6 ms | 25,000 | 502,492 |
| zdt2-30-matched | metaheuristics_jl / nsga2 | 0.5334 | 0.5326 to 0.5348 | 94.4 ms | 25,100 | 268,802 |
| zdt2-30-matched | metaheuristics_jl / nsga3 | 0.5342 | 0.5330 to 0.5363 | 232.1 ms | 25,000 | 107,311 |
| zdt2-30-matched | metaheuristics_jl / sms_emoa | 0.5386 | 0.5386 to 0.5387 | 1.12 s | 25,000 | 22,695 |
| zdt2-30-matched | metaheuristics_jl / spea2 | 0.5304 | 0.5294 to 0.5320 | 561.8 ms | 25,100 | 44,798 |
| zdt2-30-matched | moors / age_moea | 0.0000 | 0.0000 to 0.5365 | 1.4 ms | 1,100 | 76,488 |
| zdt2-30-matched | moors / ibea | 0.0000 | 0.0000 to 0.0000 | 84.5 ms | 25,100 | 303,825 |
| zdt2-30-matched | moors / nsga2 | 0.5348 | 0.5309 to 0.5355 | 25.1 ms | 25,100 | 978,702 |
| zdt2-30-matched | moors / nsga3 | 0.4920 | 0.3672 to 0.5340 | 32.5 ms | 25,100 | 731,633 |
| zdt2-30-matched | moors / revea | 0.5280 | 0.5260 to 0.5324 | 275.3 ms | 23,159 | 83,984 |
| zdt2-30-matched | moors / spea2 | 0.2672 | 0.2219 to 0.3358 | 98.7 ms | 25,100 | 259,324 |
| zdt2-30-matched | nevergrad / de | 0.1100 | 0.1100 to 0.1100 | 4.17 s | 25,000 | 6,016 |
| zdt2-30-matched | openga / nsga3 | 0.4589 | 0.3888 to 0.5316 | 216.5 ms | 25,000 | 115,431 |
| zdt2-30-matched | pygad / nsga2 | 0.5337 | 0.5329 to 0.5341 | 47.97 s | 25,034 | 523 |
| zdt2-30-matched | pygmo / moead | 0.4788 | 0.4131 to 0.5001 | 43.7 ms | 25,000 | 562,832 |
| zdt2-30-matched | pygmo / nsga2 | 0.5363 | 0.5361 to 0.5365 | 84.8 ms | 25,000 | 295,968 |
| zdt2-30-matched | pygmo / nspso | 0.5296 | 0.1100 to 0.5317 | 106.2 ms | 25,000 | 236,158 |
| zdt2-30-matched | pymoo / moead | 0.5375 | 0.5356 to 0.5382 | 5.75 s | 25,000 | 4,353 |
| zdt2-30-matched | pymoo / nsga2 | 0.5364 | 0.5358 to 0.5367 | 555.2 ms | 25,000 | 45,019 |
| zdt2-30-matched | pymoo / sms_emoa | 0.5381 | 0.5379 to 0.5383 | 669.5 ms | 25,000 | 37,299 |
| zdt2-30-matched | pymoo / spea2 | 0.5370 | 0.5368 to 0.5375 | 1.74 s | 25,000 | 14,425 |
| zdt2-30-matched | radiate / nsga2 | 0.4383 | 0.3803 to 0.4528 | 87.5 ms | 25,032 | 285,106 |
| zdt3-30-matched | deap / nsga2 | 1.3273 | 1.2446 to 1.3277 | 1.45 s | 25,000 | 17,236 |
| zdt3-30-matched | evolutionary_jl / nsga2 | 0.0000 | 0.0000 to 0.0000 | 224.2 ms | 25,000 | 112,376 |
| zdt3-30-matched | genoxide / moead | 1.3216 | 1.3196 to 1.3226 | 41.1 ms | 25,034 | 595,864 |
| zdt3-30-matched | genoxide / nsga2 | 1.3275 | 1.3271 to 1.3278 | 26.3 ms | 25,000 | 914,973 |
| zdt3-30-matched | genoxide / sms_emoa | 1.3288 | 1.3286 to 1.3289 | 48.8 ms | 25,000 | 505,086 |
| zdt3-30-matched | genoxide / spea2 | 1.3273 | 1.2442 to 1.3277 | 191.0 ms | 25,000 | 131,719 |
| zdt3-30-matched | genoxide_python / moead | 1.3222 | 1.3208 to 1.3236 | 43.7 ms | 25,035 | 573,209 |
| zdt3-30-matched | genoxide_python / nsga2 | 1.3275 | 1.3271 to 1.3278 | 28.4 ms | 25,000 | 853,496 |
| zdt3-30-matched | genoxide_python / sms_emoa | 1.3288 | 1.3286 to 1.3289 | 57.3 ms | 25,000 | 435,680 |
| zdt3-30-matched | genoxide_python / spea2 | 1.3273 | 1.2442 to 1.3277 | 189.1 ms | 25,000 | 131,706 |
| zdt3-30-matched | jenetics / moea | 0.0367 | 0.0000 to 0.0669 | 142.3 ms | 25,013 | 174,495 |
| zdt3-30-matched | jenetics / nsga2 | 1.3257 | 1.3226 to 1.3277 | 203.8 ms | 25,054 | 121,888 |
| zdt3-30-matched | jmetal / moead | 1.3253 | 1.3251 to 1.3258 | 174.8 ms | 25,000 | 142,656 |
| zdt3-30-matched | jmetal / nsga2 | 1.3274 | 1.3268 to 1.3277 | 219.9 ms | 25,000 | 113,398 |
| zdt3-30-matched | jmetal / nsga3 | 1.3256 | 1.3248 to 1.3266 | 224.9 ms | 25,000 | 110,801 |
| zdt3-30-matched | jmetal / smpso | 1.3280 | 1.3258 to 1.3288 | 61.4 ms | 25,000 | 415,646 |
| zdt3-30-matched | jmetal / sms_emoa | 1.3292 | 1.3290 to 1.3293 | 173.5 ms | 25,000 | 143,286 |
| zdt3-30-matched | jmetal / spea2 | 1.3256 | 1.2419 to 1.3264 | 2.50 s | 25,000 | 9,998 |
| zdt3-30-matched | metaheuristics_jl / moead | 0.8186 | 0.7495 to 0.9353 | 53.9 ms | 25,000 | 447,480 |
| zdt3-30-matched | metaheuristics_jl / nsga2 | 1.3229 | 1.3226 to 1.3253 | 89.8 ms | 25,100 | 274,104 |
| zdt3-30-matched | metaheuristics_jl / nsga3 | 1.3222 | 1.3197 to 1.3234 | 237.6 ms | 25,000 | 105,656 |
| zdt3-30-matched | metaheuristics_jl / sms_emoa | 1.3291 | 1.3289 to 1.3293 | 1.03 s | 25,000 | 24,260 |
| zdt3-30-matched | metaheuristics_jl / spea2 | 1.3155 | 1.3112 to 1.3175 | 623.6 ms | 25,100 | 40,282 |
| zdt3-30-matched | moors / age_moea | 1.0869 | 0.8426 to 1.3138 | 501.2 ms | 25,100 | 50,124 |
| zdt3-30-matched | moors / ibea | 0.4211 | 0.0000 to 1.2590 | 80.8 ms | 25,100 | 307,841 |
| zdt3-30-matched | moors / nsga2 | 1.0868 | 0.8406 to 1.3187 | 23.4 ms | 25,100 | 1,026,711 |
| zdt3-30-matched | moors / nsga3 | 1.0794 | 0.8414 to 1.3155 | 30.6 ms | 25,100 | 798,890 |
| zdt3-30-matched | moors / revea | 1.1874 | 0.8409 to 1.2367 | 290.9 ms | 24,477 | 83,897 |
| zdt3-30-matched | moors / spea2 | 0.8789 | 0.8331 to 1.1787 | 92.7 ms | 25,100 | 271,666 |
| zdt3-30-matched | nevergrad / de | 1.0450 | 0.9768 to 1.0900 | 11.89 s | 25,000 | 2,125 |
| zdt3-30-matched | openga / nsga3 | 1.3195 | 1.3138 to 1.3239 | 208.2 ms | 25,000 | 118,799 |
| zdt3-30-matched | pygad / nsga2 | 1.3249 | 1.3237 to 1.3252 | 48.49 s | 25,060 | 517 |
| zdt3-30-matched | pygmo / moead | 1.1719 | 1.1287 to 1.2532 | 48.5 ms | 25,000 | 509,384 |
| zdt3-30-matched | pygmo / nsga2 | 1.3274 | 1.3272 to 1.3279 | 86.0 ms | 25,000 | 292,167 |
| zdt3-30-matched | pygmo / nspso | 1.2945 | 1.2876 to 1.3091 | 121.2 ms | 25,000 | 204,929 |
| zdt3-30-matched | pymoo / moead | 1.3239 | 1.3220 to 1.3251 | 5.79 s | 25,000 | 4,314 |
| zdt3-30-matched | pymoo / nsga2 | 1.3276 | 1.3273 to 1.3280 | 552.1 ms | 25,000 | 45,124 |
| zdt3-30-matched | pymoo / sms_emoa | 1.3288 | 1.2455 to 1.3289 | 697.0 ms | 25,000 | 35,779 |
| zdt3-30-matched | pymoo / spea2 | 1.3277 | 1.3269 to 1.3279 | 1.86 s | 25,000 | 13,473 |
| zdt3-30-matched | radiate / nsga2 | 1.2068 | 1.1087 to 1.2792 | 84.8 ms | 25,028 | 292,936 |

## Instructions per evaluation

OneMax 1000 (matched), counted by Callgrind: the framework and the fitness function together, without the startup and imports.

| Library / solver | Instructions per evaluation |
|---|---|
| genoxide / ga | 3,400 |
| genetic_algorithm / evolve | 8,869 |
| genoxide_python / ga | 29,019 |
| moors / ga | 33,797 |
| openga / ga | 36,372 |
| radiate / ga | 65,947 |
| pygmo / ga | 312,958 |
| pymoo / ga | 1,016,245 |
| pygad / ga | 2,133,695 |
| deap / ga | 4,416,804 |
