# Results 20260924-025338

Seeds per scenario: 10, wall time cap per run: 60.0 s, single-threaded.
Linux, Intel(R) Core(TM) Ultra 7 265K

- genoxide 0.1.0+2d636d3 (the 0.2 development version)
- genetic_algorithm 0.27.3
- deap 1.4.4
- pygad 3.7.0
- pymoo 0.6.2

| Scenario | Library / solver | Success | Median time to target | Median evaluations | Median best | Evaluations/s | Throughput vs DEAP GA |
|---|---|---|---|---|---|---|---|
| nqueens-32-idiomatic | deap / ga | 100% (10) | 539.9 ms | 3.949e+04 | 0 | 72,114 | - |
| nqueens-32-idiomatic | genetic_algorithm / evolve | 100% (10) | 1.0 ms | 4200 | 0 | 3,657,927 | 51× |
| nqueens-32-idiomatic | genetic_algorithm / hill_climb | 100% (10) | 374 µs | 1494 | 0 | 4,309,964 | 60× |
| nqueens-32-idiomatic | genoxide / ga | 100% (10) | 601 µs | 2,870 | 0 | 4,844,437 | 67× |
| nqueens-32-idiomatic | genoxide / local_search | 100% (10) | 166 µs | 1136 | 0 | 6,770,784 | 94× |
| nqueens-32-idiomatic | pygad / ga | 100% (10) | 3.63 s | 8.749e+04 | 0 | 23,965 | 0.3× |
| nqueens-32-idiomatic | pymoo / ga | 20% (10) | 791.6 ms | 500,000 | 1 | 24,882 | 0.3× |
| nqueens-64-idiomatic | deap / ga | 100% (10) | 1.43 s | 63,456 | 0 | 45,021 | - |
| nqueens-64-idiomatic | genetic_algorithm / evolve | 100% (10) | 6.5 ms | 1.487e+04 | 0 | 2,609,949 | 58× |
| nqueens-64-idiomatic | genetic_algorithm / hill_climb | 100% (10) | 966 µs | 3728 | 0 | 3,329,606 | 74× |
| nqueens-64-idiomatic | genoxide / ga | 100% (10) | 1.7 ms | 6048 | 0 | 3,698,796 | 82× |
| nqueens-64-idiomatic | genoxide / local_search | 100% (10) | 506 µs | 2,455 | 0 | 4,921,602 | 109× |
| nqueens-64-idiomatic | pygad / ga | 100% (10) | 33.56 s | 5.169e+05 | 0 | 15,255 | 0.3× |
| nqueens-64-idiomatic | pymoo / ga | 20% (10) | 8.69 s | 1,000,000 | 2.5 | 21,819 | 0.5× |
| onemax-100-idiomatic | deap / ga | 100% (10) | 133.0 ms | 6,420 | 100 | 49,255 | - |
| onemax-100-idiomatic | genetic_algorithm / evolve | 100% (10) | 461 µs | 1952 | 100 | 3,356,164 | 68× |
| onemax-100-idiomatic | genoxide / ga | 100% (10) | 672 µs | 5,539 | 100 | 8,185,451 | 166× |
| onemax-100-idiomatic | pygad / ga | 0% (10) | - | 2.001e+05 | 94.5 | 28,469 | 0.6× |
| onemax-100-idiomatic | pymoo / ga | 100% (10) | 176.6 ms | 5,900 | 100 | 33,837 | 0.7× |
| onemax-100-matched | deap / ga | 100% (10) | 134.3 ms | 6,773 | 100 | 49,545 | - |
| onemax-100-matched | genetic_algorithm / evolve | 100% (10) | 2.1 ms | 9702 | 100 | 4,573,202 | 92× |
| onemax-100-matched | genoxide / ga | 100% (10) | 962 µs | 5184 | 100 | 5,372,494 | 108× |
| onemax-100-matched | pygad / ga | 100% (10) | 134.6 ms | 6,900 | 100 | 52,374 | 1.1× |
| onemax-100-matched | pymoo / ga | 100% (10) | 174.6 ms | 10,350 | 100 | 58,765 | 1.2× |
| onemax-1000-matched | deap / ga | 100% (10) | 15.45 s | 105,287 | 1,000 | 6,725 | - |
| onemax-1000-matched | genetic_algorithm / evolve | 100% (10) | 126.9 ms | 1.105e+05 | 1,000 | 871,003 | 130× |
| onemax-1000-matched | genoxide / ga | 100% (10) | 13.9 ms | 5.957e+04 | 1,000 | 4,278,132 | 636× |
| onemax-1000-matched | pygad / ga | 100% (10) | 11.68 s | 147,150 | 1,000 | 12,745 | 1.9× |
| onemax-1000-matched | pymoo / ga | 100% (10) | 7.92 s | 172,200 | 1,000 | 21,503 | 3.2× |
| rastrigin-10-idiomatic | deap / cma_es | 50% (10) | 53.9 ms | 105,700 | 0.5022 | 299,179 | - |
| rastrigin-10-idiomatic | deap / ga | 0% (10) | - | 500,000 | 0.05702 | 137,546 | - |
| rastrigin-10-idiomatic | genetic_algorithm / evolve | 100% (10) | 10.6 ms | 48,425 | 0.008311 | 4,582,933 | 33× |
| rastrigin-10-idiomatic | genoxide / ga | 100% (10) | 51.9 ms | 192,964 | 0.008825 | 3,714,807 | 27× |
| rastrigin-10-idiomatic | pygad / ga | 0% (10) | - | 500,050 | 0.1681 | 113,789 | 0.8× |
| rastrigin-10-idiomatic | pymoo / cma_es | 100% (10) | 1.28 s | 48,394 | 0.008202 | 38,862 | 0.3× |
| rastrigin-10-idiomatic | pymoo / de | 100% (10) | 278.0 ms | 14,950 | 0.008325 | 53,468 | 0.4× |
| rastrigin-10-idiomatic | pymoo / ga | 100% (10) | 242.2 ms | 11,700 | 0.009317 | 47,856 | 0.3× |
| rastrigin-30-idiomatic | deap / cma_es | 60% (10) | 986.4 ms | 146,100 | 0.008862 | 144,849 | - |
| rastrigin-30-idiomatic | deap / ga | 0% (10) | - | 2,000,000 | 0.3929 | 75,780 | - |
| rastrigin-30-idiomatic | genetic_algorithm / evolve | 0% (10) | - | 193,446 | 6.965 | 3,026,905 | 40× |
| rastrigin-30-idiomatic | genoxide / ga | 90% (10) | 515.7 ms | 1,540,023 | 0.009798 | 2,962,978 | 39× |
| rastrigin-30-idiomatic | pygad / ga | 0% (10) | - | 2,000,098 | 51.24 | 61,053 | 0.8× |
| rastrigin-30-idiomatic | pymoo / cma_es | 100% (10) | 10.90 s | 421,479 | 0.008558 | 38,381 | 0.5× |
| rastrigin-30-idiomatic | pymoo / de | 100% (10) | 2.31 s | 104,150 | 0.008115 | 45,142 | 0.6× |
| rastrigin-30-idiomatic | pymoo / ga | 100% (10) | 1.65 s | 65,100 | 0.009658 | 39,637 | 0.5× |
