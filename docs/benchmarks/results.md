# Results 20260924-121708

Seeds per scenario: 10, wall time cap per run: 60.0 s, single-threaded.
Linux, Intel(R) Core(TM) Ultra 7 265K

- genoxide 0.3.0+8ce5b50 (the 0.4 development version)
- genetic_algorithm 0.27.3
- deap 1.4.4
- pygad 3.7.0
- pymoo 0.6.2

| Scenario | Library / solver | Success | Median time to target | Median evaluations | Median best | Evaluations/s | Throughput vs DEAP GA |
|---|---|---|---|---|---|---|---|
| nqueens-32-idiomatic | deap / ga | 100% (10) | 539.9 ms | 3.949e+04 | 0 | 72,114 | - |
| nqueens-32-idiomatic | genetic_algorithm / evolve | 100% (10) | 1.0 ms | 4200 | 0 | 3,657,927 | 51× |
| nqueens-32-idiomatic | genetic_algorithm / hill_climb | 100% (10) | 374 µs | 1494 | 0 | 4,309,964 | 60× |
| nqueens-32-idiomatic | genoxide / ga | 100% (10) | 620 µs | 2,870 | 0 | 4,677,873 | 65× |
| nqueens-32-idiomatic | genoxide / local_search | 100% (10) | 188 µs | 1136 | 0 | 6,061,669 | 84× |
| nqueens-32-idiomatic | pygad / ga | 100% (10) | 3.63 s | 8.749e+04 | 0 | 23,965 | 0.3× |
| nqueens-32-idiomatic | pymoo / ga | 20% (10) | 791.6 ms | 500,000 | 1 | 24,882 | 0.3× |
| nqueens-64-idiomatic | deap / ga | 100% (10) | 1.43 s | 63,456 | 0 | 45,021 | - |
| nqueens-64-idiomatic | genetic_algorithm / evolve | 100% (10) | 6.5 ms | 1.487e+04 | 0 | 2,609,949 | 58× |
| nqueens-64-idiomatic | genetic_algorithm / hill_climb | 100% (10) | 966 µs | 3728 | 0 | 3,329,606 | 74× |
| nqueens-64-idiomatic | genoxide / ga | 100% (10) | 1.7 ms | 6048 | 0 | 3,701,883 | 82× |
| nqueens-64-idiomatic | genoxide / local_search | 100% (10) | 571 µs | 2,455 | 0 | 4,370,649 | 97× |
| nqueens-64-idiomatic | pygad / ga | 100% (10) | 33.56 s | 5.169e+05 | 0 | 15,255 | 0.3× |
| nqueens-64-idiomatic | pymoo / ga | 20% (10) | 8.69 s | 1,000,000 | 2.5 | 21,819 | 0.5× |
| onemax-100-idiomatic | deap / ga | 100% (10) | 133.0 ms | 6,420 | 100 | 49,255 | - |
| onemax-100-idiomatic | genetic_algorithm / evolve | 100% (10) | 461 µs | 1952 | 100 | 3,356,164 | 68× |
| onemax-100-idiomatic | genoxide / ga | 100% (10) | 454 µs | 3,183 | 100 | 7,248,175 | 147× |
| onemax-100-idiomatic | pygad / ga | 0% (10) | - | 2.001e+05 | 94.5 | 28,469 | 0.6× |
| onemax-100-idiomatic | pymoo / ga | 100% (10) | 176.6 ms | 5,900 | 100 | 33,837 | 0.7× |
| onemax-100-matched | deap / ga | 100% (10) | 134.3 ms | 6,773 | 100 | 49,545 | - |
| onemax-100-matched | genetic_algorithm / evolve | 100% (10) | 2.1 ms | 9702 | 100 | 4,573,202 | 92× |
| onemax-100-matched | genoxide / ga | 100% (10) | 987 µs | 4938 | 100 | 4,091,342 | 83× |
| onemax-100-matched | pygad / ga | 100% (10) | 134.6 ms | 6,900 | 100 | 52,374 | 1.1× |
| onemax-100-matched | pymoo / ga | 100% (10) | 174.6 ms | 10,350 | 100 | 58,765 | 1.2× |
| onemax-1000-matched | deap / ga | 100% (10) | 15.45 s | 105,287 | 1,000 | 6,725 | - |
| onemax-1000-matched | genetic_algorithm / evolve | 100% (10) | 126.9 ms | 1.105e+05 | 1,000 | 871,003 | 130× |
| onemax-1000-matched | genoxide / ga | 100% (10) | 15.4 ms | 54,038 | 1,000 | 3,234,466 | 481× |
| onemax-1000-matched | pygad / ga | 100% (10) | 11.68 s | 147,150 | 1,000 | 12,745 | 1.9× |
| onemax-1000-matched | pymoo / ga | 100% (10) | 7.92 s | 172,200 | 1,000 | 21,503 | 3.2× |
| rastrigin-10-idiomatic | deap / cma_es | 50% (10) | 53.9 ms | 105,700 | 0.5022 | 299,179 | - |
| rastrigin-10-idiomatic | deap / ga | 0% (10) | - | 500,000 | 0.05702 | 137,546 | - |
| rastrigin-10-idiomatic | genetic_algorithm / evolve | 100% (10) | 10.6 ms | 48,425 | 0.008311 | 4,582,933 | 33× |
| rastrigin-10-idiomatic | genoxide / cma_es | 100% (10) | 32.2 ms | 61,315 | 0.008057 | 1,957,732 | 14× |
| rastrigin-10-idiomatic | genoxide / de | 100% (10) | 8.0 ms | 25,750 | 0.007908 | 3,234,081 | 24× |
| rastrigin-10-idiomatic | genoxide / ga | 100% (10) | 6.3 ms | 2.274e+04 | 0.008531 | 3,540,094 | 26× |
| rastrigin-10-idiomatic | pygad / ga | 0% (10) | - | 500,050 | 0.1681 | 113,789 | 0.8× |
| rastrigin-10-idiomatic | pymoo / cma_es | 100% (10) | 1.28 s | 48,394 | 0.008202 | 38,862 | 0.3× |
| rastrigin-10-idiomatic | pymoo / de | 100% (10) | 278.0 ms | 14,950 | 0.008325 | 53,468 | 0.4× |
| rastrigin-10-idiomatic | pymoo / ga | 100% (10) | 242.2 ms | 11,700 | 0.009317 | 47,856 | 0.3× |
| rastrigin-30-idiomatic | deap / cma_es | 60% (10) | 986.4 ms | 146,100 | 0.008862 | 144,849 | - |
| rastrigin-30-idiomatic | deap / ga | 0% (10) | - | 2,000,000 | 0.3929 | 75,780 | - |
| rastrigin-30-idiomatic | genetic_algorithm / evolve | 0% (10) | - | 193,446 | 6.965 | 3,026,905 | 40× |
| rastrigin-30-idiomatic | genoxide / cma_es | 100% (10) | 1.66 s | 513,597 | 0.008268 | 310,332 | 4.1× |
| rastrigin-30-idiomatic | genoxide / de | 100% (10) | 40.4 ms | 80,150 | 0.009041 | 1,961,054 | 26× |
| rastrigin-30-idiomatic | genoxide / ga | 100% (10) | 34.7 ms | 102,781 | 0.009649 | 2,945,086 | 39× |
| rastrigin-30-idiomatic | pygad / ga | 0% (10) | - | 2,000,098 | 51.24 | 61,053 | 0.8× |
| rastrigin-30-idiomatic | pymoo / cma_es | 100% (10) | 10.90 s | 421,479 | 0.008558 | 38,381 | 0.5× |
| rastrigin-30-idiomatic | pymoo / de | 100% (10) | 2.31 s | 104,150 | 0.008115 | 45,142 | 0.6× |
| rastrigin-30-idiomatic | pymoo / ga | 100% (10) | 1.65 s | 65,100 | 0.009658 | 39,637 | 0.5× |

## Multi-objective

| Scenario | Library / solver | Median hypervolume | Range | Median time | Median evaluations | Evaluations/s |
|---|---|---|---|---|---|---|
| dtlz2-3-matched | deap / nsga2 | 0.6891 | 0.6852 to 0.6986 | 1.29 s | 25,024 | 19,394 |
| dtlz2-3-matched | deap / nsga3 | 0.7435 | 0.7413 to 0.7441 | 628.0 ms | 25,024 | 39,482 |
| dtlz2-3-matched | genoxide / moead | 0.7441 | 0.7439 to 0.7442 | 27.4 ms | 25,014 | 902,536 |
| dtlz2-3-matched | genoxide / nsga2 | 0.6984 | 0.6888 to 0.7080 | 24.9 ms | 25,036 | 980,958 |
| dtlz2-3-matched | genoxide / nsga3 | 0.7442 | 0.7428 to 0.7444 | 40.2 ms | 25,024 | 611,414 |
| dtlz2-3-matched | genoxide / sms_emoa | 0.7542 | 0.7539 to 0.7544 | 343.2 ms | 25,042 | 73,234 |
| dtlz2-3-matched | genoxide / spea2 | 0.7300 | 0.7272 to 0.7316 | 274.8 ms | 25,031 | 90,804 |
| dtlz2-3-matched | pymoo / moead | 0.7443 | 0.7440 to 0.7445 | 6.35 s | 25,025 | 3,933 |
| dtlz2-3-matched | pymoo / nsga2 | 0.6970 | 0.6877 to 0.7102 | 571.4 ms | 25,024 | 43,619 |
| dtlz2-3-matched | pymoo / nsga3 | 0.7443 | 0.7439 to 0.7445 | 629.4 ms | 25,024 | 39,571 |
| dtlz2-3-matched | pymoo / sms_emoa | 0.7545 | 0.7542 to 0.7547 | 902.4 ms | 25,024 | 27,690 |
| dtlz2-3-matched | pymoo / spea2 | 0.7321 | 0.7275 to 0.7344 | 4.21 s | 25,024 | 5,940 |
| zdt1-30-matched | deap / nsga2 | 0.8694 | 0.8689 to 0.8698 | 1.51 s | 25,000 | 16,399 |
| zdt1-30-matched | genoxide / moead | 0.8684 | 0.8676 to 0.8689 | 41.9 ms | 25,045 | 596,560 |
| zdt1-30-matched | genoxide / nsga2 | 0.8694 | 0.8690 to 0.8697 | 24.9 ms | 25,020 | 1,002,286 |
| zdt1-30-matched | genoxide / sms_emoa | 0.8715 | 0.8715 to 0.8717 | 53.0 ms | 25,052 | 470,718 |
| zdt1-30-matched | genoxide / spea2 | 0.8703 | 0.8699 to 0.8707 | 202.5 ms | 25,060 | 122,722 |
| zdt1-30-matched | pymoo / moead | 0.8699 | 0.8672 to 0.8707 | 5.93 s | 25,000 | 4,199 |
| zdt1-30-matched | pymoo / nsga2 | 0.8697 | 0.8693 to 0.8699 | 570.0 ms | 25,000 | 43,859 |
| zdt1-30-matched | pymoo / sms_emoa | 0.8716 | 0.8715 to 0.8718 | 698.9 ms | 25,000 | 34,835 |
| zdt1-30-matched | pymoo / spea2 | 0.8706 | 0.8703 to 0.8708 | 2.14 s | 25,000 | 11,514 |
| zdt3-30-matched | deap / nsga2 | 1.3273 | 1.2446 to 1.3277 | 1.48 s | 25,000 | 16,749 |
| zdt3-30-matched | genoxide / moead | 1.3216 | 1.3196 to 1.3226 | 41.5 ms | 25,034 | 602,785 |
| zdt3-30-matched | genoxide / nsga2 | 1.3273 | 1.2447 to 1.3277 | 25.1 ms | 25,062 | 971,164 |
| zdt3-30-matched | genoxide / sms_emoa | 1.3288 | 1.3286 to 1.3289 | 51.4 ms | 25,070 | 482,117 |
| zdt3-30-matched | genoxide / spea2 | 1.3272 | 1.3269 to 1.3277 | 197.5 ms | 25,050 | 126,326 |
| zdt3-30-matched | pymoo / moead | 1.3239 | 1.3220 to 1.3251 | 5.87 s | 25,000 | 4,226 |
| zdt3-30-matched | pymoo / nsga2 | 1.3276 | 1.3273 to 1.3280 | 563.2 ms | 25,000 | 44,253 |
| zdt3-30-matched | pymoo / sms_emoa | 1.3288 | 1.2455 to 1.3289 | 706.9 ms | 25,000 | 35,222 |
| zdt3-30-matched | pymoo / spea2 | 1.3277 | 1.3269 to 1.3279 | 1.89 s | 25,000 | 13,177 |
