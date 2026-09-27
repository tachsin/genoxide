---
title: Gear train design
category: integer
summary: The numbers of teeth of four gears whose ratio is closest to 1/6.931, an integer problem.
reference: "Sandgren, E. (1990). Nonlinear integer and discrete programming in mechanical design optimization. Journal of Mechanical Design 112(2): 223-229."
reference_url: https://doi.org/10.1115/1.2912596
optimum: "2.700857e-12 (squared error of the ratio)"
languages: [rust, python]
order: 74
---

# Gear train design

The gear train problem chooses the numbers of teeth of a compound gear train of four gears, from
12 to 60 each, so that its ratio T_d T_b / (T_a T_f) is closest to 1/6.931; the score is the
squared error. The example uses genoxide's `GearTrain`, on integer genes, and runs a genetic
algorithm with a population of 100, binary tournaments, uniform crossover and a mutation that
redraws each gene with probability 0.25. It stops at the minimum, 2.700857e-12, at 16 · 19 / (43 ·
49), known by evaluating all 49⁴ ≈ 5.8 million designs, and prints the error and the teeth. In
Python, `run` evaluates the problem in Rust, so both versions print the same.
