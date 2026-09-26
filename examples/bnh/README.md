---
title: BNH, a constrained two-objective problem
category: multi-objective
summary: Minimize two objectives subject to two constraints with NSGA-II, and measure the front against the optimal one.
reference: "Binh, T. T. and Korn, U. (1997). MOBES: a multiobjective evolution strategy for constrained optimization problems. Proceedings of the Third International Conference on Genetic Algorithms (Mendel 97), Brno: 176-182."
reference_url: ""
optimum: "the front f = (8t², 2(t − 5)²) for t in [0, 5]; hypervolume 9883.33 (reference point (210, 55))"
languages: [rust, python]
order: 82
---

# BNH, a constrained two-objective problem

Binh and Korn's problem minimizes f₁ = 4x₁² + 4x₂² and f₂ = (x₁ − 5)² + (x₂ − 5)² over
[−15, 30]², subject to (x₁ − 5)² + x₂² ≤ 25 and (x₁ − 8)² + (x₂ + 3)² ≥ 7.7. Its optimal
solutions are x₁ = x₂ = t for t in [0, 5], and its front f = (8t², 2(t − 5)²) runs from (0, 50) to
(200, 0). The example runs NSGA-II on genoxide's `Bnh`, whose fitness is the two objectives and the
constraint violation, so that feasible solutions win over infeasible ones (Deb's rules): a
population of 100, simulated binary crossover and polynomial mutation, both with η = 20, for 250
generations. It prints how many solutions of the final front are feasible, their IGD+ to 500
points of the optimal front, and the front's hypervolume with the reference point (210, 55),
which is 9883.33 for the whole optimal front. In Python, `run` evaluates the problem in Rust, so
both versions print the same.
