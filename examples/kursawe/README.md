---
title: Kursawe's disconnected front
category: multi-objective
summary: Minimize two objectives whose Pareto front is in three separate pieces, with SPEA2 and NSGA-II.
reference: "Kursawe, F. (1991). A variant of evolution strategies for vector optimization. Parallel Problem Solving from Nature, LNCS 496: 193-197."
reference_url: https://doi.org/10.1007/BFb0029752
optimum: "not known in closed form"
languages: [rust, python]
order: 85
---

# Kursawe's disconnected front

Kursawe's problem minimizes f₁ = Σ −10 exp(−0.2 √(xᵢ² + xᵢ₊₁²)) and
f₂ = Σ (|xᵢ|^0.8 + 5 sin(xᵢ³)) over [−5, 5]³. Its Pareto front is in three separate pieces, one of
them a single point near (−20, 0), and isn't known in closed form. The example runs SPEA2 and
NSGA-II on genoxide's `Kursawe` with the same settings: a population of 100, simulated binary
crossover with η = 15 and polynomial mutation with η = 20, for 250 generations. For each it prints
the size of the final front, how many pieces it covers (a new piece starts where two neighbors are
more than 0.5 apart), and its hypervolume with the reference point (−14, 1). In Python, `run`
evaluates the problem in Rust, so both versions print the same.
