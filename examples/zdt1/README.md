---
title: ZDT1
category: multi-objective
summary: Minimize two conflicting objectives over 30 variables, with a convex Pareto front.
reference: "Zitzler, E., Deb, K. and Thiele, L. (2000). Comparison of multiobjective evolutionary algorithms: empirical results. Evolutionary Computation 8(2): 173-195."
reference_url: https://doi.org/10.1162/106365600568202
optimum: "hypervolume 0.8767 (reference point (1.1, 1.1))"
languages: [rust, python]
order: 80
---

# ZDT1

ZDT1 is a standard test problem with two objectives to minimize over 30 variables in [0, 1]; its
Pareto front is f₂ = 1 − √f₁ for f₁ in [0, 1]. The example runs NSGA-II with a population of 100,
simulated binary crossover and polynomial mutation for 25,000 evaluations. It prints the size of
the final non-dominated front and its hypervolume with the reference point (1.1, 1.1), which is
0.8767 for the whole Pareto front. The Rust version uses genoxide's built-in ZDT1 and hypervolume;
the Python version evaluates a whole generation per call with numpy.
