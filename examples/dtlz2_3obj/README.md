---
title: DTLZ2 with 3 objectives
category: multi-objective
summary: Minimize three conflicting objectives whose Pareto front is an eighth of the unit sphere, with NSGA-III.
reference: "Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2002). Scalable multi-objective optimization test problems. Proceedings of the 2002 Congress on Evolutionary Computation, pp. 825-830."
reference_url: https://doi.org/10.1109/CEC.2002.1007032
optimum: "hypervolume 0.8074 (reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 90
---

# DTLZ2 with 3 objectives

DTLZ2 scales to any number of objectives; with 3 objectives and 12 variables in [0, 1], its Pareto
front is the part of the unit sphere with non-negative coordinates. The example runs NSGA-III with
the settings of Deb and Jain (2014, IEEE Transactions on Evolutionary Computation 18(4): 577-601):
91 reference directions from Das and Dennis's method with 12 divisions, a population of 92,
simulated binary crossover with η = 30 and polynomial mutation with η = 20, for 250 generations.
It prints the size of the final front and its hypervolume with the reference point (1.1, 1.1, 1.1),
which is 1.1³ − π/6 ≈ 0.8074 for the whole Pareto front.
