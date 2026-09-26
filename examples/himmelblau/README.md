---
title: Himmelblau's function
category: continuous
summary: Find the four global minima of a two-dimensional function by restarting a local search from random points.
reference: "Himmelblau, D. M. (1972). Applied Nonlinear Programming. McGraw-Hill."
reference_url: ""
optimum: "0 (at four points)"
languages: [rust, python]
order: 64
---

# Himmelblau's function

Himmelblau's function, (x₁² + x₂ − 11)² + (x₁ + x₂² − 7)² over [−5, 5]², is 0 at four points,
where x₁² + x₂ = 11 and x₁ + x₂² = 7: (3, 2) and three others, which genoxide computes to full
precision. A local search ends in the minimum whose basin it starts in, so restarting it from
random points finds them all. The example runs 20 hill climbers from random points, each for
1,000 steps of 10 neighbors made with Gaussian steps of 0.01, and takes the known minima from
genoxide's `problems::Himmelblau`. For each minimum it prints how many searches ended nearest to
it and the values they reached. In Python, `run` evaluates the problem in Rust, so both versions
print the same table.
