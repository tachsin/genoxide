---
title: Travelling salesman (berlin52)
category: permutation
summary: The shortest round trip through 52 locations in Berlin, a TSPLIB instance.
reference: "Reinelt, G. (1991). TSPLIB: a traveling salesman problem library. ORSA Journal on Computing 3(4): 376-384."
reference_url: https://doi.org/10.1287/ijoc.3.4.376
optimum: "7542 (tour length)"
languages: [rust, python]
order: 40
---

# Travelling salesman (berlin52)

berlin52 is a TSPLIB instance of the travelling salesman problem: 52 locations in Berlin, with
distances in TSPLIB's EUC_2D metric (the Euclidean distance rounded to the nearest integer), and
an optimal tour of length 7542. A permutation genome is the order of the visits. The example runs
local search with inversion neighbors, a random 2-opt move that reverses a segment of the tour,
and simulated annealing from a temperature of 100 cooled by 0.99996 per step, for 200,000
evaluations. It prints the length of the best tour and the tour from location 1.
