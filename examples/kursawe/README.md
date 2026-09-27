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

## The problem

Kursawe (1991) posed a problem with two objectives, both minimized, in n variables. With n = 3 and
each xᵢ in [−5, 5]:

```text
f₁ = Σ −10 exp(−0.2 √(xᵢ² + xᵢ₊₁²))    over i = 1, 2
f₂ = Σ (|xᵢ|^0.8 + 5 sin(xᵢ³))          over i = 1, 2, 3
```

This is the form that Deb, Pratap, Agarwal and Meyarivan (2002, IEEE Transactions on Evolutionary
Computation 6(2): 182-197) restate. Van Veldhuizen (1999, PhD thesis, Air Force Institute of
Technology) notes that the original is misprinted, and prints sin(xᵢ)³ instead.

At x = 0, each exponential is 1, so f₁ = −20, its minimum, and f₂ = 0. That point, (−20, 0), is one
end of the Pareto front.

## What makes it hard

The Pareto front is disconnected: for 3 variables it's the point (−20, 0) and three separate
curves, four pieces in all, and it isn't known in closed form. Deb, Pratap, Agarwal and Meyarivan
(2002) and Van Veldhuizen (1999, PhD thesis) describe three regions, and plot the point apart from
them. Between the pieces, no solution is optimal. An algorithm has to keep separate groups
of solutions on each piece, or it loses a piece.

f₂ is multimodal. The sine of xᵢ³ oscillates faster as |xᵢ| grows: near |xᵢ| = 5 it goes through a
full period for every 0.08 in xᵢ. The term |xᵢ|^0.8 has a kink at 0.

## Representation

A `Real` genome of 3 genes in [−5, 5]: the vector x. The problem is genoxide's `Kursawe`, whose
fitness is the pair (f₁, f₂).

## Algorithm

Two algorithms, with the same settings: a population of 100, simulated binary crossover with η = 15
at genoxide's default rate of 0.9, polynomial mutation with η = 20 at a rate of 1/3 per gene, and
250 generations.

- SPEA2 (Zitzler, Laumanns and Thiele, 2001, TIK-Report 103, ETH Zurich) keeps an archive of the
  best solutions. When the non-dominated solutions don't fit in it, it removes, one at a time, the
  solution nearest to another. That keeps the extremes, and spreads the front evenly.
- NSGA-II (Deb et al., 2002) sorts solutions into non-dominated fronts, and within a front prefers
  solutions with a larger crowding distance, a measure of the gap between their neighbors.

Both spread the front, in different ways, which matters on a front in pieces.

## Output

One line per algorithm: how many solutions are on its final front, how many pieces they cover, and
the front's hypervolume. The example sorts the front by f₁, and starts a new piece wherever f₁
grows by more than 0.2 from one solution to the next: along a piece it grows by at most about 0.13,
and across the gaps between pieces by 0.25 to 0.92. The hypervolume is the area that the front dominates, up to the
reference point (−14, 1). Larger is better. In Python, `run` evaluates the problem in Rust, so both
versions print the same.

The project page plays this run back.

## Good results

The front's exact hypervolume isn't known, so there is no target value. A good front covers all
four pieces: the point (−20, 0), and the three curves with many solutions on each. Both algorithms
find the four pieces with 100 solutions.
SPEA2's hypervolume, about 37.10, is a little larger than NSGA-II's, about 37.02, on this seed.
