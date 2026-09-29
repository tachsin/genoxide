---
title: Kursawe's disconnected front
category: multi-objective
summary: Minimize two objectives whose Pareto front is in four separate pieces, a point and three curves, with SPEA2 and NSGA-II.
reference: "Kursawe, F. (1991). A variant of evolution strategies for vector optimization. Parallel Problem Solving from Nature, LNCS 496: 193-197."
reference_url: https://doi.org/10.1007/BFb0029752
optimum: "not known in closed form; a reference front from much longer runs has hypervolume 37.3489 (reference point (−14, 1))"
languages: [rust, python]
order: 119
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
the front's hypervolume. Then the hypervolume of a reference front. The example sorts the front by
f₁, and starts a new piece wherever f₁ grows by more than 0.2 from one solution to the next. Along
a piece, it grows by at most about 0.13 between neighbors on these fronts; the gaps between the
pieces of the true front are 0.25 to 0.92 wide. The hypervolume is the area that the front
dominates, up to the reference point (−14, 1). Larger is better. In Python, `run` evaluates the
problem in Rust, so both versions print the same.

The true front isn't known, so I built a reference front from much longer runs: 16 runs of NSGA-II
with the same crossover and mutation, 500 solutions for 2,000 generations each, keeping every
solution evaluated in their second halves that no other one dominates. Its 309,166 points have a
hypervolume of 37.3489. A reference ten times smaller, from 8 runs of 200 solutions for 1,000
generations, has 30,732 points and 37.3463, only 0.0026 less.

[The project page](https://tachsin.gr/projects/genoxide/examples/kursawe) plays this run back.

## Good results

The front's exact hypervolume isn't known, so the target is the reference front: a hypervolume of
at least 99% of its 37.3489, or an IGD+ to it of at most 0.01 with the objectives scaled to its
range (f₁ spans 5.565, f₂ 11.627). A good front also covers all four pieces: the point (−20, 0),
and the three curves with many solutions on each.

Both algorithms find the four pieces with 100 solutions. SPEA2's hypervolume, 37.1035, is 99.3% of
the reference front's; NSGA-II's, 37.0213, 99.1%. Their scaled IGD+ to it are 0.0022 and 0.0027.
Both meet the target on every seed from 1 to 50: SPEA2 with 99.22% to 99.35% of the hypervolume
and a scaled IGD+ of at most 0.0025, NSGA-II with 99.05% to 99.20% and at most 0.0033. SPEA2's
worst seed has a larger hypervolume than NSGA-II's best.
