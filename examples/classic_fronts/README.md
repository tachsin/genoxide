---
title: Classic two-objective fronts
category: multi-objective
summary: NSGA-II on four classic two-objective problems, Schaffer's first and second, Fonseca and Fleming's and Poloni's, with convex, concave and disconnected fronts.
reference: "Deb, K., Pratap, A., Agarwal, S. and Meyarivan, T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation 6(2): 182-197."
reference_url: https://doi.org/10.1109/4235.996017
optimum: "normalized hypervolume 1.0433 (SCH1), 0.8142 (SCH2), 0.5273 (FON), at least 1.1239 (POL); reference point (1.1, 1.1)"
languages: [rust, python]
order: 81
---

# Classic two-objective fronts

## The problem

Four small problems with two objectives, both minimized. The NSGA-II paper (Deb, Pratap, Agarwal
and Meyarivan, 2002) tests on three of them, restated in its table I. Each has a Pareto front of a
different shape: the set of optimal trade-offs, where one objective can't improve without the
other getting worse.

**Schaffer 1** (SCH1) has one variable, x in [−1000, 1000]:

```text
f₁ = x²
f₂ = (x − 2)²
```

The optimal solutions are x from 0 to 2, and the front is f₂ = (√f₁ − 2)² for f₁ from 0 to 4: a
convex curve from (0, 4) to (4, 0). Schaffer (1985, Proceedings of the First International
Conference on Genetic Algorithms: 93-100) posed it. The definition and bounds here are the NSGA-II
paper's; other papers use other bounds.

**Schaffer 2** (SCH2) has one variable, x in [−5, 10], and a piecewise f₁:

```text
f₁ = −x      for x ≤ 1
     x − 2   for 1 < x ≤ 3
     4 − x   for 3 < x ≤ 4
     x − 4   for x > 4
f₂ = (x − 5)²
```

The optimal solutions are x in [1, 2) and [4, 5], and the front is in two pieces: f₂ = (f₁ − 3)²
for f₁ from −1 to 0, and f₂ = (f₁ − 1)² for f₁ from 0 to 1. At x = 2, the solution (0, 9) is
dominated by x = 4, (0, 1). It is Schaffer's (1985) too; the definition and bounds are as Van
Veldhuizen (1999, PhD thesis, Air Force Institute of Technology, table B.1) restates them, after
Srinivas and Deb (1994).

**Fonseca-Fleming** (FON) has n variables, each in [−4, 4]; here n = 3:

```text
f₁ = 1 − exp(−Σ (xᵢ − 1/√n)²)
f₂ = 1 − exp(−Σ (xᵢ + 1/√n)²)
```

The optimal solutions have all variables equal, to t in [−1/√n, 1/√n]. The front, the same for
every n, is f = (1 − exp(−(s − 1)²), 1 − exp(−(s + 1)²)) for s = t√n from −1 to 1: a concave curve
from (0, 0.9817) to (0.9817, 0). Fonseca and Fleming (1995, Evolutionary Computation 3(1): 1-16)
posed it. The definition and bounds are as Deb, Thiele, Laumanns and Zitzler (2001, TIK-Report
112) restate them, and, for 3 variables, the NSGA-II paper.

**Poloni** (POL) has two variables, each in [−π, π]:

```text
f₁ = 1 + (A₁ − B₁)² + (A₂ − B₂)²
f₂ = (x₁ + 3)² + (x₂ + 1)²
A₁ = 0.5 sin 1 − 2 cos 1 + sin 2 − 1.5 cos 2
A₂ = 1.5 sin 1 − cos 1 + 2 sin 2 − 0.5 cos 2
B₁ = 0.5 sin x₁ − 2 cos x₁ + sin x₂ − 1.5 cos x₂
B₂ = 1.5 sin x₁ − cos x₁ + 2 sin x₂ − 0.5 cos x₂
```

f₁ is 1 at (1, 2), where B = A, and f₂ is 0 at (−3, −1). The front runs from (1, 25) to about
(16.77, 0) and is disconnected; it isn't known in closed form. On a fine grid over the box, it has
two pieces, with a gap where f₂ falls from about 20.9 to about 3.1. Poloni, Giurgevich, Onesti and
Pediroda (2000, Computer Methods in Applied Mechanics and Engineering 186(2-4): 403-420) use it; it
first appeared in Poloni et al. (1996) and Poloni (1997), which, as Van Veldhuizen (1999) notes,
print it mistyped. The definition and bounds are the NSGA-II paper's.

genoxide hasn't yet checked these restatements against the original papers.

## What makes it hard

Each problem tests one thing.

- SCH1: the bounds. The optimal solutions, x in [0, 2], are 0.1% of [−1000, 1000], and far from
  them both objectives are huge. The search has to find the small interval, then spread over it.
- SCH2: a disconnected front. Between the two pieces, x in [2, 4) gives only dominated solutions,
  so the population has to keep two separate groups.
- FON: a concave front. A weighted sum of the objectives finds only its two ends; a
  multi-objective algorithm has to fill the middle. Far from the optimal solutions, both
  objectives are close to 1, a plateau with little to guide the search, and the more variables,
  the larger it is.
- POL: sines and cosines of the two variables make f₁ multimodal, and the front has a gap. The
  first piece, near (1, 25), is short, and easy to lose.

## Representation

A `Real` genome, the vector x: 1 gene in [−1000, 1000] for SCH1, 1 in [−5, 10] for SCH2, 3 in
[−4, 4] for FON, and 2 in [−π, π] for POL. The problems are genoxide's `Schaffer1`, `Schaffer2`,
`FonsecaFleming` and `Poloni`, whose fitness is the pair (f₁, f₂). In Python, `run` evaluates them
in Rust, so both versions print the same.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002). It ranks solutions by non-dominated sorting:
the first front is the solutions that no other solution beats in both objectives, the second front
those beaten only by the first, and so on. Within a front, it prefers solutions in less crowded
regions (crowding distance). Parents and children compete for the next population, so it keeps the
best solutions found so far.

The same settings on all four problems, the usual ones that genoxide's NSGA-II docs give:

- a population of 100, for 250 generations, as in the NSGA-II paper;
- simulated binary crossover with η = 15, at genoxide's default rate of 0.9;
- polynomial mutation with η = 20, at a rate of 1/n per gene for n variables, one gene per child
  on average.

## Output

A table with a line per problem: how many solutions are on the final front, its normalized
hypervolume, and its IGD+.

The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4):
257-271) is the area that the front dominates, up to a reference point. Larger is better. The
objectives of the four problems have very different scales, so the example first maps each to
[0, 1] with the problem's ideal point, the best value of each objective on the optimal front, and
its nadir point, the worst: (f − ideal) / (nadir − ideal). The reference point is then (1.1, 1.1)
for all four, and the curve on the example's page shares one axis. The ideal and nadir points are
(0, 0) and (4, 4) for SCH1, (−1, 0) and (1, 16) for SCH2, and (0, 0) and (0.9817, 0.9817) for FON.
genoxide doesn't give POL's, so the example takes them from the ends of the front: the ideal point
is (1, 0), the least f₁ and f₂, and the nadir point (16.7723, 25), f₁ at (−3, −1) and f₂ at (1, 2).

The normalized hypervolume depends on the front's shape: a convex front leaves more of the box
dominated than a concave one. Compare each problem's value with its whole front's, not with the
other problems'.

IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over 500 points of the
optimal front, the distance to the nearest point of the found front, counting only the objectives
in which the found point is worse. It is in each problem's own units, not normalized. 0 means that
the found front covers the optimal one. Smaller is better. POL's front isn't known, so it has no
IGD+ (`-`).

[The project page](https://tachsin.gr/projects/genoxide/examples/classic-fronts) plays this run back.

## Good results

A good front has 100 solutions, spread over the whole optimal front, and on SCH2 and POL over each
of its pieces. No set of 100 points reaches the hypervolume of the whole, continuous front. The
table compares the run with the 100 points of the optimal front that genoxide's `optimal_front(100)`
gives, evenly spaced in x for SCH1 and SCH2 and in s for FON:

| Problem | Run's hypervolume | 100 points' | Whole front's | Run's IGD+ | 100 points' |
|---|---|---|---|---|---|
| SCH1 | 1.0392 | 1.0399 | 1.0433 | 0.0082 | 0.0068 |
| SCH2 | 0.8114 | 0.8119 | 0.8142 | 0.0087 | 0.0068 |
| FON | 0.5195 | 0.5219 | 0.5273 | 0.0038 | 0.0022 |
| POL | 1.1228 | | at least 1.1239 | | |

SCH1's whole front has 1.1² − 1/6 = 1.0433: the box minus the area under the curve (1 − √f₁)². For
POL, a grid of 2000 × 2000 points over the box dominates 1.1239, so the whole front's is at least
that. On every problem, the run's front has 100 solutions, and it covers both pieces of SCH2's and
POL's fronts. Its hypervolume is within 0.4% of the whole front's on SCH1, SCH2 and POL, and 1.5%
on FON, whose front fills more slowly: it has 100 solutions only after about 100 generations.
