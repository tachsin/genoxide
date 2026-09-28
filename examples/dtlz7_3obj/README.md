---
title: DTLZ7 with 3 objectives
category: multi-objective
summary: Minimize three conflicting objectives whose Pareto front is four disconnected regions, with NSGA-III and NSGA-II.
reference: "Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2001). Scalable Test Problems for Evolutionary Multi-Objective Optimization. TIK-Report 112, Computer Engineering and Networks Laboratory, ETH Zürich."
reference_url: https://sop.tik.ee.ethz.ch/publicationListFiles/dtlz2001a.pdf
optimum: "f₃ = 6 − φ(f₁) − φ(f₂), φ(f) = f (1 + sin 3πf), with f₁ and f₂ in [0, 0.2514] or (0.6316, 0.8594]; hypervolume 1.7392 (reference point (0.9453, 0.9453, 6.6))"
languages: [rust, python]
order: 131
family: DTLZ
tab: DTLZ7
---

# DTLZ7 with 3 objectives

## The problem

Deb, Thiele, Laumanns and Zitzler built test problems that scale to any number of objectives M,
first in a technical report (TIK-Report 112, ETH Zürich, 2001), then in a paper (Proceedings of
the 2002 Congress on Evolutionary Computation: 825-830). The two number the problems differently
from DTLZ5 on. This page uses the report's numbering, the common one, as genoxide does. The
report's DTLZ7 is the paper's DTLZ6.

DTLZ7 has M + k − 1 variables in [0, 1]; with M = 3 objectives and the suggested k = 20, that's 22
variables. All three objectives are minimized:

```text
f₁ = x₁
f₂ = x₂
g  = 1 + 9 (x₃ + … + x₂₂) / 20
h  = 3 − (f₁ (1 + sin 3πf₁) + f₂ (1 + sin 3πf₂)) / (1 + g)
f₃ = (1 + g) h
```

This is the report's eq. 27 (p. 22) and the paper's eq. 10 (p. 829). g is 1, its smallest, where
x₃ = … = x₂₂ = 0. There, with φ(f) = f (1 + sin 3πf),

```text
f₃ = 6 − φ(f₁) − φ(f₂)
```

A larger φ(f₁) lowers f₃, and a larger f₁ is worse. So a value of f₁ is Pareto optimal only where
φ is larger there than at every smaller value. genoxide's docs derive where that is: f₁ in
[0, A] or (B, C], with A = 0.2514 and C = 0.8594 the first two local maxima of φ, and B = 0.6316
where φ climbs back to φ(A). Between A and B, φ is smaller than φ(A): a solution there is dominated
by the one at f₁ = A. The same holds for f₂, so the front is four separate regions, one for each
pair of ranges, low or high in f₁ and in f₂. f₃ runs from 2.614 at (C, C) to 6 at (0, 0). With all
22 variables at 0, the solution is on the front at (0, 0, 6); with x₁ = x₂ = 1/6 and the rest 0,
φ = 1/3 for both, and it is at (0.1667, 0.1667, 5.3333).

## What makes it hard

The front is disconnected. The solutions have to be spread over four regions, with no optimal
solution between them. A region with no solutions can be lost for good: f₁ and f₂ are the genes
x₁ and x₂ themselves, and from the low range to the high one, a gene has to jump over the gap
from 0.25 to 0.63, where every solution is dominated.

Twenty distance variables have to converge to 0. A random solution has g near 5.5. Until g is near
1, the regions don't show: the front of the initial population is scattered far above them.

The regions are curved and tilted: f₃ falls from 6 to 2.614 across them, and the region with both
objectives high lies lowest.

## Representation

A `Real` genome of 22 genes in [0, 1]: the vector x. The problem is genoxide's `Dtlz7`, whose
fitness is the three objectives. In Python, `run` evaluates it in Rust.

## Algorithm

Two algorithms, with the same population of 92, simulated binary crossover with η = 30,
polynomial mutation with η = 20 at a rate of 1/22 per gene, one gene per child on average, and 250
generations. The crossover rate is genoxide's default for each: 1 for NSGA-III, as Deb and Jain
use, and 0.9 for NSGA-II.

- NSGA-III (Deb and Jain, 2014, IEEE Transactions on Evolutionary Computation 18(4): 577-601),
  with the 91 reference directions of Das and Dennis's method (1998, SIAM Journal on Optimization
  8(3): 631-657) with 12 divisions, as the DTLZ2 example runs it. It ranks solutions into
  non-dominated fronts; within the last front that fits, each solution joins the direction nearest
  to it, and directions with few members get more. Unlike DTLZ5's curve, DTLZ7's front spreads over
  a surface, as NSGA-III expects.
- NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary
  Computation 6(2): 182-197). Within a front, it prefers solutions with a larger crowding distance,
  the size of the box between their neighbors in each objective.

## Output

A line per algorithm, and one for the whole front. Each gives the size of the final front, how many
of its solutions lie in each of the four regions, its IGD+ and its hypervolume.

The regions are counted in this order: f₁ and f₂ both low, f₁ low and f₂ high, f₁ high and f₂ low,
both high. A value counts as low below (A + B) / 2 = 0.4415, halfway across the gap. A solution
near a region but just outside its range, where φ is nearly flat, counts with the region.

IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) is measured to 1,024 points of the
front from genoxide's `optimal_front`: a grid of 32 × 32 values of f₁ and f₂, shared between the
two ranges in proportion to their widths. It averages, over those points, the distance to the
nearest point of the found front, counting only the objectives in which the found point is worse.
0 means that the found front covers the regions. Smaller is better.

The hypervolume is the volume that the front dominates, up to a reference point. Larger is better.
The reference point is (0.9453, 0.9453, 6.6), 1.1 times the nadir point (0.8594, 0.8594, 6), the
worst value of each objective on the front. For the whole front, it is 1.7392, integrated
numerically: at each (f₁, f₂) in the box, the lowest f₃ that the front reaches with no larger f₁
and f₂ is 6 − Φ(f₁) − Φ(f₂), Φ(v) being the largest φ at or below v.

The plot shows the NSGA-III run, with the grid of 1,024 points as faint dots.

[The project page](https://tachsin.gr/projects/genoxide/examples/dtlz7-3obj) plays this run back.

## Good results

No finite set reaches 1.7392. A grid of 10 × 10 points of the front, 100 points, has a
hypervolume of 1.6567 and an IGD+ of 0.0222.

NSGA-III's front has 92 solutions, 29, 22, 22 and 19 in the four regions: close to the regions'
shares of the grid, 26, 23, 23 and 20 of 92. It has an IGD+ of 0.0412 and a hypervolume of 1.5990.
Its initial front has 17 solutions, and the hypervolume stays 0 until about generation 30, when the
first solutions come inside the reference box. By generation 40, all 92 solutions are
non-dominated, with an IGD+ of 0.64. The IGD+ falls to 0.115 by generation 100 and 0.049 by 190.
At the end, the median g is 1.008, not yet the front's 1: run for 500 generations, NSGA-III reaches
an IGD+ of 0.0372 and a hypervolume of 1.6174.

NSGA-II covers the four regions too, with 24, 23, 16 and 29 solutions, but less evenly, and less
well: an IGD+ of 0.0479 and a hypervolume of 1.5542.

Over seeds 1 to 20, NSGA-III covers all four regions in 19 runs, with an IGD+ of 0.0377 to 0.0421
and a hypervolume of 1.589 to 1.618. NSGA-II covers them in 18, with 0.0404 to 0.0518 and 1.543 to
1.589, and puts the most solutions in the region with both objectives high in each of them. The
other runs lose two regions, the high range of one objective, by generation 40, and don't get them
back: NSGA-III's has only low values of f₂, NSGA-II's two only low values of f₁. Their IGD+ is 0.24
and their hypervolume 1.40 to 1.41.
