---
title: DTLZ5 with 3 objectives
category: multi-objective
summary: Minimize three conflicting objectives whose Pareto front is a curve, not a surface, with NSGA-II and NSGA-III.
reference: "Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2001). Scalable Test Problems for Evolutionary Multi-Objective Optimization. TIK-Report 112, Computer Engineering and Networks Laboratory, ETH Zürich."
reference_url: https://sop.tik.ee.ethz.ch/publicationListFiles/dtlz2001a.pdf
optimum: "the curve f₁ = f₂ = cos θ / √2, f₃ = sin θ for θ in [0, π/2]; hypervolume 0.1349 (reference point (0.7778, 0.7778, 1.1))"
languages: [rust, python]
order: 179
family: DTLZ
tab: DTLZ5
---

# DTLZ5 with 3 objectives

## The problem

Deb, Thiele, Laumanns and Zitzler built test problems that scale to any number of objectives M,
first in a technical report (TIK-Report 112, ETH Zürich, 2001), then in a paper (Proceedings of
the 2002 Congress on Evolutionary Computation: 825-830). The two number the problems differently
from DTLZ5 on. This page uses the report's numbering, the common one, as genoxide does. The
report's DTLZ5 isn't in the paper: the paper's DTLZ5 is the report's DTLZ6.

DTLZ5 has M + k − 1 variables in [0, 1]; with M = 3 objectives and the suggested k = 10, that's 12
variables. It is DTLZ2 with its second angle changed. All three objectives are minimized:

```text
g  = (x₃ − 0.5)² + … + (x₁₂ − 0.5)²
θ₁ = x₁ π/2
θ₂ = π (1 + 2 g x₂) / (4 (1 + g))
f₁ = (1 + g) cos θ₁ cos θ₂
f₂ = (1 + g) cos θ₁ sin θ₂
f₃ = (1 + g) sin θ₁
```

This is the report's eq. 25 (p. 20), with the mapping of the angles of its eq. 10 (p. 8).
genoxide's docs note two typos in eq. 25: it writes cos(θᵢπ/2) for cos θᵢ, and doesn't define θ₁.
The report's eq. 8 (p. 7), the problem that eq. 10 maps, has θ₁ = x₁π/2 and cos θᵢ.

As for DTLZ2, f₁² + f₂² + f₃² = (1 + g)², and g is 0 where x₃ = … = x₁₂ = 0.5. There, θ₂ = π/4
whatever x₂ is, and f₁ = f₂. So the Pareto front isn't a surface: it's the curve

```text
f₁ = f₂ = cos θ₁ / √2,   f₃ = sin θ₁,   θ₁ from 0 to π/2
```

a quarter circle in the plane f₁ = f₂, from (0.7071, 0.7071, 0) to (0, 0, 1). With x₁ = 0.5 and
g = 0, the solution is on the front at (0.5, 0.5, 0.7071).

With 4 or more objectives, the front isn't a curve. genoxide's docs cite Huband, Hingston, Barone
and While (2006, IEEE Transactions on Evolutionary Computation 10(5): 477-506, section VI-A3) for
this, checked in their paper, and its tests show a solution that no point of the curve dominates.
With 3 objectives, every solution is on the curve or dominated by it.

## What makes it hard

Three objectives, but a front of one dimension: a degenerate front. Only x₁ moves a solution
along it. x₂ matters only off the front, where g > 0, and ten variables must converge to 0.5.

Algorithms for three or more objectives often expect a front that spreads over a surface.
NSGA-III, which genoxide's docs recommend for them, spreads its solutions along reference
directions that cover the whole triangle of the objective space. The curve passes near only a few
of them. The example runs it, with the settings that the DTLZ2 example uses, against NSGA-II,
whose crowding distance measures the gaps between neighboring solutions, whatever the front's
shape.

## Representation

A `Real` genome of 12 genes in [0, 1]: the vector x. The problem is genoxide's `Dtlz5`, whose
fitness is the three objectives. In Python, `run` evaluates it in Rust.

## Algorithm

Two algorithms, with the same population of 92, simulated binary crossover with η = 30,
polynomial mutation with η = 20 at a rate of 1/12 per gene, one gene per child on average, and 250
generations. The crossover rate is genoxide's default for each: 1 for NSGA-III, as Deb and Jain
use, and 0.9 for NSGA-II.

- NSGA-III (Deb and Jain, 2014, IEEE Transactions on Evolutionary Computation 18(4): 577-601),
  with the 91 reference directions of Das and Dennis's method (1998, SIAM Journal on Optimization
  8(3): 631-657) with 12 divisions, as the DTLZ2 example runs it. It ranks solutions into
  non-dominated fronts; within the last front that fits, each solution joins the direction nearest
  to it, and directions with few members get more. Only 7 of the 91 directions, those with equal
  first and second coordinates, lie in the plane of the curve.
- NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary
  Computation 6(2): 182-197). Within a front, it prefers solutions with a larger crowding distance,
  the size of the box between their neighbors in each objective. On a curve, that spreads the
  solutions along it.

## Output

A line per algorithm, and one for the whole front. Each gives the size of the final front, its
IGD+, its hypervolume, the largest gap between its solutions along the curve, and the largest g
among them.

IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) is measured to 1,000 points of the
curve, evenly spread in θ₁, from genoxide's `optimal_front`. It averages, over those points, the
distance to the nearest point of the found front, counting only the objectives in which the found
point is worse. 0 means that the found front covers the curve. Smaller is better.

The hypervolume is the volume that the front dominates, up to a reference point. Larger is better.
The reference point is (0.7778, 0.7778, 1.1), 1.1 times the nadir point (0.7071, 0.7071, 1), the
worst value of each objective on the front. At a height f₃ = z below 1, the curve dominates a
square of side 0.7778 − √(1 − z²) / √2; above 1, the whole square of side 0.7778. So the whole
curve's hypervolume is 1.1³/2 − 1.1π/4 + 1/3 = 0.1349.

The gap is measured in θ₁, in degrees: 0° at (0.7071, 0.7071, 0) and 90° at (0, 0, 1), with θ₁ =
atan2(f₃, √(f₁² + f₂²)). The largest gap is the largest between two neighboring solutions, or
between an end of the curve and the solution nearest to it. g is √(f₁² + f₂² + f₃²) − 1: 0 on the
front.

The plot shows the NSGA-II run, with the curve as faint dots.

[The project page](https://tachsin.gr/projects/genoxide/examples/dtlz5-3obj) plays this run back.

## Good results

No finite set reaches 0.1349. 92 points evenly spread along the curve have a hypervolume of 0.1331,
an IGD+ of 0.0020 and gaps of 1.0°.

NSGA-II comes close: its front has 92 solutions, an IGD+ of 0.0032, a hypervolume of 0.1323 and a
largest gap of 2.6°. It converges fast. Its initial front has 22 solutions, with g from 0.31 to
0.92. By generation 20, all 92 solutions are non-dominated; by generation 40, the IGD+ is 0.0098,
and by generation 100, 0.0036. At the end, half of its solutions have g below 0.0004, and the
largest is 0.0121.

NSGA-III converges as well, with half of its solutions at g below 0.0004, but spreads them worse:
an IGD+ of 0.0088, a hypervolume of 0.1268 and a largest gap of 7.3°. Its solutions bunch: 13 of
them between 38.7° and 42.3°, and 11 between 82.9° and 85.8°, with gaps of up to 7.3° between the
bunches. NSGA-II's solutions are spread evenly, 7 to 12 in every 10°.

Over seeds 1 to 10, NSGA-II has an IGD+ of 0.0029 to 0.0034 and a largest gap of 2.4° to 3.7°;
NSGA-III 0.0073 to 0.0095, and 6.8° to 9.8°.
