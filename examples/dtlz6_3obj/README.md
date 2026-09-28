---
title: DTLZ6 with 3 objectives
category: multi-objective
summary: Minimize three conflicting objectives whose Pareto front is a curve, behind a distance function that only exact zeros satisfy, with NSGA-II and NSGA-III.
reference: "Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2001). Scalable Test Problems for Evolutionary Multi-Objective Optimization. TIK-Report 112, Computer Engineering and Networks Laboratory, ETH Zürich."
reference_url: https://sop.tik.ee.ethz.ch/publicationListFiles/dtlz2001a.pdf
optimum: "the curve f₁ = f₂ = cos θ / √2, f₃ = sin θ for θ in [0, π/2]; hypervolume 0.1349 (reference point (0.7778, 0.7778, 1.1))"
languages: [rust, python]
order: 110
---

# DTLZ6 with 3 objectives

## The problem

Deb, Thiele, Laumanns and Zitzler built test problems that scale to any number of objectives M,
first in a technical report (TIK-Report 112, ETH Zürich, 2001), then in a paper (Proceedings of
the 2002 Congress on Evolutionary Computation: 825-830). The two number the problems differently
from DTLZ5 on. This page uses the report's numbering, the common one, as genoxide does. The
report's DTLZ6 is the paper's DTLZ5 (section VII-E, p. 829), which states the same problem.

DTLZ6 is DTLZ5 with another distance function g, the report's eq. 26 (p. 21). With M = 3
objectives and the suggested k = 10, it has 12 variables in [0, 1]. All three objectives are
minimized:

```text
g  = x₃^0.1 + … + x₁₂^0.1
θ₁ = x₁ π/2
θ₂ = π (1 + 2 g x₂) / (4 (1 + g))
f₁ = (1 + g) cos θ₁ cos θ₂
f₂ = (1 + g) cos θ₁ sin θ₂
f₃ = (1 + g) sin θ₁
```

f₁² + f₂² + f₃² = (1 + g)², and g is 0 only where x₃ = … = x₁₂ = 0. There, θ₂ = π/4 and f₁ = f₂,
so the Pareto front is DTLZ5's curve:

```text
f₁ = f₂ = cos θ₁ / √2,   f₃ = sin θ₁,   θ₁ from 0 to π/2
```

a quarter circle in the plane f₁ = f₂, from (0.7071, 0.7071, 0) to (0, 0, 1). With x₁ = 0.5 and the
distance variables at 0, the solution is on the front at (0.5, 0.5, 0.7071).

## What makes it hard

The front is a curve, as for DTLZ5: three objectives, but one dimension.

And g is hard to bring to 0. x^0.1 falls steeply only near 0: it is 0.63 at x = 0.01, 0.1 at 10⁻¹⁰
and still 0.01 at 10⁻²⁰. A random solution has g near 9, since x^0.1 averages 1/1.1 over [0, 1].
For g below 0.1, all ten distance variables have to be below 10⁻²⁰. In practice they have to be
exactly 0, on the lower bound.

Polynomial mutation can get close. Near the lower bound, half of its moves take a gene to a random
point between the bound and the gene: a gene falls by orders of magnitude, a few mutations at a
time. But it can't take the last step alone. genoxide's polynomial mutation computes 1 − x, which
rounds to 1 below about 5.6 × 10⁻¹⁷ for genes in [0, 1]. Tried 100,000 times each, a mutation of a
gene at 10⁻¹⁶ sets it to exactly 0 a third of the time and raises it otherwise, and a mutation of a
gene at 5 × 10⁻¹⁷ or below always raises it. So exact zeros appear one gene at a time, and
crossover has to bring ten of them together in one genome.

Simulated binary crossover (SBX) can't: it spreads each gene between the values of the two parents,
and a 0 and a 2 × 10⁻¹⁷ give two children between them, neither of them 0. Uniform crossover can:
it takes each gene from either parent, unchanged.

## Representation

A `Real` genome of 12 genes in [0, 1]: the vector x. The problem is genoxide's `Dtlz6`, whose
fitness is the three objectives. In Python, `run` evaluates it in Rust.

## Algorithm

Three runs, with a population of 92, polynomial mutation with η = 20 at a rate of 1/12 per gene,
one gene per child on average, and 400 generations:

- NSGA-III (Deb and Jain, 2014, IEEE Transactions on Evolutionary Computation 18(4): 577-601), with
  SBX with η = 30, the 91 reference directions of Das and Dennis's method with 12 divisions, and a
  crossover rate of 1, as the DTLZ2 and DTLZ5 examples run it;
- NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation
  6(2): 182-197), with the same SBX at a crossover rate of 0.9, as the DTLZ5 example runs it; the
  DTLZ5 example shows that it spreads solutions along a curve better than NSGA-III;
- NSGA-II with uniform crossover, which swaps each gene between the two parents with probability
  1/2, at a crossover rate of 0.9.

Uniform crossover works here because of two properties of DTLZ6: the distance variables' optimum is
the lower bound, a value that the search reaches exactly, and each distance variable adds to g on
its own, so the zeros of different genomes can be combined. Uniform crossover makes no new values
of a gene. On a problem whose optimal values aren't shared in this way, it would do less.

## Output

A line per run, and one for the whole front. Each gives the size of the final front, its IGD+, its
hypervolume, the largest gap between its solutions along the curve, and the median g of its
solutions.

IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) is measured to 1,000 points of the
curve, evenly spread in θ₁, from genoxide's `optimal_front`. It averages, over those points, the
distance to the nearest point of the found front, counting only the objectives in which the found
point is worse. 0 means that the found front covers the curve. Smaller is better.

The hypervolume is the volume that the front dominates, up to the reference point (0.7778, 0.7778,
1.1), 1.1 times the nadir point (0.7071, 0.7071, 1). Larger is better. As for DTLZ5, the whole
curve's hypervolume is 1.1³/2 − 1.1π/4 + 1/3 = 0.1349.

The gap is measured in θ₁ = atan2(f₃, √(f₁² + f₂²)), in degrees, from 0° at (0.7071, 0.7071, 0) to
90° at (0, 0, 1), between neighboring solutions or between an end of the curve and the solution
nearest to it. g is √(f₁² + f₂² + f₃²) − 1: 0 on the front.

The plot shows the run with uniform crossover, with the curve as faint dots.

[The project page](https://tachsin.gr/projects/genoxide/examples/dtlz6-3obj) plays this run back.

## Good results

No finite set reaches 0.1349. 92 points evenly spread along the curve have a hypervolume of 0.1331,
an IGD+ of 0.0020 and gaps of 1.0°.

NSGA-II with uniform crossover reaches the front. Its front has 92 solutions, all with g = 0, an
IGD+ of 0.0036, a hypervolume of 0.1320 and a largest gap of 3.3°: as good as NSGA-II's front on
DTLZ5. Its initial front has 66 solutions with g from 8.1 to 9.7. For 240 generations, polynomial
mutation lowers the distance variables: the median g on the front is 2.8 at generation 100, 1.0
at 180 and 0.5 at 240, and the smallest variable goes from 10⁻³ to 10⁻¹⁶. The first exact zeros
appear at generation 220. Uniform crossover spreads them: 204 of the population's 920 distance
variables are 0 at generation 260, and 819 at generation 320, when the first genome has all ten.
By generation 340, every genome has all ten at 0, and the front is on the curve.

With SBX, neither algorithm gets there. After 400 generations, NSGA-III's front has a median g of
0.17, an IGD+ of 0.1667 and a hypervolume of 0.0333, and NSGA-II's a median g of 0.35, an IGD+ of
0.3405 and 0.0031. More generations don't help. Run for 3,000, both stop converging by generation
1,000: NSGA-II with every solution at g = 0.067 to 0.069, NSGA-III with its best at g = 0.080. In
the NSGA-II run, 644 of the 920 distance variables are 0 by then, but no genome has all ten. The
others sit between 2.5 × 10⁻¹⁷ and 5.1 × 10⁻¹⁷, where each adds about 0.022 to g, and mutation can
only raise them.

Over seeds 1 to 10, after 400 generations, NSGA-II with uniform crossover has an IGD+ of 0.0033 to
0.0037, with g below 0.0001 on every solution. With SBX, NSGA-III has 0.148 to 0.241, and
NSGA-II 0.214 to 0.396. NSGA-III with uniform crossover reaches g = 0 too, but spreads its
solutions as it does on DTLZ5, with an IGD+ of 0.0096 to 0.0126 and largest gaps of 8.5° to 12.4°.
