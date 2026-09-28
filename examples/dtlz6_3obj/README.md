---
title: DTLZ6 with 3 objectives
category: multi-objective
summary: Minimize three conflicting objectives whose Pareto front is a curve, behind a distance function that only falls steeply right next to the bound, with NSGA-II and NSGA-III.
reference: "Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2001). Scalable Test Problems for Evolutionary Multi-Objective Optimization. TIK-Report 112, Computer Engineering and Networks Laboratory, ETH Zürich."
reference_url: https://sop.tik.ee.ethz.ch/publicationListFiles/dtlz2001a.pdf
optimum: "the curve f₁ = f₂ = cos θ / √2, f₃ = sin θ for θ in [0, π/2]; hypervolume 0.1349 (reference point (0.7778, 0.7778, 1.1))"
languages: [rust, python]
order: 130
family: DTLZ
tab: DTLZ6
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
For g below 0.1, all ten distance variables have to be below 10⁻²⁰, and for g below 0.001, below
10⁻⁴⁰. g is 0 only where they are exactly 0, on the lower bound.

Polynomial mutation gets there, slowly. Near the lower bound, half of its moves take a gene to a
uniformly random point between the bound and the gene: a factor of e smaller on average, an order
of magnitude in about 2.3 such moves, all the way down to 5 × 10⁻³²⁴, the smallest positive
floating-point number, and from there to 0. With one gene mutated per child, and ten variables to
lower by tens of orders of magnitude, that takes hundreds of generations.

Crossover can speed it up by bringing together small values found in different genomes, since each
distance variable adds to g on its own. Uniform crossover does: it takes each gene from either
parent, unchanged. Simulated binary crossover (SBX) doesn't: as in NSGA-II's code, it leaves a gene
alone where the two parents' values are within 10⁻¹⁴ of each other, so below that it neither
spreads nor exchanges them, and each genome has to lower its own ten variables by mutation.

## Representation

A `Real` genome of 12 genes in [0, 1]: the vector x. The problem is genoxide's `Dtlz6`, whose
fitness is the three objectives. In Python, `run` evaluates it in Rust.

## Algorithm

Three runs, with a population of 92, polynomial mutation with η = 20 at a rate of 1/12 per gene,
one gene per child on average, and 1,000 generations:

- NSGA-III (Deb and Jain, 2014, IEEE Transactions on Evolutionary Computation 18(4): 577-601), with
  SBX with η = 30, the 91 reference directions of Das and Dennis's method with 12 divisions, and a
  crossover rate of 1, as the DTLZ2 and DTLZ5 examples run it;
- NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation
  6(2): 182-197), with the same SBX at a crossover rate of 0.9, as the DTLZ5 example runs it; the
  DTLZ5 example shows that it spreads solutions along a curve better than NSGA-III;
- NSGA-II with uniform crossover, which swaps each gene between the two parents with probability
  1/2, at a crossover rate of 0.9.

Uniform crossover works here because of two properties of DTLZ6: the distance variables' optimum is
the lower bound, shared by all of them, and each distance variable adds to g on its own, so the
small values of different genomes can be combined. Uniform crossover makes no new values of a gene.
On a problem whose optimal values aren't shared in this way, it would do less.

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

NSGA-II with uniform crossover reaches the front. Its front has 92 solutions, all with g below
0.0001, an IGD+ of 0.0033, a hypervolume of 0.1319 and a largest gap of 3.2°: as good as NSGA-II's
front on DTLZ5. Its initial front has 66 solutions with g from 8.1 to 9.7. The distance variables
fall steadily: the median g on the front is 2.8 at generation 100, 0.26 at 300, 0.024 at 500,
0.0025 at 700 and 0.0001 at 1,000, a factor of about 3 every 100 generations from generation 300
on, which is 5 orders of magnitude in each variable. At generation 1,000, every distance variable in
the population is below 10⁻⁴⁷. None is exactly 0 yet: by generation 3,000, they are below 10⁻¹⁴⁵.

With SBX, both algorithms converge too, but slower. After 1,000 generations, NSGA-III's front has a
median g of 0.022, an IGD+ of 0.0308 and a hypervolume of 0.1107, and NSGA-II's a median g of 0.017,
an IGD+ of 0.0193 and 0.1206. Run for 3,000, NSGA-II reaches a median g of 0.0006, an IGD+ of
0.0036 and a hypervolume of 0.1318, as with uniform crossover, and NSGA-III a median g of 0.0005
and an IGD+ of 0.0100, with its solutions spread as on DTLZ5.

Over seeds 1 to 10, after 1,000 generations, NSGA-II with uniform crossover has an IGD+ of 0.0030 to
0.0035, with g below 0.0001 on every solution. With SBX, NSGA-III has 0.0244 to 0.0356, and NSGA-II
0.0145 to 0.0237. NSGA-III with uniform crossover gets about as close to the front, with a median
g of 0.0001 to 0.0003, but spreads its solutions as it does on DTLZ5, with an IGD+ of 0.0072 to
0.0115 and largest gaps of 7.5° to 10.4°.
