---
title: WFG8
category: multi-objective
summary: Minimize two objectives whose concave front is reached through 24 parameters, where each distance parameter is biased by the parameters before it, with NSGA-II and SMS-EMOA.
reference: "Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation 10(5): 477-506."
reference_url: https://doi.org/10.1109/TEVC.2005.861417
optimum: "the quarter ellipse (f₁/2)² + (f₂/4)² = 1; hypervolume 3.3968 (reference point (2.2, 4.4))"
languages: [rust, python]
order: 123
---

# WFG8

## The problem

Huband, Hingston, Barone and While (2006), of the Walking Fish Group, built a toolkit for test
problems with any number of objectives, and nine problems from it, WFG1 to WFG9. Each problem
passes its variables through a chain of transformations down to a few values, then turns those
into the objectives. genoxide's `Wfg8` follows the paper's table XIV, with the transformations of
its table XI and the shapes of its table X; the optimal solutions are those of its section VIII-B.
It was checked against the paper as published, against its first version (Huband, Barone, While
and Hingston, 2005, EMO 2005, LNCS 3410: 280-295, the authors' corrected version), and against
the authors' C++ toolkit, version 2006.03.28, whose values genoxide's tests match.

This example uses 2 objectives and the recommended sizes: k = 4 position parameters and l = 20
distance parameters, 24 variables zᵢ in [0, 2i]. The problem first divides each by its upper
bound, yᵢ = zᵢ / 2i, then:

1. biases each distance parameter by the mean of the parameters before it:
   yᵢ ← b_param(yᵢ, mean(y₁, …, yᵢ₋₁)) for i = 5, …, 24, the means taken over the values before
   the bias;
2. shifts each distance parameter so that 0.35 maps to 0: yᵢ ← s_linear(yᵢ, 0.35), which is
   |yᵢ − 0.35| divided by 0.35 below 0.35 and by 0.65 above;
3. reduces each group to its mean: the position x₁ = mean(y₁, …, y₄), and the distance
   x₂ = mean(y₅, …, y₂₄).

Both objectives are minimized:

```text
f₁ = x₂ + 2 sin(x₁ π/2)
f₂ = x₂ + 4 cos(x₁ π/2)
```

The bias raises its value to a power set by u, the mean of the parameters before it:
b_param(y, u) = y^e, with e = 0.02 + 1.96 u for u up to 0.5, and e = 1 + 49 (2u − 1) above,
from 0.02 at u = 0 through 1 at u = 0.5 to 50 at u = 1. (Table XI writes it with the constants
A = 0.98/49.98, B = 0.02 and C = 50.)

The Pareto front is where the distance x₂ is 0, and there (f₁/2)² + (f₂/4)² = 1: a quarter
ellipse from (0, 4) to (2, 0), the same front as WFG4 to WFG7 and WFG9. The ideal point is (0, 0)
and the nadir point (2, 4). x₂ is 0 when every distance parameter comes out of the bias at 0.35:
yᵢ = 0.35^(1/e), with e set by the parameters before it. genoxide's docs give the optimal values
as the toolkit computes them, from z₅ to z₂₄ in turn:

```text
zᵢ = 2i × 0.35^(1 / (0.02 + 49.98 v(u))),   u = mean(y₁, …, yᵢ₋₁)
```

where v is b_param's middle term, so that the power is e above.

## What makes it hard

The optimal distance parameters depend on the position. The first distance parameter's u is the
mean of the four position parameters, which is x₁ itself; each later one's u includes the distance
parameters before it. With all four position parameters equal, the optimal y₅ to y₂₄ are:

| x₁ | y₅ | y₂₄ |
|---|---|---|
| 0 | 1.6 × 10⁻²³ | 1.6 × 10⁻²³ |
| 0.1 | 0.0078 | 5.4 × 10⁻⁹ |
| 0.25 | 0.128 | 0.0013 |
| 0.5 | 0.35 | 0.161 |
| 0.75 | 0.960 | 0.976 |
| 1 | 0.979 | 0.978 |

So no one set of distance parameters serves the whole front: the Pareto set is a curve through
the 24-dimensional box, not the plane zᵢ = 0.35 × 2i where WFG4 to WFG7 have theirs. On that
plane, WFG8's solutions are 0.115 from the front with the position parameters at 0.5, and 0.21 on
average over 20,000 random position parameters. The problem is non-separable: changing one
distance parameter moves the optimum of every one after it.

The ends of the front are the hardest. Near x₁ = 0 the optimal values are below 0.01, and at
x₁ = 0 they are 1.6 × 10⁻²³, a hair above the lower bound, where s_linear gives its worst value,
1. Near x₁ = 1 the power is close to 50, so y^e changes fast: y = 0.979 gives 0.35, but 0.97
gives 0.22 and 0.99 gives 0.60. Only near x₁ = 0.5, where the power is about 1, is the target
wide.

## Representation

A `Real` genome of 24 genes, the i-th in [0, 2i]: the vector z. The problem is genoxide's
`Wfg8::<2>::default()`, whose fitness is the pair (f₁, f₂). In Python,
`gx.problems.Wfg8(2)` gives the same problem, and `run` evaluates it in Rust, so both versions
print the same.

## Algorithm

The two algorithms of the WFG7 example, with the settings of the ZDT examples: a population of
100, simulated binary crossover with η = 15 at genoxide's default rate of 0.9, and polynomial
mutation with η = 20 at a rate of 1/24 per gene, one gene per child on average. Each runs for
1,000 generations, and the example reports its front after 250, 25,000 evaluations, and after
1,000.

- NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary
  Computation 6(2): 182-197), as the ZDT examples run it. It sorts solutions into non-dominated
  fronts, and within a front prefers solutions with a larger crowding distance, a measure of the
  gap between their neighbors.
- SMS-EMOA (Beume, Naujoks and Emmerich, 2007, European Journal of Operational Research 181(3):
  1653-1669), in genoxide's generational form: 100 children a generation. From the last front
  that fits only in part, it removes the solutions that add the least hypervolume.

On WFG7, SMS-EMOA reaches the front and NSGA-II nearly does. WFG8 has the same front and the same
bias, moved to the distance parameters: the example shows what that does to both.

## Output

Two lines per algorithm and budget: the size of the front, its IGD+ and its hypervolume, and then
how far its solutions are from the true front. The last line gives the whole front's hypervolume.

The distance of a point (f₁, f₂) is the d for which (f₁ − d, f₂ − d) lies on the front: the
smaller root of ((f₁ − d)/2)² + ((f₂ − d)/4)² = 1. Since WFG adds the distance x₂ to both
objectives, d is the solution's x₂. The example prints the smallest, the largest and the mean
over the front.

IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) is measured to 500 points of the
optimal front, from genoxide's `optimal_front`. It averages, over those 500 points, the distance
to the nearest point of the found front, counting only the objectives in which the found point is
worse. 0 means that the found front covers the optimal one. Smaller is better.

The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4):
257-271) is the area that the front dominates, up to a reference point. Larger is better. The
reference point is (2.2, 4.4), 1.1 times the nadir point (2, 4), as the ZDT examples use 1.1
times theirs. For the whole front, the hypervolume is the box up to the reference point less the
quarter ellipse under the front: 2.2 × 4.4 − 2π = 3.3968.

[The project page](https://tachsin.gr/projects/genoxide/examples/wfg8) plays this run back.

## Good results

A good front would have 100 solutions spread from (0, 4) to (2, 0), distances near 0, an IGD+ near
0 and a hypervolume near 3.3968; the 100 points of `optimal_front(100)` give 3.3610 and an IGD+
of 0.0051. Neither algorithm gets close, on this seed or any other tried.

Both fronts spread over the whole range of x₁ within about 100 generations, but they don't come
down to the true front. After 250 generations, NSGA-II's front has a mean distance of 0.1247 and
SMS-EMOA's 0.1073; after 1,000, 0.1109 and 0.0888. The IGD+ goes from 0.1734 to 0.1483 for
NSGA-II and from 0.1657 to 0.1397 for SMS-EMOA, and the hypervolume from 2.5378 to 2.6289 and from
2.6118 to 2.6884. SMS-EMOA is ahead from about generation 80 on, but by little.

The distance depends on the position, as the Pareto set does. After 1,000 generations, both
fronts come closest to the true one between x₁ = 0.4 and 0.5, where their smallest distances are
0.0109 for NSGA-II and 0.0043 for SMS-EMOA. From there they bend away towards both ends: at x₁
below 0.1 the distances are 0.12 to 0.16, and at x₁ above 0.9 about 0.21, the largest distance in
both runs. In the plot, the fronts lie on the true one in the middle and above it at both ends.

On seeds 1 to 10, NSGA-II has an IGD+ of 0.1589 to 0.1820 after 250 generations and 0.1394 to
0.1524 after 1,000, with mean distances of 0.1106 to 0.1299 and 0.1023 to 0.1136. SMS-EMOA has
0.1637 to 0.1778 and 0.1368 to 0.1451, with mean distances of 0.1012 to 0.1131 and 0.0888 to
0.0971. After 2,000 generations they are still at 0.1363 to 0.1479 and 0.1294 to 0.1391. MOEA/D
(Tchebycheff, 100 weight vectors, seeds 1 to 10) and SPEA2 (seeds 1 to 3) end in the same range
after 1,000 generations, at 0.1312 to 0.1443. So do the other operator settings tried with
NSGA-II on seeds 1 to 3, with IGD+ from 0.146 to 0.157 after 1,000 generations: polynomial
mutation with η = 5 or at a rate of 4/24, simulated binary crossover with η = 5, and blend
crossover (α = 0.5). Arithmetic crossover is worse, at 0.235 to 0.248. The selection and the
operators change little: the difficulty is in how the variables depend on each other.
