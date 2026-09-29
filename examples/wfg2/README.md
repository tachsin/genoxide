---
title: WFG2
category: multi-objective
summary: Minimize two objectives over 24 variables, with a convex front in six disconnected regions and distance parameters that act in pairs, with NSGA-II and SMS-EMOA.
reference: "Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation 10(5): 477-506."
reference_url: https://doi.org/10.1109/TEVC.2005.861417
optimum: "six regions of the curve f₁ = 2 (1 − cos(x₁π/2)), f₂ = 4 (1 − x₁ cos²(5πx₁)); hypervolume 6.1511 (reference point (2.2, 4.4))"
languages: [rust, python]
order: 127
family: WFG
---

# WFG2

## The problem

Huband, Hingston, Barone and While (2006) built a toolkit for test problems with any number of
objectives, and nine problems from it, WFG1 to WFG9 (their table XIV). An earlier version appeared
at EMO 2005 (Huband, Barone, While and Hingston, LNCS 3410: 280-295), and the authors released a
C++ implementation. genoxide's `Wfg2` follows the 2006 paper, and was checked against the EMO 2005
version and against the authors' C++ toolkit, version 2006.03.28, compiled and run for its tests.

Every WFG problem has n = k + l variables: k position parameters, which say where on the front a
solution lies, and l distance parameters, which say how far from it. The i-th variable zᵢ is in
[0, 2i]. The problem divides it by 2i, passes the values through a chain of transformations, and
reduces them to one value per objective. This example uses 2 objectives and the sizes the authors
recommend, k = 4 and l = 20: 24 variables. With yᵢ = zᵢ / 2i:

```text
distance parameters (i = 5, …, 24):
  yᵢ ← |yᵢ − 0.35| / 0.35 below 0.35, (yᵢ − 0.35) / 0.65 above    (s_linear: 0.35 becomes 0)
  each pair (a, b) = (y₅, y₆), (y₇, y₈), …, (y₂₃, y₂₄) becomes
    (a + b + 2 |a − b|) / 3                                         (r_nonsep)
x₁ = the mean of y₁, …, y₄                (the position)
x₂ = the mean of the 10 pair values       (the distance)
f₁ = x₂ + 2 (1 − cos(x₁π/2))
f₂ = x₂ + 4 (1 − x₁ cos²(5πx₁))
```

Both objectives are minimized. The optimal solutions have every distance parameter at 0.35 × 2i,
so that x₂ = 0. f₁ then grows with x₁, from 0 to 2. f₂ doesn't fall steadily: 1 − x₁ cos²(5πx₁)
dips and rises five times. A value of x₁ gives an optimal solution only where f₂ is below all its
values at smaller x₁. That leaves six ranges of x₁, found to the last bit in genoxide from the
shape, each ending at a local minimum of f₂ or at 1:

| Region | x₁ | f₁ | f₂ |
|---|---|---|---|
| 1 | 0 to 0.0416 | 0 to 0.0043 | 4 to 3.8951 |
| 2 | 0.1297 to 0.2096 | 0.0414 to 0.1074 | 3.8950 to 3.1805 |
| 3 | 0.3549 to 0.4050 | 0.3028 to 0.3912 | 3.1814 to 2.3900 |
| 4 | 0.5641 to 0.6034 | 0.7351 to 0.8331 | 2.3882 to 1.5933 |
| 5 | 0.7691 to 0.8025 | 1.2904 to 1.3894 | 1.5932 to 0.7949 |
| 6 | 0.9724 to 1 | 1.9133 to 2 | 0.7968 to 0 |

The Pareto front is these six pieces of a convex curve, from (0, 4) to (2, 0). The paper notes
that it is disconnected (section VIII-B). The ideal point is (0, 0) and the nadir point (2, 4).

## What makes it hard

A disconnected front. Between the regions, no solution is optimal: an algorithm has to keep a
separate group of solutions on each. The regions are unequal. The first is 0.004 wide in f₁ and
0.1 tall in f₂. The last is the longest: of 500 points evenly spread along the front, 99 are on it.

The last region is hard to reach. f₂ has a local minimum at the end of the fifth region, x₁ =
0.8025. Between it and the sixth region, from x₁ = 0.8025 to 0.9724, every solution is beaten by
the end of the fifth, and f₂ rises to its largest, 4, at x₁ = 0.9. x₁ is the mean of the four
position values, so to cross the gap they have to rise together by 0.17 on average. No small step
leads there: a search that has settled on the first five regions sees nothing better nearby.

Distance parameters in pairs. The reduction (a + b + 2 |a − b|) / 3 is 0 only when both values of
a pair are 0, and it punishes their difference. With b fixed above 0, the best a is b, not 0:
moving one parameter of a pair to its optimum alone makes the pair worse. The problem is
non-separable: the best value of a variable depends on another.

## Representation

A `Real` genome of 24 genes, the i-th in [0, 2i]: the vector z. The problem is genoxide's
`Wfg2::<2>::default()`, with k = 4 and l = 20, whose fitness is the pair (f₁, f₂). In Python,
`gx.problems.Wfg2(objectives=2)`, which `run` evaluates in Rust, so both versions print the same.

## Algorithm

Two algorithms, both with a population of 100 and simulated binary crossover with η = 15 at
genoxide's default rate of 0.9. They differ in how they mutate and how they select.

- NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary
  Computation 6(2): 182-197), as the ZDT examples run it: polynomial mutation with η = 20 at a rate
  of 1/24 per gene, one gene per child on average, for 1,000 generations. It sorts solutions into
  non-dominated fronts, and within a front prefers solutions with a larger crowding distance, a
  measure of the gap between their neighbors. At a gap in the front, the end solutions of each
  region have a neighbor across it, so they count as uncrowded and are kept.
- SMS-EMOA (Beume, Naujoks and Emmerich, 2007, European Journal of Operational Research 181(3):
  1653-1669), in genoxide's generational form: 100 children a generation. From the last front
  that fits only in part, it removes the solutions that add the least hypervolume. Its mutation is
  uniform: one gene per child, drawn anew anywhere in its range. It runs for 5,000 generations,
  and the example reports its front after 1,000 and after 5,000.

Uniform mutation is there for the last region. Polynomial mutation with η = 20 usually moves a
gene by a few hundredths of its range, and the crossover mixes the values the population already
has. A uniform draw can move one position parameter anywhere, and x₁ with it by up to 0.25: from
the fifth region across the gap in one step. Such a child has the smallest f₂ of the population, so
it isn't dominated, and it survives. The hypervolume then rewards filling the new region, which is
the longest of the six.

## Output

A line per algorithm and budget: the size of the front, how many of its solutions are on each of
the six regions (with an f₁ within 0.01 of the region's range), its IGD+ and its hypervolume.
The last line gives the whole front's hypervolume.

IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) is measured to 500 points of the
optimal front, evenly spread along it, region by region. It averages, over those 500 points, the
distance to the nearest point of the found front, counting only the objectives in which the found
point is worse. 0 means that the found front covers the optimal one. Smaller is better.

The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4):
257-271) is the area that the front dominates, up to the reference point (2.2, 4.4), 1.1 times the
nadir point, as the ZDT examples use (1.1, 1.1). Larger is better. For the whole front, it is
6.1511, found numerically from a million points of the front.

[The project page](https://tachsin.gr/projects/genoxide/examples/wfg2) plays these runs back.

## Good results

A good front has solutions on all six regions, an IGD+ near 0 and a hypervolume near 6.1511. No
set of 100 points reaches that hypervolume: 100 points of the optimal front, evenly spread along
it, give 6.1180 and an IGD+ of 0.0021. Without the last region, the most a front can have is
5.9485, and an IGD+ of about 0.081. The target here is an IGD+ of at most 0.01, or a hypervolume of
at least 99% of the whole front's, 6.0896.

NSGA-II doesn't find the last region. It converges quickly on the other five, and after 1,000
generations has 3, 22, 25, 24 and 26 solutions on regions 1 to 5, an IGD+ of 0.0862 and a
hypervolume of 5.9148, close to the 5.9485 of a perfect front without the last region. Its
solution with the smallest f₂ ends at (1.39, 0.80), the end of the fifth region, and stays there.

SMS-EMOA with uniform mutation crosses the gap at generation 233: a child lands at (1.998, 0.279),
on the last region. After 1,000 generations it has 16 solutions there, an IGD+ of 0.0125 and a
hypervolume of 6.0716. After 5,000, its front has 2, 16, 21, 22, 21 and 18 solutions on the six
regions, an IGD+ of 0.0055 and a hypervolume of 6.1172: 99.4% of the whole front's, and nearly the
6.1180 of 100 evenly spread points.

On seeds 1 to 40, SMS-EMOA with uniform mutation reaches the last region in 37 runs, all 20 of
seeds 1 to 20. Those 37 end with an IGD+ of 0.0037 to 0.0076 and a hypervolume of 6.1040 to
6.1290, at least 99.2% of the whole front's. The other three, seeds 25, 31 and 37, end like
NSGA-II, with an IGD+ of about 0.085.

Each change alone helps less. After 5,000 generations, on seeds 1 to 20, NSGA-II with uniform
mutation reaches the last region in 16 runs, and SMS-EMOA with polynomial mutation in none. With
polynomial mutation, NSGA-II misses it on all 20 seeds, and so do SPEA2 and MOEA/D on seeds 1 to
5, as does NSGA-II with the settings of Huband et al., who count WFG2 among the problems their
NSGA-II solved easily (section IX): SBX η = 10 and mutation η = 50.
