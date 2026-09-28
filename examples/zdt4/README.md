---
title: ZDT4
category: multi-objective
summary: Minimize two conflicting objectives over 10 variables, with the convex front of ZDT1 behind 21⁹ local fronts, with NSGA-II and SMS-EMOA.
reference: "Zitzler, E., Deb, K. and Thiele, L. (2000). Comparison of multiobjective evolutionary algorithms: empirical results. Evolutionary Computation 8(2): 173-195."
reference_url: https://doi.org/10.1162/106365600568202
optimum: "the front f₂ = 1 − √f₁ for f₁ in [0, 1]; hypervolume 0.8767 (reference point (1.1, 1.1))"
languages: [rust, python]
order: 103
---

# ZDT4

## The problem

Zitzler, Deb and Thiele (2000) built six test problems with two objectives from one scheme: f₁
depends on the first variable, a function g on the others, and f₂ on both. ZDT4 is the fourth. It
has 10 variables: x₁ in [0, 1], and x₂ to x₁₀ in [−5, 5]. It minimizes both objectives:

```text
f₁ = x₁
g  = 1 + 10 · 9 + Σ (xᵢ² − 10 cos(4π xᵢ))    over i = 2, …, 10
f₂ = g (1 − √(f₁ / g))
```

f₂ is that of ZDT1, and g is Rastrigin's function of the other 9 variables, plus 1. g is 1, its
smallest, when x₂ = … = x₁₀ = 0. There, f₂ = 1 − √f₁ for f₁ from 0 to 1: the convex Pareto front of
ZDT1. With x₁ = 0.25 and the rest 0, the solution is on the front at (0.25, 0.5).

## What makes it hard

Local fronts. Each term xᵢ² + 10 − 10 cos(4π xᵢ) of g has 21 local minima in [−5, 5], near
xᵢ = 0, ±0.5, ±1, …, ±5, with values near xᵢ²: 0 at 0, 0.25 at ±0.5, 1 at ±1. Every choice of a
local minimum for each of the 9 variables gives a local minimum of g, and with it a local front:
the curve f₂ = g (1 − √(f₁ / g)) for that g. That makes 21⁹, about 8 × 10¹¹, local fronts, as
Zitzler, Deb and Thiele count them. The nearest to the true front has g ≈ 1.25, with one variable
at ±0.5. Near a local front, a small change to one variable makes g worse, so the search has to
make larger jumps, from basin to basin, to go on.

A long way down. A random solution has g near 166, 90 of it from the constant, so the first fronts
lie far above the true one, with f₂ in the hundreds.

A front that shrinks to a point. For a large g, the drop √(f₁ g) in f₂ is small beside the
differences in g, so a solution with f₁ = 0 and a smaller g beats most others. The front gathers
at f₁ = 0 until g is small, and has to spread again later.

## Representation

A `Real` genome of 10 genes, x₁ in [0, 1] and the rest in [−5, 5]: the vector x. The problem is
genoxide's `Zdt4`, whose fitness is the pair (f₁, f₂). In Python, `run` evaluates it in Rust, so
both versions print the same.

## Algorithm

Two algorithms, with the same settings: a population of 100, simulated binary crossover with η = 15
at genoxide's default rate of 0.9, and polynomial mutation with η = 20 at a rate of 1/10 per gene,
one gene per child on average. Each runs for 500 generations, twice the 250 of the NSGA-II paper,
and the example reports its front after 250 and after 500.

- NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary
  Computation 6(2): 182-197), as the ZDT1 example runs it. It sorts solutions into non-dominated
  fronts, and within a front prefers solutions with a larger crowding distance, a measure of the
  gap between their neighbors.
- SMS-EMOA (Beume, Naujoks and Emmerich, 2007, European Journal of Operational Research 181(3):
  1653-1669), in genoxide's generational form: 100 children a generation. From the last front
  that fits only in part, it removes the solutions that add the least hypervolume.

Polynomial mutation lets both jump between basins. With η = 20, a mutated gene moves by 1/22 of
its range on average, away from the bounds: about 0.45 for [−5, 5], near the 0.5 between
neighboring local minima.

## Output

A line per algorithm and budget: the size of the front, its IGD+ and its hypervolume. The last
line gives the whole front's hypervolume.

IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) is measured to 500 points of the optimal
front, evenly spaced in f₁. It averages, over those 500 points, the distance to the nearest point
of the found front, counting only the objectives in which the found point is worse. 0 means that
the found front covers the optimal one. Smaller is better.

The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4):
257-271) is the area that the front dominates, up to the reference point (1.1, 1.1). Larger is
better. A front on the local front with g ≈ 1.25 has at most 0.6931. For the whole true front, it
is 1.1 × 1.1 − 1/3 = 0.8767, as for ZDT1.

[The project page](https://tachsin.gr/projects/genoxide/examples/zdt4) plays this run back.

## Good results

A good front has 100 solutions spread from (0, 1) to (1, 0), an IGD+ near 0 and a hypervolume near
0.8767. No set of 100 points reaches that hypervolume: 100 points of the optimal front, evenly
spaced in f₁, give 0.8714 and an IGD+ of 0.0023.

Neither algorithm is caught on a local front in this run: ZDT4 shows here as slowness. The first
front has g from 97 to 187. While g falls, both fronts gather at f₁ near 0 for a while, NSGA-II's
around generations 32 to 40, SMS-EMOA's from 24 to 48. NSGA-II's front first has solutions below
the nearest local front, g = 1.25, at about generation 140. It has 100 solutions from generation
160, and from 176 all of them are below it but for at most three with f₁ below 10⁻²⁸: no other
solution has a smaller f₁, so none dominates them, whatever their g. SMS-EMOA is slower at first:
its front first goes below g = 1.25 at about generation 150, and is entirely below it, with 100
solutions, from 192.

After the paper's 250 generations, NSGA-II is nearly there: its front has g from about 1.002 to
1.004, an IGD+ of 0.0038 and a hypervolume of 0.8694. SMS-EMOA's front, with g from about 1.014
to 1.019, is still spreading from f₁ = 0: it reaches only f₁ = 0.73, for an IGD+ of 0.0294 and
0.8267, and f₁ = 1 at about generation 310. After 500, both are on the front: NSGA-II ends with g
from 1.0008 to 1.0017, but for one solution with f₁ below 10⁻³⁵ and g = 1.25, an IGD+ of 0.0032
and 0.8703, SMS-EMOA with an IGD+ of 0.0028 and 0.8711.

On seeds 1 to 5, NSGA-II has an IGD+ of 0.0038 to 0.0072 after 250 generations, and 0.0030 to
0.0034 after 500; SMS-EMOA 0.0028 to 0.0294, and 0.0024 to 0.0028. After 1,000 generations,
SMS-EMOA's hypervolume, 0.8716 to 0.8721, passes that of the 100 evenly spaced points, since it
places its points where they add the most area. With the same settings, SPEA2 has 0.0040 to 0.0078
after 250 generations, no better than NSGA-II on any seed, and MOEA/D 0.0041 to 0.0098.
