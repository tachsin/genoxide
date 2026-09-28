---
title: WFG6
category: multi-objective
summary: Minimize two objectives over 24 variables, with a concave front behind a non-separable reduction that rewards agreeing parameters, with NSGA-II and SMS-EMOA.
reference: "Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation 10(5): 477-506."
reference_url: https://doi.org/10.1109/TEVC.2005.861417
optimum: "the front (f₁ / 2)² + (f₂ / 4)² = 1, a quarter ellipse from (0, 4) to (2, 0); hypervolume 3.3968 (reference point (2.2, 4.4))"
languages: [rust, python]
order: 121
family: WFG
---

# WFG6

## The problem

Huband, Hingston, Barone and While (2006) built a toolkit for test problems with any number of
objectives, and nine problems from it, WFG1 to WFG9 (their table XIV). An earlier version appeared
at EMO 2005 (Huband, Barone, While and Hingston, LNCS 3410: 280-295), and the authors released a
C++ implementation. genoxide's `Wfg6` follows the 2006 paper, and was checked against the EMO 2005
version and against the authors' C++ toolkit, version 2006.03.28, compiled and run for its tests.

Every WFG problem has n = k + l variables: k position parameters, which say where on the front a
solution lies, and l distance parameters, which say how far from it. The i-th variable zᵢ is in
[0, 2i]. The problem divides it by 2i, passes the values through a chain of transformations, and
reduces them to one value per objective. This example uses 2 objectives and the sizes the authors
recommend, k = 4 and l = 20: 24 variables. With yᵢ = zᵢ / 2i:

```text
distance parameters (i = 5, …, 24):
  yᵢ ← (0.35 − yᵢ) / 0.35 below 0.35, (yᵢ − 0.35) / 0.65 above    (s_linear: 0.35 becomes 0)
x₁ = r_nonsep(y₁, …, y₄)         (the position)
x₂ = r_nonsep(y₅, …, y₂₄)        (the distance)
  r_nonsep of m values = (Σ yⱼ + Σ |yⱼ − yₖ| over the m (m − 1) pairs j ≠ k) / c,
                         capped at 1, with c = 10 for m = 4 and c = 210 for m = 20
f₁ = x₂ + 2 sin(x₁π/2)
f₂ = x₂ + 4 cos(x₁π/2)
```

Both objectives are minimized. r_nonsep is the paper's non-separable reduction, here with its
degree of non-separability equal to the number of values, so that every value is compared with
every other. The optimal solutions have every distance parameter at 0.35 × 2i: every shifted value
is then 0, and so is x₂. The position parameters can be anything. The Pareto front is then
(f₁ / 2)² + (f₂ / 4)² = 1: a quarter of an ellipse, concave, from (0, 4) to (2, 0). The ideal
point is (0, 0) and the nadir point (2, 4). WFG4, WFG5 and WFG7 to WFG9 have the same front, and
differ in the transformations before it.

Every other point is the front moved up by its distance x₂ in both objectives. From the objectives
alone, x₂ is the smaller root of ((f₁ − x₂) / 2)² + ((f₂ − x₂) / 4)² = 1. The example prints it.

## What makes it hard

The distance parameters count together. x₂ adds up the 20 shifted values, and the 380 differences
between them. In a random genome, the differences make up most of it: each is 1/3 on average, so
the sum is about 10 + 380 / 3 ≈ 137, and x₂ about 0.65.

Agreeing is easy. The parameters can remove the differences by moving towards each other, one at a
time. If they agree on a shifted value v, x₂ = 20v / 210 = v / 10.5: about 0.048 for v = 0.5, the
average of a random start.

Moving on is hard. From 20 equal values, moving one of them towards 0 by δ lowers the sum by δ, but
opens 38 differences of δ: x₂ grows by 37δ / 210. Every change of one parameter towards the optimum
makes the solution worse. Only a move of all 20 together, by the same amount, lowers x₂, by
20δ / 210. The solutions have to follow a narrow valley along the diagonal of a 20-dimensional
space. In y, the parameters below 0.35 have to rise while those above it fall.

The position is non-separable too. x₁ = 0, the end of the front at (0, 4), needs all four position
parameters at 0; x₁ = 1, the end at (2, 0), needs them to disagree: two at 0 and two at 1.

## Representation

A `Real` genome of 24 genes, the i-th in [0, 2i]: the vector z. The problem is genoxide's
`Wfg6::<2>::default()`, with k = 4 and l = 20, whose fitness is the pair (f₁, f₂). In Python,
`gx.problems.Wfg6(objectives=2)`, which `run` evaluates in Rust, so both versions print the same.

## Algorithm

Two algorithms, with the settings of the ZDT examples: a population of 100, simulated binary
crossover with η = 15 at genoxide's default rate of 0.9, and polynomial mutation with η = 20 at a
rate of 1/24 per gene, one gene per child on average. Each runs for 1,000 generations, and the
example reports its front after 250, the NSGA-II paper's budget, and after 1,000.

- NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary
  Computation 6(2): 182-197), as the ZDT1 example runs it. It sorts solutions into non-dominated
  fronts, and within a front prefers solutions with a larger crowding distance, a measure of the
  gap between their neighbors.
- SMS-EMOA (Beume, Naujoks and Emmerich, 2007, European Journal of Operational Research 181(3):
  1653-1669), in genoxide's generational form: 100 children a generation. From the last front
  that fits only in part, it removes the solutions that add the least hypervolume.

Both select, and neither changes how children are made. The operators work gene by gene. Polynomial
mutation changes about one gene per child: a move of one parameter, which the valley punishes. SBX
recombines each gene with probability 1/2, with its own random spread, and barely changes genes
that are nearly equal in both parents. SMS-EMOA, which did best on WFG4, shows whether a different
selection helps.

## Output

Two lines per algorithm and budget. The first gives the size of the front, its IGD+ and its
hypervolume. The second gives the least and the largest distance x₂ of the front's points, and the
lowest and highest of the population's 2,000 shifted distance parameters (100 solutions, 20
each): how far they are from agreeing, and from 0. The last line gives the whole front's
hypervolume.

IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) is measured to 500 points of the
optimal front, those of genoxide's `optimal_front`: evenly spaced points of the line from (1, 0) to
(0, 1), moved onto the unit circle and stretched by 2 and 4. It averages, over those 500 points,
the distance to the nearest point of the found front, counting only the objectives in which the
found point is worse. 0 means that the found front covers the optimal one. Smaller is better.

The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4):
257-271) is the area that the front dominates, up to the reference point (2.2, 4.4): 1.1 times the
nadir point, as the ZDT examples use (1.1, 1.1). Larger is better. For the whole front, it is the
rectangle less the quarter ellipse, 2.2 × 4.4 − 2π = 3.3968.

[The project page](https://tachsin.gr/projects/genoxide/examples/wfg6) plays this run back.

## Good results

A good front has 100 solutions spread from (0, 4) to (2, 0), an IGD+ near 0 and a hypervolume near
3.3968; 100 points of the optimal front, those of `optimal_front(100)`, give 3.3610 and an IGD+ of
0.0051. Neither algorithm gets there: both find the valley, and stop in it.

The first front, of 20 solutions, is 0.45 to 0.62 behind the optimal one, with a hypervolume of
0.5410. Its shifted distance parameters range from 0 to 1, with a mean of 0.50. The first 100
generations make them agree. NSGA-II's range from 0.17 to 0.81 at generation 25, 0.40 to 0.66 at
50 and 0.46 to 0.60 at 100, their mean staying at 0.50 to 0.51, and the front comes to 0.060
behind at its closest. SMS-EMOA's agree at a lower value: their mean drifts from 0.50 to 0.42 by
generation 100, and stays there.

After 250 generations, NSGA-II's front is 0.0517 to 0.0625 behind, with its shifted distance
parameters from 0.47 to 0.54, an IGD+ of 0.0688 and a hypervolume of 2.9932. SMS-EMOA's is 0.0421
to 0.0507 behind, with an IGD+ of 0.0625 and 3.0187. From there, little changes. After 1,000,
SMS-EMOA's 2,000 shifted values are all between 0.4159 and 0.4205, and its front is 0.0406 to
0.0413 behind, near 0.418 / 10.5: every solution has nearly the same distance parameters, about
0.203 or 0.623 in y, 42% of the way from 0.35 to a bound. Its IGD+ is 0.0522 and its hypervolume
3.0894. NSGA-II's front is 0.0486 to 0.0659 behind, with an IGD+ of 0.0632 and 3.0316.

On seeds 1 to 5, after 1,000 generations, NSGA-II has an IGD+ of 0.0436 to 0.0704 and SMS-EMOA
0.0450 to 0.0620. Where a run stalls depends on where its parameters first agree. After 4,000
generations, the ranges are 0.0411 to 0.0691 and 0.0443 to 0.0611, and no run of NSGA-II,
SMS-EMOA, SPEA2 or MOEA/D (100 weight vectors, the same operators) has any point of its front
closer than 0.030 to the optimal one. The operators matter more than the selection: with blend
crossover (BLX-α, α = 0.5) in place of SBX, NSGA-II ends seeds 1 to 3 with an IGD+ of 0.0275 to
0.0493 after 1,000 generations, its front 0.016 to 0.035 behind at its closest; closer, but not
there.
