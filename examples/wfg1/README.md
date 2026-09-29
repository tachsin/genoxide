---
title: WFG1
category: multi-objective
summary: Minimize two objectives over 24 variables, with a front of convex and mixed parts behind a flat region and a strong bias that double precision can't get past, with NSGA-II.
reference: "Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation 10(5): 477-506."
reference_url: https://doi.org/10.1109/TEVC.2005.861417
optimum: "the front f₁ = 2 (1 − cos(x₁π/2)), f₂ = 4 (1 − x₁ + sin(10πx₁) / 10π) for x₁ in [0, 1]; hypervolume 6.7857 (reference point (2.2, 4.4)); no genome reaches it in double precision: the closest front a genome can have is the front moved by 0.0695, hypervolume 6.3321"
languages: [rust, python]
order: 126
family: WFG
---

# WFG1

## The problem

Huband, Hingston, Barone and While (2006) built a toolkit for test problems with any number of
objectives, and nine problems from it, WFG1 to WFG9 (their table XIV). An earlier version appeared
at EMO 2005 (Huband, Barone, While and Hingston, LNCS 3410: 280-295), and the authors released a
C++ implementation. genoxide's `Wfg1` follows the 2006 paper, and was checked against the EMO 2005
version and against the authors' C++ toolkit, version 2006.03.28, compiled and run for its tests.

Every WFG problem has n = k + l variables: k position parameters, which say where on the front a
solution lies, and l distance parameters, which say how far from it. The i-th variable zᵢ is in
[0, 2i]. The problem divides it by 2i, passes the values through a chain of transformations, and
reduces them to one value per objective. This example uses 2 objectives and the sizes the authors
recommend, k = 4 and l = 20: 24 variables. With yᵢ = zᵢ / 2i:

```text
distance parameters (i = 5, …, 24):
  yᵢ ← |yᵢ − 0.35| / 0.35 below 0.35, (yᵢ − 0.35) / 0.65 above    (s_linear: 0.35 becomes 0)
  yᵢ ← 0.8 for yᵢ from 0.75 to 0.85, stretched to fit elsewhere     (b_flat)
all 24:
  yᵢ ← yᵢ^0.02                                                     (b_poly)
x₁ = Σ 2i yᵢ / Σ 2i over i = 1, …, 4       (the position)
x₂ = Σ 2i yᵢ / Σ 2i over i = 5, …, 24      (the distance)
f₁ = x₂ + 2 (1 − cos(x₁π/2))
f₂ = x₂ + 4 (1 − x₁ − cos(10πx₁ + π/2) / 10π)
```

Both objectives are minimized. The optimal solutions have every distance parameter at 0.35 × 2i,
so that x₂ = 0. The position parameters can be anything. The Pareto front is then the curve
f₁ = 2 (1 − cos(x₁π/2)), f₂ = 4 (1 − x₁ + sin(10πx₁) / 10π) for x₁ from 0 to 1, from (0, 4) to
(2, 0). f₁ is convex in x₁. f₂ falls in five steps: its slope is 0 at x₁ = 0.2, 0.4, 0.6 and 0.8,
where the front runs flat for a moment, and the front bends between convex and concave. The ideal
point is (0, 0) and the nadir point (2, 4).

## What makes it hard

A strong bias. Every value is raised to the power 0.02, and small values become large ones:
0.5^0.02 = 0.986, 10⁻⁸ becomes 0.69, and 10⁻¹⁶ becomes 0.48. A random genome has x₁ and the
distance x₂ both near 0.98: the first population sits near f = (2.9, 1.0), above the end of the
front at (2, 0). To reach x₁ = 0.5, the position parameters must be near 0.5⁵⁰ ≈ 10⁻¹⁵ of their
range; for x₁ = 0.1, near 10⁻⁵⁰. Most of the front, the part with small f₁ and large f₂, needs
position parameters pressed against their lower bounds. Huband et al. report that the NSGA-II of
their experiments (section IX) covers little of the front after 250 generations and converges
poorly even after 25,000, and they suggest that the bias is the cause.

The same power acts on the distance: close to 0.35 × 2i isn't enough. A distance parameter whose
shifted value is 10⁻³ still counts 0.87 in x₂'s weighted mean; at 10⁻⁸, 0.69. Only the exact
value counts 0. So x₂ falls in steps, each time a parameter lands on its value to the last bit.

A flat region. Before the bias, the shifted distance values from 0.75 to 0.85 all become 0.8: in
10% of each distance parameter's range, a change makes no difference to the objectives.

Uneven weights. The reductions weigh the i-th parameter by 2i, so the last distance parameter,
i = 24, counts 4.8 times as much as the first, i = 5.

No genome reaches the front. In double precision, zᵢ / 2i is never exactly 0.35 for some i: 3,
6, 12, 24, 48, 53, and more. The nearest value is off in the last bit, by about 10⁻¹⁶, and the bias
turns that into 0.48. The recommended sizes have three such distance parameters, i = 6, 12 and 24,
so every genome has x₂ ≥ 0.48 × (12 + 24 + 48) / 580 ≈ 0.0695. The closest a genome gets is the
front moved up by 0.0695 in both objectives. The authors' toolkit computes the same, and its
README warns that WFG1's bias can strain the precision of doubles. So neither indicator below can
reach its value for the true front: IGD+ can't fall below about 0.087, and the hypervolume can't
pass 6.3321, that of the whole moved front. The target of this example is therefore that closest
front.

## Representation

A `Real` genome of 24 genes, the i-th in [0, 2i]: the vector z. The problem is genoxide's
`Wfg1::<2>::default()`, with k = 4 and l = 20, whose fitness is the pair (f₁, f₂). In Python,
`gx.problems.Wfg1(objectives=2)`, which `run` evaluates in Rust, so both versions print the same.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation
6(2): 182-197), as the ZDT1 example runs it. It sorts solutions into non-dominated fronts, and
within a front prefers solutions with a larger crowding distance, a measure of the gap between
their neighbors. A population of 100, and polynomial mutation with η = 20 at a rate of 1/24 per
gene, one gene per child on average, in two runs:

- with the ZDT examples' simulated binary crossover (SBX), η = 15, at genoxide's default rate of
  0.9, for 2,500 generations: ten times the budget of the NSGA-II paper;
- with SBX at η = 2 for 20,000 generations, 2,000,000 evaluations. A smaller η puts children
  further from their parents.

The second run is the one that reaches the target. Why η matters is in Good results.

## Output

A line per run and budget: the size of the front, its IGD+ to the optimal front and to the closest
front, and its hypervolume. Then the closest front's own IGD+ and hypervolume, and the whole
front's hypervolume.

IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over the points of a
reference front, the distance to the nearest point of the found front, counting only the
objectives in which the found point is worse. Smaller is better. The first IGD+ is to 500 points of
the optimal front, evenly spread along it: it can't fall below about 0.087. The second, in
brackets, is to the same 500 points moved by 0.0695, the closest front, with f₁ divided by 2 and
f₂ by 4, the front's range: 0 for a front on the closest one.

The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4):
257-271) is the area that the front dominates, up to the reference point (2.2, 4.4), 1.1 times the
nadir point, as the ZDT examples use (1.1, 1.1). Larger is better. For the whole front, it is
6.7857, found numerically from a million points of the front.

The closest front: the genome with the distance parameters at 0.35 × 2i, the paper's optimal
solutions, has x₂ = 0.0695. The front moved by that much in both objectives has an IGD+ of 0.0868
to the optimal front, and 100 points of it, evenly spread, a hypervolume of 6.2985: about the best
that a run with 100 solutions can do.

[The project page](https://tachsin.gr/projects/genoxide/examples/wfg1) plays these runs back.

## Good results

The target is the closest front: an IGD+ to it of at most 0.01, with the objectives scaled to the
front's range, or a hypervolume of at least 99% of its 6.3321.

With η = 2, NSGA-II reaches it. After 20,000 generations its front has 100 solutions, an IGD+ of
0.0052 to the closest front and a hypervolume of 6.2507: 98.7% of the whole closest front's, and
99.2% of what 100 evenly spread points of it give. Its IGD+ to the true front, 0.0947, is 0.008
above the closest front's own. Along the way, the IGD+ to the closest front falls from 0.1196
after 2,500 generations to 0.053 after 4,000, 0.018 after 8,000, 0.0084 after 12,000 and 0.0058
after 16,000. Most of the front has x₂ = 0.0697, next to the smallest possible 0.0695, and it
reaches down to x₁ = 0.068, f₁ = 0.011, near the upper left end (0, 4) of the front.

It does so on every seed from 1 to 40: after 20,000 generations, the largest IGD+ to the closest
front is 0.0073, and the hypervolume is 98.3% to 99.3% of the closest front's. After 16,000
generations, every seed has reached it too, the largest 0.0081; after 12,000, 37 of the 40. η = 1
does about as well, all 40 seeds with at most 0.0075 after 20,000 generations.

With η = 15, NSGA-II stalls. The first front, of 15 solutions, lies at f₁ from 2.81 to 2.93 and
f₂ near 1: near the end (2, 0) of the front, and 0.9 above it. After 2,500 generations, it has an
IGD+ of 0.3793 (0.1345 to the closest front) and a hypervolume of 4.7459, against 0.3259 and
4.9842 with η = 2; its front doesn't reach f₁ below 0.70. Given the same 20,000 generations, it
ends with 0.0206 to the closest front on this seed, and reaches the target on only 1 of the seeds
1 to 16: from 0.0027 to 0.108.

The selection isn't what holds it back. The distance parameters must land on their values to the
last bit, since the bias counts a parameter off by 10⁻⁸ as 0.69. Once the population's distance
parameters are nearly equal, crossover between them changes little, and polynomial mutation, whose
steps are a few hundredths of the range, almost never lands on a value to the last bit. With
η = 15, on seeds 1 to 6, 1 to 4 distance parameters are still off by more than 10⁻¹⁵ after 20,000
generations, and most of the front has x₂ from 0.09 to 0.18. With η = 2, SBX makes larger steps
between parents that differ, and I take it that this keeps their values apart for longer, so that
the population closes in on the exact values before it collapses. Other selections don't do
better with η = 15: on seeds 1 to 5, NSGA-II ends 2,500 generations with an IGD+ of 0.33 to 0.44
to the true front; SMS-EMOA, which keeps the solutions that add the most hypervolume, 0.45 to 0.59;
SPEA2 0.32 to 0.44; and MOEA/D (100 weight vectors, SBX η = 20) 0.57 to 0.69.
