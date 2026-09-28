---
title: WFG7
category: multi-objective
summary: Minimize two objectives whose concave front is reached through 24 parameters, where the distance parameters bias the position ones, with NSGA-II and SMS-EMOA.
reference: "Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation 10(5): 477-506."
reference_url: https://doi.org/10.1109/TEVC.2005.861417
optimum: "the quarter ellipse (f₁/2)² + (f₂/4)² = 1; hypervolume 3.3968 (reference point (2.2, 4.4))"
languages: [rust, python]
order: 132
family: WFG
---

# WFG7

## The problem

Huband, Hingston, Barone and While (2006), of the Walking Fish Group, built a toolkit for test
problems with any number of objectives, and nine problems from it, WFG1 to WFG9. Each problem
passes its variables through a chain of transformations down to a few values, then turns those
into the objectives. genoxide's `Wfg7` follows the paper's table XIV, with the transformations of
its table XI and the shapes of its table X. It was checked against the paper as published, against
its first version (Huband, Barone, While and Hingston, 2005, EMO 2005, LNCS 3410: 280-295, the
authors' corrected version), and against the authors' C++ toolkit, version 2006.03.28, whose
values genoxide's tests match.

This example uses 2 objectives and the recommended sizes: k = 4 position parameters and l = 20
distance parameters, 24 variables zᵢ in [0, 2i]. The problem first divides each by its upper
bound, yᵢ = zᵢ / 2i, then:

1. biases each position parameter by the mean of the parameters after it:
   yᵢ ← b_param(yᵢ, mean(yᵢ₊₁, …, y₂₄)) for i = 1, …, 4;
2. shifts each distance parameter so that 0.35 maps to 0: yᵢ ← s_linear(yᵢ, 0.35), which is
   |yᵢ − 0.35| divided by 0.35 below 0.35 and by 0.65 above, for i = 5, …, 24;
3. reduces each group to its mean: the position x₁ = mean(y₁, …, y₄), and the distance
   x₂ = mean(y₅, …, y₂₄).

Both objectives are minimized:

```text
f₁ = x₂ + 2 sin(x₁ π/2)
f₂ = x₂ + 4 cos(x₁ π/2)
```

The bias raises its value to a power set by u, the mean of the parameters after it:
b_param(y, u) = y^e, with e = 0.02 + 1.96 u for u up to 0.5, and e = 1 + 49 (2u − 1) above,
from 0.02 at u = 0 through 1 at u = 0.5 to 50 at u = 1. (Table XI writes it with the constants
A = 0.98/49.98, B = 0.02 and C = 50.)

The Pareto front is where the distance x₂ is 0: every distance parameter at 0.35, zᵢ = 0.35 × 2i.
There, (f₁/2)² + (f₂/4)² = 1, a quarter ellipse from (0, 4) to (2, 0). With x₁ = 0.5, the
solution is on the front at (1.4142, 2.8284). The ideal point is (0, 0) and the nadir point
(2, 4).

## What makes it hard

The position parameters decide where on the front a solution lies. The distance parameters decide
how far from the front it is: x₂ adds the same amount to both objectives. In WFG1 to WFG6, the
two groups are transformed apart. In WFG7, the bias ties them: the mean u of the parameters after a
position parameter, 20 of them distance parameters, sets the power the position parameter is
raised to. So the same position parameters put a solution elsewhere on the front as the distance
parameters change.

A check of 20,000 random genomes shows how much. In the random genomes, u is about 0.5 and often
above it, so the powers are often large and push positions towards 0: 72% of them have x₁ below
0.5. With the same position parameters and the distance parameters moved to their optimum 0.35, u
is about 0.35 and the power about 0.7, which pushes positions up: only 27% have x₁ below 0.5. A
search that converges moves its solutions along the front on the way, and has to spread them
again.

That is the whole difficulty, and it is a mild one. The optimum of each distance parameter, 0.35,
doesn't depend on the others (the problem is separable), and s_linear has a single minimum there
(unimodal). The bias changes where the solutions are on the front, not how hard the front is to
reach. WFG8 and WFG9 put the same bias on distance parameters, and there it does.

## Representation

A `Real` genome of 24 genes, the i-th in [0, 2i]: the vector z. The problem is genoxide's
`Wfg7::<2>::default()`, whose fitness is the pair (f₁, f₂). In Python,
`gx.problems.Wfg7(2)` gives the same problem, and `run` evaluates it in Rust, so both versions
print the same.

## Algorithm

Two algorithms, with the settings of the ZDT examples: a population of 100, simulated binary
crossover with η = 15 at genoxide's default rate of 0.9, and polynomial mutation with η = 20 at a
rate of 1/24 per gene, one gene per child on average. Each runs for 1,000 generations, and the
example reports its front after 250, 25,000 evaluations, and after 1,000.

- NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary
  Computation 6(2): 182-197), as the ZDT examples run it. It sorts solutions into non-dominated
  fronts, and within a front prefers solutions with a larger crowding distance, a measure of the
  gap between their neighbors.
- SMS-EMOA (Beume, Naujoks and Emmerich, 2007, European Journal of Operational Research 181(3):
  1653-1669), in genoxide's generational form: 100 children a generation. From the last front
  that fits only in part, it removes the solutions that add the least hypervolume.

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

[The project page](https://tachsin.gr/projects/genoxide/examples/wfg7) plays this run back.

## Good results

A good front has 100 solutions spread from (0, 4) to (2, 0), distances near 0, an IGD+ near 0 and
a hypervolume near 3.3968. No set of 100 points reaches that hypervolume: the 100 points of
`optimal_front(100)` give 3.3610 and an IGD+ of 0.0051.

Both algorithms converge. By generation 48, both fronts are within 0.07 of the true one; the
first frames of the plot show both fronts come down while they spread out along it. After 250
generations, NSGA-II's front has a mean distance of 0.0056, an IGD+ of 0.0113 and a hypervolume
of 3.3125. SMS-EMOA's is closer and more even, with a mean distance of 0.0012, all its points
within 0.0032, an IGD+ of 0.0063 and 3.3468.

After 1,000 generations, SMS-EMOA's front is on the true one, with a mean distance of 0.0002 and
the largest 0.0015. Its IGD+, 0.0049, and its hypervolume, 3.3646, pass those of the 100 points of
`optimal_front(100)`, since it places its points where they add the most area. NSGA-II's mean
distance falls to 0.0025, but a few points stay up to 0.0158 away. A point off the front stays in
NSGA-II's first front as long as no other point beats it in both objectives, and within a front,
NSGA-II looks only at the gaps between neighbors. SMS-EMOA ranks the points of a front by the
hypervolume each adds, and a point that moves onto the front adds more.

On seeds 1 to 10, NSGA-II has an IGD+ of 0.0108 to 0.0130 after 250 generations and 0.0077 to
0.0096 after 1,000, with mean distances of 0.0053 to 0.0068 and 0.0025 to 0.0034. SMS-EMOA has
0.0063 to 0.0083 and 0.0049 to 0.0051, with mean distances of 0.0007 to 0.0017 and 0.0001 to
0.0004. For comparison, WFG8 and WFG9, whose biases reach the distance parameters, stop far from
their fronts with the same settings: see their pages.
