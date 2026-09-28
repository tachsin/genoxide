---
title: WFG4
category: multi-objective
summary: Minimize two objectives over 24 variables, with a concave front behind many local fronts, with NSGA-II and SMS-EMOA.
reference: "Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation 10(5): 477-506."
reference_url: https://doi.org/10.1109/TEVC.2005.861417
optimum: "the front (f₁ / 2)² + (f₂ / 4)² = 1, a quarter ellipse from (0, 4) to (2, 0); hypervolume 3.3968 (reference point (2.2, 4.4))"
languages: [rust, python]
order: 109
---

# WFG4

## The problem

Huband, Hingston, Barone and While (2006) built a toolkit for test problems with any number of
objectives, and nine problems from it, WFG1 to WFG9 (their table XIV). An earlier version appeared
at EMO 2005 (Huband, Barone, While and Hingston, LNCS 3410: 280-295), and the authors released a
C++ implementation. genoxide's `Wfg4` follows the 2006 paper, and was checked against the EMO 2005
version and against the authors' C++ toolkit, version 2006.03.28, compiled and run for its tests.

Every WFG problem has n = k + l variables: k position parameters, which say where on the front a
solution lies, and l distance parameters, which say how far from it. The i-th variable zᵢ is in
[0, 2i]. The problem divides it by 2i, passes the values through a chain of transformations, and
reduces them to one value per objective. This example uses 2 objectives and the sizes the authors
recommend, k = 4 and l = 20: 24 variables. With yᵢ = zᵢ / 2i:

```text
all 24:
  yᵢ ← s_multi(yᵢ, 30, 10, 0.35)
     = (1 − cos(122π d) + 40 d²) / 12, with d = (0.35 − yᵢ) / 0.7 below 0.35,
                                            and (yᵢ − 0.35) / 1.3 above
x₁ = the mean of y₁, …, y₄       (the position)
x₂ = the mean of y₅, …, y₂₄      (the distance)
f₁ = x₂ + 2 sin(x₁π/2)
f₂ = x₂ + 4 cos(x₁π/2)
```

Both objectives are minimized. The shift s_multi is 0 at 0.35 and 1 at 0 and 1. The optimal
solutions have every distance parameter at 0.35 × 2i, so that x₂ = 0. The position parameters can
be anything. The Pareto front is then (f₁ / 2)² + (f₂ / 4)² = 1: a quarter of an ellipse, concave,
from (0, 4) to (2, 0). The ideal point is (0, 0) and the nadir point (2, 4). WFG5 to WFG9 have the
same front, and differ in the transformations before it.

Every other point is the front moved up by its distance x₂ in both objectives. From the objectives
alone, x₂ is the smaller root of ((f₁ − x₂) / 2)² + ((f₂ − x₂) / 4)² = 1. The example prints it.

## What makes it hard

Many local fronts. Each distance parameter's shift has its global minimum, 0, at 0.35, and 60
local minima: 30 below 0.35, 0.0115 apart, and 30 above, 0.0213 apart. The j-th from 0.35, on
either side, has a value of about 0.000895 j², and between neighbors are hills of 0.167 or more.
Every choice of a minimum for each of the 20 distance parameters, 61²⁰ ≈ 5 × 10³⁵ of them, puts
the solutions on a local front: the optimal front moved up by the mean of the chosen minima's
values.

The local fronts are close, and hard to leave. One distance parameter at the nearest local minimum
puts a solution 0.000895 / 20 ≈ 0.00004 behind the front. To reach the global minimum, that
parameter has to cross a hill, and land near the bottom of a basin 0.016 wide, from 0.3443 to
0.3607. A small step makes the solution worse; a large one rarely lands at the bottom.

The position is multimodal too. The position parameters get the same shift, so x₁ = 0, the end of
the front at (0, 4), needs all four at 0.35, and each point between has its own basins.

Separable, though. Each parameter can be improved alone, and a solution that improves one keeps
the others. The search is slowed, not trapped.

## Representation

A `Real` genome of 24 genes, the i-th in [0, 2i]: the vector z. The problem is genoxide's
`Wfg4::<2>::default()`, with k = 4 and l = 20, whose fitness is the pair (f₁, f₂). In Python,
`gx.problems.Wfg4(objectives=2)`, which `run` evaluates in Rust, so both versions print the same.

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

With η = 20, a mutated gene moves by 1/22 of its range on average, away from the bounds: about
0.045 in y, more than the width of a basin. A mutation can jump a hill, but seldom lands at the
bottom of the next basin. The two algorithms differ in what they keep: a solution a little closer
to the front adds a little hypervolume, which SMS-EMOA counts, while NSGA-II keeps any
non-dominated solution, however far behind, and prefers those that fill gaps.

## Output

Two lines per algorithm and budget. The first gives the size of the front, its IGD+ and its
hypervolume. The second gives the least and the largest distance x₂ of the front's points, and
where the population's 2,000 distance parameters (100 solutions, 20 each) are: at the global
minimum of their shift, at the local minima next to it (j = 1), or further. The last line gives the
whole front's hypervolume.

IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) is measured to 500 points of the
optimal front, those of genoxide's `optimal_front`: evenly spaced points of the line from (1, 0) to
(0, 1), moved onto the unit circle and stretched by 2 and 4. It averages, over those 500 points,
the distance to the nearest point of the found front, counting only the objectives in which the
found point is worse. 0 means that the found front covers the optimal one. Smaller is better.

The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4):
257-271) is the area that the front dominates, up to the reference point (2.2, 4.4): 1.1 times the
nadir point, as the ZDT examples use (1.1, 1.1). Larger is better. For the whole front, it is the
rectangle less the quarter ellipse, 2.2 × 4.4 − 2π = 3.3968.

[The project page](https://tachsin.gr/projects/genoxide/examples/wfg4) plays this run back.

## Good results

A good front has 100 solutions spread from (0, 4) to (2, 0), an IGD+ near 0 and a hypervolume near
3.3968. No set of 100 points reaches that: the 100 points of `optimal_front(100)` give 3.3610 and
an IGD+ of 0.0051.

The first front, of 26 solutions, has a hypervolume of 1.4236. After 100 generations, NSGA-II's
has 3.1881 and SMS-EMOA's 3.1364; from there, both close in on the front slowly, as the distance
parameters move from basin to basin. After 250 generations, NSGA-II's front is 0.0071 to 0.0221
behind the optimal one, with an IGD+ of 0.0178 and a hypervolume of 3.2691. Of its population's
distance parameters, 365 are at the global minimum and 639 at the local minima next to it.
SMS-EMOA is ahead: 0.0019 to 0.0167 behind, an IGD+ of 0.0102, and 593 at the global minimum.

After 1,000 generations, NSGA-II has an IGD+ of 0.0121 and a hypervolume of 3.3144. Its front is
still up to 0.0184 behind at some points, which it keeps because they fill gaps, and 662 of its
distance parameters are two or more minima from the global one. SMS-EMOA's front is 0.0007 to
0.0021 behind, with an IGD+ of 0.0055 and a hypervolume of 3.3563, near those of the 100 points of
the optimal front. Only 101 of its distance parameters are two or more minima away; 939 are at the
global minimum and 960 at the next ones, each of which adds about 0.00004 to its solution's
distance.

On seeds 1 to 5, NSGA-II has an IGD+ of 0.0178 to 0.0226 after 250 generations, and 0.0093 to
0.0121 after 1,000; SMS-EMOA 0.0090 to 0.0118, and 0.0053 to 0.0057. After 4,000 generations,
SMS-EMOA's hypervolume, 3.3637 to 3.3651, passes that of the 100 points of `optimal_front(100)`,
with an IGD+ of 0.0049 to 0.0050, while NSGA-II is at 0.0080 to 0.0088. With the same settings,
SPEA2 is between the two, with 0.0080 to 0.0110 after 1,000 generations. MOEA/D (100 weight
vectors, the same operators) gets closest to the front, its whole front within 0.0011 of it after
4,000 generations, but spreads its points less evenly: an IGD+ of 0.0061 to 0.0064.
