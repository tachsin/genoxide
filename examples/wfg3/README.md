---
title: WFG3
category: multi-objective
summary: Minimize two objectives over 24 variables, with a linear front and distance parameters that act in pairs, with NSGA-II and SMS-EMOA.
reference: "Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation 10(5): 477-506."
reference_url: https://doi.org/10.1109/TEVC.2005.861417
optimum: "the segment from (0, 4) to (2, 0); hypervolume 5.68 (reference point (2.2, 4.4))"
languages: [rust, python]
order: 115
---

# WFG3

## The problem

Huband, Hingston, Barone and While (2006) built a toolkit for test problems with any number of
objectives, and nine problems from it, WFG1 to WFG9 (their table XIV). An earlier version appeared
at EMO 2005 (Huband, Barone, While and Hingston, LNCS 3410: 280-295), and the authors released a
C++ implementation. genoxide's `Wfg3` follows the 2006 paper, and was checked against the EMO 2005
version and against the authors' C++ toolkit, version 2006.03.28, compiled and run for its tests.

Every WFG problem has n = k + l variables: k position parameters, which say where on the front a
solution lies, and l distance parameters, which say how far from it. The i-th variable zᵢ is in
[0, 2i]. The problem divides it by 2i, passes the values through a chain of transformations, and
reduces them to one value per objective. WFG3 has the transformations of WFG2 and a linear shape.
This example uses 2 objectives and the sizes the authors recommend, k = 4 and l = 20: 24
variables. With yᵢ = zᵢ / 2i:

```text
distance parameters (i = 5, …, 24):
  yᵢ ← |yᵢ − 0.35| / 0.35 below 0.35, (yᵢ − 0.35) / 0.65 above    (s_linear: 0.35 becomes 0)
  each pair (a, b) = (y₅, y₆), (y₇, y₈), …, (y₂₃, y₂₄) becomes
    (a + b + 2 |a − b|) / 3                                         (r_nonsep)
x₁ = the mean of y₁, …, y₄                (the position)
x₂ = the mean of the 10 pair values       (the distance)
f₁ = x₂ + 2 x₁
f₂ = x₂ + 4 (1 − x₁)
```

Both objectives are minimized. The optimal solutions have every distance parameter at 0.35 × 2i,
so that x₂ = 0. The Pareto front is then the segment from (0, 4) to (2, 0), where
f₁ / 2 + f₂ / 4 = 1. Any other solution has f₁ / 2 + f₂ / 4 = 1 + 0.75 x₂: the distance moves it
away from the front, along the diagonal. The ideal point is (0, 0) and the nadir point (2, 4).

WFG3 was meant to be degenerate: to have a front of lower dimension than usual, a line for any
number of objectives. With 2 objectives, that means nothing, since every front of 2 objectives is
a curve, and WFG3's is the segment above. With 3 or more, the paper's line is optimal, but
Ishibuchi, Masuda and Nojima (2016, IEEE Transactions on Evolutionary Computation 20(5): 807-813)
show that it isn't the whole front: solutions off the optimal distance are optimal too. With 3
objectives, for instance, (3, 1, 1), with the distance x₃ = 1, is beaten by no other solution.
genoxide's tests confirm that point; its docs note that the letter itself hasn't yet been checked.
genoxide gives no front, ideal point or nadir point for WFG3 with 3 or more objectives, since the
front isn't known. This example runs 2 objectives, where the front is known exactly.

## What makes it hard

Distance parameters in pairs. The reduction (a + b + 2 |a − b|) / 3 is 0 only when both values of
a pair are 0, and it punishes their difference. With b fixed above 0, the best a is b, not 0:
moving one parameter of a pair to its optimum alone makes the pair worse. The problem is
non-separable: the best value of a variable depends on another. Huband et al. call this
reduction easier than the one of WFG6 and WFG9, which ties all the distance parameters together.

Little else. The front is connected and straight, and the position is the plain mean of four
values, without a bias. Huband et al. count WFG3 among the problems their NSGA-II solved easily
(section IX). What is left to see is how close to the front and how evenly an algorithm puts its
solutions: on a straight front, gaps and clusters are easy to see.

## Representation

A `Real` genome of 24 genes, the i-th in [0, 2i]: the vector z. The problem is genoxide's
`Wfg3::<2>::default()`, with k = 4 and l = 20, whose fitness is the pair (f₁, f₂). In Python,
`gx.problems.Wfg3(objectives=2)`, which `run` evaluates in Rust, so both versions print the same.

## Algorithm

Two algorithms, with the settings of the ZDT examples: a population of 100, simulated binary
crossover with η = 15 at genoxide's default rate of 0.9, and polynomial mutation with η = 20 at a
rate of 1/24 per gene, one gene per child on average. Each runs for 1,000 generations, and the
example reports its front after 250 and after 1,000.

- NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary
  Computation 6(2): 182-197), as the ZDT1 example runs it. It sorts solutions into non-dominated
  fronts, and within a front prefers solutions with a larger crowding distance, a measure of the
  gap between their neighbors.
- SMS-EMOA (Beume, Naujoks and Emmerich, 2007, European Journal of Operational Research 181(3):
  1653-1669), in genoxide's generational form: 100 children a generation. From the last front
  that fits only in part, it removes the solutions that add the least hypervolume. A solution's
  share of the hypervolume grows with the gaps to its neighbors, so it keeps the solutions apart.

## Output

A line per algorithm and budget: the size of the front, its IGD+, its hypervolume, and the
largest gap between neighbors on it. The last line gives the whole front's hypervolume and the
gap between 100 evenly spaced points of it.

IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) is measured to 500 points of the
optimal front, evenly spaced. It averages, over those 500 points, the distance to the nearest
point of the found front, counting only the objectives in which the found point is worse. 0 means
that the found front covers the optimal one. Smaller is better.

The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4):
257-271) is the area that the front dominates, up to the reference point (2.2, 4.4), 1.1 times the
nadir point, as the ZDT examples use (1.1, 1.1). Larger is better. For the whole front, it is
2.2 × 4.4 less the triangle under the segment, 4: 5.68.

The largest gap is the largest distance between two neighbors, with the front sorted by f₁. The
segment is √20 ≈ 4.472 long, so 100 evenly spaced points on it are 0.0452 apart.

[The project page](https://tachsin.gr/projects/genoxide/examples/wfg3) plays this run back.

## Good results

A good front has 100 solutions spread from (0, 4) to (2, 0), an IGD+ near 0 and a hypervolume near
5.68. No set of 100 points reaches that hypervolume: 100 evenly spaced points of the segment give
5.6396 and an IGD+ of 0.0067.

Both algorithms converge quickly. The first front has 18 solutions and a hypervolume of 2.68.
NSGA-II's has 100 solutions from about generation 32, SMS-EMOA's from about 48. After 250
generations, NSGA-II has an IGD+ of 0.0339 and a hypervolume of 5.4776, SMS-EMOA 0.0133 and 5.5891.

SMS-EMOA gets closer and spreads more evenly. After 1,000 generations, its solutions have a
distance x₂ of 0.0012 to 0.0022, and its gaps are 0.025 to 0.063: an IGD+ of 0.0083 and a
hypervolume of 5.6289. NSGA-II's solutions have x₂ from 0.0056 to 0.021, and its gaps range from
0.0005 to 0.123, near copies in some places and holes in others: an IGD+ of 0.0176 and a
hypervolume of 5.5732. From the last front that fits only in part, NSGA-II keeps the solutions
with the largest crowding distances, computed once for the front: it can drop two neighbors at
once, and leave a hole where they were.

On seeds 1 to 5, after 1,000 generations, NSGA-II has an IGD+ of 0.0145 to 0.0177 and a
hypervolume of 5.5729 to 5.5931, SMS-EMOA 0.0083 to 0.0103 and 5.6158 to 5.6289. With the same
settings, SPEA2 has 0.0118 to 0.0178, and MOEA/D (100 weight vectors, SBX η = 20) 0.0105 to
0.0125.
