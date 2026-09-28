---
title: WFG5
category: multi-objective
summary: Minimize two objectives over 24 variables, with a concave front and deceptive parameters that lead every run to a front 0.05 behind it, with NSGA-II and SMS-EMOA.
reference: "Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation 10(5): 477-506."
reference_url: https://doi.org/10.1109/TEVC.2005.861417
optimum: "the front (f₁ / 2)² + (f₂ / 4)² = 1, a quarter ellipse from (0, 4) to (2, 0); hypervolume 3.3968 (reference point (2.2, 4.4))"
languages: [rust, python]
order: 130
family: WFG
---

# WFG5

## The problem

Huband, Hingston, Barone and While (2006) built a toolkit for test problems with any number of
objectives, and nine problems from it, WFG1 to WFG9 (their table XIV). An earlier version appeared
at EMO 2005 (Huband, Barone, While and Hingston, LNCS 3410: 280-295), and the authors released a
C++ implementation. genoxide's `Wfg5` follows the 2006 paper, and was checked against the EMO 2005
version and against the authors' C++ toolkit, version 2006.03.28, compiled and run for its tests.

Every WFG problem has n = k + l variables: k position parameters, which say where on the front a
solution lies, and l distance parameters, which say how far from it. The i-th variable zᵢ is in
[0, 2i]. The problem divides it by 2i, passes the values through a chain of transformations, and
reduces them to one value per objective. This example uses 2 objectives and the sizes the authors
recommend, k = 4 and l = 20: 24 variables. With yᵢ = zᵢ / 2i:

```text
all 24:
  yᵢ ← s_decept(yᵢ, 0.35, 0.001, 0.05)
     = |yᵢ − 0.35| / 0.001            within 0.001 of 0.35
       0.05 + 0.95 yᵢ / 0.349         below 0.349
       0.05 + 0.95 (1 − yᵢ) / 0.649   above 0.351
x₁ = the mean of y₁, …, y₄       (the position)
x₂ = the mean of y₅, …, y₂₄      (the distance)
f₁ = x₂ + 2 sin(x₁π/2)
f₂ = x₂ + 4 cos(x₁π/2)
```

Both objectives are minimized. The optimal solutions have every distance parameter at 0.35 × 2i,
so that x₂ = 0. The position parameters can be anything. The Pareto front is then
(f₁ / 2)² + (f₂ / 4)² = 1: a quarter of an ellipse, concave, from (0, 4) to (2, 0). The ideal
point is (0, 0) and the nadir point (2, 4). WFG4 and WFG6 to WFG9 have the same front, and differ
in the transformations before it.

Every other point is the front moved up by its distance x₂ in both objectives. From the objectives
alone, x₂ is the smaller root of ((f₁ − x₂) / 2)² + ((f₂ − x₂) / 4)² = 1. The example prints it.

## What makes it hard

Deception. The shift s_decept has its global minimum, 0, at 0.35, in a narrow V: it rises to 1 at
0.349 and 0.351. Outside the V, it falls in a straight line to 0.05 at 0 and at 1, the two
deceptive minima. Everywhere but in the V, the slope leads away from 0.35, towards a bound. A
solution with its distance parameters at the bounds is 0.05 behind the front, on a local front: the
optimal front moved up by 0.05 in both objectives.

A tiny target. A distance parameter beats 0.05 only within 0.00005 of 0.35: a window of 1/10,000
of its range. Landing in the V but outside that window makes the solution worse, not better. With
η = 20, polynomial mutation of a parameter at 0 lands in the window about twice in 10 million
tries, and of a parameter at 1, practically never. Twenty parameters must each find it.

The position is deceptive too. The position parameters get the same shift. At the bounds, their
mean is at least 0.05, so the end of the front at (0, 4) is out of reach: the nearest a solution
there gets is f₁ = 0.05 + 2 sin(0.025π) = 0.207. The other end, x₁ = 1, needs every position
parameter where the shift is 1: at the edges of the V, 0.349 or 0.351.

WFG5 is separable: each parameter can be changed alone. That doesn't help here, since each
parameter alone is deceptive.

## Representation

A `Real` genome of 24 genes, the i-th in [0, 2i]: the vector z. The problem is genoxide's
`Wfg5::<2>::default()`, with k = 4 and l = 20, whose fitness is the pair (f₁, f₂). In Python,
`gx.problems.Wfg5(objectives=2)`, which `run` evaluates in Rust, so both versions print the same.

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

SMS-EMOA is the algorithm that did best on WFG4, whose front is the same. Here it shows whether a
different selection helps against deception. Both are expected to be deceived, and the example
shows how far from the front they end.

## Output

Two lines per algorithm and budget. The first gives the size of the front, its IGD+ and its
hypervolume. The second gives the least and the largest distance x₂ of the front's points, and
how many of the population's 2,000 distance parameters (100 solutions, 20 each) are within 0.001 of
the bounds 0 and 1, and within 0.001 of 0.35, in the V. The last line gives the whole front's
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

The deceptive front, 0.05 behind, has a hypervolume of 2.15 × 4.35 − 2π = 3.0693, and 100 points
of it, those of `optimal_front(100)` moved up by 0.05, have 3.0335 and an IGD+ of 0.0631.

[The project page](https://tachsin.gr/projects/genoxide/examples/wfg5) plays this run back.

## Good results

A good front would have 100 solutions spread from (0, 4) to (2, 0), an IGD+ near 0 and a
hypervolume near 3.3968; 100 points of the optimal front give 3.3610 and an IGD+ of 0.0051. Neither
algorithm comes near. Both are deceived, as expected: they end on the deceptive front, 0.05 behind
the true one.

The first front, of 17 solutions, has a hypervolume of 0.9577. The distance parameters slide down
their slopes quickly: after 250 generations, 1,454 of NSGA-II's 2,000 are within 0.001 of a bound,
and 1,978 of SMS-EMOA's. None is in the V. NSGA-II's front is 0.0509 to 0.0603 behind, with an IGD+
of 0.0722 and a hypervolume of 2.9620; SMS-EMOA's 0.0502 to 0.0515 behind, with 0.0719 and 2.9681.

After 1,000 generations, the fronts sit on the deceptive front: SMS-EMOA's is 0.0500 to 0.0507
behind, with 1,993 of its distance parameters at the bounds, an IGD+ of 0.0681 and a hypervolume of
2.9909. NSGA-II's is 0.0500 to 0.0624 behind, with an IGD+ of 0.0704 and 2.9736. Still, no distance
parameter is in the V. Both are further from the true front than 100 points of the deceptive front,
because the position is deceptive too. NSGA-II's front begins at f₁ = 0.2070: all four position
parameters of that solution are at 1, to 0.00001. SMS-EMOA's begins at 0.1890: one of its position
parameters is 0.000017 from 0.35, in the window, and SMS-EMOA keeps it for the hypervolume it adds
at the end of the front. At the other end, near (2.05, 0.05), both have their position parameters
0.001 to 0.002 from 0.35, just at the rim of the V, where the shift is near 1, as that end needs.
The search finds the V, but only its rim.

The deception holds on every seed and for every algorithm tried. On seeds 1 to 5, after 4,000
generations, NSGA-II, SMS-EMOA, SPEA2 and MOEA/D (100 weight vectors, the same operators) all end
with the front 0.0500 behind at its closest point, and IGD+ values from 0.0650 to 0.0695. A longer
run spreads the points better along the deceptive front; it doesn't bring them closer to the true
one.
