---
title: Viennet 1
category: multi-objective
summary: Minimize three objectives of two variables, each a squared distance to a point, whose Pareto front is a curved triangle, with NSGA-III.
reference: "Viennet, R., Fonteix, C. and Marc, I. (1996). Multicriteria optimization using a genetic algorithm for determining a Pareto set. International Journal of Systems Science 27(2): 255-260."
reference_url: https://doi.org/10.1080/00207729608929211
optimum: "the image of the triangle with corners (0, 1), (0, −1) and (1, 0); hypervolume about 33.52 (reference point (4.4, 5.4, 4.2))"
languages: [rust, python]
order: 162
family: Viennet
---

# Viennet 1

## The problem

Viennet, Fonteix and Marc (1996) posed three problems with two variables and three objectives. This
is the first, VNT1. genoxide's `Viennet1` uses the definition and bounds that Van Veldhuizen
restates (1999, PhD thesis AFIT/DS/ENG/99-01, Air Force Institute of Technology, table B.1);
genoxide hasn't yet checked them against the original paper.

Over x₁ and x₂ in [−2, 2], all three objectives are minimized:

```text
f₁ = x₁² + (x₂ − 1)²
f₂ = x₁² + (x₂ + 1)² + 1
f₃ = (x₁ − 1)² + x₂² + 2
```

Each objective is the squared distance from x to a point, plus a constant: f₁ to (0, 1), f₂ to
(0, −1) and f₃ to (1, 0). Each point is its objective's minimum: f₁ = 0 at (0, 1), f₂ = 1 at
(0, −1) and f₃ = 2 at (1, 0).

The optimal trade-offs are the points of the triangle with these three corners. From a point
outside it, moving to the nearest point of the triangle brings x closer to all three corners, so it
improves every objective. Inside it, every step moves away from at least one corner. The Pareto
front is the triangle's image: a curved triangle in the space of the objectives, with corners
(0, 5, 4), (4, 1, 4) and (2, 3, 2), the images of (0, 1), (0, −1) and (1, 0). At the triangle's
center, (1/3, 0), the objectives are (1.11, 2.11, 2.44).

## What makes it hard

VNT1 is the easiest of Viennet's three problems. What it tests is a front that is a surface, not a
curve. Covering a surface evenly takes many more solutions than covering a curve, and an algorithm
has to spread them in three directions at once.

The surface is curved, and the objectives have different ranges on it: f₁ and f₂ go from their
minimum to 4 above it, f₃ only 2. An algorithm that spreads solutions by distances between raw
objective values crowds them along f₁ and f₂.

The optimal solutions fill a triangle of area 1 in a box of area 16, so a random solution is optimal
once in 16 draws. The initial population of the first run has 18 non-dominated solutions, and only 8
of them are optimal.

A point just outside the triangle is worse than its nearest point in the triangle, but better than
most other points in at least one objective. Only the points of a thin region between it and the
triangle beat it in all three. If the population holds none of them, the point stays on the
population's front, though it isn't optimal.

## Representation

A `Real` genome of 2 genes in [−2, 2]: the point x. The problem is genoxide's `Viennet1`, whose
fitness is the three objectives. In Python, `run` evaluates it in Rust.

## Algorithm

NSGA-III (Deb and Jain, 2014, IEEE Transactions on Evolutionary Computation 18(4): 577-601). Like
NSGA-II, it ranks solutions into non-dominated fronts, and parents and children compete for the
next population. Instead of crowding distance, it spreads the front along reference directions. It
first normalizes the objectives by the best values and the extreme points it has found, so their
different ranges don't matter. Each solution joins the direction nearest to it, and directions with
few members get more. genoxide's docs recommend it for three or more objectives.

The example runs it twice, for 50 generations each, with simulated binary crossover with η = 30, as
Deb and Jain use, and polynomial mutation with η = 20 at a rate of 0.5 per gene, one of the two
genes per child on average. The reference directions come from Das and Dennis's method (1998, SIAM
Journal on Optimization 8(3): 631-657): all points (a/H, b/H, c/H) with a + b + c = H, for H
divisions.

- 12 divisions, 91 directions, as Deb and Jain use for 3 objectives, and a population of 92, the
  multiple of four just above 91: 4,692 evaluations.
- 30 divisions: 496 directions, and a population of 496, one solution per direction: 25,296
  evaluations.

The second run is there for the target: a front within 1% of the optimal one, measured by IGD+
(see Good results). 91 points can't cover the curved triangle that closely. 496 can.

Both fronts need few generations. With 91 directions, all 92 solutions are non-dominated from
generation 3 on, and the hypervolume reaches 32.1 at generation 4. After that, it moves between
about 31.8 and 32.2: NSGA-III keeps about one solution per direction, and swaps solutions between
neighboring directions without regard to the hypervolume. With 496 directions, all 496 are
non-dominated from generation 4, when the hypervolume reaches 33.05; it stays between 33.0 and
33.05.

## Output

A line per run: how many solutions are on its final front, the front's hypervolume and its IGD+.
The last line gives the whole front's hypervolume.

The hypervolume is the volume that the front dominates, up to a reference point. Larger is better.
The reference point here is (4.4, 5.4, 4.2): the nadir point (4, 5, 4), the worst value of each
objective on the front, plus a tenth of each objective's range from the ideal point (0, 1, 2), the
best values. genoxide's `ideal_point` and `nadir_point` give both. With it, the extreme solutions
count too. For the whole front, the hypervolume is about 33.52: a 4,001 × 4,001 grid of the
variables gives 33.517.

IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) is measured to 1,035 points of the
optimal front, from genoxide's `optimal_front`: the images of points spread evenly over the triangle
of optimal solutions. IGD+ averages, over those points, the distance to the nearest point of the
found front, counting only the objectives in which the found point is worse. 0 means that the found
front covers the optimal one. Smaller is better. The objectives' ranges on the front differ, 4 for
f₁ and f₂ and 2 for f₃, so the example also gives IGD+ scaled: with each objective scaled to [0, 1]
over its range on the front, from the ideal point to the nadir point.

[The project page](https://tachsin.gr/projects/genoxide/examples/viennet1) plays the second run
back.

## Good results

The target is a front close to the whole optimal one: a scaled IGD+ of at most 0.01, the front
within about 1% of the objectives' ranges, or a hypervolume of at least 99% of the whole front's,
33.18. No finite set of solutions reaches the whole front's hypervolume of 33.52. For comparison,
the images of 91 points spread evenly over the triangle of optimal solutions, as many as the
reference directions, have a hypervolume of 32.46 and an IGD+ of 0.0505, scaled 0.0152; for a
scaled IGD+ of 0.01, it takes about 200 points.

With 91 directions, the run's front has 92 solutions, a hypervolume of 32.22 and an IGD+ of 0.065,
scaled 0.019: it covers the optimal front nearly as well as those 91 points. 63 of its solutions
lie in the triangle. The other 29 lie just outside it, typically 0.07 away and at most 0.16: no
other solution of the population beats them. Running longer doesn't change that: after 400
generations, the IGD+ is about 0.079.

With 496 directions, the front has 496 solutions, a hypervolume of 33.03 (98.5% of the whole
front's), and an IGD+ of 0.0257, scaled 0.0076: within the target. 387 of its solutions lie in the
triangle, and the other 109 at most 0.12 outside it. Over seeds 1 to 20, every run reaches the
target, with a scaled IGD+ of 0.0072 to 0.0080 and a hypervolume of 32.99 to 33.06; after 400
generations, the range is the same. With 91 directions, the scaled IGD+ is 0.0174 to 0.0229.
