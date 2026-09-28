---
title: Viennet 3
category: multi-objective
summary: Minimize three objectives of two variables, two of which depend only on the distance from the origin, so that the Pareto front is two separate curves, with NSGA-III.
reference: "Viennet, R., Fonteix, C. and Marc, I. (1996). Multicriteria optimization using a genetic algorithm for determining a Pareto set. International Journal of Systems Science 27(2): 255-260."
reference_url: https://doi.org/10.1080/00207729608929211
optimum: "not known in closed form: two curves; hypervolume about 5.3255 (reference point (9.016, 17.2407, 0.2036))"
languages: [rust, python]
order: 111
---

# Viennet 3

## The problem

Viennet, Fonteix and Marc (1996) posed three problems with two variables and three objectives. This
is the third, VNT3. genoxide's `Viennet3` uses the definition and bounds that Van Veldhuizen
restates (1999, PhD thesis AFIT/DS/ENG/99-01, Air Force Institute of Technology, table B.1; its
table 5.3 has wider bounds); genoxide hasn't yet checked them against the original paper.

Over x₁ and x₂ in [−3, 3], with r² = x₁² + x₂², all three objectives are minimized:

```text
f₁ = r²/2 + sin(r²)
f₂ = (3x₁ − 2x₂ + 4)²/8 + (x₁ − x₂ + 1)²/27 + 15
f₃ = 1/(r² + 1) − 1.1 exp(−r²)
```

f₁ and f₃ depend only on the distance from the origin, where both are smallest: f₁ = 0 and
f₃ = −0.1. There, f₂ = 17.037. f₂ is a convex quadratic, smallest, 15, at (−2, −1), where
f₁ = 1.541 and f₃ = 0.159. Its contours are long, thin ellipses along the line 3x₁ − 2x₂ + 4 = 0.

Neither f₁ nor f₃ grows steadily with the distance. f₁ falls where cos(r²) < −1/2: for r² from 2π/3
to 4π/3 (2.09 to 4.19), and again from 8.38 to 10.47 and from 14.66 to 16.76. f₃ rises to 0.196 at
r² = 2.7, and falls after that.

All points of a circle around the origin have the same f₁ and f₃. Only the point of the circle
where f₂ is smallest can be optimal: it beats every other point of the circle. So the optimal
solutions lie on a curve, and the Pareto front is a curve in the space of the objectives, not a
surface. genoxide gives no closed form for it: `optimal_front` is `None`. Computed numerically, from
the point where f₂ is smallest on each of 60,001 circles, keeping the points that no other beats, it
is two curves:

- near the origin, for r² up to 1.50: from (0, 17.037, −0.1), where f₁ and f₃ are smallest, to
  about (1.748, 15.005, 0.155), at x = (−1.22, 0.15). Along it, f₂ falls by 2 and f₁ and f₃ rise;
- farther out, for r² from 4π/3 = 4.19 to 17.15: from about (1.228, 15.0001, 0.176), at
  x = (−1.88, −0.82), where f₁ has a local minimum, through f₂'s minimum at (−2, −1), along f₂'s
  valley to the box's edge x₁ = −3, and to about (7.58, 15.09, 0.055) at x = (−3, −2.86). Along
  it, f₂ stays within 0.09 of 15, and f₁ reaches 8.196 at r² = 14π/3 = 14.66.

No optimal solution has r² between 1.50 and 4.19: a point of the outer curve beats each point there
in all three objectives.

The ideal point, the best value of each objective on the front, is (0, 15, −0.1). The nadir point,
the worst, is (8.1964, 17.0370, 0.1760): f₁'s local maximum at r² = 14π/3, 7π/3 + √3/2; f₂ at the
origin, 460/27; and f₃ at the start of the outer curve, r² = 4π/3, where it's
1/(1 + 4π/3) − 1.1 exp(−4π/3). genoxide's `ideal_point` and `nadir_point` give both, and the
example computes its reference point from them.

## What makes it hard

The front is degenerate: two curves, where a problem with three objectives usually has a surface.
The optimal solutions have no area, so a random solution is never optimal. A solution next to
a curve is beaten only by the solutions of a thin region between it and the curve. A finite
population rarely holds one, so solutions next to the curves stay on the population's front.

In the variables, the gap between r² = 1.50 and 4.19 separates the two curves.
The objectives' ranges on the front differ widely: 8.2 for f₁, 2.04 for f₂ and 0.28 for f₃. And
f₂'s range on the front is small beside its range over the box, from 15 to 61.9.

## Representation

A `Real` genome of 2 genes in [−3, 3]: the point x. The problem is genoxide's `Viennet3`, whose
fitness is the three objectives. In Python, `run` evaluates it in Rust.

## Algorithm

NSGA-III (Deb and Jain, 2014, IEEE Transactions on Evolutionary Computation 18(4): 577-601). Like
NSGA-II, it ranks solutions into non-dominated fronts, and parents and children compete for the
next population. Instead of crowding distance, it spreads the front along reference directions. It
first normalizes the objectives by the best values and the extreme points it has found, so their
different ranges don't matter. Each solution joins the direction nearest to it, and directions with
few members get more. genoxide's docs recommend it for three or more objectives. Its directions
cover a surface; on a curve, many of them have no solution near them, and the population spreads
along the curves instead.

The settings:

- 91 reference directions from Das and Dennis's method (1998, SIAM Journal on Optimization 8(3):
  631-657) with 12 divisions: all points (a/12, b/12, c/12) with a + b + c = 12, as Deb and Jain
  use for 3 objectives;
- a population of 92, the multiple of four just above 91;
- simulated binary crossover with η = 30, as Deb and Jain use, and polynomial mutation with η = 20
  at a rate of 0.5 per gene, one of the two genes per child on average;
- 50 generations, 4,692 evaluations.

The front needs about 10 generations. All 92 solutions are non-dominated from generation 7 on, and
the hypervolume reaches 5.26 at generation 10 and its highest, 5.280, at generation 14. After that,
it drifts down a little, to 5.255 at generation 50: NSGA-III keeps the solutions spread along the
directions, without regard to the hypervolume. A run of 400 generations ends at about 5.25.

## Output

Two lines. The first gives how many solutions are on the final front, and how many of them are near
the origin, with r² < 3, or farther out: on each of the two curves, or next to it.

The second gives the front's hypervolume: the volume that it dominates, up to a reference point.
Larger is better. The reference point here is (9.016, 17.2407, 0.2036): the nadir point plus a tenth
of each objective's range from the ideal point, so that the extreme solutions count too, rounded to
4 decimals. For the whole front, the hypervolume is about 5.3255: 43,192 points of the two curves
give 5.3254, and a 4,001 × 4,001 grid of the variables gives 5.3255.

[The project page](https://tachsin.gr/projects/genoxide/examples/viennet3) plays this run back.

## Good results

No finite set of solutions reaches 5.3255. A good front has solutions along both curves, from end
to end, and close to them.

The run's front has 92 solutions and a hypervolume of about 5.2553, within 1.3% of the whole
front's. 68 of its solutions are near the origin, up to r² = 1.39 of the inner curve's 1.50. The
other 24 are on the outer curve, for r² from 5.08 to 15.45: its two ends, from 4.19 to 5.08 and from
15.45 to 17.15, have none. Most solutions are on the curves or very near them. For half of them, f₂
is less than 0.002 above its minimum on their circle; for 90%, less than 0.02; for the worst, 0.21
above.
