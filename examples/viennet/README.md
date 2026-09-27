---
title: Viennet's three-objective problems
category: multi-objective
summary: Minimize three objectives of two variables on Viennet's three problems with NSGA-III, and compare the fronts by their normalized hypervolume.
reference: "Viennet, R., Fonteix, C. and Marc, I. (1996). Multicriteria optimization using a genetic algorithm for determining a Pareto set. International Journal of Systems Science 27(2): 255-260."
reference_url: https://doi.org/10.1080/00207729608929211
optimum: "VNT1: the image of the triangle (0, 1), (0, −1), (1, 0); VNT2 and VNT3: not known in closed form"
languages: [rust, python]
order: 92
---

# Viennet's three-objective problems

## The problem

Viennet, Fonteix and Marc (1996) posed three problems with two variables and three objectives, all
minimized. genoxide's `Viennet1`, `Viennet2` and `Viennet3` use the definitions and bounds that Van
Veldhuizen restates (1999, PhD thesis AFIT/DS/ENG/99-01, Air Force Institute of Technology, table
B.1); genoxide hasn't yet checked them against the original paper.

VNT1, over x₁ and x₂ in [−2, 2]:

```text
f₁ = x₁² + (x₂ − 1)²
f₂ = x₁² + (x₂ + 1)² + 1
f₃ = (x₁ − 1)² + x₂² + 2
```

Each objective is the squared distance to a point, (0, 1), (0, −1) or (1, 0), plus a constant.
Moving towards one point moves away from the others. The optimal solutions are the triangle with
these three corners, and the Pareto front is its image: a curved surface from (0, 5, 4) to
(4, 1, 4) and (2, 3, 2).

VNT2, over x₁ and x₂ in [−4, 4]:

```text
f₁ = (x₁ − 2)²/2 + (x₂ + 1)²/13 + 3
f₂ = (x₁ + x₂ − 3)²/36 + (−x₁ + x₂ + 2)²/8 − 17
f₃ = (x₁ + 2x₂ − 1)²/175 + (2x₂ − x₁)²/17 − 13
```

These are quadratics too, but their contours are ellipses, stretched in different directions.
Their minima are 3 at (2, −1), −17 at (2.5, 0.5) and −13 at (0.5, 0.25). genoxide gives no closed
form for the front.

VNT3, over x₁ and x₂ in [−3, 3], with r² = x₁² + x₂²:

```text
f₁ = r²/2 + sin(r²)
f₂ = (3x₁ − 2x₂ + 4)²/8 + (x₁ − x₂ + 1)²/27 + 15
f₃ = 1/(r² + 1) − 1.1 exp(−r²)
```

f₁ and f₃ depend only on the distance from the origin, where both are smallest: 0 and −0.1. f₂ is
smallest, 15, at (−2, −1). Its front isn't known in closed form either.

## What makes it hard

With three objectives, the front is a surface, not a curve, and covering it evenly takes many more
solutions. The three problems' objectives also have very different scales: VNT2's f₁ varies by
about 1.2 over its front and its f₂ by about 0.5, around −17.

VNT3's sine makes f₁ fall as r² grows where cos(r²) < −1/2, for r² from 2π/3 to 4π/3 and a period
later. On a fine grid of the variables, no optimal solution has r² between about 1.5 and 4.2: the
optimal solutions leave out a ring around the origin. Most of the front lies within 0.1 of f₂'s
minimum, 15, while f₁ runs from about 1.2 to 8.2: a thin band, plus a part near the origin where f₂
rises to 17.04 as f₁ and f₃ reach their minima.

## Representation

A `Real` genome of 2 genes, within the problem's bounds: the point x. The problems are genoxide's
`Viennet1`, `Viennet2` and `Viennet3`, whose fitness is the three objectives. In Python, `run`
evaluates them in Rust.

## Algorithm

NSGA-III (Deb and Jain, 2014, IEEE Transactions on Evolutionary Computation 18(4): 577-601). Like
NSGA-II, it ranks solutions into non-dominated fronts, and parents and children compete. Instead of
crowding distance, it spreads the front along reference directions. It first normalizes the
objectives by the best values and the extreme points it has found, so their scales don't matter.
Each solution joins the direction nearest to it, and directions with few members get more.

The settings, the same for the three problems:

- 91 reference directions from Das and Dennis's method (1998, SIAM Journal on Optimization 8(3):
  631-657) with 12 divisions, as Deb and Jain use for 3 objectives;
- a population of 92, the multiple of four just above 91;
- simulated binary crossover with η = 30, and polynomial mutation with η = 20 at a rate of 0.5 per
  gene, one of the two genes per child on average;
- 30 generations, 2,852 evaluations per problem. The fronts change little after about 15.

## Output

A table with a row per problem: how many solutions are on the final front, its normalized
hypervolume and, for VNT1, its IGD+.

The normalized hypervolume maps each objective to [0, 1], with 0 at the ideal point, the best value
of each objective on the front, and 1 at the nadir point, the worst. It is then the volume that the
front dominates, up to the reference point (1.1, 1.1, 1.1). Larger is better, and no front reaches
1.1³ = 1.331. The mapping puts the three problems on one scale, but their fronts have different
shapes, so the best possible value differs between them.

VNT1's ideal and nadir points come from genoxide's `ideal_point` and `nadir_point`: (0, 1, 2) and
(4, 5, 4). genoxide doesn't give them for VNT2 and VNT3, so the example does. Their ideal points
are the objectives' minima: (3, −17, −13) and (0, 15, −0.1). VNT2's objectives are convex, so its
optimal solutions are the minimizers of weighted sums of the objectives, and the worst value of each
objective on the front is at the minimum of another, as for VNT1: (4.2452, −16.4766, −12.0531).
VNT3's nadir point, (8.1964, 17.0370, 0.1762), is an estimate from the non-dominated points of a
2,001 × 2,001 grid of the variables.

IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) needs the optimal front, known only
for VNT1. The example takes 1,035 points of it, the images of points spread evenly over the
triangle of optimal solutions. IGD+ averages, over those points, the distance to the nearest point
of the found front, counting only the objectives in which the found point is worse, in the
objectives' own units. 0 means that the found front covers the optimal one. Smaller is better.

[The project page](https://tachsin.gr/projects/genoxide/examples/viennet) plays this run back.

## Good results

A finite front doesn't reach the normalized hypervolume of the whole front. For VNT1, 20,100 points
of the optimal front give about 1.046. The images of 91 points spread evenly over the triangle of
optimal solutions, as many as the reference directions, give 1.0144, with an IGD+ of 0.0505. For
VNT2 and VNT3, the non-dominated points of a 2,001 × 2,001 grid of the variables give about 1.2546
and 1.1556.

The runs' fronts have 92 solutions each. VNT1's has a normalized hypervolume of about 1.004 and an
IGD+ of about 0.069, on objectives that span 2 to 4. VNT2's has about 1.240, and VNT3's about 1.142,
each within 1.5% of the grid's front.
