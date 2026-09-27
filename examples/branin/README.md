---
title: Branin
category: continuous
summary: Find the three global minima of Branin's function by restarting a local search from random points.
reference: "Branin, F. H. (1972). Widely convergent method for finding multiple solutions of simultaneous nonlinear equations. IBM Journal of Research and Development 16(5): 504-522."
reference_url: "https://doi.org/10.1147/rd.165.0504"
optimum: "0.397887 (at three points)"
languages: [rust, python]
order: 65
---

# Branin

## The problem

Branin's function, also called RCOS, is a function of two variables to minimize:

```text
f(x₁, x₂) = (x₂ − 5.1 x₁² / (4π²) + 5 x₁ / π − 6)² + 10 (1 − 1 / (8π)) cos x₁ + 10
```

with x₁ in [−5, 10] and x₂ in [0, 15]. It comes from Branin's (1972) paper on finding several
solutions of a system of nonlinear equations. genoxide takes the definition and the bounds from Yao,
Liu and Lin's (1999) restatement, function f17; they are not yet checked against the original.

The function has two parts. The square is 0 along a curved valley, the parabola

```text
x₂ = 5.1 x₁² / (4π²) − 5 x₁ / π + 6
```

and grows quickly away from it. The rest, 10 (1 − 1 / (8π)) cos x₁ + 10, depends on x₁ only, and
runs between 10 − 10 (1 − 1 / (8π)) = 5 / (4π) and 20 − 5 / (4π). At the origin,
f = 36 + 20 − 5 / (4π) ≈ 55.6. The largest value in the box is about 308, at the corner (−5, 0).

## What makes it hard

The function has three global minima, all of the same value, 5 / (4π) ≈ 0.397887. They are where
the square is 0 and cos x₁ = −1, at x₁ = −π, π and 3π:

| x₁ | x₂ | f |
|---|---|---|
| −π ≈ −3.14159 | 12.275 | 0.397887 |
| π ≈ 3.14159 | 2.275 | 0.397887 |
| 3π ≈ 9.42478 | 2.475 | 0.397887 |

Yao, Liu and Lin (1999) and Jamil and Yang (2013) print the third point as (3π, 2.425). That is a
misprint: there the square is 0.05² and the value is 0.0025 higher. genoxide's `problems::Branin`
derives the three points from the formula and gives them to full precision.

Along the floor of the valley, f = 10 (1 − 1 / (8π)) cos x₁ + 10: it rises to about 19.6 at x₁ = 0
and x₁ = 2π, between the minima. So the valley is split into three basins, one per minimum. Finding
one minimum is easy: a search that goes downhill falls into the valley and follows it to the nearest
of the three. Finding all three takes several searches, or a method that keeps several points
apart. The basins are not the same size. In 1,000 searches with this example's settings (seeds 1 to
1,000), 34% end at (−π, 12.275), 40% at (π, 2.275) and 26% at (3π, 2.475).

## Representation

A `Real` genome of 2 genes, the point (x₁, x₂) itself, with x₁ in [−5, 10] and x₂ in [0, 15]. The
fitness is f, to minimize. The function, its bounds and its three minima are genoxide's
`problems::Branin`.

## Algorithm

Thirty independent local searches, each from a random point, with seeds 1 to 30. Each is a hill
climber, with the settings of the [Himmelblau example](../himmelblau/):

- a step makes 10 neighbors of the current point, each with Gaussian noise on both genes, of
  standard deviation 0.001 of the gene's range, 0.015;
- the search moves to the best neighbor only if it is strictly better;
- it stops after 1,000 steps, 10,001 evaluations with the starting point.

With steps that small, a search follows its basin down to the minimum at the bottom: first down the
steep walls into the valley, then along the valley floor. Restarting from random points is the
simplest way to find several minima: each start lands in some basin, and basins are found about in
proportion to their size. The searches that end within 2% of the bounds' width of each other, in
both genes, are counted as one minimum.

With the smallest basin at 26%, 30 searches all miss it with a probability of 0.74³⁰, about 1 in
10,000. A population method, such as a GA or differential evolution, also finds a global minimum,
but its population usually gathers at one of the three and loses the others. Independent searches
keep them apart.

## Output

The first line gives the number of searches, the bounds and the length of each search. Then a table
has a row per minimum the searches reached: the best point found there, rounded to 3 decimals, how
many searches ended there, and the best value, to 5 significant digits. The rows go from the lowest
value to the highest, then by x₁, and a row within 0.001 of the global minimum is marked `global`.
In Python, `run` evaluates the function in Rust, so both versions print the same table.

The page's plot shows the 30 searches as points on the function's contour, and a curve of the best
and the median value's distance above the minimum, 5 / (4π), on a logarithmic axis.

[The project page](https://tachsin.gr/projects/genoxide/examples/branin) plays this run back.

## Good results

A good result reaches all three minima, each with a value close to 0.397887. The run finds all
three: 7 searches end at (−π, 12.275), 18 at (π, 2.275) and 5 at (3π, 2.475). Every search ends in
a global minimum, since the function has no other minima in the box. The third point is at
x₂ = 2.475, not the misprinted 2.425.

At the start, the best of the 30 random points is 0.13 above the minimum, and the median one 23.5
above it. After 1,000 steps, the best is within 1e-8 of the minimum, and the median within 2e-7.
The step size is fixed, so near a minimum few neighbors are better than the current point, and
progress slows down, as in the Himmelblau example.
