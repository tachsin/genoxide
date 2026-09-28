---
title: Goldstein-Price
category: continuous
summary: Find the global minimum of the Goldstein-Price function, and the local minima that trap a search, by restarting a local search from random points.
reference: "Goldstein, A. A. and Price, J. F. (1971). On descent from local minima. Mathematics of Computation 25(115): 569-574."
reference_url: "https://doi.org/10.1090/S0025-5718-1971-0312365-X"
optimum: "3 (at (0, −1))"
languages: [rust, python]
order: 56
---

# Goldstein-Price

## The problem

The Goldstein-Price function (Goldstein and Price, 1971) is a product of two factors, each a
polynomial of degree 4 in two variables:

```text
f(x₁, x₂) = [1 + (x₁ + x₂ + 1)² (19 − 14x₁ + 3x₁² − 14x₂ + 6x₁x₂ + 3x₂²)]
          × [30 + (2x₁ − 3x₂)² (18 − 32x₁ + 12x₁² + 48x₂ − 36x₁x₂ + 27x₂²)]
```

to minimize with both variables in [−2, 2]. The paper gives no bounds: these are Dixon and Szegö's
(1978), as restated in Yao, Liu and Lin (1999, function f18). At the origin,
f = (1 + 19) × 30 = 600.

The paper found the minimum, 3 at (0, −1), numerically, and lists three local minima:

| x₁ | x₂ | f |
|---|---|---|
| 0 | −1 | 3, the global minimum |
| −0.6 | −0.4 | 30 |
| 1.8 | 0.2 | 84 |
| 1.2 | 0.8 | 840 |

genoxide's `problems::GoldsteinPrice` proves that 3 is the global minimum. Each factor depends on
one combination of the variables only. With s = x₁ + x₂, the first factor is

```text
A(s) = 1 + (s + 1)² (3s² − 14s + 19) ≥ 1
```

and with t = 2x₁ − 3x₂, the second is

```text
B(t) = 30 + t² (3t² − 16t + 18) ≥ 3
```

Both bounds are met only at s = −1 and t = 3, that is at (0, −1).

## What makes it hard

The function spans a huge range. It is 3 at the minimum and 600 at the origin, and near the corner
(−2, 2) it is about 1,000,000; its largest value in the box is about 1,015,700, at (−1.74, 2). Half
the box is above 6,500. The contour on the example's page is shaded on a logarithmic scale, or the
basins would not show.

Its local minima are traps. Since f = A(s) · B(t), with both factors positive, its gradient is 0
exactly where both factors are flat. A has its minima at s = −1 (A = 1) and s = 2 (A = 28), and a
maximum at s = 1 (A = 33); B has its minima at t = 3 (B = 3) and t = 0 (B = 30), and a maximum at
t = 1 (B = 35). Every combination is a stationary point of f:

- the four minima: 1 × 3 = 3, 1 × 30 = 30, 28 × 3 = 84 and 28 × 30 = 840, the paper's four;
- four saddle points, the passes between the basins: 35 at (−0.4, −0.6), 99 at (1.2, −0.2), 980 at
  (1.4, 0.6) and 990 at (0.6, 0.4);
- one maximum, 1,155 at (0.8, 0.2).

These nine are all its stationary points, so the paper's list of minima is complete. A search that
goes only downhill and falls into the basin of 30, 84 or 840 stays there. The basin of the global
minimum is not the largest. In 1,000 searches with this example's settings (seeds 1 to 1,000), 32%
end at the global minimum, 40% at 30, 9% at 84 and 19% at 840.

## Representation

A `Real` genome of 2 genes, the point (x₁, x₂) itself, each in [−2, 2]. The fitness is f, to
minimize. The function, its bounds and its global minimum are genoxide's `problems::GoldsteinPrice`.

## Algorithm

Thirty independent local searches, each from a random point, with seeds 1 to 30. Each is a hill
climber, with the settings of the [Himmelblau example](../himmelblau/):

- a step makes 10 neighbors of the current point, each with Gaussian noise on both genes, of
  standard deviation 0.001 of the gene's range, 0.004;
- the search moves to the best neighbor only if it is strictly better;
- it stops after 1,000 steps, 10,001 evaluations with the starting point.

With steps that small, a search follows its basin down to the minimum at the bottom. Restarting from
random points is the simplest way to find the global minimum despite the traps: with 32% of the
starts in its basin, 30 searches all miss it with a probability of 0.68³⁰, about 1 in 100,000. The
restarts also show the local minima, about in proportion to their basins. The searches that end
within 2% of the bounds' width of each other, in both genes, are counted as one minimum.

The steps don't need to adapt to the huge range: a hill climber compares values only, never their
differences, so a basin 1,000 times steeper than another is followed the same way.

## Output

The first line gives the number of searches, the bounds and the length of each search. Then a table
has a row per minimum the searches reached: the best point found there, rounded to 3 decimals, how
many searches ended there, and the best value, to 5 significant digits. The rows go from the lowest
value to the highest, and a row within 0.001 of the global minimum is marked `global`. In Python,
`run` evaluates the function in Rust, so both versions print the same table.

The page's plot shows the 30 searches as points on the function's contour, and a curve of the best
and the median value's distance above the global minimum, 3, on a logarithmic axis.

[The project page](https://tachsin.gr/projects/genoxide/examples/goldstein-price) plays this run
back.

## Good results

A good result reaches the global minimum, 3 at (0, −1), with a value close to 3. The run finds it
with 15 of its 30 searches, within 1e-7. The other 15 end in the paper's three local minima: 10 at
30, 3 at 84 and 2 at 840, at the paper's points to 3 decimals and its values to 5 digits. Together,
the searches reach all four minima of the function.

At the start, the best of the 30 random points is 13 above the minimum, and the median one about
4,200 above it. By step 300, the median has settled at 16.5, halfway between the 15 searches at 3
and the 10 at 30: an error of 13.5 on the plot, which stays there.
