---
title: Branin, Goldstein-Price and the six-hump camel
category: continuous
summary: Find the minima of three two-dimensional test functions by restarting a local search from random points.
reference: "Dixon, L. C. W. and Szegö, G. P. (eds.) (1978). Towards Global Optimisation 2. North-Holland."
reference_url: ""
optimum: "0.397887 (Branin, at three points), 3 (Goldstein-Price), −1.0316285 (six-hump camel, at two points)"
languages: [rust, python]
order: 66
---

# Branin, Goldstein-Price and the six-hump camel

## The problem

Three functions of two variables, x₁ and x₂, each to minimize within bounds. They are classic test
functions for global optimization, collected in Dixon and Szegö's *Towards Global Optimisation 2*
(1978). genoxide takes their bounds, and the definitions of the first and the third, from Yao, Liu
and Lin's (1999) restatement, functions f17, f18 and f16.

- **Branin's function** (Branin, 1972), also called RCOS:

  ```text
  f(x₁, x₂) = (x₂ − 5.1 x₁² / (4π²) + 5 x₁ / π − 6)² + 10 (1 − 1 / (8π)) cos x₁ + 10
  ```

  with x₁ in [−5, 10] and x₂ in [0, 15]. The square is 0 along a curved valley, and the cosine is
  lowest, −1, at x₁ = −π, π and 3π. The minimum, 5 / (4π) ≈ 0.397887, is where both happen: at
  (−π, 12.275), (π, 2.275) and (3π, 2.475). Yao, Liu and Lin (1999) and Jamil and Yang (2013)
  print the third as (3π, 2.425), where the value is 0.0025 higher.
- **The Goldstein-Price function** (Goldstein and Price, 1971) is a product of two factors, each a
  polynomial of degree 4:

  ```text
  f(x₁, x₂) = [1 + (x₁ + x₂ + 1)² (19 − 14x₁ + 3x₁² − 14x₂ + 6x₁x₂ + 3x₂²)]
            × [30 + (2x₁ − 3x₂)² (18 − 32x₁ + 12x₁² + 48x₂ − 36x₁x₂ + 27x₂²)]
  ```

  with both variables in [−2, 2], Dixon and Szegö's bounds (the paper gives none). The minimum is 3
  at (0, −1). The paper also lists three local minima: 30 at (−0.6, −0.4), 84 at (1.8, 0.2) and 840
  at (1.2, 0.8).
- **The six-hump camel-back function** (Dixon and Szegö, 1978):

  ```text
  f(x₁, x₂) = (4 − 2.1x₁² + x₁⁴ / 3) x₁² + x₁x₂ + (−4 + 4x₂²) x₂²
  ```

  with both variables in [−5, 5], Yao, Liu and Lin's bounds; other papers use x₁ in [−3, 3] and x₂
  in [−2, 2]. It has six local minima, in pairs symmetric about the origin. The lowest pair,
  −1.0316285, is at ±(0.089842, −0.712656), and the next, −0.2154638, at ±(1.70361, −0.79608).

The functions, their bounds and their global minima are genoxide's `problems::Branin`,
`problems::GoldsteinPrice` and `problems::SixHumpCamel`, which give the minima to full precision.
genoxide's definitions of Branin's function and the camel are checked against Yao, Liu and Lin's
restatement, not yet against the originals.

## What makes it hard

Each function has several minima, in separate basins. A search that converges to one point finds
the minimum of the basin it starts in: where it starts decides where it ends.

- Branin's function has three global minima, all of the same value. Finding one is easy; finding
  all three takes several searches, or a method that keeps several points apart.
- The Goldstein-Price function spans a huge range: from 3 at the minimum to about 1,000,000 near
  the corner (−1.74, 2), and 600 at the origin. Its local minima are traps: a search that falls into
  the basin of 30, 84 or 840 stays there.
- The six-hump camel has six minima, two of them global, and grows fast away from the origin, to
  about 6,400 at the corners: most of the box is steep walls around the six basins. Four of the six
  minima are not global, and a search can end in any of them.

## Representation

A `Real` genome of 2 genes, the point itself, with each function's bounds. The fitness is the
function's value, to minimize.

## Algorithm

For each function, 30 independent local searches, each from a random point, with seeds 1 to 30.
Each is a hill climber, with the settings of the [Himmelblau example](../himmelblau/):

- a step makes 10 neighbors of the current point, each with Gaussian noise on both genes, of
  standard deviation 0.001 of the gene's range (0.015 for Branin's function, 0.004 for the
  Goldstein-Price function, 0.01 for the camel);
- the search moves to the best neighbor only if it is strictly better;
- it stops after 1,000 steps, about 10,000 evaluations.

With steps that small, a search follows its basin down to the minimum at the bottom. Restarting from
random points is the simplest way to find several minima: each start lands in some basin, and
basins are found about in proportion to their size. The searches that end within 2% of the bounds'
width of each other, in both genes, are counted as one minimum.

## Output

The first line gives the number of searches and their length. Then comes a table per function, with
its bounds, and a row per minimum the searches reached: the best point found there, rounded to 3
decimals, how many searches ended there, and the best value, to 5 significant digits. The rows go
from the lowest value to the highest, and the global minima are marked `global`. In Python, `run`
evaluates the functions in Rust, so both versions print the same tables.

For Branin's function, the three rows are the three global minima. For the Goldstein-Price function,
they are the global minimum and the paper's three local minima. For the camel, they are all six
minima: the searches also find the third pair, 2.1043 at ±(1.607, 0.569), which genoxide's docs
don't list.

The page's plot shows the three functions' searches side by side, and a curve per function: its
best value so far minus its global minimum, on one logarithmic axis.

[The project page](https://tachsin.gr/projects/genoxide/examples/minima-2d) plays this run back.

## Good results

A good result reaches every global minimum, with a value close to it, and shows which local minima
catch the searches. The run finds all three minima of Branin's function, 5 to 18 searches each; the
Goldstein-Price minimum with 15 searches, and its three local minima with the other 15; and all six
minima of the camel, both global ones among them. Each function's best value is within 1e-7 of its
global minimum: the step size is fixed, so progress slows down near a minimum, as in the Himmelblau
example.
