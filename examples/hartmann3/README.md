---
title: Hartmann 3-D
category: continuous
summary: Find the local minima of Hartmann's function in 3 dimensions, four Gaussian wells in the unit cube, by restarting a local search from random points.
reference: "Hartman, J. K. (1973). Some experiments in global optimization. Naval Research Logistics Quarterly 20(3): 569-576. Constants as tabulated in Dixon, L. C. W. and Szegö, G. P. (1978). The global optimisation problem: an introduction. In Towards Global Optimisation 2, North-Holland: 1-15."
reference_url: "https://doi.org/10.1002/nav.3800200316"
optimum: "−3.86278 at (0.11461, 0.55565, 0.85255) (best known)"
languages: [rust, python]
order: 69
family: Hartmann
tab: 3-D
---

# Hartmann 3-D

## The problem

Hartmann's function in 3 dimensions is a sum of four Gaussian wells, to minimize:

```text
f(x) = −Σᵢ₌₁⁴ cᵢ exp(−Σⱼ₌₁³ aᵢⱼ (xⱼ − pᵢⱼ)²),   each xⱼ in [0, 1]
```

Well i is centered near pᵢ, has depth about cᵢ, and a width per gene set by aᵢⱼ: the larger aᵢⱼ,
the narrower the well along gene j.

| i | cᵢ | aᵢ₁, aᵢ₂, aᵢ₃ | pᵢ₁, pᵢ₂, pᵢ₃ |
|---|---|---|---|
| 1 | 1 | 3, 10, 30 | 0.3689, 0.1170, 0.2673 |
| 2 | 1.2 | 0.1, 10, 35 | 0.4699, 0.4387, 0.7470 |
| 3 | 3 | 3, 10, 30 | 0.1091, 0.8732, 0.5547 |
| 4 | 3.2 | 0.1, 10, 35 | 0.03815, 0.5743, 0.8828 |

The function is Hartman's (1973); the constants are those that Dixon and Szegö (1978) tabulate in
their collection of test problems, which made it a standard one. Hartman's report (1972, published
in 1973) defines the form, with constants drawn at random that it doesn't print, and has no problem
in 3 or 6 dimensions. Dixon and Szegö's book isn't online: genoxide takes the constants from Yao,
Liu and Lin's (1999, table XII) reprint, which prints p₄₁ as 0.038150; Jamil and Yang (2013)
misprint p₂₂ as 0.4837.

genoxide's `problems::Hartmann3` gives the minimum as −3.862782147820755 at (0.11461433858967196,
0.5556488499718569, 0.8525469535208658): the point where the gradient is 0, computed to 40 digits
by Newton's method. Later papers quote −3.86278 from Dixon and Szegö, and the point as (0.114614,
0.555649, 0.852547). It's the best known minimum, not proven global: the lowest of the local
minima that searches from many random points find.

## What makes it hard

Not much, for a global method: the function is smooth, and the deepest well has the largest basin.
Its difficulty is that it has several local minima, and a method that goes downhill from one
point ends in the nearest. Three local minima draw nearly every search:

| x₁ | x₂ | x₃ | f |
|---|---|---|---|
| 0.11461 | 0.55565 | 0.85255 | −3.86278 |
| 0.10934 | 0.86052 | 0.56412 | −3.08976 |
| 0.36872 | 0.11756 | 0.26757 | −1.00082 |

The first is in well 4, the deepest, made deeper by well 2: at the minimum, their terms are 3.087
and 0.700, well 3's 0.076. Well 2 is so wide along x₁ (a₂₁ = 0.1) that it reaches the minimum from
0.36 away along x₁. The second minimum is well 3, and the third well 1, whose weight is only 1. Far
from the wells the function is nearly flat: at the corner (1, 1, 0), it is −0.00004.

The genes don't count the same: a₃ is 30 or 35 in every well, and a₁ as low as 0.1. So the wells
are narrow along x₃ and wide along x₁: moving x₁ by 0.3 barely changes f near the minimum, while
moving x₃ by 0.1 costs about a quarter of it.

## Representation

A `Real` genome of 3 genes, each in [0, 1]: the point x itself. The fitness is f(x), to minimize.
The function, its bounds and its best known minimum are genoxide's `problems::Hartmann3`.

## Algorithm

Thirty independent local searches, each from a random point, with seeds 1 to 30, as in the
[Branin example](../branin/). Each is a hill climber:

- a step makes 10 neighbors of the current point, each with Gaussian noise on every gene, of
  standard deviation 0.001 of the gene's range;
- the search moves to the best neighbor only if it is strictly better;
- it stops after 1,000 steps, 10,001 evaluations with the starting point.

Each search follows its basin down to the minimum at its bottom, so the searches show the basins
and how large they are. The searches that end within 0.02 of each other in every gene are counted as
one minimum.

## Output

The first line gives the number of searches, the bounds and the length of each search. Then a table
has a row per minimum the searches reached: the best point found there, rounded to 3 decimals, how
many searches ended there, and the best value, to 5 decimals. The rows go from the lowest value to
the highest, and the row within 0.001 of the best known minimum is marked `global`. In Python,
`run` evaluates the function in Rust, so both versions print the same table.

The page's plot shows the 30 searches at their (x₁, x₂), over the contour of the lowest value of f
over x₃ at each (x₁, x₂), and a curve of the best and the median value's distance above the best
known minimum, on a logarithmic axis.

[The project page](https://tachsin.gr/projects/genoxide/examples/hartmann3) plays this run back.

## Good results

A good result reaches the minimum −3.86278 at (0.115, 0.556, 0.853). The run does: 11 of the 30
searches end there, 5 at −3.08976 and 14 at −1.00082, and the best is within 3e-8 of the minimum
after 1,000 steps. In 1,000 searches with the same settings (seeds 1 to 1,000), 48% end at the
minimum, 26% at −3.08976 and 27% at −1.00082: the basin of the minimum is the largest, but more
than half the searches from a random point miss it. The restarts are what make the method reliable:
30 searches all miss it with a probability of 0.52³⁰, about 3 in 10⁹, and in each of 20 groups of
30 searches (seeds 1 to 600), 10 to 18 end within 1e-6 of the minimum.

For a population method this is an easy function. CMA-ES with genoxide's defaults and no restarts,
in runs not shown here, reached the minimum to within 1e-6 from 29 of seeds 1 to 30, with about
430 evaluations each; the other run ended at −1.00082.
