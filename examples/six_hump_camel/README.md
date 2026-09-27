---
title: Six-hump camel
category: continuous
summary: Find the six minima of the six-hump camel-back function, two of them global, by restarting a local search from random points.
reference: "Dixon, L. C. W. and Szegö, G. P. (eds.) (1978). Towards Global Optimisation 2. North-Holland."
reference_url: ""
optimum: "−1.0316285 (at two points)"
languages: [rust, python]
order: 67
---

# Six-hump camel

## The problem

The six-hump camel-back function is a polynomial in two variables, to minimize:

```text
f(x₁, x₂) = (4 − 2.1x₁² + x₁⁴ / 3) x₁² + x₁x₂ + (−4 + 4x₂²) x₂²
```

with both variables in [−5, 5]. It is one of the test functions of Dixon and Szegö's *Towards Global
Optimisation 2* (1978). genoxide takes the definition and the bounds from Yao, Liu and Lin's (1999)
restatement, function f16; they are not yet checked against the original. Yao, Liu and Lin's box is
wider than the x₁ in [−3, 3], x₂ in [−2, 2] of other papers.

The first term is a polynomial in x₁ alone, of degree 6, with a dip at 0 and two more near
x₁ = ±1.65. The last is a polynomial in x₂ alone, of degree 4, with two dips near x₂ = ±0.7. The
middle term, x₁x₂, couples them. Replacing (x₁, x₂) by (−x₁, −x₂) leaves every term unchanged, so
the function is symmetric about the origin, and its minima come in pairs. At the origin, f = 0. At
(1, 1), f = 97 / 30 ≈ 3.23.

## What makes it hard

The function has six local minima, three pairs, which give it its name. Its stationary points solve
∂f/∂x₂ = x₁ − 8x₂ + 16x₂³ = 0, so x₁ = 8x₂ − 16x₂³, and then ∂f/∂x₁ = 0, a polynomial of degree 15
in x₂. It has 15 real roots, 15 stationary points. Classified by their second derivatives, computed
for this page:

| kind | value | points |
|---|---|---|
| minimum (global) | −1.0316285 | ±(0.089842, −0.712656) |
| minimum | −0.2154638 | ±(1.70361, −0.79608) |
| minimum | 2.1042503 | ±(1.60710, 0.56865) |
| saddle | 0 | the origin |
| saddle | 0.5437186 | ±(1.10921, −0.76827) |
| saddle | 2.2293572 | ±(1.63807, 0.22867) |
| saddle | 2.2294708 | ±(1.29607, 0.60508) |
| maximum | 2.4962954 | ±(1.23023, 0.16233) |

genoxide's `problems::SixHumpCamel` gives the global minimum, −1.0316284534898774, at both points,
computed to 40 digits by Newton's method and rounded, and its docs name the second pair. Yao, Liu
and Lin's value, −1.0316285, agrees to 8 digits; their x₁, 0.08983, is 0.089842 to 5 digits. The
docs don't list the third pair, at 2.1042503: it is the shallowest, only 0.125 below the saddles
next to it, but it is a minimum, and a search can end there.

Two of the six minima are global. A search that converges to one point finds the minimum of the
basin it starts in, and four of the six basins hold a minimum that is not global. Away from the
origin, the function grows fast, as x₁⁶ / 3 and 4x₂⁴: it is about 6,400 at the corners, and above
549 on half of the box. The six minima lie within |x₁| < 1.8 and |x₂| < 0.8, about 6% of the box;
the rest is steep walls around their basins.

In 1,000 searches with this example's settings (seeds 1 to 1,000), 24% end at a global minimum,
12% at each, 42% at the second pair, and 34% at the third, shallow pair.

## Representation

A `Real` genome of 2 genes, the point (x₁, x₂) itself, each in [−5, 5]. The fitness is f, to
minimize. The function, its bounds and its two global minima are genoxide's
`problems::SixHumpCamel`.

## Algorithm

Thirty independent local searches, each from a random point, with seeds 1 to 30. Each is a hill
climber, with the settings of the [Himmelblau example](../himmelblau/):

- a step makes 10 neighbors of the current point, each with Gaussian noise on both genes, of
  standard deviation 0.001 of the gene's range, 0.01;
- the search moves to the best neighbor only if it is strictly better;
- it stops after 1,000 steps, 10,001 evaluations with the starting point.

With steps that small, a search follows its basin down to the minimum at the bottom: first down the
steep walls towards the origin, then into one of the six basins. Restarting from random points is
the simplest way to find several minima: each start lands in some basin, and basins are found about
in proportion to their size. With 24% of the starts in a global basin, 30 searches all miss both
global minima with a probability of 0.76³⁰, about 1 in 4,000. The searches that end within 2% of
the bounds' width of each other, in both genes, are counted as one minimum.

## Output

The first line gives the number of searches, the bounds and the length of each search. Then a table
has a row per minimum the searches reached: the best point found there, rounded to 3 decimals, how
many searches ended there, and the best value, to 5 significant digits. The rows go from the lowest
value to the highest, then by x₁, and a row within 0.001 of the global minimum is marked `global`.
In Python, `run` evaluates the function in Rust, so both versions print the same table.

The page's plot shows the 30 searches as points on the function's contour, and a curve of the best
and the median value's distance above the global minimum, −1.0316285, on a logarithmic axis.

[The project page](https://tachsin.gr/projects/genoxide/examples/six-hump-camel) plays this run
back.

## Good results

A good result reaches both global minima, each with a value close to −1.0316285, and shows which
other minima catch the searches. The run finds all six minima: both global ones, with 2 and 8
searches; the second pair, −0.21546, with 6 and 7; and the third pair, 2.1043, with 3 and 4. Each
pair's points are the opposites of each other, to the 3 decimals printed.

At the start, the best of the 30 random points is 2.9 above the global minimum, and the median one
about 640 above it. The best search is within 1e-5 of the minimum by step 50, and within 1e-9 after
1,000 steps. By step 250, the median has settled at −0.21546, a search in the second pair, 0.816
above the global minimum on the plot.
