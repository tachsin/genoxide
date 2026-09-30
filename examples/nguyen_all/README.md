---
title: Nguyen-1 to 12
category: genetic programming
summary: All twelve of Nguyen's symbolic regression problems, by the same genetic programming search, with and without linear scaling.
reference: "Uy, N. Q., Hoai, N. X., O'Neill, M., McKay, R. I. and Galván-López, E. (2011). Semantically-based crossover in genetic programming: application to real-valued symbolic regression. Genetic Programming and Evolvable Machines 12(2): 91-119."
reference_url: https://doi.org/10.1007/s10710-010-9121-2
optimum: "Each formula, exactly (an RMSE of 0, up to rounding); this search recovers 11 of the 12 in some runs"
languages: [rust]
order: 258
family: Nguyen
tab: All twelve
---

# Nguyen-1 to 12

## The problems

The twelve symbolic regression problems of Uy et al. (2011), known as the Nguyen problems: find a
formula from points of it, all with Koza's function set (addition, subtraction, multiplication,
protected division, sine, cosine, the exponential and a protected logarithm) and no constants.

| Problem | Target | Points |
|---|---|---|
| Nguyen-1 | x³ + x² + x | 20 in [−1, 1] |
| Nguyen-2 | x⁴ + x³ + x² + x | 20 in [−1, 1] |
| Nguyen-3 | x⁵ + x⁴ + x³ + x² + x | 20 in [−1, 1] |
| Nguyen-4 | x⁶ + x⁵ + x⁴ + x³ + x² + x | 20 in [−1, 1] |
| Nguyen-5 | sin(x²) cos(x) − 1 | 20 in [−1, 1] |
| Nguyen-6 | sin(x) + sin(x + x²) | 20 in [−1, 1] |
| Nguyen-7 | ln(x + 1) + ln(x² + 1) | 20 in [0, 2] |
| Nguyen-8 | √x | 20 in [0, 4] |
| Nguyen-9 | sin(x) + sin(y²) | 100 in [−1, 1]² |
| Nguyen-10 | 2 sin(x) cos(y) | 100 in [−1, 1]² |
| Nguyen-11 | xʸ | 100 in [0, 1]² |
| Nguyen-12 | x⁴ − x³ + y²/2 − y | 100 in [−1, 1]² |

A formula is recovered when its error is at the level of rounding (10⁻¹⁰ of the values' standard
deviation) on the training points and on five times as many test points from the same range that the
search never sees. The problems' names, targets, sampling and function set are as McDermott et al.
(2012) restate them, not yet checked against the paper. Their points come from fixed seeds of
genoxide's random stream: another library's differ, and so can how hard a problem is.

The Nguyen-1, 5 and 9 tabs have a page each: the problems this search recovers in nearly every run,
without being as easy as Nguyen-10 and 11.

There's no Python version: the Python package has no genetic programming yet.

## What makes them hard

Different things. The polynomials grow harder with their degree: each term is a product the search
has to build, and near misses of sines and exponentials fit the smooth curve closely. The formulas of
Nguyen-5, 7 and 12 need constants the set doesn't have (− 1, the 1s of x + 1, the ½ of y²/2), built
from x / x or not at all. Nguyen-8, √x, has no square root in the set: it needs
exp(½ ln x), and the ½ again. And the two-variable problems have twice the leaves to choose from,
but their formulas are sums and products of one term per variable, which a search can improve a part
at a time.

Linear scaling (the error of a + b × tree, a and b fitted by least squares; Keijzer 2003) removes a
constant offset and factor: it supplies Nguyen-5's − 1 and Nguyen-10's 2. But it also makes every
tree with the right shape up to a shift and a stretch as good as the formula, and on the polynomials
many trees of sines and exponentials have that shape.

## Representation and algorithm

The search of the Nguyen-1, 5 and 9 pages and of Koza's quartic, on each problem: trees of the
problem's primitives (`gp::regression::problems::Nguyen1` to `Nguyen12`) with Koza's limits (depth
17, 1024 nodes) and ramped half-and-half initialization, in eight islands of 500 trees in a ring that
pass their two best trees on every 10 generations. Each island has tournaments of 7, subtree
crossover at a rate of 0.9 and subtree mutation at a rate of 0.1. The fitness is the RMSE on the
training points (`gp::regression::Regression`), without and then with linear scaling, and each run
stops at exact recovery or after 200 generations.

## Output

A line per problem, with the result of its run without linear scaling (the tree itself) and with it:
"recovered at" and the generation of exact recovery, or the test RMSE of the best tree when the run
ended without it. It's one run per problem and setting (the islands' seeds 100 to 107); the table
below gives 20.

## Good results

Over 20 runs per problem and setting (the islands' seeds 100 s + 0 to 7 for s = 1 to 20, the first
being `output.txt`'s), the runs that recovered the formula:

| Problem | Tree itself | Linear scaling |
|---|---|---|
| Nguyen-1 | 20 | 13 |
| Nguyen-2 | 10 | 7 |
| Nguyen-3 | 11 | 0 |
| Nguyen-4 | 3 | 0 |
| Nguyen-5 | 8 | 18 |
| Nguyen-6 | 19 | 4 |
| Nguyen-7 | 3 | 0 |
| Nguyen-8 | 12 | 1 |
| Nguyen-9 | 20 | 19 |
| Nguyen-10 | 20 | 20 |
| Nguyen-11 | 20 | 20 |
| Nguyen-12 | 0 | 0 |

Every problem but Nguyen-12 is recovered in some runs, and 8 in most runs with the better setting.
Linear scaling helps where the target has an offset the set can't build cheaply (Nguyen-5), costs
little where the formula is a short sum (Nguyen-9 to 11), and hurts on the rest: on the polynomials,
Nguyen-6, 7 and 8, the scaled error rewards trees of the right shape, and the search settles on
one that isn't the formula. Nguyen-10 needs the 2 from scaling or builds it, in 6 generations in the median
against 0 or 1 with scaling; Nguyen-10 and 11 are recovered in the first generations, some in the
initial population. The runs that don't recover a formula end on trees of 54 to 823 nodes, 262 in the median, the
bloat that the [accuracy against size](../accuracy_and_size/) page trades off against the error.

The same polynomial is also harder here than in [Koza's quartic](../koza_quartic/): Nguyen-2 is the
quartic, recovered in 10 of 20 runs here and 96 of 100 there, with the same search. Only the 20
points differ: Nguyen-2's lie mostly in [−1, 0.3], where the quartic is small and flat and near
misses fit it as well as the formula, while Koza's include nine in [0.5, 1], where it rises steeply.
