---
title: CEC 2006 g14
category: constrained
summary: A sum of logarithmic terms in 10 variables under 3 linear equality constraints, with bounds open at 0 and some optimal genes a thousand times smaller than others, solved by CMA-ES with Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "−47.7648884594915 (best known, with the equalities met within 0.0001)"
languages: [rust, python]
order: 90
---

# CEC 2006 g14

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 problems,
g01 to g24, from the literature, with their best known solutions and rules for comparing algorithms.
g14 is the fourteenth, the report's equations 28 and 29 (page 7). The report takes it from
Himmelblau (1972, Applied Nonlinear Programming, McGraw-Hill).

There are 10 variables, x1 to x10, each in (0, 10]. With s = x1 + x2 + … + x10, the problem is

```text
minimize   f(x) = Σᵢ xᵢ (cᵢ + ln(xᵢ / s))
subject to h1(x) = x1 + 2 x2 + 2 x3 + x6 + x10 − 2 = 0
           h2(x) = x4 + 2 x5 + x6 + x7 − 1 = 0
           h3(x) = x3 + x7 + x8 + 2 x9 + x10 − 1 = 0
```

with c = (−6.089, −17.164, −34.054, −5.914, −24.721, −14.986, −24.1, −10.708, −26.662, −22.179).
Each term xᵢ ln(xᵢ / s) is negative, since xᵢ < s, and the constants cᵢ reward some variables more
than others. The three equalities are linear: each is a plane, and together they leave a flat set of
7 dimensions.

The bounds are open at 0, since the logarithm of 0 is undefined. genoxide closes them at
`f64::MIN_POSITIVE`, 2.2 · 10⁻³⁰⁸, the smallest positive normal number, where every logarithm is
finite. At 0 itself, a term is 0 · ln 0 and the fitness is invalid.

Real-valued samples almost never meet an equality exactly. The report counts an equality as met when
|h(x)| ≤ 0.0001, and so does genoxide's `G14`: each plane becomes a slab 0.0002 thick in its h. The
best known value is f* = −47.7648884594915, at the report's x* = (0.0406684113216282,
0.147721240492452, 0.783205732104114, 0.00141433931889084, 0.485293636780388, 0.000693183051556082,
0.0274052040687766, 0.0179509660214818, 0.0373268186859717, 0.0968844604336845). It meets the
equalities within the tolerance only, and genoxide's docs mark it as a best known value, not a
proven optimum. As printed, the report's x* exceeds the tolerance by 1e-15 in h1, from rounding its
digits.

## What makes it hard

The feasible region has no volume without the tolerance, and hardly any with it: a random point of
the box lies in all three slabs practically never. A search has to reach them first, guided by the
violation.

The minimum is also badly scaled. Its genes range from 0.78 down to 0.0014 and 0.00069, on a range
of 10: the smallest is 14,000 times smaller than the range, and a search must place them to a few
digits. Near 0, the logarithms are steep: the slope of the term xᵢ ln(xᵢ / s) grows without bound as
xᵢ approaches 0, so small genes change f far more than their size suggests.

One thing makes it easier. f is convex: Σ xᵢ ln(xᵢ / s) is the relative entropy of x to the vector
whose every entry is s, which is convex in x, and Σ cᵢ xᵢ is linear. The slabs are convex too, so
the problem has no local minima but the global one, and a search can't be trapped away from it.

## Representation

A `Real` genome of 10 genes, x1 to x10, in [2.2 · 10⁻³⁰⁸, 10]. genoxide's `problems::cec2006::G14`
is the fitness: the value f(x) and the total constraint violation,
max(0, |h1| − 0.0001) + max(0, |h2| − 0.0001) + max(0, |h3| − 0.0001), 0 for a feasible solution.
The tolerance is the report's, `genoxide::problems::cec2006::EQUALITY_TOLERANCE`.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults: a population of 4 + ⌊3 ln 10⌋ = 10, a step size of 0.3 of each gene's range, a random
start and no restarts. A sample outside the bounds is drawn again, up to 100 times, and then clipped
to them: to 2.2 · 10⁻³⁰⁸ at the lower bound, never to 0. Deb's rules rank the samples.

The run has the report's budget of 500,000 evaluations, and stops once its best solution is feasible
with an error f(x) − f* of at most 1e-8, an absolute error: 2e-10 of |f*|. The report counts a run
as successful with an error of at most 1e-4; the example asks for more.

Why CMA-ES: its covariance matrix learns the directions of the slabs, and the scales of the genes,
from 0.78 to 0.0007. On a convex problem, it doesn't need restarts. With seeds 1 to 25, in the same
budget:

| Algorithm | Runs that met the target | Evaluations (median, range) |
|---|---|---|
| CMA-ES | 25 of 25 | 17,290 (15,180 to 19,580) |
| SHADE | 25 of 25 | 54,500 (50,600 to 62,900) |
| L-SHADE | 24 of 25 | 78,595 (73,845 to 82,235) |

SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78) is genoxide's default differential
evolution. It misses the target in 9 of 25 runs on g11 and in all on g13, whose equalities are
curved, but solves g14: the slabs are flat, so the difference between two feasible solutions points
along them, not off them. It takes three times as many evaluations as CMA-ES. L-SHADE (Tanabe and
Fukunaga, 2014, IEEE CEC 2014: 1658-1665), its variant with a population that shrinks over the
budget, starts with 18 × 10 = 180 solutions and is slower still; one of its runs ended feasible
but 2.0 above f*.

## Output

The first line names the run. The second gives what stopped it, after how many evaluations, the
error f(x) − f* and whether the best solution is feasible: "< 1e-8" means the run met its target.
The third gives the evaluations to the first feasible solution, and to an error of 1e-4, the
report's criterion of success. The fourth compares f(x) with f*, to 6 significant digits. The fifth
gives the solution, to 4 significant digits, and the last |h| of each equality, met when it's at
most 0.0001. In Python, `run` evaluates the problem in Rust, so both versions print the same.

The page's plot shows each variable on its range, and each constraint's state from its violation,
max(0, |h| − 0.0001): violated until the best solution reaches the slabs, and met from then on. Its
curve shows the error f − f* of the best feasible solution, and of the population's median, on a log
scale. The median's curve has gaps: in about 40 % of the generations, half or more of the 10 samples
fall outside the slabs.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g14) plays this run back.

## Good results

A good run is feasible and ends within 1e-4 of f*, the report's success. A value below f* is
possible, since f* is only the best known, but the runs here end just above it.

The recorded run finds its first feasible solution after 1,920 evaluations, with an error of 4.9. It
then descends in stages. From 3,700 to 5,500 evaluations, the error stays between 2.3 and 2.8, while
x4 and then x6 dip far below their optimal values, to 0.00006 and 0.00005, and come back. It falls
to 1.0 after 7,000 evaluations and to 0.001 after 10,900. The run meets the report's criterion after
11,330 evaluations and its target after 15,650. The solution is x = (0.04067, 0.1477, 0.7832,
0.001413, 0.4853, 0.0006958, 0.02740, 0.01796, 0.03733, 0.09689), within 1.1e-5 of the report's x*
in every gene, with each |h| at 0.0001, on the edges of the tolerance.
