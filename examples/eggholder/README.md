---
title: Eggholder
category: continuous
summary: Minimize the eggholder function, a two-dimensional landscape of deep local minima whose deepest lies on the edge of the box, from 30 seeds with CMA-ES and particle swarms.
reference: "Whitley, D., Mathias, K., Rana, S. and Dzubera, J. (1996). Evaluating evolutionary algorithms. Artificial Intelligence 85(1-2): 245-276."
reference_url: "https://doi.org/10.1016/0004-3702(95)00124-7"
optimum: "−959.6407 at (512, 404.2318) (best known)"
languages: [rust, python]
order: 78
---

# Eggholder

## The problem

The eggholder function is a function of two variables to minimize:

```text
f(x₁, x₂) = −(x₂ + 47) sin √|x₂ + x₁ / 2 + 47| − x₁ sin √|x₁ − (x₂ + 47)|,   x₁, x₂ in [−512, 512]
```

It comes from Whitley, Mathias, Rana and Dzubera (1996, section 4.2), who built test functions
that are hard for evolutionary algorithms: it is their F101, with this formula, on [−512, 511] with
10 bits per variable, and without a minimum in 2 dimensions. The name and the bounds [−512, 512]
are those of Mishra (2006, MPRA paper 2718), which later papers follow. On Whitley et al.'s
[−512, 511], x₁ = 512 is outside the box, and the minimum is the next one below, −956.9182.

genoxide's `problems::Eggholder` gives the minimum as −959.6406627208508 at (512,
404.2318051137578): on the bound x₁ = 512, where the derivative in x₂ is 0, computed to 40 digits by
Newton's method. Mishra (2006) gives it as 959.64 at (512, 404.2319), and Jamil and Yang (2013)
repeat it: the sign is lost, and x₂ is 404.2318 to 4 decimals. It's the best known minimum, not
proven global: the best of the minima that local searches from the lowest points of a fine grid
find.

## What makes it hard

Each term is a sine of the square root of a distance, times a factor that grows across the box: the
function is a field of wells whose depth grows towards the edges, and whose width grows with the
distance from two lines, x₂ = −x₁ / 2 − 47 and x₂ = x₁ − 47, where the square roots have a kink.
On a grid of 4,097 × 4,097 points over the box, more than a thousand points are lower than their
eight neighbors, and the deepest minima are spread out near the edges:

| x₁ | x₂ | f |
|---|---|---|
| 512 | 404.2318 | −959.6407 |
| 482.3533 | 432.8790 | −956.9182 |
| 479.0454 | 434.5059 | −955.2552 |
| 439.4810 | 453.9774 | −935.3380 |
| −465.6942 | 385.7167 | −894.5789 |
| 347.3270 | 499.4154 | −888.9491 |

The best minimum is on the bound. The next two are inside the box, 41 and 45 away from it and 2.7
and 4.4 worse, on the two sides of the kink x₂ = x₁ − 47. The fifth is at the other side of the
box, 978 away.

## Representation

A `Real` genome of 2 genes, each in [−512, 512]: the point (x₁, x₂) itself. The fitness is f, to
minimize. The function, its bounds and its best known minimum are genoxide's `problems::Eggholder`.

## Algorithm

Three algorithms, each from seeds 1 to 30, with a budget of 50,000 evaluations per run and a target
within 1e-6 of the best known minimum:

- particle swarm optimization (Kennedy and Eberhart, 1995, Proceedings of ICNN'95: 1942-1948) with
  80 particles and Clerc and Kennedy's constriction coefficients (2002, IEEE Transactions on
  Evolutionary Computation 6(1): 58-73), on a ring: each particle follows the best of itself and
  its two neighbors, so that good points spread slowly and the swarm explores for longer;
- the same swarm with the global topology, in which every particle follows the best point of the
  whole swarm;
- for contrast, CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) with
  IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776), with genoxide's defaults: it
  samples a population of 6 from a normal distribution and adapts its mean, step size and
  covariance matrix, and a run that has converged starts again from a random point with twice the
  population.

The swarm is larger than the usual 20 to 50 particles, and the budget larger than most runs need:
with 40 particles and 20,000 evaluations, the ring didn't reach the minimum in 16 of 1,000 runs
(seeds 1 to 1,000), and with 80 particles and 50,000 evaluations it reaches it in all of them.

The runs whose best points end within 2% of the bounds' width of each other, in both genes, are
counted as one group.

## Output

The first line gives the best known minimum, the seeds and the budget. Then a table has a row per
group of runs: the best point found in it, to 1 decimal, its value, and how many runs of each
algorithm end there, from the lowest value to the highest. The last row counts the runs that come
within 1e-6 of the best known minimum. In Python, `run` evaluates the function in Rust, so both
versions print the same table.

The page's plot shows the 30 runs of the swarm with the global topology, each at its best point so
far, over the function's contour, and a curve of the best and the median run's distance above the
best known minimum, on a logarithmic axis.

[The project page](https://tachsin.gr/projects/genoxide/examples/eggholder) plays this run back.

## Good results

A good result reaches −959.6407 in every run. The swarm on a ring does, in all 30 runs. The swarm
with the global topology reaches it in 27, and ends twice at −894.58, on the other side of the box,
and once at −821.20.

CMA-ES with IPOP restarts reaches it in none of its 30 runs. Its restarts converge into many
different local minima, and the corner's narrow wells are rarely among them. The best points of its
runs near the corner, down to −940.64, are samples that fell near a deep minimum on the way, without
the run converging there: CMA-ES's best point isn't always where it converged.

Over seeds 1 to 1,000, the swarm on a ring reaches the minimum in every run, after at most 32,800
evaluations, and with the global topology in 93% of the runs. Over seeds 1 to 300, CMA-ES with IPOP
restarts reaches it in 2%.
