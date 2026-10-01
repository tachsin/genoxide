---
title: Continuation, from a smooth maximum to the exact one
category: local
summary: Find the center of the smallest ball around 420 points by minimizing a smoothed largest distance, sharper in each of 4 stages, Adam's state kept from one to the next, to the exact center.
reference: "Pólya, G. (1913). Sur un algorithme toujours convergent pour obtenir les polynomes de meilleure approximation de Tchebycheff pour une fonction continue quelconque. Comptes Rendus de l'Académie des Sciences 157: 840-843."
reference_url: ""
optimum: "the center c, at largest distance 1"
languages: [rust, python]
order: 286
---

# Continuation, from a smooth maximum to the exact one

## The problem

The smallest ball around a set of points: the point x whose largest distance to them is least, a
minimax problem,

```text
minimize  max_i |x − aᵢ|
```

in 10 dimensions, with 420 points made so that the answer is known exactly: 20 points at distance
1 from a center c along each axis, c ± eᵢ, and 400 points within 0.3 of c, all on one side of it
(each coordinate moved by up to 0.3/√10, upwards).

The answer is c, at largest distance 1. From any other point x = c + u, the axis point opposite
the largest component of u, c − sign(uᵢ) eᵢ, is at squared distance |u|² + 2|uᵢ| + 1 > 1; and the
near points are at most 0.3 from c.

## What makes it hard

The largest distance has a kink wherever two points tie for the largest, and at the answer all 20
axis points tie: no gradient leads to it. The usual remedy smooths the maximum by a p-norm,

```text
F_p(x) = (Σᵢ |x − aᵢ|^2p)^(1/2p)
```

which is smooth, and tends to the largest distance as p grows (Pólya used it in 1913 to reach
best uniform approximations through least p-th powers). But a small p averages over the points:
the minimum of F₂ is pulled 1.3e-2 towards the 400 near points. A large p from the start is sharp
everywhere, with the steep sides of a near-kink. So the problem is solved in stages, p = 2, 4, 8
and 16, each started where the last ended: continuation.

The last stage ends at the exact answer, because the stages' minima approach c fast. Each F_p is
convex, and at c the axis points' pulls cancel in opposite pairs, so only the near points pull,
each weighted by (its squared distance / 1)^(p−1) ≤ 0.09^(p−1) against the axis points' 1: for
p = 16, 0.09¹⁵ ≈ 2e-16 each, which moves the minimum less than 10⁻¹³ from c, below the run's
tolerance.

## Representation

A `Real` genome of 10 genes in [−2, 2], starting from (−1.5, …, −1.5). The fitness is F_p², the
smoothed largest squared distance, to minimize, with its gradient: `Differentiable` in Rust, and
`gradient=True` in Python, where the function returns the value and the gradient. The stage's p
is a shared value the fitness function reads: an `Arc<AtomicU32>` in Rust, a variable in Python.
With M the largest squared distance, F_p² = M (Σ rᵢ^p)^(1/p) with rᵢ ≤ 1, which can't overflow;
p = 2^k, so the powers are k squarings and the root k square roots, which round the same on every
platform, and both versions sum in the same order.

## Algorithm

`Continuation` around `FirstOrder` with `Step::Adam` (Kingma and Ba 2015, β₁ = 0.9, β₂ = 0.999,
learning rate 0.05), through 4 stages. A stage ends when Adam has converged (no component of the
gradient above 1e-11, or a step within 1e-12); then the `on_stage` closure sets the next p, Adam's
point is evaluated again on the changed function, and Adam goes on from there with its state:
its averages of the gradient and of its square, and the step count of their corrections
(`Keep::State`, the default). The run stops by itself (`StopReason::Converged`) after the last
stage.

Then, as contrasts, the same stages keeping only the point (`Keep::Point`: Adam's averages start
again at 0 in each stage), and p = 16 from the start.

## Output

A row per stage: its p, its steps and evaluations (the first evaluation of a stage re-evaluates
its start), F_p at its end, and the distance to the center (the largest difference of a
coordinate) where it ended.

The stages close in on the center: 1.3e-2 at p = 2, 2.3e-5 at p = 4, then 5.1e-11 and 1.1e-11,
the last within the tolerance of the answer, its largest distance 1 + 1.1e-11. The value F₁₆ can't
show such digits: it changes by less than its rounding within about 1e-8 of c, and Adam, which
steps by the gradient alone, still resolves the center to 1e-11.

[The project page](https://tachsin.gr/projects/genoxide/examples/continuation) plays the three
runs back: the distance to the center and the stage's p at every step.

## Good results

The answer is c, at largest distance 1. The stages converge in 1,251 steps and 1,255 evaluations,
within 1.1e-11 of c.

| Run | Steps (evaluations) | Distance to c |
|---|---|---|
| 4 stages, Adam's state kept (this example) | 1,251 (1,255) | 1.1e-11 |
| 4 stages, only the point kept | 1,853 (1,857) | 9.7e-13 |
| p = 16 from the start | 517 (518) | 2.6e-12 |
| p = 2 alone | 529 (530) | 1.3e-2 |

Keeping the state saves a third of the steps: with only the point kept, each stage restarts Adam's
averages at 0, and its first steps are then a full learning rate long, 0.05 per coordinate, away
from a point already within 2.3e-5 or 5e-11 of the stage's minimum, which takes 390 to 470 steps
to undo. With the state kept, the later stages take 409, 284 and 29 steps.

On this problem, p = 16 from the start costs fewer steps still: F₁₆ is convex, and its minimum
is already c, so the smoother stages have nothing to show it. Continuation pays where the sharp
function is hard to start on (several local minima, or gradients that vanish far from the answer,
as with sharp projections and penalty weights): the smooth stages lead it to the right minimum,
and each later stage starts close to its own. This example shows the mechanism, the state carried
over and its measured effect, on a problem whose answer is known to the last digit.
