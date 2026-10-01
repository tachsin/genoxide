---
title: Continuation by Gaussian smoothing
category: local
summary: Minimize a tilted Rastrigin function with L-BFGS-B through 6 stages of its Gaussian smoothing, from convex to exact, to the global minimum, where L-BFGS-B alone from the same start stays in the nearest basin.
reference: "Blake, A. and Zisserman, A. (1987). Visual Reconstruction. MIT Press."
reference_url: ""
optimum: "f* = 0.82084153378296, computed gene by gene to the last bit"
languages: [rust, python]
order: 286
---

# Continuation by Gaussian smoothing

## The problem

A Rastrigin function tilted by a quadratic centered off its lattice, in 10 dimensions:

```text
f(x) = Σᵢ (xᵢ − aᵢ)² + 10 (1 − cos 2πxᵢ),   a = (1.3, −0.7, 2.2, −1.6, 0.35, 3.25, −2.8, 0.7, −0.3, 1.8)
```

Each gene has a well near every integer, and the quadratic makes the well nearest aᵢ the deepest.
The function is separable, so its global minimum is computed exactly, gene by gene: in each well
around an integer k near aᵢ, where the term is convex (|x − k| ≤ 1/4), the root of its derivative
2(x − aᵢ) + 20π sin 2πx by bisection to the last bit, and of those the lowest. The minimum is
f* = 0.82084153378296, at x ≈ (1.0015, −0.9985, 2.0010, −1.9980, 0.0018, 3.0013, −2.9990, 0.9985,
−0.0015, 1.9990).

## What makes it hard

A well near every integer of the box [−5, 5] in every gene: 11¹⁰, about 2.6 × 10¹⁰ local minima,
each a trap for a local method. From xᵢ = −3, L-BFGS-B settles in the well near −3 of every gene.

Smoothing removes them. The function averaged over a Gaussian of standard deviation σ,
E[f(x + σz)], has a closed form, since E[cos 2π(x + σz)] = e^(−2π²σ²) cos 2πx:

```text
f_σ(x) = Σᵢ (xᵢ − aᵢ)² + σ² + 10 (1 − e^(−2π²σ²) cos 2πxᵢ)
```

Its second derivative is at least 2 − 40π² e^(−2π²σ²), so f_σ is convex for σ above 0.517, with
its minimum near a; and f₀ = f. Graduated smoothing (Blake and Zisserman's graduated
non-convexity) follows the minimum from the convex function to the exact one: σ = 0.6, 0.4, 0.3,
0.2, 0.1 and 0, each stage started where the last ended. As σ falls, the wells come back around
the point, and it slides into the one nearest a, the deepest.

The stages move the point: the smoothed minimum lies between aᵢ and the deepest well, and closes
in on the well as σ falls (0.38 from the global minimum at σ = 0.6, then 4.1e-2, 9.6e-3, 2.4e-3
and 4.4e-4); the last stage, on f itself, ends at it.

## Representation

A `Real` genome of 10 genes in [−5, 5], starting from xᵢ = −3. The fitness is f_σ, to minimize,
with its gradient, 2(xᵢ − aᵢ) + 20π e^(−2π²σ²) sin 2πxᵢ: `Differentiable` in Rust and
`gradient=True` in Python. The stage's σ is a value the fitness function reads, an
`Arc<AtomicU64>` in Rust and a variable in Python. The cosines, sines and exponential are
genoxide's portable ones, and the terms are summed in the same order in both, so both versions
take the same steps.

## Algorithm

`Continuation` around `Lbfgsb`, through 6 stages. A stage ends when L-BFGS-B has converged (its
projected gradient within 1e-10); then the `on_stage` closure sets the next σ, the point is
evaluated again on the new function, and L-BFGS-B goes on from it. It drops its curvature pairs
between stages by default: they describe the last stage's function. The run stops by itself
(`StopReason::Converged`) after the last stage.

Then, as contrasts from the same start: σ = 0 at once, L-BFGS-B on f alone; and the stages with
the curvature pairs kept (`keep_pairs(true)`).

## Output

A row per stage: its σ, its rounds and evaluations (the first evaluation of a stage re-evaluates
its start), the stage's own minimum value, and the distance to the global minimum (the largest
difference of a gene) where it ended. The last stage ends within 2.7e-15 of the global minimum,
at f* to all 14 printed digits.

[The project page](https://tachsin.gr/projects/genoxide/examples/continuation) plays the three
runs back: the distance to the global minimum and the stage's σ at every round.

## Good results

The global minimum f* = 0.82084153378296, reached in 36 evaluations.

| Run, from xᵢ = −3 | Rounds (evaluations) | Result |
|---|---|---|
| 6 stages of smoothing (this example) | 30 (36) | f*, within 2.7e-15 of the global minimum |
| σ = 0 from the start | 5 (6) | trapped at f = 146.38192099, 145.56 above f*, 6.0 from the global minimum |
| 6 stages, L-BFGS-B's pairs kept | 50 (56) | within 4.4e-9 of the global minimum |

L-BFGS-B alone converges fast, to the wrong minimum: the well it starts in. The stages cost 30
evaluations more, and find the global one. Keeping the curvature pairs across stages costs 20
evaluations more and digits of accuracy: the pairs describe the last stage's function, whose
curvature changes most where the wells come back, and mislead the first steps of the next.

Smoothing finds the global minimum here because the deepest well is the one nearest the minimum
of the convex smoothing, a property of this function (a quadratic with a periodic ripple), not of
smoothing in general: for functions without that structure, graduated smoothing leads to a good
minimum, not always the best.
