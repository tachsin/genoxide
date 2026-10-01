---
title: Nelder-Mead on Rosenbrock
category: local
summary: Follow Rosenbrock's curved valley from the classic start (−1.2, 1) to its minimum with the Nelder-Mead simplex method, which needs no derivatives.
reference: "Nelder, J. A. and Mead, R. (1965). A simplex method for function minimization. The Computer Journal 7(4): 308-313."
reference_url: "https://doi.org/10.1093/comjnl/7.4.308"
optimum: "0 (at (1, 1))"
languages: [rust, python]
order: 280
---

# Nelder-Mead on Rosenbrock

## The problem

Rosenbrock's function in two dimensions:

```text
f(x₁, x₂) = 100 (x₂ − x₁²)² + (x₁ − 1)²
```

Its minimum is 0, at (1, 1). Rosenbrock (1960) started his searches from (−1.2, 1), where f is
24.2, and this example starts there too. The search is limited to x₁ in [−2, 2] and x₂ in [−1, 3],
the region where the plot is drawn; the bounds never come into play.

## What makes it hard

The minimum lies at the end of a narrow valley along the parabola x₂ = x₁². From (−1.2, 1), the
way to (1, 1) goes around the bend of the valley: first down and to the right, then up. Along the
floor f falls slowly, while across it f rises fast, so a method has to keep turning to stay inside.
A method that only compares values, without a gradient to point down the slope, has to work out
the direction from the points it tries.

## Representation

A `Real` genome of 2 genes, x₁ in [−2, 2] and x₂ in [−1, 3]. The fitness is f, to minimize:
genoxide's `problems::Rosenbrock` in 2 dimensions, used here with this smaller box.

## Algorithm

`NelderMead`: the simplex method of Nelder and Mead (1965), as Lagarias, Reeds, Wright and Wright
(1998) state it precisely. In two dimensions the simplex is a triangle. Each iteration takes its
worst vertex and tries the point opposite it, through the midpoint of the other two (the
reflection):

- if that point is better than the best vertex, a point twice as far out (the expansion);
- if it's no better than the second worst, a point between the midpoint and the better of the
  reflection and the worst vertex (a contraction);
- if even the contraction isn't better, the whole triangle shrinks towards its best vertex.

The accepted point replaces the worst vertex. The triangle stretches along the valley when steps
succeed, and contracts across it when they fail, so it turns with the bend without any derivative.
The coefficients are Gao and Han's (2012), which in two dimensions are the standard ones: 1 for
the reflection, 2 for the expansion, 1/2 for contractions and shrinks.

The first triangle is the start (−1.2, 1) and two copies moved by 0.1 of each gene's range: (−0.8,
1) and (−1.2, 1.4). The run has converged when every vertex is within 1e-9 of that first step
(1e-10 of the range) of the best vertex in each gene; the engine then stops with
`StopReason::Converged`.

A round is one batch of evaluations: one trial point, or the two vertices of a shrink. Then the same
run again with speculative asks: each iteration evaluates the reflection, the expansion and both
contractions in one round, so that a parallel engine can evaluate them at once.

## Output

A row every 20 rounds and the last: the evaluations, the best value, and the size of the triangle
(the largest difference between a vertex and the best one in a gene, as a fraction of its range).
The last line is the speculative run. In Python, `run` evaluates the function in Rust, so both
versions print the same.

The best value falls slowly at first, while the triangle travels along the valley, then fast once
it's at the minimum: from round 120 on, each 20 rounds gain about two or three digits. The size
grows again around round 80, when the triangle stretches along the straighter part of the
valley.

[The project page](https://tachsin.gr/projects/genoxide/examples/nelder-mead) plays this run back,
with the triangle drawn on the contour of the function.

## Good results

The minimum is 0 at (1, 1). The run converges after 234 evaluations, at (1, 1) to 10 decimal
places, with f = 1.7e-20. Speculative asks take the same triangles to the same end in 126 rounds
instead of 231, for 507 evaluations: worth it only when a round of 4 points in parallel costs about
as much as one.

From 1,000 random starts in the box, every run converges with f below 1e-12, the worst at 8.9e-20.
