---
title: Nelder-Mead with restarts on Himmelblau
category: local
summary: Find all four global minima of Himmelblau's function with a Nelder-Mead search that restarts from a random point each time it converges.
reference: "Himmelblau, D. M. (1972). Applied Nonlinear Programming. McGraw-Hill."
reference_url: ""
optimum: "0 (at four points)"
languages: [rust, python]
order: 281
---

# Nelder-Mead with restarts on Himmelblau

## The problem

Himmelblau's function (1972) is a sum of two squares in two variables:

```text
f(x₁, x₂) = (x₁² + x₂ − 11)² + (x₁ + x₂² − 7)²
```

It is 0 exactly where both squares are 0, at four points: (3, 2), (−2.805118, 3.131313),
(−3.779310, −3.283186) and (3.584428, −1.848127). These are its four global minima.

## What makes it hard

The four minima lie in four separate basins. A local method converges to the minimum of the basin
it starts in, so one run finds one minimum, and which one depends on the start. Finding all four
takes runs from starts in all four basins.

## Representation

A `Real` genome of 2 genes, each in [−5, 5], the box around the four minima. The fitness is f, to
minimize: genoxide's `problems::Himmelblau`, which also gives the four minima, computed to full
precision.

## Algorithm

`NelderMead` with `local::Restarts::Random { times: 19 }`: a Nelder-Mead simplex search (see the
[Nelder-Mead on Rosenbrock](../nelder_mead/) example for its steps) that starts again from a new
random point in the box each time its simplex has converged, 19 times. A run has converged when every
vertex of its triangle is within 1e-10 of the range of the best vertex in each gene. After the last
run, the engine stops with `StopReason::Converged`.

Each run starts from a random point with a triangle of 0.1 of the range on each side, 1 in both
genes, and follows its basin down. The best vertex of each run when it converges is assigned to the
nearest known minimum.

## Output

The first line gives the number of runs. Then a table has a row per known minimum: its
coordinates, how many runs ended at it, and the range of values they reached, from the best to the
worst. The last line gives the evaluations of all 20 runs. In Python, `run` evaluates the function in
Rust, so both versions print the same table.

[The project page](https://tachsin.gr/projects/genoxide/examples/nelder-mead-himmelblau) plays the
runs back: the triangle on the contour of the function, and where each run so far converged.

## Good results

Each minimum is worth 0. A good result finds all four, each within 1e-6 of 0. The 20 runs find all
four, 4 to 6 runs each, and every run ends between 2.1e-18 and 6.7e-18, after 141 evaluations on
average: 2,815 in all.

With seeds 1 to 300, every run of every seed ends at one of the minima, at most 2.4e-17 from 0, and
294 of the 300 seeds find all four. In the other 6, none of the 20 random starts fell in one of the
basins. More restarts make that rarer.

The [Himmelblau](../himmelblau/) example finds the same four minima with 20 hill climbers of
Gaussian steps, 30,001 evaluations each, ending between 3.4e-10 and 2.1e-7: the simplex adapts its
size as it closes in, where fixed steps can't.
