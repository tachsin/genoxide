---
title: L-BFGS-B on a minimum at the bound
category: local
summary: Minimize Rosenbrock's function in a box that cuts its valley, so that the minimum lies on the box's edge, and land on it exactly with L-BFGS-B.
reference: "Byrd, R. H., Lu, P., Nocedal, J. and Zhu, C. (1995). A limited memory algorithm for bound constrained optimization. SIAM Journal on Scientific Computing 16(5): 1190-1208."
reference_url: "https://doi.org/10.1137/0916069"
optimum: "0.25 (at (0.5, 0.25), on the bound x₁ = 0.5)"
languages: [rust, python]
order: 283
---

# L-BFGS-B on a minimum at the bound

## The problem

Rosenbrock's function in two dimensions,

```text
f(x₁, x₂) = 100 (x₂ − x₁²)² + (x₁ − 1)²,
```

in the box x₁ ∈ [−2, 0.5], x₂ ∈ [−1, 3], from the classic start (−1.2, 1). The unconstrained
minimum, (1, 1), is outside the box: the bound x₁ ≤ 0.5 cuts the valley before its end.

The minimum in the box is on that bound. With x₁ = 0.5, f = 100 (x₂ − 0.25)² + 0.25 is least at
x₂ = 0.25, so the minimum is f = 0.25 at (0.5, 0.25). There the gradient is (−1, 0): f falls only
by growing x₁, beyond the bound. That is the optimality condition of a bound-constrained problem (a
gradient that points out of the box, or is 0, in every gene), so (0.5, 0.25) is the minimum.

## What makes it hard

The minimum isn't a point where the gradient is 0, so a method can't find it by driving the
gradient to 0; it has to learn which bounds hold at the minimum (the active set) and land on them.
A method that only steps in the open box approaches the bound ever more closely without reaching
it.

## Representation

A `Real` genome of 2 genes with the bounds of the box. The fitness is f, to minimize: genoxide's
`problems::Rosenbrock` in 2 dimensions, with its analytic gradient, in this box.

## Algorithm

`Lbfgsb` with its defaults. Each iteration of L-BFGS-B (Byrd, Lu, Nocedal and Zhu, 1995) models
f by a quadratic and finds the first minimum of that model along the path of steepest descent,
bent at the bounds: the generalized Cauchy point. The genes that path holds at a bound are the
active set; the step then minimizes the model over the other genes, projected back into the box
(Morales and Nocedal, 2011), and a line search along it never leaves the box. Once x₁ reaches 0.5,
it stays there: the path of steepest descent holds it at the bound, and the steps move x₂ alone.

A run has converged when the projected gradient, the step of steepest descent cut back at the
bounds, is at most 1e-5 in every gene: at a minimum on a bound, the component that points out of
the box doesn't count.

Then, for contrast, Nelder-Mead from the same start in the same box: it mirrors trial points that
leave the box back into it.

## Output

The current point after each round: the evaluations, x₁, x₂, f, and the largest component of the
projected gradient. A round is one point the line search tries. Then where each method ended. In
Python, `run` evaluates the function in Rust, so both versions print the same.

The first rounds follow the valley up and to the right, as on the full function. At round 26, x₁
reaches 0.5 exactly and stays there; two rounds later x₂ is 0.25 and the projected gradient is 0.

[The project page](https://tachsin.gr/projects/genoxide/examples/lbfgsb-bounds) plays the run
back: the points the search stood at, on the contour of the function in the box.

## Good results

The minimum in the box is 0.25 at (0.5, 0.25). L-BFGS-B ends exactly there, (0.5, 0.25) with
f = 0.25 and a projected gradient of 0, after 29 evaluations, and stops as converged.

Nelder-Mead ends at (0.49999999999999956, 0.24999999980867693), f = 0.25000000000000044, after
257 evaluations: close, but on the inside of the bound, and nine times the evaluations without a
gradient to use.
