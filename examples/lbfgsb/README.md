---
title: L-BFGS-B on Rosenbrock in 100 dimensions
category: local
summary: Minimize Rosenbrock's function in 100 dimensions to 1e-10 with L-BFGS-B, with its analytic gradient and with forward differences, and see what the gradient costs.
reference: "Byrd, R. H., Lu, P., Nocedal, J. and Zhu, C. (1995). A limited memory algorithm for bound constrained optimization. SIAM Journal on Scientific Computing 16(5): 1190-1208."
reference_url: "https://doi.org/10.1137/0916069"
optimum: "0 (at (1, …, 1))"
languages: [rust, python]
order: 282
---

# L-BFGS-B on Rosenbrock in 100 dimensions

## The problem

Rosenbrock's function, chained over 100 variables:

```text
f(x) = Σᵢ₌₁⁹⁹ 100 (xᵢ₊₁ − xᵢ²)² + (xᵢ − 1)²
```

Its minimum is 0, at (1, …, 1). Both runs start from the classic start, (−1.2, 1) repeated, where
f = 24,926, and stop once f ≤ 1e-10.

## What makes it hard

Each term couples a variable to the next, so the valley of the two-dimensional function becomes a
curved valley in 100 dimensions: a step that improves one pair spoils the next. The way to the
minimum bends at every variable, and a method has to learn that curvature as it goes. Here a
gradient-based method needs the gradient, a vector of 100 derivatives, at every point it tries.

## Representation

A `Real` genome of 100 genes in [−30, 30], the box of genoxide's `problems::Rosenbrock`, which
also gives the analytic gradient. The box never comes into play.

## Algorithm

`Lbfgsb`: L-BFGS-B (Byrd, Lu, Nocedal and Zhu, 1995), with the default memory of 10 correction
pairs. Each iteration models the function by its gradient and the curvature of the last 10 steps
(a limited-memory BFGS matrix), steps to the model's minimum within the box, and searches along
that step with the Moré-Thuente line search, which tries the full step first. Near the minimum,
the full step is accepted at nearly every iteration.

The gradient comes from one of two sources:

- **Analytic** (`Gradients::Auto`, as the problem supplies it): the function computes its gradient
  with its value, one evaluation per point.
- **Forward differences** (`Gradients::Forward`): 100 more points per point, each moving one
  gene by about 1.5e-8 of its value, evaluated in the same round. Their gradient is accurate to
  about 7 digits.

The tolerances are turned off (`gradient_tolerance(0.0)`, `function_tolerance(0.0)`), so each
run goes on until `Stop::target(1e-10)`.

## Output

A row every 50 rounds and at the end of each run: the evaluations so far and the best value, for
each source of the gradient. A round is one point the line search tries, with its gradient. Then
each run's iterations and evaluations, and the ratio of the evaluations. In Python, `run`
evaluates the problem in Rust, gradient included, so both versions print the same.

Round for round the two runs are nearly the same: forward differences change the gradient in its
eighth digit, and the path differs only near the end. The cost is not: 101 evaluations per round
against 1.

[The project page](https://tachsin.gr/projects/genoxide/examples/lbfgsb) plays the runs back:
the best value and the evaluations per round, for both sources of the gradient.

## Good results

The minimum is 0 at (1, …, 1). With the analytic gradient, the run reaches f = 3.0e-11 in 520
iterations and 613 evaluations. With forward differences, it reaches f = 9.9e-11 in 523 iterations
and 61,206 evaluations: 100 times as many, of which 60,600 compute the gradient.

Supply the gradient whenever there's one: for n genes, forward differences cost n times the
evaluations. They cost nothing to set up, so they suit a function without a formula for its
gradient, a few genes, or a fast function.
