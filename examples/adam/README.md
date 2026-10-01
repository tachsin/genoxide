---
title: Adam with a learning-rate schedule
category: local
summary: Smooth 100,000 noisy points into a curve, its 100,000 values the parameters, with Adam and a learning rate lowered by control, to the exact answer.
reference: "Kingma, D. P. and Ba, J. (2015). Adam: a method for stochastic optimization. ICLR 2015."
reference_url: "https://arxiv.org/abs/1412.6980"
optimum: "the curve x* the data are made from, at distance 0"
languages: [rust, python]
order: 285
---

# Adam with a learning-rate schedule

## The problem

Smoothing: 100,000 noisy points dᵢ, at tᵢ evenly spread over [0, 1], become a curve xᵢ that
follows them without their noise, by penalized least squares:

```text
f(x) = Σᵢ (xᵢ − dᵢ)² + λ Σᵢ (xᵢ₊₁ − xᵢ)²,    λ = 50
```

The first sum keeps the curve near the data, the second keeps it smooth. Each of the 100,000
values xᵢ is a parameter.

The data are made so that the answer is known exactly. The curve

```text
x*ᵢ = sin(2π tᵢ) + 0.3 sin(10π tᵢ) + εᵢ,    εᵢ uniform in [−0.01, 0.01]
```

is smooth but for a little noise of its own, and the data are dᵢ = x*ᵢ + λ (L x*)ᵢ, with L the
Laplacian of the chain of points ((L x)ᵢ = 2xᵢ − xᵢ₋₁ − xᵢ₊₁ inside, one neighbor at the ends). The
gradient of f is 2(x − d) + 2λ L x, so it's 0 exactly where (I + λ L) x = d: at x*, the only
minimum, since f is a convex quadratic. The data's noise is the curve's, amplified up to 1 + 4λ =
201 times by L: about ±2 around a curve of amplitude 1.3.

## What makes it hard

Many parameters, and coupled ones: each value is pulled toward its data point and toward its two
neighbors. The Hessian, 2(I + λ L), has curvatures from 2 to 2(1 + 4λ) = 402, so the curve's
rough components settle in a few steps and its smooth ones in hundreds. A method that stores a
matrix of the parameters (10¹⁰ entries) is out of the question, and a line search would cost
evaluations every step.

## Representation

A `Real` genome of 100,000 genes in [−10, 10], starting from a flat curve, all zeros. The fitness
is f, to minimize, with its gradient: `Differentiable` in Rust, and `gradient=True` in Python, where
the function returns the value and the gradient. The gradient is computed in the same order of
operations in both, so both versions take the same steps.

## Algorithm

`FirstOrder` with `Step::Adam`: Adam (Kingma and Ba 2015, Algorithm 1), with β₁ = 0.9, β₂ = 0.999
and ε = 1e-8. Each step keeps averages of the gradient and of its square per value, corrects them
for their start at 0, and moves each value by about the learning rate α in the direction of the
averaged gradient, scaled by its typical size. One gradient per step, and the memory of two vectors.

The learning rate starts at 0.05 and is halved every 500 steps, by `Engine::control`
(`set_learning_rate` in Rust, `running.learning_rate` in Python). The run stops by itself
(`StopReason::Converged`) when no component of the gradient exceeds 1e-9.

Then the same number of steps again, with the learning rate kept at 0.05, as a contrast.

## Output

A row every 100 steps and the last: the learning rate, the loss f, the largest component of the
gradient, and the distance to the answer, the largest |xᵢ − x*ᵢ|.

For 500 steps at 0.05, Adam gets within about 1e-2 of the answer and stays there: its steps are
too long for the stiffest components of the curve, which it keeps overshooting, so the gradient
doesn't shrink. Halved to 0.025, the steps fit, and every 100 steps gain more than two digits,
until the gradient is below 1e-9 after 925 steps: every value is then within 2.5e-12 of the
answer. With the learning rate kept at 0.05, the same 925 steps end within 1.3e-2.

[The project page](https://tachsin.gr/projects/genoxide/examples/adam) plays both runs back: the
distance to the answer and the learning rate at every step.

## Good results

The minimum is f(x*) = 50,609.073…, at distance 0 from x*. The run converges in 925 steps, one
gradient each, within 2.5e-12 of x*; a step takes about 1.5 ms (the method's own share, 0.7 ms,
is linear in the parameters).

Adam's strength is a step of about α per parameter whatever the scale of its gradient, and an
average that smooths noisy gradients (mini-batches): on this deterministic, well-scaled quadratic
the others do as well or better. With the same convergence test, from the same start:

| Step rule | Steps to converge |
|---|---|
| Adam, α = 0.05 halved every 500 steps (this example) | 925 |
| Adam, α constant at 0.025 or 0.04 | none in 200,000 (stays within 4e-3 to 6e-3) |
| Gradient descent, α = 0.0049 (just below the stability limit 2 / 402) | 2,202 |
| Polyak's momentum, α = 0.005, μ = 0.9 | 433 |
| Nesterov's accelerated gradient, α = 0.003, μ = 0.9 | 367 |

`Lbfgsb`, the quasi-Newton method with a line search, learns the curvatures from its last
gradients: from the same start, with the gradient tolerance at 1e-9, it stops after 93 iterations
and 218 evaluations, within 4.0e-7 of the answer, when its line search can no longer lower the
loss: near the minimum, the loss of 50,609 changes by less than its rounding, while the gradient
still points the way. It's the better choice for a smooth deterministic function, where a few
digits of the answer suffice or the loss is small. Adam and momentum follow the gradient alone,
which brings them closer here, and suit the cases a line search can't serve: gradients that are
noisy estimates, such as a model trained on mini-batches, or evaluations too costly to spend on a
line search.
