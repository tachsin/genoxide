---
title: Bayesian optimization of Branin
category: bayesian
summary: Find a global minimum of Branin's function in 31 evaluations, with a Gaussian process that chooses each point by the log expected improvement and a final L-BFGS-B polish of the model's mean.
reference: "Jones, D. R., Schonlau, M. and Welch, W. J. (1998). Efficient global optimization of expensive black-box functions. Journal of Global Optimization 13(4): 455-492."
reference_url: "https://doi.org/10.1023/A:1008306431147"
optimum: "5 / (4π) ≈ 0.397887 (at three points)"
languages: [rust, python]
order: 290
---

# Bayesian optimization of Branin

## The problem

Branin's function, in the form Dixon and Szegö (1978) give it, is

```text
f(x₁, x₂) = (x₂ − 5.1 x₁² / (4π²) + 5 x₁ / π − 6)² + 10 (1 − 1 / (8π)) cos x₁ + 10
```

on x₁ in [−5, 10] and x₂ in [0, 15]. Its three global minima, all worth 5 / (4π) ≈ 0.397887, are at
(−π, 12.275), (π, 2.275) and (3π, 2.475). It's one of the problems on which Jones, Schonlau and
Welch (1998) introduced efficient global optimization, the method this example runs: Bayesian
optimization with the expected improvement.

Here the function stands for an expensive one, a simulation that runs for minutes or an
experiment, where every evaluation counts. The question isn't how fast the search runs but how
few evaluations it needs.

## What makes it hard

Nothing, for a method with thousands of evaluations to spend: the [Branin](../branin/) example
finds all three minima with 30 hill climbers of 10,001 evaluations each. With tens of
evaluations, every point has to be chosen with care: where the function is probably low, and
where too little is known to tell. The values range from 0.4 to over 300, while the minima's
basins are narrow valleys.

## Representation

A `Real` genome of 2 genes, x₁ in [−5, 10] and x₂ in [0, 15]: genoxide's `problems::Branin`,
whose box and minima are those above, evaluated in Rust in both languages.

## Algorithm

`Bo`, genoxide's Bayesian optimization, with its defaults:

- **The initial design**: 2(n + 1) = 6 points of a Latin hypercube, one in each sixth of each
  gene's range.
- **The model**: a Gaussian process with a constant mean and Matérn's 5/2 kernel, a length scale
  per gene, fitted to every evaluation so far by maximizing its marginal likelihood (Rasmussen and
  Williams 2006), without noise, since the function is deterministic: the model passes through
  every value.
- **The next point**: the maximum of the log expected improvement (Ament et al. 2023), the
  logarithm of how much a point is expected to beat the best value under the model, computed
  so that it keeps a gradient where the expected improvement itself underflows. It's evaluated
  at 1,000 random points, then maximized by L-BFGS-B from the best 10 of them and from the best
  point so far.

After 30 evaluations, a Gaussian process of all of them is fitted, and its posterior mean,
smooth and cheap, is minimized by L-BFGS-B with its gradient from the best point found. The
function is evaluated once at the result, the 31st evaluation.

## Output

The first lines give the problem and the method. Then a row per evaluation: its number, the point,
the function's value and its distance above the global minimum. Generation 0 is the six points of
the design; each later row is a point the model chose. The last lines give the polish: the
point where the model's mean is lowest, the mean there, the function's value, its distance above
the nearest minimum, and the best of the 31 evaluations. In Python, `run` evaluates the function
in Rust and the model is genoxide's, so both versions print the same rows.

[The project page](https://tachsin.gr/projects/genoxide/examples/bayesian-optimization) plays the
run back: at each step, the model's mean over the box with the points so far, the log expected
improvement that chose the next point, and a curve of the best value's distance above the minimum.

## Good results

A good result is within 1e-4 of a global minimum, 0.397887, in tens of evaluations. The first
point the model chose, the 7th evaluation, is already 2.5 above it; the 16th is 0.043 above it,
near (π, 2.275), and from there the search visits all three basins, the 24th within 1.3e-3 of
(−π, 12.275), the 29th within 9.6e-5 of (3π, 2.475). The model of the 30 evaluations has its
lowest mean at (9.422974, 2.475867), 0.397917, and the function there is 0.397909, 2.1e-5 above
the minimum: 31 evaluations in all.

Over seeds 1 to 20, the search reaches 1e-3 in a median of 30 evaluations, every seed within 80.
After 30 evaluations, 4 searches are within 1e-4, and 14 with the polish; after 35, 18, and 20
with the polish. The polish costs one evaluation: the model, which passes through every value,
is most precise near the best points, where the search has evaluated most.
