---
title: MMA on a million variables
category: local
summary: Minimize a sum of a million terms under one constraint with the method of moving asymptotes, the gradient-based method for very many variables and few constraints, to the closed-form minimum.
reference: "Svanberg, K. (1987). The method of moving asymptotes: a new method for structural optimization. International Journal for Numerical Methods in Engineering 24(2): 359-373."
reference_url: "https://doi.org/10.1002/nme.1620240207"
optimum: "(Σ √cₖ)² / V, at xⱼ = V √cⱼ / Σ √cₖ"
languages: [rust, python]
order: 282
---

# MMA on a million variables

## The problem

A budget V shared among n = 1,000,000 items, each with a cost cⱼ that falls as its share xⱼ grows:

```text
minimize   Σⱼ cⱼ / xⱼ
subject to Σⱼ xⱼ ≤ V
           0.01 ≤ xⱼ ≤ 10
```

with cⱼ = 1 + (j mod 9), from 1 to 9, and V = n, a share of 1 on average. The function is convex
and the constraint linear, so the minimum is where the gradients balance: cⱼ / xⱼ² = λ for every
j, with the constraint active. That gives the minimum in closed form,

```text
xⱼ = V √cⱼ / Σₖ √cₖ,   f* = (Σₖ √cₖ)² / V,   λ = (Σₖ √cₖ)² / V²
```

about 0.466 for the items of cost 1 and 1.399 for those of cost 9, inside the bounds. The example
computes it, and compares the run with it.

## What makes it hard

A million variables. A method that keeps an n × n matrix, such as BFGS or SQP, would need 8
terabytes for it; one that estimates the gradient by finite differences, a million evaluations per
gradient. Population-based methods need many evaluations per variable. What's left are methods
that use the gradient and keep a few vectors of n values: here, with a constraint, the method of
moving asymptotes.

## Representation

A `Real` genome of a million genes in [0.01, 10]. The fitness function is a
`Constrained::differentiable` closure: it returns the value, and writes the gradient
`−cⱼ / xⱼ²`, the constraint's value `Σ xⱼ − V` (at most 0) and its gradient, all ones.

## Algorithm

`Mma`: Svanberg's method of moving asymptotes (1987), as his notes on MMA and GCMMA (2007)
describe it. Each iteration replaces the function and the constraint by convex approximations
around the current point, built from their values and gradients there and from two asymptotes per
variable: each term is `p / (u − x) + q / (x − l)`, so the approximation is separable, a sum of
functions of one variable each. The asymptotes l and u move with the iterates: nearer where a
variable oscillates, which adds curvature and damps it, farther where it moves steadily.

The approximate problem is solved through its dual, in the constraint's single multiplier λ: for a
given λ, each variable's minimizer has a closed form, so the dual function and its derivatives are
sums over the variables, and a Newton method on λ takes a few of them. An iteration is then a few
passes over the million variables, with no matrix of them, and the run keeps 15 values per
variable. The sums run in fixed chunks, here on several threads (`parallel_sums(true)`), with the same
results as one after the other.

The run starts from xⱼ = 0.5 for every item, half the budget, with the asymptotes half the range
away. It stops when it has converged: when the KKT conditions hold to 1e-9 or a step moves no
variable by more than 1e-10 of its range.

## Output

A row every 4 iterations: the best value and the largest relative error of a variable of the
current point against the closed form. Then the iterations, the value against the minimum, the
largest error of a variable of the best point, the budget it uses, and the multiplier against λ.
In Python, the fitness function is numpy's, with its sums in the same order as Rust's
(`np.cumsum`), so both versions print the same.

The first iterations overshoot: the approximations of the far asymptotes are nearly linear, and
the steps move every variable as far as allowed, past the budget. The best stays the initial point
for 9 iterations, while the asymptotes close in on the oscillating variables. Then the iterates
converge: from iteration 20 on, each four iterations gain about three digits.

The run takes about 2.5 seconds on 20 threads, 5 on one.

## Good results

The minimum is f* = 4,601,497.0170, at xⱼ = V √cⱼ / Σ √cₖ. The run converges after 31
iterations and 32 evaluations, one per iteration: the best point's variables are within 2.7e-9 of
the closed form, relative to each, and its value agrees with f* to all eleven digits printed. It
uses the budget to 3e-6 (to 3e-12 of it), feasible, and the multiplier is λ to 1.5e-10.

Beyond 1e-11 relative the value can't be compared: its sum of a million terms and that of f* are
each rounded by about that much, and they differ in their last digits even at the same point.
