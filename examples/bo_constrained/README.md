---
title: Constrained Bayesian optimization
category: bayesian
summary: Find the minimum of Gramacy et al.'s toy problem, on the boundary of a wavy constraint, to within 1e-5 in 21 evaluations, with a Gaussian process per constraint and the probability of feasibility.
reference: "Gramacy, R. B., Gray, G. A., Le Digabel, S., Lee, H. K. H., Ranjan, P., Wells, G. and Wild, S. M. (2016). Modeling an augmented Lagrangian for blackbox constrained optimization. Technometrics 58(1): 1-11."
reference_url: "https://doi.org/10.1080/00401706.2015.1014065"
optimum: "0.599788 at (0.195123, 0.404665)"
languages: [rust, python]
order: 293
---

# Constrained Bayesian optimization

## The problem

The toy problem of Gramacy et al. (2016, section 1): minimize a linear function of two variables
in the unit square, subject to two nonlinear constraints,

```text
minimize   f(x) = x₁ + x₂,   x in [0, 1]²
subject to c₁(x) = 3/2 − x₁ − 2x₂ − ½ sin(2π(x₁² − 2x₂)) ≤ 0
           c₂(x) = x₁² + x₂² − 3/2 ≤ 0
```

The paper gives its global minimizer as about (0.1954, 0.4044), where f is about 0.5998, c₁ is
active and c₂ isn't, and two local minimizers, about (0.7197, 0.1411), f ≈ 0.8609, and (0, 0.75)
on the bound. `tests/reference/gramacy_toy.py` solves c₁ = 0 with the Lagrange condition, which
says the two partial derivatives of c₁ are equal there, with mpmath from the paper's points:
the minimum is 0.5997880520 at (0.1951226835, 0.4046653685), c₂ = −1.298 there, and a scan of the
box on a grid of 1/2000 confirms it's the global one. The paper's coordinates differ from these in
the fourth digit, and its f agrees: along the boundary, f changes slowly.

The objective is known and cheap here, but the method treats it as a black box, as for a
simulation whose output and constraints all come from one expensive run.

## What makes it hard

The minimum lies on the boundary of c₁, whose sine makes the feasible region's edge a wave: the
objective falls toward the infeasible corner (0, 0), so the best points are exactly where the
constraint stops them, and a search that ignores the constraint goes the wrong way. Bayesian
optimization has only the evaluations to learn where that boundary is.

## Representation

A `Real` genome of 2 genes in [0, 1]. The fitness function is a `constraint::Constrained` with 2
constraints: it writes c₁ and c₂ into a slice and returns x₁ + x₂, and genoxide makes the fitness
`(score, violation)` of it, the violation the sum of the positive values, while `Bo` reads the
values one by one. In Python, the function returns `(value, g)` and `run` takes `constraints=2`.
Both use genoxide's portable sine, so both versions give the same run.

## Algorithm

`Bo` with its defaults. With the constraints' values, it fits a Gaussian process to the objective
and one to each constraint, and maximizes the log expected improvement over the best feasible
point plus the logarithm of the probability that a point is feasible, `P(c₁ ≤ 0) P(c₂ ≤ 0)` under
the constraints' models: the expected constrained improvement of Gardner et al. (2014), the
product of the expected improvement and the probability of feasibility, through its logarithm.
Until a feasible point is evaluated, the search maximizes the probability of feasibility alone.
The run stops within 1e-5 of the minimum, or after 60 evaluations.

As a contrast, the same search with a fitness function that returns only `(score, violation)`:
without the values, `Bo` models the score alone, and only the best point found is chosen by Deb's
rules.

## Output

The first lines give the problem and the method. Then a row per evaluation: its number, the point,
x₁ + x₂, both constraints' values (feasible at 0 or below) and the best feasible value's distance
above the minimum so far. Evaluations 1 to 6 are the initial design; the rest are the points the
models chose. The last lines give the best feasible point found and how far it is from the
minimum, and the contrast's feasible points and best.

[The project page](https://tachsin.gr/projects/genoxide/examples/bo-constrained) plays the run back:
at each step, the probability of feasibility over the square with the points so far, and the
acquisition that chose the next point.

## Good results

A good result is feasible and within 1e-5 of 0.599788. Three points of the design are feasible,
the best 0.96; the models then probe the line x₁ = 0, where the objective is lowest, and from
evaluation 11 on, they walk the boundary c₁ = 0 to the minimum: 1.3e-2 above it at evaluation 11,
3.9e-4 at 15, 8.4e-6 at 21, (0.194733, 0.405064), with c₁ just below 0. That point is 5.6e-4 from
the minimizer: along the boundary, f changes slowly.

Over seeds 1 to 20, every search came within 1e-5, after 18 to 39 evaluations, half of them within
26; within 1e-3, after 13 to 35, half within 19.

The contrast doesn't come close: with only the violation, the search models x₁ + x₂ alone and
heads for the infeasible corner, where the function is lowest. In its 21 evaluations, 6 points are
feasible, and the best of them is 0.944530, 3.4e-1 above the minimum.
