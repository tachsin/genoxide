---
title: Car side impact
category: constrained
summary: The lightest car body whose side passes the European side-impact test, from seven panel thicknesses under ten limits on the crash dummy's injuries and the structure's velocities.
reference: "Gu, L., Yang, R. J., Tho, C. H., Makowski, M., Faruque, O. and Li, Y. (2001). Optimisation and robustness for crashworthiness of side impact. International Journal of Vehicle Design 26(4): 348-360."
reference_url: https://doi.org/10.1504/IJVD.2001.005210
optimum: "23.585658 (weight), best known"
languages: [rust, python]
order: 76
---

# Car side impact

## The problem

In a side-impact test, a barrier hits the side of a car, and a crash-test dummy in the seat records
the loads on its body. Gu, Yang, Tho, Makowski, Faruque and Li (2001) made a car body as light as
possible while it still passes the European side-impact test. Crash simulations are slow, so they
fitted a response surface, a low-degree polynomial of the thicknesses, to each response of a set of
simulations. The optimization then works on the polynomials.

The seven design variables are thicknesses, in millimetres, of parts of the body's side:

| Gene | Part | Range |
|---|---|---|
| x₁ | B-pillar inner | 0.5 to 1.5 |
| x₂ | B-pillar reinforcement | 0.45 to 1.35 |
| x₃ | floor side inner | 0.5 to 1.5 |
| x₄ | cross members | 0.5 to 1.5 |
| x₅ | door beam | 0.875 to 2.625 |
| x₆ | door beltline reinforcement | 0.4 to 1.2 |
| x₇ | roof rail | 0.4 to 1.2 |

The B-pillar is the post between the front and the rear door. The weight to minimize is linear in
the thicknesses:

```text
1.98 + 4.9 x₁ + 6.67 x₂ + 6.98 x₃ + 4.01 x₄ + 1.78 x₅ + 0.00001 x₆ + 2.73 x₇
```

Ten constraints keep the dummy's injuries and the intrusion into the car within limits:

| Constraint | Limit |
|---|---|
| the abdomen load | 1 kN |
| the upper, middle and lower chest velocities | 0.32 m/s each |
| the upper, middle and lower rib deflections | 32 mm each |
| the pubic force | 4 kN |
| the velocity of the B-pillar's middle point | 9.9 mm/ms |
| the velocity of the front door | 15.7 mm/ms |

Each response is a polynomial of degree 1 or 2 in the thicknesses; genoxide's docs of
[`CarSideImpact`](https://docs.rs/genoxide/latest/genoxide/problems/engineering/struct.CarSideImpact.html)
give them all. genoxide follows the restatement of Jain and Deb (2014, IEEE Transactions on
Evolutionary Computation 18(4): 602-622, appendix of the authors' version), which hasn't yet been
checked against the original. There the weight is the first of three objectives. The original has
four more variables, two materials and the barrier's height and hitting position, which the
surfaces fix. Two coefficients are as published in the restatement and the implementations that
follow it, and differ from the eleven-variable restatements: the abdomen load's 0.0092928 x₃, and
the lower chest's 0.031296 x₃. Neither constraint is active at the best designs, so neither changes
them.

The restatement gives no optimum. genoxide's best known weight, 23.585658, is the corner described
below, stored to the last bit: SLSQP from 2,000 random starting points finds no other minimum. It
isn't proven optimal.

## What makes it hard

The constraints are nonlinear: they contain products of thicknesses and a square. Only 18 % of
random designs meet all ten. The lower rib deflection rules out 62 % of them, the pubic force 60 %
and the lower chest velocity 50 %. Most responses grow as the parts get thinner, so a lighter
design comes closer to the limits: the search has to reach the boundary of the feasible region and
stay on it.

The lightest designs sit in a corner. Four thicknesses are at their lower bounds: the B-pillar
inner, the floor side inner, the door beam and the roof rail. Three responses are at their limits:
the lower rib deflection, the pubic force and the front door's velocity. That's seven active
constraints for seven genes, which fix the design.

One gene barely counts. The door beltline reinforcement x₆ adds 0.00001 to the weight per
millimetre: taking it from its lowest feasible value, about 0.8842, to its upper bound, 1.2, adds
only 3.2e-6. Of the ten limits, only the front door's velocity needs it larger. The weight hardly
steers the search toward the best x₆, and a search that converges early leaves x₆ wherever it
happens to be.

## Representation

A `Real` genome of 7 genes, the thicknesses, within their ranges. The fitness is the weight and the
total constraint violation: the sum over the ten constraints of how far each response exceeds its
limit, 0 for a feasible design. genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000,
Computer Methods in Applied Mechanics and Engineering 186: 311-338): a feasible design beats an
infeasible one, two feasible ones compare by weight, and two infeasible ones by violation.

## Algorithm

L-SHADE (Tanabe and Fukunaga, 2014, IEEE CEC 2014: 1658-1665), a differential evolution that adapts
its scale factor and crossover rate from successful trials. Its population starts at 18 times the
number of genes, 126, and shrinks linearly to 4 over the budget of 20,000 evaluations. genoxide's
`De::l_shade` takes L-SHADE's settings, so the example only gives it the budget.

Its shrinking population gets to the best known weight sooner than SHADE (Tanabe and Fukunaga,
2013, IEEE CEC 2013: 71-78) with genoxide's defaults, as in the welded beam example, whose
population stays at 100. With seeds 1 to 5, L-SHADE comes within 1e-12 of the best known weight,
relative to it, after 18,200 to 19,800 evaluations. SHADE, which doesn't restart on the way,
ends 2e-11 to 9e-11 above it after 30,000 evaluations.

CMA-ES, which solves the cantilever beam example, puts the other six thicknesses on their bounds and
limits, but leaves x₆ where it happens to be: between 0.93 and 1.12 with seeds 1 to 5. After 30,000
evaluations, it's 2e-8 to 1e-7 above the best known weight, relative to it.

## Output

The first line gives the weight of the best design, the evaluations, and the best known weight. The
second gives the design's constraint violation, 0 when it's feasible, and the relative gap to the
best known weight, (weight − best known) / best known; a negative gap is a lighter design. Then
comes a row per part: its thickness in the best design and its range. The last rows give each
response of the best design next to its limit. In Python, `run` evaluates the problem in Rust, so
both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/car-side-impact) plays this run back.

## Good results

The best known weight is 23.585657980780084, at (0.5, 1.225732, 0.5, 1.207111, 0.875, 0.884189,
0.4). The run ends at 23.585657981, with no violation: a relative gap of 3.1e-13, 7e-12 in weight.
Of the other four seeds, two end at it exactly and two 3e-16 above it.

At the best known design, all seven constraints of the corner hold with equality. With x₁, x₃, x₅
and x₇ on their lower bounds, the lower rib deflection gives
x₂ = (46.36 − 4.4505 · 0.5 − 32) / 9.9 = 1.225732, the pubic force gives x₄ = 1.207111, and the
front door's velocity gives x₆ = 0.884189.
genoxide stores the corner to the last bit: each of x₂, x₄ and x₆ is the smallest floating-point
number whose limit holds as genoxide computes it, so its violation is exactly 0. The run's design
is the same to the printed digits.

At the corner, the weight's gradient is a combination of the seven active constraints' gradients
with positive coefficients, the Lagrange multipliers, so no feasible move nearby makes the design
lighter: it's a local minimum. The constraints aren't convex, so that doesn't prove it's the global one.

genoxide's earlier best known design had x₆ = 0.884329, where the front door's velocity is 1.0e-4
below its limit: 3.3e-9 heavier, a gain far below the accuracy of response surfaces fitted to crash
simulations. The run finds the corner exactly, down to the gene that barely counts.
