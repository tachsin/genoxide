---
title: Cantilever beam
category: constrained
summary: The lightest beam of five hollow square segments that carries a load at its free end, under a limit on its deflection, a convex problem with a proven minimum.
reference: "Fleury, C. and Braibant, V. (1986). Structural optimization: a new dual method using mixed variables. International Journal for Numerical Methods in Engineering 23(3): 409-428."
reference_url: https://doi.org/10.1002/nme.1620230307
optimum: "1.339956361 (weight), proven"
languages: [rust, python]
order: 75
---

# Cantilever beam

## The problem

A cantilever is a beam held at one end and free at the other. This one is made of five hollow square
segments of equal length, joined end to end. The first segment is fixed to a rigid support, and a
vertical load acts at the free end of the fifth. The design variables are the widths x₁ to x₅ of the
segments, from the support to the free end. Fleury and Braibant (1986) posed the problem of making
the beam as light as possible while limiting how far its free end deflects. genoxide's
`CantileverBeam` follows the restatement of Yang, Huyck, Karamanoglu and Khan (2013, International
Journal of Bio-Inspired Computation 5(6): 329-335, eqs. 35-37), which hasn't yet been checked
against the original:

```text
minimize   0.0624 (x₁ + x₂ + x₃ + x₄ + x₅)
subject to 61/x₁³ + 37/x₂³ + 19/x₃³ + 7/x₄³ + 1/x₅³ ≤ 1
           0.01 ≤ xᵢ ≤ 100
```

The formulas read as a beam whose walls have a fixed thickness. A segment's weight then grows with
its width, and its bending stiffness with the cube of its width. The coefficients 61, 37, 19, 7 and
1 are 5³ − 4³, 4³ − 3³, 3³ − 2³, 2³ − 1³ and 1³. By beam theory, a segment k segment lengths from
the load adds to the deflection of the free end in proportion to k³ − (k − 1)³, divided by its
stiffness. The segment at the support carries the largest bending moment, and counts 61 times as
much as the last one. The constant 0.0624 and the limit 1 absorb the material, the lengths and the
load; the restatement gives no units.

The problem is convex: the weight is linear, and the constraint is a convex function of positive
widths. Its minimum follows from the Lagrange conditions. At the minimum, aᵢ / xᵢ⁴ is the same for
every segment, where aᵢ are the coefficients 61 to 1. That gives xᵢ = s^(1/3) aᵢ^(1/4), with s =
Σ aᵢ^(1/4), and the weight 0.0624 s^(4/3) ≈ 1.339956361. Yang et al. (2013) derive it this way and
report 1.339956367 at (6.0160159, 5.3091739, 4.4943296, 3.5014750, 2.15266533); genoxide computes
the closed form. It's a proven minimum.

## What makes it hard

Not much, for an optimizer that knows the formulas: the problem is convex, with a single constraint.
For a search that only sees the fitness, the difficulty is precision.

The constraint is active at the minimum: the lightest beam deflects exactly as far as the limit
allows. The search has to approach a curved boundary that it can't cross. Near the minimum, the
weight changes little along that boundary. A beam that stays on the boundary but misses the
minimum's widths by a small δ is heavier only in proportion to δ². The weight can then be right to
9 digits while the widths are right to only 4 or 5.

The bounds are wide. The minimum's widths lie between 2.2 and 6.0, in a box up to 100. Most of the
box is feasible and heavy: 88 % of random designs meet the deflection limit, and the lightest
feasible one of a million random designs weighs 1.93, 44 % above the minimum. A single segment can
break the limit on its own: x₁ below 61^(1/3) ≈ 3.94 does, whatever the others are.

## Representation

A `Real` genome of 5 genes, the widths x₁ to x₅, each in [0.01, 100]. The fitness is the weight and
the constraint violation, max(0, g(x)) for the constraint written as g(x) = 61/x₁³ + … + 1/x₅³ − 1
≤ 0; it's 0 for a feasible beam. genoxide compares fitnesses with Deb's feasibility rules (Deb,
2000, Computer Methods in Applied Mechanics and Engineering 186: 311-338): a feasible beam beats an
infeasible one, two feasible ones compare by weight, and two infeasible ones by violation.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, its step size and its covariance matrix, which
learns how the genes interact. It uses genoxide's defaults: a population of 4 + ⌊3 ln 5⌋ = 8, a
step size of 0.3 of each gene's range, a random start and no restarts. It ranks each population by
Deb's rules, and runs for 8,000 evaluations.

CMA-ES suits a smooth problem with a single minimum, where the widths interact through the
constraint. With seeds 1 to 5, it comes within 1e-8 of the minimum, relative to it, after 3,300 to
4,200 evaluations, and within 1e-12 after 5,000 to 7,400. SHADE (Tanabe and Fukunaga, 2013, IEEE
CEC 2013: 71-78), with genoxide's defaults as in the welded beam example, needs 33,700 to 35,800
evaluations for 1e-8. Its restarts then begin whenever the population's weights agree to about
1e-8, and after 100,000 evaluations it's still about 1e-9 above the minimum, with widths off by up
to 1e-4. L-SHADE (Tanabe and Fukunaga, 2014,
IEEE CEC 2014: 1658-1665), with a budget of 20,000 evaluations, reaches 1e-12 after 15,700 to
16,400.

## Output

The first line gives the weight of the best beam, the evaluations, and the minimum. The second gives
the beam's constraint violation; 0 means it's feasible. Then comes a row per segment, from the
support to the free end: the best beam's width and the minimum's. The last line gives the
constraint's left side, which the limit caps at 1. In Python, `run` evaluates the problem in Rust,
so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/cantilever-beam) plays this run back.

## Good results

The minimum weight is 1.339956361, at widths 6.016016, 5.309174, 4.494330, 3.501475 and 2.152665,
from the support to the free end. The run reaches it to the 9 printed decimals, with every width
right to the 6 printed decimals and no violation. The deflection constraint is exactly at its
limit, 1.000000. The first feasible beam comes with the first population: the search approaches
the boundary from inside.

The widths shrink from the support to the free end, since the segments near the support bend under
larger moments. Each segment's width follows the fourth root of its coefficient: the minimum makes
the last bit of material in every segment reduce the deflection by the same amount.
