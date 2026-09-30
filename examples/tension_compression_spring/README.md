---
title: Tension/compression spring
category: constrained
summary: The lightest coil spring whose deflection, shear stress, surge frequency and outer diameter stay within their limits.
reference: "Belegundu, A. D. (1982). A Study of Mathematical Programming Methods for Structural Optimization. PhD thesis, University of Iowa."
reference_url: ""
optimum: "0.012665 (weight), best known"
languages: [rust, python]
order: 72
---

# Tension/compression spring

## The problem

A helical coil spring is wound from wire and loaded along its axis. The design has three
variables: the diameter d of the wire, the mean diameter D of the coils, both in inches, and the
number N of active coils, those that deflect under the load. The goal is the lightest spring. The
weight is taken as

```text
(N + 2) D d²
```

which is proportional to the volume of the wire: N + 2 coils (the active ones and one closed coil
at each end), each πD long, of cross-section πd²/4.

Four constraints, written g(x) ≤ 0, keep the spring usable:

```text
g1 = 1 − D³ N / (71785 d⁴)                                     the deflection
g2 = (4D² − dD) / (12566 (D d³ − d⁴)) + 1 / (5108 d²) − 1      the shear stress
g3 = 1 − 140.45 d / (D² N)                                     the surge frequency
g4 = (D + d) / 1.5 − 1                                         the outer diameter
```

- g1: under its load, the spring must deflect at least a given amount. The deflection grows with
  D³ N / d⁴: a thin wire, wide coils and many of them make a soft spring.
- g2: the shear stress in the wire must stay below the allowed stress.
- g3: the surge frequency, the frequency of waves along the spring, must stay above a limit, to
  avoid resonance.
- g4: the outer diameter, D + d, can't exceed 1.5 inches.

The problem comes from Belegundu (1982) and Arora (1989, Introduction to Optimum Design,
McGraw-Hill). genoxide's `TensionCompressionSpring` uses the definition and bounds that Coello
Coello restates (2000, Computers in Industry 41(2): 113-127): d in [0.05, 2], D in [0.25, 1.3] and
N in [2, 15]. N is continuous, as in the restatement. The best known weight, 0.012665, at d =
0.051690, D = 0.356750 and N = 11.287126, comes from Cagnina, Esquivel and Coello Coello (2008,
Informatica 32: 319-326). It's printed to 5 digits, and it isn't proven optimal. The published
design, printed to 6 digits, exceeds g2 by 2e-5.

## What makes it hard

The constraints pull against each other. A lighter spring needs less wire: a thinner wire, smaller
coils or fewer of them. But a thinner wire raises the shear stress (g2), and smaller or fewer coils
make the spring too stiff to deflect enough (g1). At the best designs found, both g1 and g2 hold
with equality: the lightest spring lies where their two curved boundaries meet. g3 and g4 have
slack.

The feasible region is small. Of 2 million designs drawn at random within the bounds, 0.75 % are
feasible; g1 alone holds for 1.7 %. The bounds are also far wider than the good designs: the best d
is 0.0517, in the lowest 0.1 % of its range [0.05, 2].

Near the minimum, the weight hardly changes along the curve where g1 and g2 meet: two feasible
designs with D = 0.3567 and D = 0.3581 differ in weight by 6e-8, in the sixth significant digit. A
search gets close fast and settles the last digits slowly.

## Representation

A `Real` genome of 3 genes, (d, D, N), within the bounds above. The fitness is the weight and the
total constraint violation, the sum of max(0, g) over the four constraints, 0 for a feasible
design. genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in
Applied Mechanics and Engineering 186: 311-338): a feasible design beats an infeasible one, two
feasible ones compare by weight, and two infeasible ones by violation. The rules need no penalty
weights.

## Algorithm

SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78), a differential evolution that adapts its
scale factor and crossover rate from successful trials, with genoxide's defaults: its published
population of 100, and a restart after 200 generations without progress. Differential evolution
builds each trial design from differences between designs of the population, so its steps shrink
as the population gathers along the curve of the minimum. It runs for 50,000 evaluations.

## Output

The first line gives the weight of the best design and the best known weight. The second gives its
constraint violation; 0 means it's feasible. The third gives the design: d, D and N. The last names
the constraints at their limit, within 1e-6 of 0. In Python, `run` evaluates the problem in Rust,
so both versions print the same.

The plot shows each variable on its range, and each constraint's value g: satisfied with its slack,
active (within 1e-6 of its limit) or violated. The best design is feasible from the first generation
on: the best of the 100 random designs of the first population is feasible. The weight falls to
0.0127 after about 10,000 evaluations and to 0.012666 after about 35,000. From about 26,000
evaluations, g1 and g2 come within 1e-6 of their limits, first in turn and from about 40,000 on
together; g3 and g4 keep their slack throughout. From about 49,000 evaluations, the median weight
matches the best to the 6 digits that the plot shows. The population is still moving along the
curve of the minimum when the budget runs out, without a restart: the best weight falls by 1.3e-7
over the last 5,000 evaluations.

[The project page](https://tachsin.gr/projects/genoxide/examples/tension-compression-spring) plays this run back.

## Good results

The best known weight is 0.012665, rounded. The run finds a feasible spring of weight 0.0126653
(0.01266530 to 7 significant digits), at d = 0.051747, D = 0.358122 and N = 11.207099, close to
the published design. It's above 0.012665 because that value is rounded, and 6e-8 above the
lightest spring that seeds 1 to 5 find, 0.01266523, because it's still settling its last digits.
The published design weighs 0.0126651, a little less, but it exceeds g2 by 2e-5: it's slightly
infeasible.

In the run's design, the deflection (g1) and the shear stress (g2) are at their limits. The surge
frequency is five times its limit, and the outer diameter is 0.41 inch, far below 1.5.

Runs with seeds 2 to 5 end between 0.01266523 and 0.01266548, on different points of the same
curve.
