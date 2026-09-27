---
title: Speed reducer
category: constrained
summary: Golinski's lightest gearbox, whose gear teeth and shafts stay within their stress and deflection limits, a mixed discrete-continuous design.
reference: "Golinski, J. (1973). An adaptive optimization system applied to machine synthesis. Mechanism and Machine Theory 8(4): 419-436."
reference_url: https://doi.org/10.1016/0094-114X(73)90018-9
optimum: "2996.348165 (weight), best known"
languages: [rust, python]
order: 73
---

# Speed reducer

## The problem

A speed reducer is a gearbox. A small gear, the pinion, on one shaft drives a larger gear on a
second shaft, which turns slower with a larger torque. Golinski posed the design of the lightest
such gearbox (1970, Journal of Mechanisms 5(3): 287-309; 1973). There are seven design variables,
in cm except for the teeth:

| Variable | Gene | Meaning | Bounds |
|---|---|---|---|
| b | x₁ | face width of the gears | [2.6, 3.6] |
| m | x₂ | module of the teeth (the pitch diameter per tooth) | [0.7, 0.8] |
| z | x₃ | number of teeth on the pinion, an integer | [17, 28] |
| l₁ | x₄ | length of shaft 1 between its bearings | [7.3, 8.3] |
| l₂ | x₅ | length of shaft 2 between its bearings | [7.8, 8.3] |
| d₁ | x₆ | diameter of shaft 1 | [2.9, 3.9] |
| d₂ | x₇ | diameter of shaft 2 | [5.0, 5.5] |

The weight to minimize adds up the gears and the two shafts:

```text
0.7854 b m² (3.3333 z² + 14.9334 z − 43.0934) − 1.508 b (d₁² + d₂²)
  + 7.4777 (d₁³ + d₂³) + 0.7854 (l₁ d₁² + l₂ d₂²)
```

Eleven constraints, written g(x) ≤ 0, keep the gearbox sound:

```text
g1  = 27 / (b m² z) − 1                                    bending stress of the teeth
g2  = 397.5 / (b m² z²) − 1                                surface stress of the teeth
g3  = 1.93 l₁³ / (m z d₁⁴) − 1                             deflection of shaft 1
g4  = 1.93 l₂³ / (m z d₂⁴) − 1                             deflection of shaft 2
g5  = √((745 l₁ / (m z))² + 16.9·10⁶) / (110 d₁³) − 1      stress in shaft 1
g6  = √((745 l₂ / (m z))² + 157.5·10⁶) / (85 d₂³) − 1      stress in shaft 2
g7  = m z / 40 − 1                                         size of the pinion
g8  = 5 m / b − 1                                          face width at least 5 modules
g9  = b / (12 m) − 1                                       face width at most 12 modules
g10 = (1.5 d₁ + 1.9) / l₁ − 1                              shaft 1 long enough for its diameter
g11 = (1.1 d₂ + 1.9) / l₂ − 1                              shaft 2 long enough for its diameter
```

The first two keep the teeth from breaking or wearing: their bending stress and the contact stress
on their surfaces. g3 to g6 limit how far each shaft bends and the stress in it. g7 to g11 are
rules of proportion: g7 limits the pinion's pitch diameter, m z, to 40.

genoxide's `SpeedReducer` uses the definition, bounds and best known solution that Cagnina,
Esquivel and Coello Coello restate (2008, Informatica 32: 319-326, appendix, problem E03). The best
known weight is 2996.348165, at b = 3.5, m = 0.7, z = 17, l₁ = 7.3, l₂ = 7.8, d₁ = 3.350214 and d₂ =
5.286683. It isn't proven optimal. The published design, printed to 6 digits, exceeds g5 by 6.0e-7
and g6 by 1.3e-7, and weighs 2996.347849.

The literature's formulations differ (Ray, 2003, AIAA Journal 41(3): 556-558). Some print 7.477
for 7.4777 and 1.5079 for 1.508, which don't give this value. Some let l₂ go down to 7.3, where the
minimum is 2994.471 (Lin, Tsai, Hu and Chang, 2013, Mathematical Problems in Engineering 419043). A
weight from another paper is comparable only if its formulation is the same.

## What makes it hard

The number of teeth is an integer, and the weight is a step function of it. One more tooth on the
pinion, 18 instead of 17, adds 177 to the weight of the best design.

The feasible region is small. Of 2 million designs drawn at random within the bounds, 0.2 % are
feasible. g8 alone rules out 99 %: the face width must be at least 5 modules, b ≥ 5m, and with b ≤
3.6 that leaves only m ≤ 0.72 of the range [0.7, 0.8].

The minimum lies on a vertex of the feasible region, where seven limits hold at once: four genes at
their lower bounds (m, z, l₁ and l₂) and three constraints (g5, g6 and g8). The shaft stresses set
the shaft diameters, and g8 sets the face width to 5 modules. A search has to reach this corner
from inside the feasible region, where a step toward it can cross a constraint.

## Representation

A `Real` genome of 7 genes, (b, m, z, l₁, l₂, d₁, d₂), within the bounds above. `SpeedReducer`
rounds the third gene to the nearest integer when it evaluates a genome, and its `design` method
gives the rounded design. Any real-valued algorithm can then search the genes: a gene of 17.13 is 17
teeth.

The fitness is the weight and the total constraint violation, the sum of max(0, g) over the eleven
constraints, 0 for a feasible design. genoxide compares fitnesses with Deb's feasibility rules
(Deb, 2000, Computer Methods in Applied Mechanics and Engineering 186: 311-338): a feasible design
beats an infeasible one, two feasible ones compare by weight, and two infeasible ones by violation.
The rules need no penalty weights.

## Algorithm

SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78), a differential evolution that adapts its
scale factor and crossover rate from successful trials, with genoxide's defaults: its published
population of 100, and a restart after 200 generations without progress. It runs for 50,000
evaluations.

Differential evolution suits a minimum at the bounds: genoxide's DE sets a trial gene that falls
outside its bounds halfway between its parent's gene and the bound. The population can then close
in on a bound, halving the distance at each such step, without leaving the box.

## Output

The first line gives the weight of the best design and the best known weight. The second gives its
constraint violation; 0 means it's feasible. The next three give the design: the face width, the
module and the number of teeth, then each shaft's length and diameter. The last names the
constraints at their limit, within 1e-6 of 0. In Python, `run` evaluates the problem in Rust, so
both versions print the same.

The plot shows each variable on its range, and each constraint's value g: satisfied with its slack,
active (within 1e-6 of its limit) or violated. The best of the first 100 random designs exceeds g5,
the stress in shaft 1; within 400 evaluations, the best design is feasible. Between about 11,000 and
13,000 evaluations, g6, g8 and then g5 reach their limits. After about 15,000 evaluations, the
design matches the best known one to the 6 digits that the plot shows. SHADE then restarts twice,
after about 20,000 and 38,500 evaluations (the median weight jumps back up), and each time the
population comes back to the same design.

[The project page](https://tachsin.gr/projects/genoxide/examples/speed-reducer) plays this run back.

## Good results

The run finds a feasible design of weight 2996.348167, 2e-6 above the best known 2996.348165: the
design b = 3.5, m = 0.7, z = 17, l₁ = 7.3, l₂ = 7.8, d₁ = 3.350215 and d₂ = 5.286683. The shaft
stresses (g5, g6) and the least face width (g8) are at their limits. Setting d₁ and d₂ so that g5
and g6 hold exactly gives 2996.348165, the best known value: that's the minimum at this vertex.

Runs with seeds 2 to 5 end between 2996.348168 and 2996.348170, at the same design.
