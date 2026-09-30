---
title: Pressure vessel design
category: constrained
summary: The cheapest cylindrical pressure vessel of a given volume, a mixed discrete-continuous engineering design problem.
reference: "Sandgren, E. (1990). Nonlinear integer and discrete programming in mechanical design optimization. Journal of Mechanical Design 112(2): 223-229."
reference_url: https://doi.org/10.1115/1.2912596
optimum: "6059.714335 (cost)"
languages: [rust, python]
order: 90
---

# Pressure vessel design

## The problem

Sandgren (1990) designs a cylindrical pressure vessel capped at both ends by hemispherical heads.
The goal is the lowest total cost of material, forming and welding. There are four design variables,
in inches: the thickness of the shell T_s, the thickness of the heads T_h, the inner radius R and
the length L of the cylinder. The thicknesses come in multiples of 0.0625 inch, the available steel
plates. The radius and the length are continuous.

In the form that Coello Coello (2000, Computers in Industry 41(2): 113-127) restates, in the
notation of Kannan and Kramer (1994, Journal of Mechanical Design 116(2): 405-411):

```text
minimize   0.6224 T_s R L + 1.7781 T_h R² + 3.1661 T_s² L + 19.84 T_s² R
subject to T_s ≥ 0.0193 R
           T_h ≥ 0.00954 R
           π R² L + 4/3 π R³ ≥ 1,296,000
           L ≤ 240
```

The first two constraints are the least thicknesses for the radius. The third is a volume of at
least 1,296,000 cubic inches, 750 cubic feet. The fourth limits the length.

## What makes it hard

The problem mixes discrete and continuous variables, under constraints. The volume constraint pushes
R and L up, the cost pulls them down, and the thickness constraints make a larger radius need
thicker plates. At the optimum, two constraints bind: the shell is exactly as thin as its radius
allows, and the volume is exactly 1,296,000. The other two have slack.

The discrete thicknesses make the cost a step function of them. Between plate sizes, it doesn't
change; at each size, it jumps.

## Representation

A `Real` genome of 4 genes: T_s and T_h in [0.0625, 6.1875] (1 to 99 plates), R and L in [10, 200].
genoxide's `PressureVessel` rounds the first two genes to the nearest whole plate when it evaluates
a genome, and its `design` method gives the rounded design. Any real-valued algorithm can then
search the genes.

The fitness is the cost and the total constraint violation, the sum of how far each constraint is
exceeded, in its own units. genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000,
Computer Methods in Applied Mechanics and Engineering 186: 311-338): a feasible design beats an
infeasible one, two feasible ones compare by cost, and two infeasible ones by violation.

## Algorithm

SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78), a differential evolution that adapts its
scale factor and crossover rate from successful trials, with genoxide's defaults: its published
population of 100, and a restart after 200 generations without progress. It stops once the cost is
within 1e-10 of the minimum, relative to its size, or after 200,000 evaluations.

The discrete thicknesses leave local minima: designs with other plates, whose best radius and
length cost more. With 50,000 evaluations, two fifths of the runs ended in one of them, 6090.526
with seeds 1, 6 and 7, or 6370.780 with seeds 2, 4, 14 and 20. Restarts get them out, given time:
with 200,000 evaluations, every seed tried ends at the minimum (see Good results).

## Output

The first line gives the cost of the best design, the evaluations the run took, and the known
minimum. The second gives the design's constraint violation; 0 means it's feasible. The third gives
the design: the shell and head thicknesses, rounded to whole plates, the radius and the length. In
Python, `run` evaluates the problem in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/pressure-vessel) plays this run back.

## Good results

The minimum cost is 6059.714335, at T_s = 0.8125 (13 plates), T_h = 0.4375 (7 plates), R = 42.098446
and L = 176.636596. Yang et al. (2013, International Journal of Bio-Inspired Computation 5(6):
329-335) proved it globally optimal. The run finds it, 5·10⁻⁷ above it, with no violation, and
meets the 1e-10 target after 140,998 evaluations. It first settles in a local minimum, 6090.526,
with a shell of 14 plates: its best design is there from about 32,000 evaluations. The population
restarts twice, and after the second restart, at about 110,000 evaluations, the best design moves
to the minimum's plates.

With seeds 1 to 500, every run ends on the minimum's plates, 13 and 7. 498 meet the target, after
33,600 to 199,000 evaluations, half of them within 46,500. The other two stop at the budget,
5·10⁻⁶ and 0.00014 above the minimum: a relative gap of at most 2·10⁻⁸.
