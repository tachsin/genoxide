---
title: Welded beam design
category: constrained
summary: The cheapest welded beam under stress, buckling and deflection limits, in the two forms of the literature.
reference: "Ragsdell, K. M. and Phillips, D. T. (1976). Optimal design of a class of welded structures using geometric programming. Journal of Engineering for Industry 98(3): 1021-1025."
reference_url: https://doi.org/10.1115/1.3438995
optimum: "1.7248523085993899 (seven constraints) and 2.3811341169090015 (five constraints), best known"
languages: [rust, python]
order: 111
trace_note: "Recorded from the seeded run below, on the first form, WeldedBeam."
---

# Welded beam design

## The problem

A bar is welded to a rigid support and carries a load of 6000 lb at its free end, 14 inches away.
Ragsdell and Phillips (1976) posed the problem of choosing its dimensions for the lowest cost. There
are four variables, in inches: the weld's thickness h and length l, and the bar's height t and
thickness b. In both forms of the literature, the cost is

```text
1.10471 h² l + 0.04811 t b (14 + l)
```

It grows with the size of the weld (h² l) and the volume of the bar (t b (14 + l)). The constraints
limit the shear stress in the weld to 13,600 psi, the bending stress in the bar to 30,000 psi and
the deflection of its end to 0.25 inch. The load must stay below the bar's buckling load, and the
weld can't be thicker than the bar (h ≤ b).

Two forms of the problem circulate in the literature:

- `WeldedBeam` has seven constraints: the five above, a limit on a cost-like term, 0.10471 h² +
  0.04811 t b (14 + l) ≤ 5, and h ≥ 0.125. It follows Rao (1996, Engineering Optimization, Wiley) as
  restated by Coello Coello (2000, Computers in Industry 41(2): 113-127). Its best known cost is
  1.7248523085993899: Cagnina, Esquivel and Coello Coello (2008, Informatica 32: 319-326) report
  1.724852, and a local solver (SLSQP) started from their design, printed to 6 digits and slightly
  infeasible, reaches it to full precision with every constraint met.
- `WeldedBeamRagsdell` has five, after Ragsdell and Phillips (1976) as restated by Deb (2000,
  Computer Methods in Applied Mechanics and Engineering 186: 311-338). Its best known cost is
  2.3811341169090015, a local solver's (SLSQP) minimum started from the design that genoxide's
  SHADE finds, with the active constraints tightened by 1e-7 and h = b, so that it meets them
  exactly. It isn't proven optimal. Reklaitis, Ravindran and Ragsdell
  (1983, Engineering Optimization: Methods and Applications, Wiley) report 2.38116.

The two forms also differ in the constants of their shear stress and buckling formulas, so the same
beam has different stresses in each. Their best costs aren't comparable.

## What makes it hard

The constraints are nonlinear, and several bind at once. At the best designs that the example finds
for both forms, four hold with equality: the shear stress, the bending stress, the buckling load and
h ≤ b. The deflection has slack. A search has to approach a corner where four curved boundaries
meet, from inside the feasible region or across it.

## Representation

A `Real` genome of 4 genes, (h, l, t, b), within each form's bounds: h and b in [0.1, 2] and l and t
in [0.1, 10] for `WeldedBeam`; h in [0.125, 10] and the others in [0.1, 10] for
`WeldedBeamRagsdell`. The fitness is the cost and the total constraint violation, the sum of how far
each constraint is exceeded. genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000): a
feasible design beats an infeasible one, two feasible ones compare by cost, and two infeasible ones
by violation.

## Algorithm

SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78), a differential evolution that adapts its
scale factor and crossover rate from successful trials, with genoxide's defaults: its published
population of 100, and a restart after 200 generations without progress. On each form, it stops
once the cost is within 1e-10 of the best known, relative to its size, or after 100,000
evaluations.

## Output

Two lines per form. The first gives the form, the cost of the best design, its constraint violation
(0 means feasible), the evaluations the run took and the best known cost. The second gives the design: h, l, t and b. In both, h
equals b: the constraint h ≤ b binds. In Python, `run` evaluates the problems in Rust, so both
versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/welded-beam) plays back the first form's run. Its
progress curve is the cost's error to the best known cost, on a log axis.

## Good results

On the seven-constraint form, the run meets the target after 45,000 evaluations, at 1.724852 to 7
digits, with no violation. On the five-constraint form, it meets it after 53,800, at 2.381134, also
with no violation. That's
below the 2.38116 that Reklaitis, Ravindran and Ragsdell report, whose design, published to 4
digits, costs 2.38151 as printed.

With seeds 1 to 20, every run meets its target: after 42,100 to 48,000 evaluations on the
seven-constraint form, 45,700 at the median, and after 49,700 to 56,700 on the five-constraint
form, 52,350 at the median.
