---
title: CEC 2006 g21
category: constrained
summary: The linear function x1 of 7 variables under a nonlinear inequality and 5 nonlinear equalities, which leave one free variable and two local optima, solved with SHADE and Deb's feasibility rules at an ε level that falls to 0, with which 98 % of the runs reach the better optimum.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "193.724510070035 (best known, with the equalities met within 0.0001)"
languages: [rust, python]
order: 97
family: "CEC 2006"
tab: g21
---

# CEC 2006 g21

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 test
problems, g01 to g24, with their best known solutions and rules for comparing algorithms. g21 is the
twenty-first, the report's equations 43 and 44 (page 13). The report takes it from Epperly's
collection of global optimization test problems with solutions (the report's reference 6). It has 7
variables:

```text
x1 in [0, 1000]    x2, x3 in [0, 40]    x4 in [100, 300]    x5 in [6.3, 6.7]
x6 in [5.9, 6.4]   x7 in [4.5, 6.25]
```

Minimize f(x) = x1, subject to one inequality, g(x) ≤ 0, and five equalities, each h(x) = 0:

```text
g1 = −x1 + 35 x2^0.6 + 35 x3^0.6
h1 = −300 x3 + 7500 x5 − 7500 x6 − 25 x4 x5 + 25 x4 x6 + x3 x4
h2 = 100 x2 + 155.365 x4 + 2500 x7 − x2 x4 − 25 x4 x7 − 15536.5
h3 = −x5 + ln(−x4 + 900)
h4 = −x6 + ln(x4 + 300)
h5 = −x7 + ln(−2 x4 + 700)
```

The report counts an equality as met when |h(x)| ≤ 0.0001, and so does genoxide's `G21`. The best
known value is f* = 193.724510070035, at the report's x*:

```text
x = (193.724510070035, 5.6e−27, 17.3191887294085, 100.047897801387, 6.68445185362378,
     5.99168428444265, 6.21451648886070)
```

where g1 is active and every equality is at the edge of the tolerance, |h| = 0.0001. The value is
the best known, not proven optimal.

### One free variable

The equalities leave less freedom than they seem to. h3, h4 and h5 give x5, x6 and x7 from x4. h1
factors as (x4 − 300) (x3 − 25 (x5 − x6)) = 0, so x3 = 25 (x5 − x6) for x4 below 300. h2 factors as
(100 − x4) (x2 − 155.365 + 25 x7) = 0, so x2 = 155.365 − 25 x7 for x4 above 100, and any x2 at x4 =
100. And f = x1 is least with g1 active, x1 = 35 x2^0.6 + 35 x3^0.6. With the equalities met
exactly, f is a function of x4 alone.

That function has a minimum at each end. At x4 = 100, x2 = 0 and x3 = 25 ln 2 meet every constraint
exactly, with f = 35 (25 ln 2)^0.6 = 193.788. It then rises, to 330.6 near x4 = 293, and falls again
to 325.1 at x4 = 299.53, where x2 reaches its upper bound 40: past it, h2 would need x2 above 40.
The tolerance of 0.0001 lets the best known solution go 0.064 below 193.788.

## What makes it hard

The feasible region has no volume: random points never meet five equalities. The report's table 3
gives a feasible share of 0.0000 %. A search starts outside, guided only by the violation, and must
reach a thin layer around a curve in 7 dimensions, the curve that x4 traces.

Then the curve has two ends, and a search that reaches the layer near x4 = 300 finds the other
minimum, f = 324.70, 131 above f*, with x2 at 40 and x3 near 0. Moving along the curve to the good
end means passing the maximum near x4 = 293: from there, every step toward it is worse, and Deb's
rules never take a worse feasible step. With Deb's rules alone, where a search first reaches the
layer decides where it ends.

And the best known solution has x2 = 5.6·10⁻²⁷. f grows like 35 x2^0.6, whose slope is infinite at
0: an error below 1e-8 needs x2 below about 10⁻¹⁶.

## Representation

A `Real` genome of 7 genes, x1 to x7, within the bounds above. genoxide's `problems::cec2006::G21`
is the fitness: the value f(x) and the total constraint violation, the sum of max(0, g1(x)) and of
max(0, |h(x)| − 0.0001) over the equalities, 0 for a feasible solution.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.
Before the first feasible solution, the search is a minimization of the violation, which leads to
the region.

## Algorithm

SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78) is genoxide's default differential
evolution: current-to-pbest/1 mutation with an archive, and a memory of the scale factor F and the
crossover rate CR that worked. It uses genoxide's defaults, but a population of 30 instead of 100,
with restarts when the population has converged or stagnated, and a trial outside the bounds brought
back halfway between its parent and the bound, which lets x2 approach 0 geometrically.

SHADE compares solutions with Deb's rules at an ε level, the ε constrained method of Takahama and
Sakai (2006, "Constrained optimization by the ε constrained differential evolution with
gradient-based mutation and feasible elites", IEEE CEC 2006), whose εDE won the CEC 2006
competition: a violation up to ε counts as none. Two solutions within ε compare by value, and the
others as Deb's rules have it. ε starts at 1,000 and follows Takahama and Sakai's schedule,
ε(t) = 1000 (1 − t / 150,000)⁵ after t evaluations, down to 0 at 150,000. The example applies it
with genoxide's features: the fitness function returns the violation beyond ε, and the engine's
`control` lowers ε every 10 generations and scores the population again (`reevaluate`), since the
old violations no longer compare with the new ones. The re-evaluations count in the budget. From
150,000 evaluations on, the rules are Deb's, and the problem the report's. While ε is large, the
value leads the population along the whole curve, toward the good end, before the layer is thin.

The run has the report's budget of 500,000 evaluations, and stops once ε is 0 and its best solution
is feasible with an absolute error f(x) − f* of at most 1e-8. The report counts a run as successful
with an error of at most 1e-4; the example asks for more.

With Deb's rules alone, no algorithm of genoxide solves g21 every time: each run reaches one end of
the curve or the other. With 100 seeds, SHADE with a population of 50 met the target on 68, after a
median of 44,700 evaluations (from 37,900 to 211,750); the other 32 ended at the minimum of 324.70.
With the default population of 100, it met the target on 58, after a median of 96,000; L-SHADE,
whose population shrinks over the budget, on 66. The restarts don't change this: with the default
population, runs met the target on the same 15 of 25 seeds with and without them. CMA-ES with IPOP
restarts (Hansen and Ostermeier, 2001; Auger and Hansen, 2005) came within 3·10⁻⁷ of f* on 14 of 25
seeds, but within 1e-8 on none: it converges pressed against the bound x2 = 0 and the edges of five
tolerances at once.

At an ε level, SHADE met the target on 97 to 99 % of the runs with populations of 30 to 50 and ε
starting at 100 or 1,000, and on 94 to 95 % with a population of 20, in 200 to 1,000 runs each. It
needs its restarts: without them, no run met the target.

## Output

The first line names the run. The second gives what stopped it, after how many evaluations, the
error f(x) − f* and whether the best solution is feasible: "< 1e-8" means the run met its target.
The third gives when the best solution, by the ε level, was first feasible without it, and when its
error first met the report's criterion of success. The fourth compares f(x) with f*, to 6
significant digits. The fifth gives the solution, with the genes near 0 in scientific notation, and
the last the constraints: for g1, "active" on the boundary (|g| ≤ 1e-6), else the value of g,
negative when it's satisfied; for h1 to h5, "active" when the equality is met within the tolerance,
else by how much |h| exceeds it. In Python, the fitness function evaluates the problem in Rust, a
generation at a time, so both versions print the same.

The page's plot shows each variable on its range, and each constraint's state: violated, active or
satisfied (an equality met within the tolerance shows as active). Its curve shows the error f − f*
of the best solution, and of the population's median, on a log scale, measured without the ε level.
The best's curve begins once the best is feasible, and the median's once half the population is:
after the end of the ε schedule.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g21) plays this run back.

## Good results

A good run is feasible and ends within 1e-4 of f*, the report's success. The runs that fail end at
the other minimum, f = 324.70, 131 above f*, or on their way along the curve.

The recorded run's best, by the ε level, uses the tolerance ε allows until the end of the schedule.
It is first feasible without ε after 167,399 evaluations, within the report's criterion after
180,299, and meets its target after 185,909. The solution is the report's x* to 6 digits, with
x2 = 8.0·10⁻¹⁸, g1 active and every equality met.

With seeds 1 to 1,000, 981 runs meet the target, after 162,599 to 485,819 evaluations (184,168 for
half of them), and one more ends within 1e-4 of f*. Of the other 18, 11 end at the other minimum,
and 7 between 193.75 and 274.49, still moving along the curve when the budget runs out.
