---
title: CEC 2006 g19
category: constrained
summary: A cubic in 15 variables under 5 nonlinear inequalities, all five active at the best known solution with eight variables on their bounds, solved by SHADE with Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "32.6555929502463 (best known)"
languages: [rust, python]
order: 95
family: "CEC 2006"
tab: g19
---

# CEC 2006 g19

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 test
problems, g01 to g24, with their best known solutions and rules for comparing algorithms. g19 is the
nineteenth, the report's equations 39 and 40 (page 11), with the data of its table 1 (page 12). The
report takes it from Himmelblau (1972, Applied Nonlinear Programming, McGraw-Hill). It has 15
variables, each in [0, 10]. Call the last five y1 to y5 (y_j = x_(10+j)).

Minimize

```text
f(x) = Σᵢ Σⱼ cᵢⱼ yᵢ yⱼ + 2 Σⱼ dⱼ yⱼ³ − Σₖ bₖ xₖ
```

with i and j from 1 to 5 and k from 1 to 10, subject to five inequalities, each g(x) ≤ 0:

```text
gⱼ = −2 Σᵢ cᵢⱼ yᵢ − 3 dⱼ yⱼ² − eⱼ + Σₖ aₖⱼ xₖ        j = 1, …, 5
```

The data:

```text
b = (−40, −2, −0.25, −4, −4, −1, −40, −60, 5, 1)

 j     1     2     3     4     5          j     1     2     3     4     5
 eⱼ   −15   −27   −36   −18   −12         a1ⱼ  −16    2     0     1     0
 c1ⱼ   30   −20   −10    32   −10         a2ⱼ   0    −2     0    0.4    2
 c2ⱼ  −20    39    −6   −31    32         a3ⱼ −3.5    0     2     0     0
 c3ⱼ  −10    −6    10    −6   −10         a4ⱼ   0    −2     0    −4    −1
 c4ⱼ   32   −31    −6    39   −20         a5ⱼ   0    −9    −2     1   −2.8
 c5ⱼ  −10    32   −10   −20    30         a6ⱼ   2     0    −4     0     0
 dⱼ     4     8    10     6     2         a7ⱼ  −1    −1    −1    −1    −1
                                          a8ⱼ  −1    −2    −3    −2    −1
                                          a9ⱼ   1     2     3     4     5
                                          a10ⱼ  1     1     1     1     1
```

c is symmetric. The first ten variables enter f only linearly, through b, and the constraints only
through a; the last five enter f as a quadratic and a cubic.

The best known value is f* = 32.6555929502463, at the report's x*:

```text
x3 = 3.94599045143234, x5 = 3.28317734584542, x6 = 9.99999999999999822,
y = (0.370764847417014, 0.278456024942956, 0.523838487672241, 0.388620152510323,
     0.298156764974679)
```

with x1, x2, x4 and x7 to x10 below 3·10⁻¹⁵, that is, at their lower bound 0. All five constraints
are active there, to 10⁻¹⁴. The report's table 3 counts no active constraint for g19: genoxide's
docs record the difference. The value is the best known, not proven optimal.

## What makes it hard

Not the feasible region: the report estimates that 33.4761 % of the box is feasible, and a sample of
2 million random points for this page found 33.47 %. A search starts with feasible solutions.

The difficulty is the corner where the best known solution lies. Of the 15 variables, seven sit at
their lower bound 0 and x6 at its upper bound 10, and all five constraints are active: 13 of 15
directions are pinned, and the search must find the one point where they meet. The constraints are
not convex (−3 dⱼ yⱼ² curves them), so the region near the corner is curved, and each gene at 0 has
to get there to about 10⁻¹⁰ before the error falls below 1e-8.

## Representation

A `Real` genome of 15 genes, x1 to x15, each within [0, 10]. genoxide's `problems::cec2006::G19` is
the fitness: the value f(x) and the total constraint violation, the sum of max(0, g(x)) over the
five constraints, 0 for a feasible solution.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.

## Algorithm

SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78) is genoxide's default differential
evolution: current-to-pbest/1 mutation with an archive, and a memory of the scale factor F and the
crossover rate CR that worked. It uses genoxide's defaults: a population of 100, restarts when the
population has converged or stagnated, and a trial outside the bounds brought back between its
parent and the bound. Deb's rules decide between a trial and its parent.

The run has the report's budget of 500,000 evaluations, and stops once its best solution is feasible
with an absolute error f(x) − f* of at most 1e-8. The report counts a run as successful with an
error of at most 1e-4; the example asks for more.

Why SHADE: with 25 seeds, it met the target on every run, after a median of 87,200 evaluations (from
81,600 to 96,700). Its bound handling moves a gene halfway to the bound at each step that crosses
it, so the seven genes that belong at 0 approach it geometrically. CMA-ES (Hansen and Ostermeier,
2001, Evolutionary Computation 9(2): 159-195) met the target on 17 of 25 seeds; the other eight
stopped short, at errors from 1.4e-8 to 2.7e-5, when its distribution converged in the corner. With
IPOP restarts it met it on all 25, after a median of 149,304 evaluations; L-SHADE, whose population
shrinks over the budget, after a median of 154,655.

## Output

The first line names the run. The second gives what stopped it, after how many evaluations, the
error f(x) − f* and whether the best solution is feasible: "< 1e-8" means the run met its target.
The third gives the evaluations to the first feasible solution, and to an error of 1e-4, the
report's criterion of success. The fourth compares f(x) with f*, to 6 significant digits. The fifth
gives the solution, with the genes near 0 in scientific notation, and the last the five constraints:
"active" for a constraint on its boundary (|g| ≤ 1e-6), else the value of g, negative when it's
satisfied. In Python, `run` evaluates the problem in Rust, so both versions print the same.

The page's plot shows each variable on its range, and each constraint's state: violated, active or
satisfied. Its curve shows the error f − f* of the best feasible solution, and of the population's
median, on a log scale. The best's curve begins at the first feasible solution, and the median's
once half the population is feasible.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g19) plays this run back.

## Good results

A good run is feasible and ends within 1e-4 of f*, the report's success. SHADE meets the target of
1e-8 with every seed tried.

With seed 1, the first 100 random solutions include feasible ones, the best of them 2,780 above f*.
The error falls steadily, by about a factor of 10 every 5,000 evaluations once the population is
near the corner: 136 after 4,900 evaluations, 0.14 after 43,300, and within the report's criterion
after 66,000. The run meets the target after 87,200. The solution is the report's x* to 5 or 6
digits, with the seven genes of the lower bound from 3·10⁻¹² to 1.7·10⁻¹⁰, and all five constraints
active.
