---
title: CEC 2006 g02
category: constrained
summary: A rugged function of 20 variables with many local optima under two inequalities, the second problem of the CEC 2006 competition, solved with SHADE and a larger population.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "−0.80361910412559, best known (not proven)"
languages: [rust, python]
order: 78
---

# CEC 2006 g02

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 test
problems, g01 to g24, with their best known solutions and rules for comparing algorithms. g02 is
the second. The report takes it from Koziel and Michalewicz (1999, Evolutionary algorithms,
homomorphous mappings, and constrained parameter optimization, Evolutionary Computation 7(1):
19-44). It has no physical meaning: the variables are x1 to x20, and the report names the
constraints g1 and g2.

The source maximizes a function; the report, and genoxide, minimize its negative:

```text
f(x) = −| Σ cos⁴(xi) − 2 Π cos²(xi) | / √( Σ i xi² )        (sums and product over i = 1 … 20)
```

subject to two inequalities, each g(x) ≤ 0:

```text
g1 = 0.75 − Π xi          (the product of the variables is at least 0.75)
g2 = Σ xi − 7.5 n         (their sum is at most 150, with n = 20)
```

Each variable lies in (0, 10]. genoxide closes the interval at 1.5e-154, the smallest positive
value whose square doesn't underflow, so that f stays finite at the corner of the box.

The best known value is −0.80361910412559, at a point the report gives. The first eight variables
are near 3, from 3.162 down to 2.922, and the other twelve near 0.46, from 0.495 down to 0.440.
g1 is nearly active there: the product of the variables is 0.75. No one has proven that it's the
global minimum.

The report's rules give each run 500,000 evaluations. A run succeeds when it finds a feasible
solution within 0.0001 of the best known value.

## What makes it hard

The landscape is rugged. Each cos⁴(xi) and cos²(xi) has a period of π, so the numerator rises and
falls about three times along each variable's range. With 20 variables, the peaks and valleys
combine into a very large number of local optima. The denominator weights the i-th variable by i,
so the best solutions keep the later variables small, but the constraint g1 keeps the product of
all twenty at least 0.75.

The constraints aren't what makes it hard: the report estimates the feasible share of the box from
random points at 99.997 %. The best known solution does lie on g1's boundary, so the search still
has to approach it from inside.

## Representation

A `Real` genome of 20 genes in [1.5e-154, 10]. The fitness is the value f(x) and the total
constraint violation, the sum of max(0, g(x)) over the two inequalities: 0 for a feasible
solution.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.

## Algorithm

SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78), a differential evolution that adapts its
scale factor and crossover rate from successful trials, with a population of 300 instead of its
published 100. The larger population keeps more valleys in play for longer before it gathers in
one. When the population has converged, genoxide restarts it: every individual but the best is
replaced by a random one.

The run has seed 1 and uses the report's whole budget of 500,000 evaluations, without stopping at
the best known value, to see whether anything beats it.

In 20 runs each, with seeds 1 to 20, stopping within 1e-8 of the best known value, relative to its
size:

| Algorithm | Runs within 1e-8 | Evaluations to get there |
|---|---|---|
| SHADE, population 300 | 20 | 291,600 to 463,800 |
| L-SHADE, a population shrinking from 360 to 4 over the budget | 15 | 278,700 to 322,600 |

The five L-SHADE runs that miss end between −0.8022 and −0.7981, in other valleys. SHADE with its
published population of 100 reaches a relative gap of 1e-4 with seed 1, and four runs with seeds 1
to 4 end between −0.80362 and −0.80306. In four runs each, the others end further away: a GA with
simulated binary crossover and polynomial mutation between −0.802 and −0.769, CMA-ES with restarts
(IPOP or BIPOP) between −0.793 and −0.765, and particle swarm optimization between −0.736 and
−0.379.

## Output

The first line names the run. The second gives the best value, whether it's feasible, and the best
known value. The third gives the gap to the best known value, relative to its size. The next two
give the best solution, x1 to x20, to 3 decimals, and the last the constraints active at it: those
with |g(x)| ≤ 1e-6.

In Python, `run` evaluates the problem in Rust, so both versions print the same.

The page's plot shows each variable on its range, and each constraint's state: violated, active or
satisfied. Its curve shows the error f − f* of the best feasible solution, and of the population's
median, on a log scale, where f* is the best known value.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g02) plays this run back.

## Good results

A gap of 0 to the best known value, −0.803619, is as good as anyone has found. The run gets within
1e-8 of it after about 310,000 evaluations, and ends about 1e-15 above it, at the report's solution
to 3 decimals. g1 is active there, and g2 far from its limit: the sum of the variables is 30, not
150.

The population doesn't restart: it goes on refining the best to the end of the budget, and its
median's curve follows the best's down. A run can't prove that the best known value is the
minimum, but none of the 20 SHADE runs above, nor the L-SHADE ones, beats it.
