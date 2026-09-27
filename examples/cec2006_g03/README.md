---
title: CEC 2006 g03
category: constrained
summary: The largest product of 10 variables on the unit sphere, an equality constraint that SHADE can't follow, solved with CMA-ES and Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "−1.0005001 (−1.0001⁵) at xi = 0.316244 for the report's tolerance 0.0001, proven"
languages: [rust, python]
order: 79
---

# CEC 2006 g03

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 test
problems, g01 to g24, with their best known solutions and rules for comparing algorithms. g03 is
the third. The report takes it from Michalewicz, Nazhiyath and Michalewicz (1996, A note on
usefulness of geometrical crossover for numerical optimization problems, Proceedings of the 5th
Annual Conference on Evolutionary Programming: 305-312). It has no physical meaning: the variables
are x1 to x10, and the report names the one constraint h1.

The source maximizes a product; the report, and genoxide, minimize its negative:

```text
f(x) = −(√n)ⁿ x1 x2 … x10        with n = 10, so (√n)ⁿ = 100,000
```

subject to one equality:

```text
h1 = x1² + x2² + … + x10² − 1 = 0
```

Each variable lies in [0, 1]. The feasible solutions lie on the unit sphere.

The report counts an equality as met when |h1| ≤ 0.0001. The product is largest with equal
genes, as far out as the tolerance allows: at x1² + … + x10² = 1.0001, so each xi = √(1.0001 / 10)
= 0.316244. The minimum there is −1.0001⁵ = −1.0005001, the report's value. genoxide derives it
from the definition, and it's proven. Without the tolerance, it would be −1 at xi = 1/√10 =
0.316228.

The report's rules give each run 500,000 evaluations. A run succeeds when it finds a feasible
solution within 0.0001 of the minimum.

## What makes it hard

The feasible region is a thin shell: the points whose sum of squares lies between 0.9999 and
1.0001. Its thickness is about 0.0001 in the distance from the origin. The report's estimate of its
share of the box, from random points, is 0.0000 %. No random solution is feasible.

Once a search reaches the shell, it has to move along it to the minimum. The shell is curved, so
a straight step along it leaves it unless the step is short. A step that leaves the shell makes a
solution infeasible, and Deb's rules rank it below every feasible one. Steps along the shell must
be short, or bent to follow it.

## Representation

A `Real` genome of 10 genes in [0, 1]. The fitness is the value f(x) and the constraint violation,
max(0, |h1(x)| − 0.0001): 0 for a feasible solution. The tolerance is the report's,
`genoxide::problems::cec2006::EQUALITY_TOLERANCE`. `G03::with_tolerance` could make the shell
thicker and the problem easier, but it would no longer be the report's problem.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), with genoxide's
defaults: 10 samples per generation for 10 genes, from a random mean, with an initial step of 0.3
of each gene's range. It samples each generation from a normal distribution and learns its
covariance matrix from the best samples' steps. Once the best samples lie in the shell, the steps
it learns from lie along the shell, so the distribution becomes flat across it and keeps more of
its samples inside. That's what a curved, thin feasible region needs. Restarts (IPOP or BIPOP)
aren't needed: with seeds 1 to 3, they give the same runs, which meet the target before they
could restart.

The run has the report's budget of 500,000 evaluations, and seed 1. It stops early once its best
solution is feasible and within 1e-8 of the minimum, relative to its size.

SHADE, the differential evolution that solves g01 and g02, fails here. Differential evolution
builds a trial from the difference between two solutions. Between two points of a curved shell,
the difference points off the shell, so a trial from a large difference is infeasible and loses to
its parent. The steps that survive are tiny, and the population stops moving along the shell.

With the same budget, in five runs each, with seeds 1 to 5:

| Algorithm | Best value (the minimum: −1.0005001) |
|---|---|
| CMA-ES | −1.0005001 in every run, after 16,170 to 26,110 evaluations (20 runs) |
| GA, SBX, Gaussian mutation, (μ + λ) | −1.00041 to −0.99972 |
| GA, arithmetic crossover, polynomial mutation | −1.00014 to −0.99866 |
| GA, SBX, polynomial mutation, (μ + λ) | −0.9976 to −0.402 |
| GA, SBX, polynomial mutation | −0.986 to −0.959 |
| Particle swarm optimization, 40 particles | −1.00048 to −0.767 |
| SHADE | −0.909 to −0.691 |
| L-SHADE | −0.575 to −0.349 |

The GAs have 100 individuals, tournaments of 2, simulated binary crossover (SBX) with η 15 or
arithmetic crossover, and polynomial mutation with η 20 or Gaussian mutation with σ 0.01, each
changing a gene with probability 0.1. Two runs besides CMA-ES's meet the report's criterion of
success, within 0.0001 of the minimum: the swarm's with seed 1 and the GA's with Gaussian mutation
and seed 1. The others end short of it.

## Output

The first line names the run. The second gives the best value, whether it's feasible, and the
evaluations the run took. The third gives the best solution, x1 to x10, and the fourth the value
of h1 at it: 0.0001, the outer edge of the tolerance. In Python, `run` evaluates the problem in
Rust, so both versions print the same.

The page's plot shows each variable on its range, and the constraint's state from its violation,
max(0, |h1| − 0.0001): violated until the best solution reaches the shell, and met from then on.
Its curve shows the error f − f* of the best feasible solution, and of the population's median, on
a log scale. The best's curve begins at the first feasible solution. The median's has gaps: in
about 60 % of the generations, half or more of the 10 samples fall outside the shell.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g03) plays this run back.

## Good results

A gap of 0 to the minimum −1.0005001 is the best possible. CMA-ES reaches the shell after 730
evaluations, with f = −0.028. It then moves along the shell slowly, with steps small enough to stay
in it: the error falls from 0.97 to 0.09 over the next 12,700 evaluations. Near the point of
equal genes, it converges fast, and it meets the 1e-8 target after 20,820 evaluations. All ten
genes are 0.3162 or 0.3163, and h1 is 0.0001: the solution sits on the outer edge of the
tolerance, where the product is largest. With seeds 1 to 20, every run meets the target, after
16,170 to 26,110 evaluations.
