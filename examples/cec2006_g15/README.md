---
title: CEC 2006 g15
category: constrained
summary: A quadratic in 3 variables on the arc where a sphere meets a plane, two equality constraints met within 0.0001, solved by CMA-ES with IPOP restarts and Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "961.715022289961 (best known, with the equalities met within 0.0001)"
languages: [rust, python]
order: 91
family: "CEC 2006"
tab: g15
---

# CEC 2006 g15

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24
problems, g01 to g24, from the literature, with their best known solutions and rules for comparing
algorithms. g15 is the fifteenth, the report's equations 30 and 31 (page 7). The report takes it
from Himmelblau (1972, Applied Nonlinear Programming, McGraw-Hill).

There are 3 variables, each in [0, 10]. The problem is

```text
minimize   f(x) = 1000 − x1² − 2 x2² − x3² − x1 x2 − x1 x3
subject to h1(x) = x1² + x2² + x3² − 25 = 0
           h2(x) = 8 x1 + 14 x2 + 7 x3 − 56 = 0
```

h1 is a sphere of radius 5 around the origin, and h2 a plane. They meet in a circle, and the
bounds keep the part of it where no variable is negative: an arc, from near (0, 1.64, 4.72) to
near (4.85, 1.23, 0). The feasible set is this arc, one degree of freedom left of three.

Real-valued samples almost never meet an equality exactly. The report counts an equality as met
when |h(x)| ≤ 0.0001, and so does genoxide's `G15`: the arc becomes a thin tube. The best known
value is f* = 961.715022289961, at x = (3.51212812611795133, 0.216987510429556135,
3.55217854929179921). It meets both equalities within the tolerance only, with each |h| at
0.0001, and genoxide's docs mark it as a best known value, not a proven optimum.

## What makes it hard

The feasible set has no volume without the tolerance, and hardly any with it: of 10 million
random points in the box, none was feasible. A search must first reach the tube from outside,
guided only by the violation.

Once inside, it can only move along the arc. The tube is 0.0001 wide in each h, and the arc bends,
so a step in almost any direction leaves it. A step along the arc must be short enough to stay
inside while the arc curves away.

The objective hardly changes along the arc. Over its whole length, f lies between 961.7 and
972.3, and near the minimum it is flat. The tube also shifts the minimum: on the exact arc, the
least value is about 961.71517, 0.00015 above f*. The best solutions sit on the tube's wall, each
|h| at 0.0001, where f is a little lower.

## Representation

A `Real` genome of 3 genes, x1 to x3, within [0, 10]. genoxide's `problems::cec2006::G15` is the
fitness: the value f(x) and the total constraint violation,
max(0, |h1| − 0.0001) + max(0, |h2| − 0.0001), 0 for a feasible solution.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults: a population of 4 + ⌊3 ln 3⌋ = 7, a step size of 0.3 of each gene's range and a random
start. With IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776), a run that has
converged starts again from a random point, with twice the population. A sample outside the bounds
is drawn again, up to 100 times, and then clipped to them. Deb's rules rank the samples.

The run has the report's budget of 500,000 evaluations, and stops once its best solution is
feasible with an error f(x) − f* of at most 1e-8, an absolute error. The report counts a run as
successful with an error of at most 1e-4; the example asks for more.

Why CMA-ES: its covariance matrix can learn the direction of the tube, so its samples spread along
the arc rather than across it. With seeds 1 to 25, IPOP-CMA-ES met the target on all 25 runs, after
a median of 31,535 evaluations (8,918 to 50,855), none of them after a restart: without restarts,
CMA-ES met it on all 25 too. SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78), genoxide's
default differential evolution, met it on 18 of 25 runs, after a median of 135,700 evaluations. Of
the other 7, 5 ended between 0.00027 and 3.0 above f*, and 2 never found a feasible solution.

## Output

The first line names the run. The second gives what stopped it, after how many evaluations and
restarts, the error f(x) − f* and whether the best solution is feasible: "< 1e-8" means the run met
its target. The third gives when the best solution was first feasible, and when its error first
met the report's criterion of success. The fourth compares f(x) with f*, to 6 significant digits.
The fifth gives the solution, to 4 significant digits, and the last the two equalities. An
equality met within the tolerance is "active", else the line gives how far |h| exceeds 0.0001. In
Python, `run` evaluates the problem in Rust, so both versions print the same.

The page shows each variable on its range, and each constraint's state. Its curve shows the error
f − f* of the best feasible solution, and of the population's median, on a log scale. The best's
curve begins at the first feasible solution, and the median's once half the population is
feasible.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g15) plays this run back.

## Good results

A good run is feasible and ends within 1e-4 of f*, the report's success. A value below f* is
possible, since f* is only the best known, but the runs here end just above it.

The recorded run finds its first feasible solution after 525 evaluations, about 2.1 above f*, at x ≈
(4.53, 0.37, 2.07). It then walks along the arc toward smaller x1 and larger x3, at a nearly steady
pace in the short steps that the tube allows: x1 shrinks by about 0.1 every 2,200 evaluations, and
the error is 0.02 after 20,000. Near the minimum the steps shrink, and the error falls by a factor
of 10 about every 1,000 evaluations. It is within 1e-4 of f* after 23,765 evaluations, and meets its
target after 26,880, without a restart. The solution is x = (3.512, 0.2170, 3.552), the best known
point to 4 digits, and both equalities are met with |h| at 0.0001, on the edge of the tolerance, as
at the best known point.
