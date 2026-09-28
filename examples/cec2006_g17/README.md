---
title: CEC 2006 g17
category: constrained
summary: A discontinuous, piecewise linear function of 6 variables under 4 nonlinear equality constraints, with a local optimum that traps most runs of every algorithm, searched with SHADE and Deb's feasibility rules.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "8853.5338748065 (best known, with the equalities met within 0.0001; the report prints 8853.53967480648)"
languages: [rust, python]
order: 93
---

# CEC 2006 g17

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24
problems, g01 to g24, from the literature, with their best known solutions and rules for comparing
algorithms. g17 is the seventeenth, the report's equations 35 and 36 (page 10). The report takes it
from Himmelblau (1972, Applied Nonlinear Programming, McGraw-Hill).

There are 6 variables:

```text
x1 in [0, 400]    x2 in [0, 1000]    x3, x4 in [340, 420]    x5 in [−1000, 1000]
x6 in [0, 0.5236]
```

The objective is f(x) = f1(x1) + f2(x2), with two piecewise linear parts:

```text
f1(x1) = 30 x1 for x1 < 300,  31 x1 from 300
f2(x2) = 28 x2 for x2 < 100,  29 x2 for 100 ≤ x2 < 200,  30 x2 from 200
```

The report defines f1 for x1 < 400 and f2 for x2 < 1000, but its bounds include 400 and 1000:
genoxide's `G17` extends the last pieces to them. There are four equality constraints:

```text
h1 = −x1 + 300 − (x3 x4 / 131.078) cos(1.48477 − x6) + (0.90798 x3² / 131.078) cos(1.47588) = 0
h2 = −x2 − (x3 x4 / 131.078) cos(1.48477 + x6) + (0.90798 x4² / 131.078) cos(1.47588) = 0
h3 = −x5 − (x3 x4 / 131.078) sin(1.48477 + x6) + (0.90798 x4² / 131.078) sin(1.47588) = 0
h4 = 200 − (x3 x4 / 131.078) sin(1.48477 − x6) + (0.90798 x3² / 131.078) sin(1.47588) = 0
```

h1, h2 and h3 fix x1, x2 and x5 from x3, x4 and x6, and h4 ties those three together. The feasible
set is a surface: two degrees of freedom left of six. Real-valued samples almost never meet an
equality exactly. The report counts an equality as met when |h(x)| ≤ 0.0001, and so does
genoxide's `G17`: the surface becomes a thin layer.

### The best known value

The report prints f(x*) = 8853.53967480648, but its x* evaluates to 8853.53401643571 under its own
definition. genoxide's docs trace the printed value to the organizers' code, which evaluates f1 and
f2 at the right-hand sides of h1 and h2, the values that x1 and x2 would take with the equalities
met exactly, instead of at x1 and x2. Within the tolerance, those differ from x1 and x2 by h1 and
h2. At x*, h1 is 0.0000953 and h2 0.0001, and 30 · 0.0000953 + 28 · 0.0001 = 0.00566, the gap
between the two values.

Later papers give 8853.5338748065, a little lower. The report's x* reaches it with x1 lowered to
201.78446249355, where h1 reaches the tolerance. genoxide uses it as f*; its source is unverified,
and it isn't proven optimal. It lies at x = (201.78446249355, 99.9999999999999005,
383.071034852773266, 420, −10.9076584514292652, 0.0731482312084287128).

## What makes it hard

The objective jumps. f1 jumps by 300 where x1 reaches 300, and f2 by 100 where x2 reaches 100 and
by 200 where it reaches 200. A step across a jump changes f far more than steps elsewhere.

The best known solution sits on the edge of a jump: x2 is just below 100, on the piece 28 x2. A
step to x2 = 100 costs 100. x4 is at its upper bound 420, and the equalities h1, h2 and h4 are on
the edge of the tolerance. The optimum is a corner of the surface, pressed against a jump, a bound
and the tolerance at once.

The feasible layer is thin: of 10 million random points in the box, none was feasible. A search
must first reach it from outside, guided only by the violation, and then move along a curved
surface in small steps.

And there's a local optimum. At f = 8927.5917, 74.06 above f*, x1 ≈ 107.8 and x2 ≈ 196.3, on the
pieces 30 x1 and 29 x2, in another region of the surface. A search that settles there stays: the
runs of SHADE that reached it never left it within the budget. Most runs of every algorithm tried
here end there.

## Representation

A `Real` genome of 6 genes, x1 to x6, within the report's bounds. genoxide's
`problems::cec2006::G17` is the fitness: the value f(x) and the total constraint violation,
the sum of max(0, |h| − 0.0001) over the four equalities, 0 for a feasible solution.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.

## Algorithm

SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78), a differential evolution that adapts its
scale factor and crossover rate from successful trials, with genoxide's defaults: its published
population of 100, and a restart after 200 generations without progress.

The run has the report's budget of 500,000 evaluations, and stops once its best solution is
feasible with an error f(x) − f* of at most 1e-8, an absolute error. The report counts a run as
successful with an error of at most 1e-4; the example asks for more.

No algorithm of genoxide solves g17 reliably. With seeds 1 to 25 and the same budget and target:

| Algorithm | Met the target | Within 1e-4 of f* |
|---|---|---|
| SHADE | 7 | 7 |
| SHADE with JADE's strategy, current-to-pbest with p = 0.1 | 9 | 9 |
| L-SHADE | 8 | 8 |
| CMA-ES with IPOP restarts | 6 | 8 |
| CMA-ES with BIPOP restarts | 8 | 12 |
| GA, simulated binary crossover and polynomial mutation | 0 | 0 |

Every run of SHADE either met the target or ended at the local optimum, 8927.59, and so did most
failed runs of the others, except the GA's. SHADE's successful runs took a median of 207,400
evaluations (164,200 to 268,000). BIPOP-CMA-ES (Hansen, 2009) alternates large populations with
small ones, and some of its runs ended between 1e-8 and 1e-4 above f*. With initial step sizes from
0.2 to 1.0 of the ranges, it met the target on 5 to 13 of 25 runs: tuning doesn't make it reliable,
and 25 runs can't tell these algorithms apart. The GA, with a population of 100, found no feasible
solution. SHADE with a population of 300 met the target on none. The example shows SHADE, whose runs
either succeed or fail clearly; seed 1 is one of its successes.

## Output

The first line names the run. The second gives what stopped it, after how many evaluations, the
error f(x) − f* and whether the best solution is feasible: "< 1e-8" means the run met its target.
The third gives when the best solution was first feasible, and when its error first met the
report's criterion of success. The fourth compares f(x) with f*, to 6 significant digits. The fifth
gives the solution, to 4 significant digits: x2 prints as 100.00, but lies just below 100, on the
piece 28 x2, as the sixth line says. The last gives the four equalities. An equality met within
the tolerance is "active", else the line gives how far |h| exceeds 0.0001. In Python, `run`
evaluates the problem in Rust, so both versions print the same.

The page shows each variable on its range, and each constraint's state. Its curve shows the error
f − f* of the best feasible solution, and of the population's median, on a log scale. The best's
curve begins at the first feasible solution, and the median's once half the population is
feasible.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g17) plays this run back.

## Good results

A good run is feasible and ends within 1e-4 of f*, the report's success. A value below f* is
possible, since f* is only the best known, but the runs here end just above it. Measured against
the report's printed value, 8853.53967, such a run would be 0.0058 below it.

The recorded run needs 38,400 evaluations to find its first feasible solution, about 24 above f*,
with x2 ≈ 89.5 on the piece 28 x2. It then raises x2 toward the jump at 100 and lowers x1: after
64,000 evaluations, x2 is just below 100 and the error 8.0. From there it slides along the edge of
the jump, x2 staying just below 100 while x3 and x4 rise and x5 falls, and the error falls
steadily. Once x4 reaches 420, after about 138,000 evaluations, the run converges quickly: it
is within 1e-4 of f* after 145,900 evaluations, and meets its target after 164,200.

The solution is x = (201.8, 100.00, 383.1, 420.0, −10.91, 0.07315), the best known point to 4
digits, with x2 below 100 and x4 at its bound. All four equalities are met: h1, h2 and h4 with |h|
at 0.0001, on the edge of the tolerance, as at the best known point, and h3 with |h| ≈ 0.000016.
