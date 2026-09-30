---
title: CEC 2006 g20
category: constrained
summary: A linear function of 24 variables under 6 inequalities and 14 equalities that no solution meets, the report's best known included; SHADE with Deb's feasibility rules finds a solution a little less infeasible than the report's.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "none feasible; the report's best known, 0.2049794002, violates a constraint by 0.1438"
languages: [rust, python]
order: 116
family: "CEC 2006"
tab: g20
---

# CEC 2006 g20

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 test
problems, g01 to g24, with their best known solutions and rules for comparing algorithms. g20 is the
twentieth, the report's equations 41 and 42 (page 12), with the data of its table 2 (page 13). The
report takes it from Himmelblau (1972, Applied Nonlinear Programming, McGraw-Hill). It has 24
variables, each in [0, 10], in two groups of twelve: x1 to x12 and x13 to x24.

Minimize

```text
f(x) = Σᵢ aᵢ xᵢ
```

subject to six inequalities, each g(x) ≤ 0, and fourteen equalities, each h(x) = 0, with S = Σⱼ xⱼ
over all 24, L = Σⱼ₌₁¹² xⱼ/bⱼ and V = Σⱼ₌₁₃²⁴ xⱼ/bⱼ:

```text
gᵢ = (xᵢ + x₁₂₊ᵢ) / (S + eᵢ)                         i = 1, 2, 3
gᵢ = (xᵢ₊₃ + xᵢ₊₁₅) / (S + eᵢ)                       i = 4, 5, 6
hᵢ = x₁₂₊ᵢ / (b₁₂₊ᵢ V) − cᵢ xᵢ / (40 bᵢ L)             i = 1, …, 12
h13 = S − 1
h14 = Σᵢ₌₁¹² xᵢ/dᵢ + k V − 1.671                       k = 0.7302 · 530 · 14.7/40
```

The data, where a and b repeat for i = 13 to 24:

```text
 i     aᵢ       bᵢ       cᵢ      dᵢ      eᵢ          i     aᵢ      bᵢ       cᵢ      dᵢ
 1   0.0693   44.094   123.7   31.244   0.1         7    0.06   62.501   49.7   56.708
 2   0.0577   58.12     31.7   36.12    0.3         8    0.1    84.94     7.1   82.7
 3   0.05     58.12     45.7   34.784   0.4         9    0.12  133.425    2.1   80.8
 4   0.2     137.4      14.7   92.7     0.3        10    0.18   82.507   17.7   64.517
 5   0.26    120.9      84.7   82.7     0.6        11    0.1    46.07     0.85  49.4
 6   0.55    170.9      27.7   91.6     0.3        12    0.09   60.097    0.64  49.1
```

The report counts an equality as met when |h(x)| ≤ 0.0001, and so does genoxide's `G20`. The
equalities are undefined, 0/0, where x1 to x12 or x13 to x24 are all 0: there the fitness is
invalid.

### No feasible solution

The report gives a best known solution, f = 0.2049794002 (its table 4), and says it is "a little
infeasible", and that no feasible solution has been found. Its x* meets the fourteen equalities
within the tolerance, but g1 = 0.1438: x13 = 0.158 is far from 0.

In fact no solution is feasible, as genoxide's docs derive. Each gᵢ is a sum of two variables over a
positive number, so g1 to g6 hold only where twelve variables are 0: x1, x2, x3, x7, x8, x9 and
their partners x13, x14, x15, x19, x20, x21. h14 then asks for V ≥ 0.0115, and h1 to h12 tie each of
x16, x17, x18, x22, x23, x24 to its partner in the first group, in a way that makes their sum at
least 109 times V: 1.26 or more, where h13 wants the sum of all 24 to be 1.

So the question for g20 is how little a solution can violate. genoxide's `G20` keeps the report's
solution as its best known, `optimum()`, not proven, and infeasible.

## What makes it hard

Nothing can succeed: a run can only trade one violation for another. The report's solution gives up
on g1, puts 0.158 in x13, which g1 wants at 0, and meets everything else.

That trade lives at the edge of where the problem is defined. In the report's x*, x1 to x12 are all
below 2·10⁻¹⁷: they are the amounts of a mixture that h1 to h12 compare only as ratios, and the
ratios still count when the amounts are almost nothing. A search has to shrink twelve variables by
many orders of magnitude while keeping their ratios right.

## Representation

A `Real` genome of 24 genes, x1 to x24, each within [0, 10]. genoxide's `problems::cec2006::G20` is
the fitness: the value f(x) and the total constraint violation, the sum of max(0, g(x)) over the
inequalities and of max(0, |h(x)| − 0.0001) over the equalities.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. With no feasible solution, the whole
run is a minimization of the violation, and f plays no part.

## Algorithm

SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78) is genoxide's default differential
evolution: current-to-pbest/1 mutation with an archive, and a memory of the scale factor F and the
crossover rate CR that worked. It uses genoxide's defaults: a population of 100, restarts when the
population has converged or stagnated, and a trial outside the bounds brought back halfway between
its parent and the bound. Deb's rules decide between a trial and its parent.

The run has the report's budget of 500,000 evaluations, and no target: it runs to the end.

Why SHADE: with 25 seeds, 24 runs ended at the same violation, 0.1437119 to 7 digits, and the other
at 0.1437121, a little below the report's 0.14375. L-SHADE, whose population shrinks over the
budget, ended there too on all 25 seeds. CMA-ES with IPOP restarts (Hansen and Ostermeier, 2001,
Evolutionary Computation 9(2): 159-195; Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776) ended
within 0.00031 of it, and CMA-ES without restarts between 0.14 and 0.38. That different algorithms
end at the same point suggests it is the least violation there is, but that isn't proven.

## Output

The first line names the run. The second gives when it stopped and the violation of its best
solution. The third gives the violation of the report's best known solution, and the evaluations
after which the run's best was less violated. The fourth compares f(x) with the report's value, to 6
significant digits. The fifth gives the solution, with the genes near 0 in scientific notation, and
the last the constraints: for g1 to g6, "active" on the boundary (|g| ≤ 1e-6), else the value of g;
for h1 to h14, "active" when the equality is met within the tolerance, else by how much |h| exceeds
it. In Python, `run` evaluates the problem in Rust, so both versions print the same.

The page's plot shows each variable on its range, and each constraint's state: violated, active or
satisfied (an equality met within the tolerance shows as active). Its curve shows the constraint
violation of the best solution, and of the population's median, on a log scale: with no feasible
solution, the error f − f* has no meaning.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g20) plays this run back.

## Good results

No run is feasible. A good run ends at a violation of 0.14371, a little below the 0.14375 of the
report's solution. The example's run gets there: with seeds 1 to 100, SHADE ends between 0.14371188
and 0.14371293 every time, at 0.14371188 in 96 runs, and L-SHADE at 0.1437118794 on all 100, the
least violation found.

With seed 1, the violation falls from 169 in the first random population to 0.33 after 32,100
evaluations and 0.150 after 64,100. The run passes the report's solution after 91,200 evaluations
and settles at 0.14371 after about 115,000. The rest of the budget goes to restarts, which bring new
random solutions into the population and don't find anything better. The solution is the report's,
to 3 or 4 digits: x13 = 0.1581, x17 = 0.5309, x22 = 0.3110, x23 and x24 near 6·10⁻⁵, x1 to x12 below
1·10⁻¹², g1 = 0.1437 and every other constraint met. Its value is 0.204975, just below the report's
0.204979, which doesn't count under Deb's rules.
