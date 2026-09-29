---
title: CEC 2006 g22
category: constrained
summary: The linear function x1 of 22 variables under 19 equalities with coefficients up to 10⁷, which leave three variables free; L-SHADE on all 22 doesn't find a feasible solution, and SHADE on the three free ones, with the equalities solved in order, beats the report's best known in every run.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "236.430975504001 (the report's best known, with the equalities met within 0.0001); 236.370313314566 with every equality met exactly"
languages: [rust, python]
order: 98
family: "CEC 2006"
tab: g22
---

# CEC 2006 g22

## The problem

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 test
problems, g01 to g24, with their best known solutions and rules for comparing algorithms. g22 is the
twenty-second, the report's equations 45 and 46 (pages 13 and 14). The report takes it from
Epperly's collection of global optimization test problems with solutions (the report's reference 6).
It has 22 variables:

```text
x1 in [0, 20000]           x2, x3, x4 in [0, 10⁶]      x5, x6, x7 in [0, 4·10⁷]
x8 in [100, 299.99]        x9 in [100, 399.99]         x10 in [100.01, 300]
x11 in [100, 400]          x12 in [100, 600]           x13, x14, x15 in [0, 500]
x16 in [0.01, 300]         x17 in [0.01, 400]          x18 to x22 in [−4.7, 6.25]
```

Minimize f(x) = x1, subject to one inequality, g(x) ≤ 0, and nineteen equalities, each h(x) = 0:

```text
g1  = −x1 + x2^0.6 + x3^0.6 + x4^0.6
h1  = x5 − 100000 x8 + 10⁷               h11 = x9 − x12 + x17
h2  = x6 + 100000 x8 − 100000 x9         h12 = −x18 + ln(x10 − 100)
h3  = x7 + 100000 x9 − 5·10⁷             h13 = −x19 + ln(−x8 + 300)
h4  = x5 + 100000 x10 − 3.3·10⁷          h14 = −x20 + ln(x16)
h5  = x6 + 100000 x11 − 4.4·10⁷          h15 = −x21 + ln(−x9 + 400)
h6  = x7 + 100000 x12 − 6.6·10⁷          h16 = −x22 + ln(x17)
h7  = x5 − 120 x2 x13                    h17 = −x8 − x10 + x13 x18 − x13 x19 + 400
h8  = x6 − 80 x3 x14                     h18 = x8 − x9 − x11 + x14 x20 − x14 x21 + 400
h9  = x7 − 40 x4 x15                     h19 = x9 − x12 − 4.60517 x15 + x15 x22 + 100
h10 = x8 − x11 + x16
```

The report counts an equality as met when |h(x)| ≤ 0.0001, and so does genoxide's `G22`. The
report's best known value is f* = 236.430975504001, at an x* whose equalities are all within the
tolerance and where g1 is −2.2·10⁻⁷, nearly active. It improved on the 382.902205 known before, as
Takahama and Sakai (2006, IEEE CEC 2006) report; it isn't proven optimal.

### Three free variables

Nineteen equalities in 22 variables leave three free, and here they can be solved in order. h1 to
h6, h10 and h11 are linear: they give x5, x6, x7, x10, x11, x12, x16 and x17 from x8 and x9 (x10 =
430 − x8, x11 = 440 − x9 + x8, x17 = 160). h12 to h16 give x18 to x22 as logarithms, h17 to h19 then
give x13, x14 and x15, and h7 to h9 give x2, x3 and x4. With g1 active, x1 = x2^0.6 + x3^0.6 +
x4^0.6, and f is a function of x8 and x9 alone, where x10 ≤ 300 and x11 ≤ 400 ask for x8 ≥ 130 and
x9 ≥ x8 + 40.

That function is least at the corner x8 = 130, x9 = 170, as a grid over the two and a local search
made for this page find: f = 236.370313314566, with every equality met exactly, 0.0607 below the
report's best known (whose x8 is 130.075 and x9 170.817). genoxide's docs derive it and its tests
check it; `G22` keeps the report's value as its `optimum()`, the value the report's rules compare
with.

## What makes it hard

The feasible set is a thin layer around a 3-dimensional surface in 22 dimensions: the report's table
3 gives a feasible share of 0.0000 %. And the equalities work at very different scales. x5, x6 and
x7 range up to 4·10⁷, and h1 to h6 multiply x8 to x12 by 100,000, yet each must be met to within
0.0001: a relative precision of about 10⁻¹¹ in x5, x6 and x7. A search that moves the variables one
sample at a time, without solving the equalities, has to hit that layer by chance.

No algorithm of the CEC 2006 session solved g22: in the organizers' comparison of the results (Liang
and Suganthan, July 2006), none reached the best known value in any run. εDE (Takahama and Sakai,
2006) found feasible solutions in all its runs, with a mutation that follows the gradients of the
constraints, which genoxide's algorithms don't use.

## Representation

The example runs two searches. The first has a `Real` genome of 22 genes, x1 to x22, within the
bounds above, as the report poses the problem. genoxide's `problems::cec2006::G22` is the fitness:
the value f(x) and the total constraint violation, the sum of max(0, g1(x)) and of
max(0, |h(x)| − 0.0001) over the equalities, 0 for a feasible solution.

The second has a `Real` genome of 3 genes, x1, x8 and x9, within their bounds. Its fitness solves
the equalities in order, as above, for the other 19 variables, each kept within its bounds, and
evaluates the 22 with `G22`: the same value and violation as the first search's. A variable that
its bounds cut, such as x10 = 430 − x8 above 300 when x8 is below 130, leaves its equality unmet,
and the violation says by how much: the bounds of the 19 solved variables become constraints on the
three free ones. The logarithms are genoxide's (`genoxide::math::ln`, the same to the bit on every
platform), which `G22` uses too; the Python example takes them from `G22`'s constraints h12 to h16,
with x18 to x22 at 0, so both versions compute the same numbers.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. Until a feasible solution appears, the
search is a minimization of the violation, and f plays no part.

## Algorithm

L-SHADE (Tanabe and Fukunaga, 2014, IEEE CEC 2014: 1658-1665) searches the 22 variables. It is
SHADE with a population that shrinks linearly over a known budget: current-to-pbest/1 mutation with
an archive, and a memory of the scale factor F and the crossover rate CR that worked. genoxide's
`De::l_shade` starts it with 18 · 22 = 396 individuals and ends it with 4. A trial outside the
bounds is brought back halfway between its parent and the bound, and Deb's rules decide between a
trial and its parent. It has the report's budget of 500,000 evaluations, and no target.

SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78) searches x1, x8 and x9, with genoxide's
defaults: its published population of 100, and a restart after 200 generations without progress.
It stops once its best solution is feasible and within 1e-8 of 236.370313314566, the least value
with every equality met exactly, or after 500,000 evaluations.

On the 22 variables, none of genoxide's algorithms found a feasible solution in 25 seeds each, and
L-SHADE came the closest. It ended at a violation between 8.5 and 53 on 12 seeds, and between
4,100 and 19,300 on the other 13. SHADE ended between 4,800 and 27,000, CMA-ES with IPOP restarts
(Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195; Auger and Hansen, 2005, IEEE
CEC 2005: 1769-1776) between 72 and 301. Deb's rules add up the violations in their own units, so
that h1 to h6, in units of 10⁵ to 10⁷, drown the rest: the runs meet those and miss the logarithms.
With each constraint's violation divided by its typical size, the median |g| or |h| over 1,000
random points, SHADE and L-SHADE found a feasible solution in each of 16 runs, but ended between
f = 243 and 317, far above f*: moving along a layer 0.0001 thick in 22 dimensions is slow. Solving
the equalities takes the layer away, and leaves a search in 3 dimensions.

## Output

The first two lines give the run on the 22 variables: its name, and its best solution's violation
when it stopped. The next two give the run on x1, x8 and x9, and what stopped it. The fifth
compares f(x) with 236.370313, the least value with every equality met, and the sixth with the
report's f*. The seventh gives the solution, all 22 variables, and the last the constraints: for
g1, "active" on the boundary (|g| ≤ 1e-6), else the value of g, negative when it's satisfied; for
h1 to h19, "active" when the equality is met within the tolerance, else by how much |h| exceeds it.
In Python, `run` evaluates the problem in Rust, and the second run's fitness function evaluates the
22 variables in Rust, a generation at a time, so both versions print the same.

The page plays the second run back. Its plot shows each of the 22 variables on its range, and each
constraint's state: violated, active or satisfied (an equality met within the tolerance shows as
active). Its curve shows the error f − 236.370313314566 of the best feasible solution, and of the
population's median, on a log scale.

[The project page](https://tachsin.gr/projects/genoxide/examples/cec2006-g22) plays this run back.

## Good results

A good run is feasible and ends within 1e-4 of f*, the report's success. f* is only the best known,
and a value below it is possible: 236.370313, with every equality met exactly, is 0.0607 below it.

L-SHADE on the 22 variables, with seed 1, ends at a violation of 53.53, after 500,003 evaluations.
Its solution meets the linear equalities h1 to h11, and h17 and h18, within the tolerance, but not
the five logarithms h12 to h16, nor h19: it has met the equalities of the large variables, and not
those of the small ones. With seeds 1 to 1,000, no run is feasible: the violations end between 2.2
and 40,400, 6,870 for half of them.

SHADE on x1, x8 and x9, with seed 1, starts with feasible solutions in its first random population,
the best at f = 437.5, and meets its target after 25,700 evaluations: f = 236.370313, 9.9·10⁻⁹
above the least value with every equality met, and 0.060662 below the report's best known. Its
solution is the corner x8 = 130, x9 = 170, where x10 and x11 are at their upper bounds, 300 and 400,
with every equality met and g1 active. With seeds 1 to 1,000, every run meets the target, after
23,400 to 27,500 evaluations (25,800 for half of them).
