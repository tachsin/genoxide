---
title: Engineering designs and CEC 2006
category: constrained
summary: SHADE with Deb's feasibility rules on five engineering designs and the CEC 2006 problems g01 to g06, against each proven or best known optimum.
reference: "Liang, J. J., Runarsson, T. P., Mezura-Montes, E., Clerc, M., Suganthan, P. N., Coello Coello, C. A. and Deb, K. (2006). Problem Definitions and Evaluation Criteria for the CEC 2006 Special Session on Constrained Real-Parameter Optimization. Technical report, Nanyang Technological University, Singapore."
reference_url: "https://github.com/P-N-Suganthan/CEC2006"
optimum: "each problem's proven or best known value f*: a gap of 0"
languages: [rust, python]
order: 76
---

# Engineering designs and CEC 2006

## The problem

Eleven constrained problems of real variables, each to minimize: five engineering designs and the
first six problems of the CEC 2006 competition. Each comes from genoxide's `problems`, with its
bounds, its constraints, its optimum or best known value f* and its source; the docs of
[`genoxide::problems::engineering`](https://docs.rs/genoxide/latest/genoxide/problems/engineering/)
and [`genoxide::problems::cec2006`](https://docs.rs/genoxide/latest/genoxide/problems/cec2006/) give
each formula. Most engineering originals aren't openly available, so the definitions are those of
later papers that restate them, named below.

The engineering designs:

- The tension/compression spring: the lightest coil spring, from its wire diameter, its mean coil
  diameter and its number of active coils. Four constraints limit its deflection, its shear stress,
  its surge frequency and its outer diameter. After Belegundu (1982, PhD thesis, University of
  Iowa) and Arora (1989, Introduction to Optimum Design, McGraw-Hill), as restated by Coello Coello
  (2000, Computers in Industry 41(2): 113-127).
- Golinski's speed reducer: the lightest gearbox, from 7 variables: the face width, the module of
  the teeth, the number of teeth on the pinion (an integer, rounded when evaluated), and the lengths
  and diameters of two shafts. Eleven constraints limit the bending and surface stresses of the
  teeth, the deflections and stresses of the shafts, and the proportions. After Golinski (1970,
  Journal of Mechanisms 5(3): 287-309; 1973, Mechanism and Machine Theory 8(4): 419-436), as
  restated by Cagnina, Esquivel and Coello Coello (2008, Informatica 32: 319-326).
- The three-bar truss: the least volume of a planar truss of three bars under a load, from the
  cross-sections of the bars. Three constraints limit the stresses in the bars. After Nowacki (1974)
  and Ray and Saini (2001, Engineering Optimization 33(6): 735-748), as restated by Yang and Gandomi
  (2012, Engineering Computations 29(5): 464-483).
- The cantilever beam: the lightest beam of five hollow square segments with a load at its free
  end, from the widths of the segments. One constraint limits its deflection. After Fleury and
  Braibant (1986, International Journal for Numerical Methods in Engineering 23(3): 409-428), as
  restated by Yang, Huyck, Karamanoglu and Khan (2013, International Journal of Bio-Inspired
  Computation 5(6): 329-335).
- The car side impact: the lightest car body whose side withstands the European side-impact test,
  from 7 panel thicknesses. Ten constraints, response surfaces fitted to crash simulations, limit
  the loads on the dummy, its rib deflections and chest velocities, and the velocities of the
  B-pillar and the front door. After Gu, Yang, Tho, Makowski, Faruque and Li (2001, International
  Journal of Vehicle Design 26(4): 348-360), as restated by Jain and Deb (2014, IEEE Transactions on
  Evolutionary Computation 18(4): 602-622).

The CEC 2006 special session on constrained optimization (Liang et al., 2006) collected 24 problems,
g01 to g24, from the literature, with their best known solutions and rules for comparing algorithms:
500,000 evaluations per run, an equality met when |h(x)| ≤ 0.0001, and a run successful when it
finds a feasible solution with f − f* ≤ 0.0001. The first six:

| Problem | Variables | Constraints | f* | Character |
|---|---|---|---|---|
| g01 | 13 | 9 linear inequalities | −15, proven | a quadratic, with 6 constraints active at the minimum |
| g02 | 20 | 2 nonlinear inequalities | −0.803619, best known | highly multimodal, with many local optima |
| g03 | 10 | 1 nonlinear equality | −1.0005001, proven | a product on the unit sphere |
| g04 | 5 | 6 nonlinear inequalities | −30665.539, proven | Himmelblau's nonlinear problem, a quadratic |
| g05 | 4 | 2 linear inequalities, 3 nonlinear equalities | 5126.4967, best known | a cubic, on a curve set by the equalities |
| g06 | 2 | 2 nonlinear inequalities | −6961.8139, proven | a cubic, in a thin crescent between two circles |

The report takes g01 and g06 from Floudas and Pardalos (1990, LNCS 455), g02 from Koziel and
Michalewicz (1999, Evolutionary Computation 7(1): 19-44), g03 from Michalewicz, Nazhiyath and
Michalewicz (1996), g04 from Himmelblau (1972, Applied Nonlinear Programming, McGraw-Hill) and g05
from Hock and Schittkowski (1981, Test Examples for Nonlinear Programming Codes, Springer). g02 and
g03 are maximizations, negated. g03's minimum, −(1.0001)⁵, lies on the outer edge of the equality's
tolerance; without the tolerance it would be −1.

A best known value isn't a proven optimum: the spring's, the speed reducer's, the car side impact's,
g02's and g05's may be beaten. The spring's, 0.012665, is printed to 5 digits, and the published
design exceeds one constraint by 2e-5. The speed reducer's published design, printed to 6 digits,
exceeds two constraints slightly. The three-bar truss's and the cantilever beam's minima are derived
in closed form.

## What makes it hard

Every minimum lies on the boundary of the feasible region, with some constraints active. The search
has to approach a boundary it can't cross.

The feasible regions are often tiny. The report estimates each one's share of the box from random
points: 0.0111 % for g01, 0.0066 % for g06, and 0.0000 % for g03 and g05, whose equalities leave
only a thin shell around a surface. For g02 it's 99.997 %, but g02's landscape is rugged, with many
local optima. The speed reducer's integer number of teeth makes its weight a step function of that
gene.

## Representation

A `Real` genome within each problem's bounds, from 2 genes (g06, the truss) to 20 (g02). The fitness
is the value and the total constraint violation: the sum of max(0, g(x)) over the inequalities and
of max(0, |h(x)| − 0.0001) over the equalities, 0 for a feasible solution. The engineering
constraints keep the units of the papers that state them.

genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible solution beats an infeasible one, two feasible
ones compare by value, and two infeasible ones by violation. The rules need no penalty weights.

## Algorithm

SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78), a differential evolution that adapts its
scale factor and crossover rate from successful trials, with genoxide's defaults: its published
population of 100, and a restart after 200 generations without progress.

One run per problem, with seed 1. The engineering designs have a budget of 50,000 evaluations each,
enough for SHADE to converge; the CEC 2006 problems have the report's 500,000. A run stops early
once its best solution is feasible and within 1e-8 of f*, relative to |f*|. The last generation can
pass the budget by less than a population.

## Output

A row per problem: its budget, the best value found and f*, to 6 significant digits, whether f* is
a proven optimum or the best known value, the gap and whether the best solution is feasible. The gap
is relative, (f − f*) / |f*|, since the values range from 0.0127 to −30665.5. "< 1e-8" means the run
met its target and stopped early.

In Python, `run` evaluates the problems in Rust, so both versions print the same table. g02 and g05
call the platform's `cos` and `sin`, whose last bit can differ between operating systems; their runs
can then take different paths, but the table stays the same on Windows and Linux.

The page's plot shows each problem's error f − f* against the evaluations, on a log scale. A curve
begins at the first feasible solution, and ends where its run stopped.

[The project page](https://tachsin.gr/projects/genoxide/examples/engineering-suite) plays this run back.

## Good results

A gap of 0 is the best possible, and a best known value can be beaten. SHADE meets the 1e-8 target
on eight problems: the speed reducer, the truss, the cantilever beam, the car side impact, g01, g04,
g05 and g06. The truss stops first, after about 10,000 evaluations. g05 is the slowest: its first
feasible solution comes after about 40,000 evaluations, and it meets the target after about 230,000.

The spring's gap, 1.8e-5, is the rounding of its best known value: SHADE's design weighs 0.0126652,
and runs with seeds 2 to 4 end at the same value, to 6 digits.

g02's gap, 1.0e-4, is 8.1e-5 in f, within the report's criterion of success. The run stops
improving after about 200,000 evaluations. With seeds 2 to 8, six runs end within a relative gap of
1e-6 of the best known value, and one at 7e-4.

g03 is the failure: the run ends at −0.791, far from the minimum of −1.0005. Deb's rules first
drive the population onto the thin shell of the equality, wherever it meets it. On the shell, a
trial point made from differences between points on a curved surface falls off it unless the step
is tiny, and loses to its feasible parent. The best value improves in steps, from −0.19 after 5,000
evaluations to −0.79 after 300,000, and then stops. With seeds 2 to 8, the runs end between −0.91
and −0.69.
