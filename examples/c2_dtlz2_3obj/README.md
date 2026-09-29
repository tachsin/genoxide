---
title: C2-DTLZ2 with 3 objectives
category: multi-objective
summary: Minimize three objectives of DTLZ2 where only the parts of the spherical front inside four small spheres are feasible, a disconnected front, with NSGA-III.
reference: "Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization algorithm using reference-point based nondominated sorting approach, part II: handling constraints and extending to an adaptive approach. IEEE Transactions on Evolutionary Computation 18(4): 602-622."
reference_url: https://doi.org/10.1109/TEVC.2013.2281534
optimum: "the parts of the unit sphere within 0.4 of (1, 0, 0), (0, 1, 0), (0, 0, 1) and (1, 1, 1)/√3; the 58 feasible target points' hypervolume is 0.6535 (reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 155
family: C-DTLZ
tab: C2-DTLZ2
---

# C2-DTLZ2 with 3 objectives

## The problem

Jain and Deb (2014) extended NSGA-III to constrained problems and built, for its tests, the
constrained DTLZ problems: DTLZ problems of any number of objectives with constraints of three
types. Type 2 makes parts of the front infeasible. C2-DTLZ2 is DTLZ2 with one constraint, which
keeps feasible only the inside of M + 1 spheres of radius r, centered at the front's corners and
at its middle:

```text
minimize   f₁ = (1 + g) cos(x₁π/2) cos(x₂π/2)
           f₂ = (1 + g) cos(x₁π/2) sin(x₂π/2)
           f₃ = (1 + g) sin(x₁π/2)
           g = Σᵢ₌₃¹² (xᵢ − 0.5)²
subject to min{ minᵢ [(fᵢ − 1)² + Σ_{j≠i} fⱼ² − r²],  Σᵢ (fᵢ − 1/√3)² − r² } ≤ 0,  r = 0.4
x in [0, 1]¹²
```

DTLZ2's front is the unit sphere, where the last ten variables are 0.5 and g = 0. The front of
C2-DTLZ2 is the parts of it inside the spheres: four round patches, one at each corner and one in
the middle, with infeasible space between them. r is 0.4 for 3 objectives and 0.5 for more.

The paper prints the constraint as max{maxᵢ […], […]} and with no inequality. Its text makes
"only the region of objective space that lies inside each of the M + 1 hyper-spheres of radius r"
feasible, and its figure 7 shows the same: only the minimum above, at most 0 inside one of the
spheres, gives that, and genoxide's `C2Dtlz2` uses it. The paper's table V confirms the reading:
it counts 58 of the 91 reference directions with a Pareto-optimal solution, and 58 of them meet
the unit sphere inside a sphere here (and 80 of 210 for 5 objectives, as the table says too). The
definition was checked in the paper's accepted manuscript (section V-C; the journal's final text
wasn't compared), with 12 variables (k = 10), as the paper uses.

## What makes it hard

The front is disconnected, and the population has to hold solutions on all four patches. NSGA-III
spreads solutions along its reference directions, and 33 of the 91 directions meet the sphere in
infeasible space: their niches have no optimal solution. Solutions that would sit between the
patches are infeasible, and the constraint pushes them into the patches. Near the front, g is
small and smooth: DTLZ2 has no local fronts.

Tanabe and Oyama (2017, "A note on constrained multi-objective optimization benchmark problems",
IEEE CEC 2017: 1127-1134) note that C2-DTLZ2 is also solved by an algorithm that ignores the
constraint and keeps the feasible solutions at the end, since its patches are parts of DTLZ2's
front.

## Representation

A `Real` genome of 12 genes in [0, 1]. The problem's fitness is the three objectives and the
constraint violation, 0 when it's feasible.

## Algorithm

NSGA-III (Deb and Jain, 2014) with Jain and Deb's constraint handling: parents are paired at
random, except that of two infeasible ones the smaller violation wins; the next population takes
the feasible solutions first, sorted into non-dominated fronts and spread along the reference
directions, and fills the rest with the least infeasible ones. The settings are the paper's:

- the 91 reference directions of Das and Dennis's method with 12 divisions, and a population of
  92;
- simulated binary crossover with η = 30, at a rate of 1;
- polynomial mutation with η = 20, at a rate of 1/n per gene for n genes: 1/12;
- 250 generations.

## Output

The first line gives the size of the final front and how many of its solutions are feasible.

NSGA-III aims at one solution per reference direction: its targets are the points where the
directions meet the front, the 58 of the 91 that are feasible. The second line counts the targets
that a solution comes within 0.02 of. The front's nadir point is (1, 1, 1) and its ideal point
the origin, so the objectives need no scaling.

The third gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to the
58 targets: the mean, over the targets, of the distance to the nearest solution, counting only the
objectives in which the solution is worse. It's the measure of the paper, which uses IGD to the
same targets. Then the front's hypervolume, the volume it dominates up to the reference point
(1.1, 1.1, 1.1), as a share of the 58 targets' hypervolume, 0.6535. The population has 92
solutions for 58 targets, and the others fill the patches between the targets: the front's
hypervolume can pass the targets'. In Python, `run` evaluates the problem in Rust, so both
versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/c2-dtlz2-3obj) plays this run
back, with the population's infeasible solutions; the true front is drawn as the feasible parts
of the sphere.

## Good results

A good front is all feasible, on all four patches, with a solution at each of the 58 targets and
an IGD+ well under 0.01.

The run's front has 92 solutions, all feasible, reaching all 58 targets, with an IGD+ of 0.0009 and
102.64% of the targets' hypervolume. Over seeds 1 to 20, 19 runs reach all 58 targets and one 56,
with an IGD+ from 0.0008 to 0.0021 and 102.3% to 103.8% of the targets' hypervolume. The paper
reports a median IGD of 0.0026 over its 20 runs.
