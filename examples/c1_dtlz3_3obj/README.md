---
title: C1-DTLZ3 with 3 objectives
category: multi-objective
summary: Minimize three objectives of DTLZ3 with an infeasible shell between the radii 4 and 9 in front of the spherical front; NSGA-III with constraint handling stops at the shell, and NSGA-III on the objectives alone passes it and reaches the front.
reference: "Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization algorithm using reference-point based nondominated sorting approach, part II: handling constraints and extending to an adaptive approach. IEEE Transactions on Evolutionary Computation 18(4): 602-622."
reference_url: https://doi.org/10.1109/TEVC.2013.2281534
optimum: "DTLZ3's front, the unit sphere, all feasible; the 91 target points' hypervolume is 0.7449 (reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 194
family: C-DTLZ
tab: C1-DTLZ3
---

# C1-DTLZ3 with 3 objectives

## The problem

Jain and Deb (2014) extended NSGA-III to constrained problems and built, for its tests, the
constrained DTLZ problems: DTLZ problems of any number of objectives with constraints of three
types. Type 1 keeps the front and puts an infeasible barrier before it. C1-DTLZ3 is DTLZ3 with one
constraint:

```text
minimize   f₁ = (1 + g) cos(x₁π/2) cos(x₂π/2)
           f₂ = (1 + g) cos(x₁π/2) sin(x₂π/2)
           f₃ = (1 + g) sin(x₁π/2)
           g = 100 (10 + Σᵢ₌₃¹² ((xᵢ − 0.5)² − cos(20π (xᵢ − 0.5))))
subject to (Σ fᵢ² − 16) (Σ fᵢ² − r²) ≥ 0,  r = 9
x in [0, 1]¹²
```

The objectives lie on a sphere of radius 1 + g, and the front is the unit sphere, where the last
ten variables are 0.5 and g = 0. The constraint makes the shell between the radii 4 and r = 9
infeasible (r is 9 for 3 objectives, 12.5 for 5 and 8, and 15 for 10 and 15): g from 3 to 8. The
front, well inside, is all feasible.

The problem is genoxide's `C1Dtlz3`, checked against the paper's eq. 5 and its radii (section V-B)
in its accepted manuscript (the journal's final text wasn't compared), with 12 variables (k = 10),
as the paper uses.

## What makes it hard

DTLZ3's g has 3¹⁰ − 1 local optima, whose local fronts are spheres of radius 1 + g for whole
numbers g. An algorithm usually climbs down them one step at a time, g from 8 to 7 to 6: here
those steps land in the infeasible shell. A population that follows the constraint strictly comes
to rest on the shell's outer edge, g = 8, where every solution is feasible and non-dominated, and
the next local front is infeasible. To get through, an offspring has to jump from g ≥ 8 to g ≤ 3
at once, for example by crossing over two parents whose off-center variables are in different
places.

Tanabe and Oyama (2017, "A note on constrained multi-objective optimization benchmark problems",
IEEE CEC 2017: 1127-1134) point out that C1-DTLZ1, C1-DTLZ3 and C2-DTLZ2 are solved by an
algorithm that ignores the constraints: their fronts are feasible, and the barrier only stops an
algorithm that respects it.

## Representation

A `Real` genome of 12 genes in [0, 1]. The problem's fitness is the three objectives and the
constraint violation, 0 when it's feasible.

## Algorithm

NSGA-III (Deb and Jain, 2014) with the paper's settings, for 1,500 generations, twice:

- the 91 reference directions of Das and Dennis's method with 12 divisions, and a population of
  92;
- simulated binary crossover with η = 30, at a rate of 1;
- polynomial mutation with η = 20, at a rate of 1/n per gene for n genes: 1/12.

The first run handles the constraint as Jain and Deb do: parents are paired at random, except that
of two infeasible ones the smaller violation wins; the next population takes the feasible
solutions first, sorted into non-dominated fronts and spread along the reference directions, and
fills the rest with the least infeasible ones.

The second run ignores the constraint: its fitness is the three objectives alone, and the final
front is checked against the constraint afterwards.

## Output

For each run, the first line gives the size of the final front and how many of its solutions are
feasible.

NSGA-III aims at one solution per reference direction: its targets are the 91 points where the
directions meet the unit sphere. The second line counts the targets that a solution comes within
0.02 of. The front's nadir point is (1, 1, 1) and its ideal point the origin, so the objectives
need no scaling.

The third gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to the
91 targets: the mean, over the targets, of the distance to the nearest solution, counting only the
objectives in which the solution is worse. It's the measure of the paper, which uses IGD to the
same targets. Then the front's hypervolume, the volume it dominates up to the reference point
(1.1, 1.1, 1.1), as a share of the 91 targets' hypervolume, 0.7449. In Python, `run` evaluates
the problem in Rust, and the second run evaluates it in batches, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/c1-dtlz3-3obj) plays the second
run back, with the solutions inside the shell drawn as infeasible.

## Good results

A good front is all feasible, with a solution at each of the 91 targets, an IGD+ well under 0.01
and a hypervolume close to the targets'.

The first run stops at the shell: 92 feasible solutions on the sphere of radius 9, an IGD+ of
8.004 and no hypervolume inside the reference point. Over seeds 1 to 20, 19 runs stop there after
1,500 generations, and one gets through, to an IGD+ of 0.0055. The paper reports that 13 of its 20
runs of NSGA-III got through in 1,000 generations, and a median IGD of 0.0081.

The second run passes the shell and reaches the front: 92 solutions, all feasible, reaching all
91 targets, with an IGD+ of 0.0008 and 99.83% of the targets' hypervolume. Over seeds 1 to 20,
every run ends with 91 or 92 feasible solutions within 0.02 of 90 or 91 targets, with an IGD+ from
0.0005 to 0.0054.
