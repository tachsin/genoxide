---
title: Car side impact, three objectives
category: multi-objective
summary: Minimize a car's weight, the pubic force on a passenger and the mean velocity of the B-pillar and the front door in a side impact, subject to ten constraints, with NSGA-III and Jain and Deb's settings.
reference: "Jain, H. and Deb, K. (2014). An evolutionary many-objective optimization algorithm using reference-point based nondominated sorting approach, part II: handling constraints and extending to an adaptive approach. IEEE Transactions on Evolutionary Computation 18(4): 602-622."
reference_url: https://doi.org/10.1109/TEVC.2013.2281534
optimum: "not known in closed form; ideal point (23.585658, 3.58525, 10.610644); genoxide's reference front has a hypervolume of 0.8687 in objectives scaled by the ideal point and its estimated nadir point (42.768, 4.0, 12.5212) (reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 241
---

# Car side impact, three objectives

## The problem

The car side impact problem of Gu et al. (2001) designs a car body for the European side-impact
test: its seven genes are the thicknesses of the B-pillar inner and reinforcement, the floor side
inner, the cross members, the door beam, the door beltline reinforcement and the roof rail, and
response surfaces fitted to crash simulations give the dummy's loads, velocities and rib
deflections. Jain and Deb (2014, section V-F and appendix) made it a problem of three objectives:

```text
minimize   f₁ = 1.98 + 4.9x₁ + 6.67x₂ + 6.98x₃ + 4.01x₄ + 1.78x₅ + 0.00001x₆ + 2.73x₇   the weight
           f₂ = F = 4.72 − 0.5x₄ − 0.19x₂x₃                          the pubic force, in kN
           f₃ = (V_MBP + V_FD) / 2                                   the mean velocity, in mm/ms
           V_MBP = 10.58 − 0.674x₁x₂ − 0.67275x₂,  V_FD = 16.45 − 0.489x₃x₇ − 0.843x₅x₆
subject to the ten constraints of the single-objective problem: the abdomen load ≤ 1 kN, the
           upper, middle and lower chest velocities ≤ 0.32 m/s, the upper, middle and lower rib
           deflections ≤ 32 mm, F ≤ 4 kN, V_MBP ≤ 9.9 mm/ms and V_FD ≤ 15.7 mm/ms
x₁, x₃, x₄ in [0.5, 1.5], x₂ in [0.45, 1.35], x₅ in [0.875, 2.625], x₆, x₇ in [0.4, 1.2]
```

The definition was checked in Jain and Deb's accepted manuscript; Gu et al.'s paper couldn't be
read. genoxide's `CarSideImpact` in `multi::problems::engineering` shares the single-objective
`CarSideImpact`'s code, whose docs note two coefficients that the eleven-variable form would give
differently.

The front isn't known. Its ideal point, from genoxide's SHADE, is (23.585658, 3.58525, 10.610644):
the single-objective problem's least weight, and the least pubic force and mean velocity, both at
bounds. genoxide's reference front, the non-dominated designs of about 12,000 runs of SHADE, each
minimizing an achievement scalarizing function along one of Das and Dennis's directions, and of
sixteen runs of NSGA-III and SMS-EMOA of 2,000 generations, has worst values (42.768, 4.0, 12.5212),
the estimated nadir point, and a hypervolume of 0.8687 in the scaled objectives below: a lower bound
on the whole front's.

## What makes it hard

Ten constraints, and a front whose shape is far from the plane of the reference directions: Jain and
Deb found solutions for only 95 of their 153 directions, the others pointing where the front isn't
(their figure 19). The pubic force's constraint, F ≤ 4, is also the front's edge.

## Representation

A `Real` genome of 7 genes. The problem is genoxide's `multi::problems::engineering::CarSideImpact`
(`gx.problems.multi_engineering.CarSideImpact` in Python), whose fitness is the three objectives and
the total constraint violation. Solutions compare by constrained dominance.

## Algorithm

NSGA-III with Jain and Deb's settings: the 153 reference directions of Das and Dennis's method with
16 divisions, a population of 156, for 500 generations, simulated binary crossover with η = 30 (at a
rate of 1) and polynomial mutation with η = 20 at a rate of 1/7 per gene.

## Output

The first line gives the size of the final front and how many of its solutions are feasible, the
second the range of each objective on it. The third gives its hypervolume up to the reference point
(1.1, 1.1, 1.1), in objectives scaled to [0, 1] by the ideal point and the estimated nadir point, as
a share of the reference front's. In Python, `run` evaluates the problem in Rust, so both versions
print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/car-side-impact-3obj) plays this
run back.

## Good results

A good front is feasible and spread over the whole surface, from the lightest car to the least pubic
force and mean velocity. 156 points of the reference front, chosen one by one for the most
hypervolume, give 96.22% of its hypervolume: no set of 156 does much better.

The run's front has 156 feasible solutions, with weights from 23.665 to 42.766, pubic forces from
3.5853 to 4 and mean velocities from 10.6108 to 12.5129, and 93.94% of the reference front's
hypervolume, 97.6% of what 156 chosen points reach. Over seeds 1 to 20, every run ends between
93.74% and 94.19%.
