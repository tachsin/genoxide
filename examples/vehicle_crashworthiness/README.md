---
title: Vehicle crashworthiness
category: multi-objective
summary: Minimize a car's mass, the deceleration in a full frontal crash and the toe board's intrusion in an offset one, three response surfaces fitted to crash simulations, with SMS-EMOA.
reference: "Liao, X., Li, Q., Yang, X., Zhang, W. and Li, W. (2008). Multiobjective optimization for crash safety design of vehicles using stepwise regression model. Structural and Multidisciplinary Optimization 35(6): 561-569."
reference_url: https://doi.org/10.1007/s00158-007-0163-x
optimum: "not known in closed form; ideal point (1661.7078, 6.1428, 0.0394), at bounds; genoxide's reference front has a hypervolume of 1.0525 in objectives scaled by the ideal point and its estimated nadir point (1695.161, 10.736, 0.264) (reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 227
---

# Vehicle crashworthiness

## The problem

Liao, Li, Yang, Zhang and Li (2008) designed the front of a car for crash safety. Its five genes are
the thicknesses, in mm, of five reinforcing members around the front, and response surfaces fitted
by stepwise regression to crash simulations give three objectives:

```text
minimize   f₁ = 1640.2823 + 2.3573285x₁ + 2.3220035x₂ + 4.5688768x₃ + 7.7213633x₄ + 4.4559504x₅
                                                                       the mass, in kg
           f₂ = 6.5856 + 1.15x₁ − 1.0427x₂ + 0.9738x₃ + 0.8364x₄ − 0.3695x₁x₄ + 0.0861x₁x₅
                + 0.3628x₂x₄ − 0.1106x₁² − 0.3437x₃² + 0.1764x₄²
                                     the integral of the deceleration in the full frontal crash
           f₃ = −0.0551 + 0.0181x₁ + 0.1024x₂ + 0.0421x₃ − 0.0073x₁x₂ + 0.024x₂x₃ − 0.0118x₂x₄
                − 0.0204x₃x₄ − 0.008x₃x₅ − 0.0241x₂² + 0.0109x₄²
                                     the toe board's intrusion in the 40% offset frontal crash
x₁ … x₅ in [1, 3]
```

Liao et al.'s paper couldn't be read. The surfaces and bounds are those of Tanabe and Ishibuchi
(2020, problem RE3-5-4) and, the same, of de Carvalho and Sichman (2018, OptMAS 2018, eqs. 1-3);
they are genoxide's `VehicleCrashworthiness`, still to be checked against the original. Deb and Jain
(2014, part I, figure 30) draw NSGA-III's front over masses from about 1660 to 1700, where only 40
of 91 reference directions found a solution.

The front isn't known. Its ideal point is at bounds: the least mass with every member at 1 mm, the
least deceleration at (1, 3, 3, 1, 1) and the least intrusion at (1, 1, 3, 3, 3). genoxide's
reference front, the non-dominated designs of about 10,000 runs of SHADE, each minimizing an
achievement scalarizing function along one of Das and Dennis's directions, and of sixteen runs of
SMS-EMOA and NSGA-III of 2,000 generations, has worst values (1695.161, 10.736, 0.264), the
estimated nadir point, and a hypervolume of 1.0525 in the scaled objectives below: a lower bound on
the whole front's.

## What makes it hard

The front covers only part of the plane of the reference directions, as Deb and Jain found, and some
of its parts are small: a run can miss one and keep its population elsewhere, since nothing
dominates the solutions it has.

## Representation

A `Real` genome of 5 genes, the thicknesses. The problem is genoxide's
`multi::problems::engineering::VehicleCrashworthiness`
(`gx.problems.multi_engineering.VehicleCrashworthiness` in Python). It has no constraints besides
the bounds.

## Algorithm

SMS-EMOA, which keeps the solutions that add the most hypervolume, with a population of 92 for 500
generations, simulated binary crossover (η = 20, at a rate of 0.9) and polynomial mutation (η = 20,
at a rate of 1/5 per gene).

## Output

The first line gives the size of the final front and how many of its solutions are feasible (all
are), the second the range of each objective on it. The third gives its hypervolume up to the
reference point (1.1, 1.1, 1.1), in objectives scaled to [0, 1] by the ideal point and the estimated
nadir point, as a share of the reference front's. In Python, `run` evaluates the problem in Rust, so
both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/vehicle-crashworthiness) plays this
run back.

## Good results

A good front reaches the three minima and covers every part of the front. 92 points of the reference
front, chosen one by one for the most hypervolume, give 99.29% of its hypervolume.

The run's front has 92 solutions, reaching the least mass and deceleration, but intrusions only down
to 0.0523 of the least 0.0394, with masses up to 1684.3 and 98.78% of the reference front's
hypervolume, 99.5% of what 92 chosen points reach. Over seeds 1 to 20, the runs end in two groups,
at 98.78% and at 99.2%, but for seed 5, which misses a part of the front and ends at 95.15%. Longer
runs, 2,000 generations, and other mutation rates end the same.
