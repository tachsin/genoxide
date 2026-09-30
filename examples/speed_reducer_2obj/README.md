---
title: Speed reducer, two objectives
category: multi-objective
summary: Minimize the volume of a gearbox and the stress in its first shaft, subject to eleven constraints, with an integer number of teeth, with NSGA-II.
reference: "Kurpati, A., Azarm, S. and Wu, J. (2002). Constraint handling improvements for multiobjective genetic algorithms. Structural and Multidisciplinary Optimization 23(3): 204-213."
reference_url: https://doi.org/10.1007/s00158-002-0178-2
optimum: "not known in closed form; from a volume of 2771.9151 at the stress limit of 1300 to a stress of 694.70574 at a volume of 5777.9203 (best known); genoxide's reference front has a hypervolume of 1.1804 in scaled objectives (reference point (1.1, 1.1))"
languages: [rust, python]
order: 223
---

# Speed reducer, two objectives

## The problem

Golinski's speed reducer, a gearbox of two shafts, with two objectives: its volume and the stress in
its first shaft. The genes are those of the single-objective problem: the face width x₁, the teeth's
module x₂, the number of teeth on the pinion x₃ (an integer), the shafts' lengths between bearings
x₄ and x₅, and their diameters x₆ and x₇:

```text
minimize   f₁ = 0.7854 x₁x₂² (10x₃²/3 + 14.933 x₃ − 43.0934) − 1.508 x₁ (x₆² + x₇²)
                + 7.477 (x₆³ + x₇³) + 0.7854 (x₄x₆² + x₅x₇²)
           f₂ = √((745 x₄ / (x₂x₃))² + 1.69·10⁷) / (0.1 x₆³)
subject to 1/(x₁x₂²x₃) ≤ 1/27,  1/(x₁x₂²x₃²) ≤ 1/397.5         the teeth's stresses
           x₄³/(x₂x₃x₆⁴) ≤ 1/1.93,  x₅³/(x₂x₃x₇⁴) ≤ 1/1.93      the shafts' deflections
           x₂x₃ ≤ 40,  5 ≤ x₁/x₂ ≤ 12                           proportions
           x₄ ≥ 1.5x₆ + 1.9,  x₅ ≥ 1.1x₇ + 1.9                  the shafts' lengths
           f₂ ≤ 1300,  √((745 x₅ / (x₂x₃))² + 1.575·10⁸) / (0.1 x₇³) ≤ 1100
x₁ in [2.6, 3.6], x₂ in [0.7, 0.8], x₃ in [17, 28], x₄, x₅ in [7.3, 8.3], x₆ in [2.9, 3.9],
x₇ in [5, 5.5]
```

Kurpati, Azarm and Wu's paper (2002), the source, couldn't be read. The definition and bounds are
those of Tanabe and Ishibuchi (2020, problem RE3-7-5 and its constrained form CRE2-7-4), which Saad,
Emam and Houssein (2025, *Scientific Reports* 15: 5126, eqs. 22-23) restate the same, but for the
second shaft's constant, 1.275·10⁸ there: genoxide follows 1.575·10⁸, Golinski's 157.5·10⁶ for the
same shaft. The constants differ from the single-objective `SpeedReducer`'s (7.477 and 14.933 for
7.4777 and 14.9334, stress limits of 1300 and 1100 for 1100 and 850, x₅ down to 7.3).

The front isn't known in closed form. Its ends, found with genoxide's SHADE, give genoxide's
`SpeedReducer` in `multi::problems::engineering` its ideal and nadir points: the lightest gearbox,
2771.9151 at (3.5, 0.7, 17, 7.3, 7.4, 3.1687580, 5), where the first shaft's stress is at its limit
of 1300, and the least stress, 694.70574, at (3.6, 0.72, 28, 7.75, 7.4, 3.9, 5), whose lightest
gearbox has a volume of 5777.9203.

genoxide's reference front, the non-dominated designs of about 6,000 ε-constraint problems (the
least volume at a stress of at most ε, and the least stress at a volume of at most ε, for ε evenly
spread between the ends), each solved by SHADE with 30,000 or 60,000 evaluations, and of eight
NSGA-II runs of 1,000 generations, has a hypervolume of 1.1804 in the scaled objectives below: a
lower bound on the whole front's.

## What makes it hard

Eleven constraints, several of them active along the front, and an integer number of teeth, which
the problem rounds. The front's far end is nearly flat: from a volume of about 4000 to 5778 the
stress falls by little more than 1, so a front can stop well short of the least stress and lose
almost nothing in hypervolume.

## Representation

A `Real` genome of 7 genes, the third rounded to the nearest integer; `design` gives the rounded
design. The problem is genoxide's `multi::problems::engineering::SpeedReducer`
(`gx.problems.multi_engineering.SpeedReducer` in Python), whose fitness is the two objectives and
the total constraint violation. Solutions compare by constrained dominance.

## Algorithm

NSGA-II with a population of 100 for 250 generations, simulated binary crossover (η = 20, at a rate
of 0.9) and polynomial mutation (η = 20, at a rate of 1/7 per gene).

## Output

The first line gives the size of the final front and how many of its solutions are feasible, the
second the range of each objective on it. The third gives its hypervolume up to the reference point
(1.1, 1.1), in objectives scaled to [0, 1] by the ideal point (2771.9151, 694.70574) and the nadir
point (5777.9203, 1300), as a share of the reference front's. In Python, `run` evaluates the problem
in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/speed-reducer-2obj) plays this run
back.

## Good results

A good front is feasible and spread from the lightest gearbox to a stress near 694.7. 100 points of
the reference front, chosen one by one for the most hypervolume, give 99.95% of its hypervolume.

The run's front has 100 feasible solutions with 17 to 22 teeth, from a volume of 2772.05 at a stress
of 1299.36 to a stress of 695.87 at 4016.30, with 99.86% of the reference front's hypervolume: it
stops at the flat end, 1.2 above the least stress. Over seeds 1 to 20, every run ends between 99.82%
and 99.91%.
