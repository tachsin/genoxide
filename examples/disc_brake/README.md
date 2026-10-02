---
title: Disc brake
category: multi-objective
summary: Minimize the mass of a multiple-disc brake and its stopping time, subject to five constraints, with an integer number of friction surfaces that splits the front into pieces, with NSGA-II.
reference: "Osyczka, A. and Kundu, S. (1995). A new method to solve generalized multicriteria optimization problems using the simple genetic algorithm. Structural Optimization 10(2): 94-99."
reference_url: https://doi.org/10.1007/BF01743536
optimum: "not known in closed form; from 0.1274 kg stopping in 16.654925 s to 2.0710401 s at 2.793 kg, both ends derived; genoxide's reference front has a hypervolume of 1.0853 in scaled objectives (reference point (1.1, 1.1))"
languages: [rust, python]
order: 238
---

# Disc brake

## The problem

A multiple-disc brake should be light and stop fast. Its design variables are the discs' inner and
outer radii r and R, in mm, the engaging force F, in N, and the number of friction surfaces s, an
integer:

```text
minimize   f₁ = 4.9·10⁻⁵ (R² − r²)(s − 1)                     the mass, in kg
           f₂ = 9.82·10⁶ (R² − r²) / (F s (R³ − r³))           the stopping time, in s
subject to R − r ≥ 20                                          the discs' width
           2.5 (s + 1) ≤ 30                                    the brake's length
           F / (3.14 (R² − r²)) ≤ 0.4                          the pressure
           2.22·10⁻³ F (R³ − r³) / (R² − r²)² ≤ 1              the temperature
           0.0266 F s (R³ − r³) / (R² − r²) ≥ 900              the torque
r in [55, 80], R in [75, 110], F in [1000, 3000], s in [2, 20]
```

Osyczka and Kundu's paper (1995) and Ray and Liew's (2002), the problem's sources, couldn't be read:
the definition and bounds are those of Yang, Karamanoglu and He (2013, *Procedia Computer Science*
18: 861-868, eqs. 10-12), and the same in Saad, Emam and Houssein (2025, *Scientific Reports* 15:
5126, eqs. 24-25). Tanabe and Ishibuchi's (2020, problem RE3-4-3) drops the length constraint and
bounds s by [11, 20], though their note says the constraint gives s ≤ 11: a different problem.

The front isn't known in closed form, but its ends follow from the definition. The lightest brake
has the narrowest discs at the smallest radii, r = 55 and R = 75, and two surfaces: 0.1274 kg,
stopping in 16.654925 s at the largest force, 3000. The fastest has the widest discs, r = 80 and R =
110, the most surfaces the length allows, 11, and the largest force: 2.0710401 s at 2.793 kg. Both
meet every constraint. They give genoxide's `DiscBrake` its ideal and nadir points.

genoxide's reference front, the non-dominated designs of about 6,000 ε-constraint problems (the
least mass at a stopping time of at most ε, and the least time at a mass of at most ε, for ε evenly
spread between the ends), each solved by SHADE with 30,000 or 60,000 evaluations, and of eight
NSGA-II runs of 1,000 generations, has a hypervolume of 1.0853 in the scaled objectives below: a
lower bound on the whole front's.

## What makes it hard

The number of surfaces is an integer: the genome is real, and the problem rounds its fourth gene to
the nearest integer, so the front is ten pieces, one for each number of surfaces from 2 to 11, and a
solution moves between pieces in jumps. The mass grows with s − 1 and the stopping time falls with
s, so every piece is needed. The force stays near its upper bound of 3000 along the whole front, and
the radii trade mass against stopping time within each piece.

## Representation

A `Real` genome of 4 genes: r, R, F and s, which the problem rounds; `design` gives the rounded
design. The problem is genoxide's `multi::problems::engineering::DiscBrake`
(`gx.problems.multi_engineering.DiscBrake` in Python), whose fitness is the two objectives and the
total constraint violation. Solutions compare by constrained dominance.

## Algorithm

NSGA-II with a population of 100 for 250 generations, simulated binary crossover (η = 20, at a rate
of 0.9) and polynomial mutation (η = 20, at a rate of 1/4 per gene).

## Output

The first line gives the size of the final front and how many of its solutions are feasible, the
second the range of each objective on it. The third gives its hypervolume up to the reference point
(1.1, 1.1), in objectives scaled to [0, 1] by the ideal point (0.1274, 2.0710401) and the nadir
point (2.793, 16.654925), as a share of the reference front's. In Python, `run` evaluates the
problem in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/disc-brake) plays this run back.

## Good results

A good front is feasible, with solutions on every piece from 2 to 11 surfaces. 100 points of the
reference front, chosen one by one for the most hypervolume, give 99.79% of its hypervolume.

The run's front has 100 feasible solutions on all ten pieces (23 with two surfaces, 26 with eleven,
2 to 14 with each number between), from 0.1303 kg to the fastest brake, 2.0710 s, with 99.60% of the
reference front's hypervolume. Over seeds 1 to 20, every run ends between 99.41% and 99.61%.
