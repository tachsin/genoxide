---
title: Welded beam, two objectives
category: multi-objective
summary: Minimize the cost of a welded beam and the deflection of its end, subject to its stress and buckling limits, with NSGA-II.
reference: "Deb, K., Pratap, A. and Moitra, S. (2000). Mechanical component design for multiple objectives using elitist non-dominated sorting GA. Parallel Problem Solving from Nature (PPSN VI), LNCS 1917: 859-868."
reference_url: https://doi.org/10.1007/3-540-45356-3_84
optimum: "not known in closed form; from a cost of 2.3811341 at a deflection of 0.0157592 to a deflection of 0.00043904 at a cost of 36.421245 (best known); genoxide's reference front has a hypervolume of 1.1434 in scaled objectives (reference point (1.1, 1.1))"
languages: [rust, python]
order: 221
---

# Welded beam, two objectives

## The problem

A bar is welded to a support and carries 6000 lb at 14 in from it. The weld's thickness h and length
l and the bar's height t and thickness b, in inches, should make the beam cheap and its end stiff:

```text
minimize   f₁ = 1.10471 h² l + 0.04811 t b (14 + l)       the cost
           f₂ = δ = 2.1952 / (t³ b)                        the end deflection, in in
subject to τ ≤ 13,600 psi                                  the weld's shear stress
           σ = 504,000 / (t² b) ≤ 30,000 psi               the bar's bending stress
           h ≤ b
           P_c = 64,746.022 (1 − 0.0282346 t) t b³ ≥ 6000  the buckling load
h, b in [0.125, 5], l, t in [0.1, 10]
```

with τ the combination of the weld's primary and torsional shear stresses of Ragsdell and Phillips's
form. The definition is Deb, Pratap and Moitra's (2000, eq. 2), checked in the authors' preprint,
KanGAL report 200002: the single-objective `WeldedBeamRagsdell`'s constraints, whose code genoxide
shares, without its deflection limit, which becomes the second objective. Tanabe and Ishibuchi's
restatement (2020, problem CRE2-4-2) takes the shear stress and buckling load of the other form in
the literature, genoxide's `WeldedBeam`: a different front.

The front isn't known in closed form. Its ends, found with genoxide's SHADE, are the cheapest
design, the single-objective problem's (whose deflection limit isn't active there), at a cost of
2.3811341 and a deflection of 0.0157592, and the stiffest, t = 10 and b = 5, with a deflection of
2.1952 / 5000 = 0.00043904, whose cheapest design costs 36.421245. They give the ideal and nadir
points of genoxide's `WeldedBeam` in `multi::problems::engineering`.

genoxide's reference front, the non-dominated designs of about 6,000 ε-constraint problems (the
least cost at a deflection of at most ε, and the least deflection at a cost of at most ε, for ε
evenly spread between the ends), each solved by SHADE with 30,000 or 60,000 evaluations, and of
eight NSGA-II runs of 2,000 generations, has a hypervolume of 1.1434 in the scaled objectives below:
a lower bound on the whole front's.

## What makes it hard

The cheapest designs lie on the constraints' boundaries: all four constraints, the shear stress, the
bending stress, h ≤ b and the buckling load, are active at the cheapest one. The front is a steep
hyperbola: the deflection falls from 0.0158 to 0.0040 while the cost doubles, and only from 0.00083
to 0.00044 while it goes from 20 to 36.4. NSGA-II finds the stiff, expensive part quickly; getting
the cheap end right, on the constraints, takes long.

## Representation

A `Real` genome of 4 genes: h, l, t and b. The problem is genoxide's
`multi::problems::engineering::WeldedBeam` (`gx.problems.multi_engineering.WeldedBeam` in Python),
whose fitness is the two objectives and the total constraint violation. Solutions compare by
constrained dominance, the rule of the NSGA-II paper.

## Algorithm

NSGA-II with a population of 100 for 2,000 generations, simulated binary crossover (η = 20, at a
rate of 0.9) and polynomial mutation (η = 20, at a rate of 1/2 per gene). The paper ran 100
generations.

## Output

The first line gives the size of the final front and how many of its solutions are feasible, the
second the range of each objective on it. The third gives its hypervolume up to the reference point
(1.1, 1.1), in objectives scaled to [0, 1] by the ideal point (2.3811341, 0.00043904) and the nadir
point (36.421245, 0.0157592), as a share of the reference front's. In Python, `run` evaluates the
problem in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/welded-beam-2obj) plays this run
back.

## Good results

A good front is feasible and spread from a cost near 2.38 to a deflection near 0.000439. 100 points
of the reference front, chosen one by one for the most hypervolume, give 99.85% of its hypervolume.

The run's front has 100 feasible solutions, from a cost of 2.4371 to a deflection of 0.000439 at
36.4326, with 99.64% of the reference front's hypervolume. Over seeds 1 to 20, every run ends
between 99.58% and 99.67%. The cheap end is the last to come: with 250 generations, the runs end
between 97.7% and 99.6%, and the paper's 100 are shorter still. Deb, Pratap and Moitra's NSGA-II
front reached a cost of 2.79.
