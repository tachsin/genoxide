---
title: Two-bar truss
category: multi-objective
summary: Minimize the volume of a truss of two bars and the larger stress in them, under a stress limit, whose front has two pieces derived from the definition, with NSGA-II.
reference: "Deb, K., Pratap, A. and Moitra, S. (2000). Mechanical component design for multiple objectives using elitist non-dominated sorting GA. Parallel Problem Solving from Nature (PPSN VI), LNCS 1917: 859-868."
reference_url: https://doi.org/10.1007/3-540-45356-3_84
optimum: "the front f₁ f₂ = 400 for stresses from 10⁵ down to 4000√5, then x₂ = 0.01 and y from 2 to 3, down to 8000√10/3 ≈ 8432.74; hypervolume 1.0663 in objectives scaled by the ideal and nadir points (reference point (1.1, 1.1))"
languages: [rust, python]
order: 200
---

# Two-bar truss

## The problem

Two bars, AC and BC, hang from supports A and B and join at C, y metres below them, where a load of
100 kN hangs. A is 4 m and B 1 m to the side of C. The truss should use as little material as
possible and stress its bars as little as possible:

```text
minimize   f₁ = x₁ √(16 + y²) + x₂ √(1 + y²)           the volume, in m³
           f₂ = max(σ_AC, σ_BC)                         the larger stress, in kPa
           σ_AC = 20 √(16 + y²) / (y x₁),  σ_BC = 80 √(1 + y²) / (y x₂)
subject to max(σ_AC, σ_BC) ≤ 10⁵
x₁, x₂ in [0, 0.01] m² (the bars' sections), y in [1, 3] m
```

The definition is Deb, Pratap and Moitra's (2000, eq. 1), checked in the authors' preprint, KanGAL
report 200002; the bounds on the sections are from its text. The preprint calls x₁ and x₂ the bars'
lengths, but they are their cross-sections. Deb and Srinivasan (2006, KanGAL report 2005007) restate
the problem with the same bounds and derive its front.

The optimal front, derived from the definition (and by Deb and Srinivasan): on it both bars carry
the same stress S. For a given y and S the least volume is then `(400 + 100y²) / (yS)`, smallest at
y = 2, so the first piece of the front is the hyperbola f₁ f₂ = 400, from (0.004, 10⁵) down to S =
4000√5 ≈ 8944.27, where x₂ reaches its bound of 0.01. Less stress needs a deeper truss: x₂ stays at
0.01 and y grows from 2 to 3, with `f₂ = 8000 √(1 + y²) / y` and `f₁ = (4 + y²) / (80 √(1 + y²))`,
down to the least stress, 8000√10/3 ≈ 8432.74, at a volume of 0.0513870. genoxide's `TwoBarTruss`
gives both pieces as its optimal front.

## What makes it hard

The two objectives differ by seven orders of magnitude and the front is strongly curved: along the
hyperbola, the stress falls from 10⁵ to under 10⁴ while the volume grows tenfold. The second piece
is short but different in kind: its solutions have a bar at its bound and a depth other than 2. A
bar with no section, at the bound 0, has an infinite stress.

## Representation

A `Real` genome of 3 genes: x₁, x₂ and y. The problem is genoxide's
`multi::problems::engineering::TwoBarTruss` (`gx.problems.multi_engineering.TwoBarTruss` in Python),
whose fitness is the two objectives and the constraint violation.

## Algorithm

NSGA-II with a population of 100 for 250 generations, simulated binary crossover (η = 20, at a rate
of 0.9) and polynomial mutation (η = 20, at a rate of 1/3 per gene). The paper ran NSGA-II with a
population of 100 for 100 generations.

## Output

The first line gives the size of the final front and how many of its solutions are feasible, the
second the range of each objective on it. The third gives its IGD+ (Ishibuchi et al., 2015, EMO
2015, LNCS 9019: 110-125) to 2,000 points of the optimal front, and its hypervolume up to the
reference point (1.1, 1.1), as a share of the whole front's, from 100,000 of its points, 1.0663.
Both use objectives scaled to [0, 1] on the front by its ideal point (0.004, 8432.74) and nadir
point (0.0513870, 10⁵). In Python, `run` evaluates the problem in Rust, so both versions print the
same.

[The project page](https://tachsin.gr/projects/genoxide/examples/two-bar-truss) plays this run back,
over the optimal front.

## Good results

A good front is feasible and spread over both pieces, from a stress of 10⁵ down to 8432.74. 100
points of the optimal front, spread evenly along it, give 99.70% of its hypervolume and an IGD+ of
0.0015.

The run's front has 100 feasible solutions, 93 on the first piece and 7 on the second, from a volume
of 0.004012 at a stress of 99,962 to a stress of 8432.7: an IGD+ of 0.0034 and 99.35% of the whole
front's hypervolume. Over seeds 1 to 20, every run ends the same way: IGD+ from 0.0032 to 0.0037,
and 99.30% to 99.40% of the hypervolume. 100 generations, the paper's, are already as good: 99.27%
to 99.40%.

Deb, Pratap and Moitra report NSGA-II solutions from (0.00407, 99,755) to (0.05304, 8439); the last
is dominated by the optimal front's end, (0.0513870, 8432.74). Deb and Srinivasan tabulate three
solutions, printed to three or four digits, which evaluate to within 0.3% of their values;
genoxide's tests check them.
