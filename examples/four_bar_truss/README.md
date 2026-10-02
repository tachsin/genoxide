---
title: Four-bar truss
category: multi-objective
summary: Minimize the volume of a truss of four bars and the displacement of its loaded joint, whose convex front has three pieces derived from the definition, with NSGA-II.
reference: "Stadler, W. and Dauer, J. (1993). Multicriteria optimization in engineering: a tutorial and survey. In Structural Optimization: Status and Promise, Progress in Astronautics and Aeronautics 150, AIAA: 209-249."
reference_url: https://doi.org/10.2514/5.9781600866234.0209.0249
optimum: "the front from (1400, 0.04) to (2200 + 600√2, (2√2 − 2)/300) ≈ (3048.53, 0.0027614), in three pieces with x₃ = √2; hypervolume 0.8891 in objectives scaled by the ideal and nadir points (reference point (1.1, 1.1))"
languages: [rust, python]
order: 240
---

# Four-bar truss

## The problem

A truss of four bars carries a load F = 10 kN at a joint. The bars' cross-sections x₁ … x₄, in cm²,
should give the truss the least volume and its loaded joint the least displacement:

```text
minimize   f₁ = L (2x₁ + √2 x₂ + √2 x₃ + x₄)                    the volume, in cm³
           f₂ = (FL/E) (2/x₁ + 2√2/x₂ − 2√2/x₃ + 2/x₄)          the displacement, in cm
F = 10 kN, L = 200 cm, E = 2·10⁵ kN/cm², σ = 10 kN/cm²
x₁, x₄ in [F/σ, 3F/σ] = [1, 3], x₂, x₃ in [√2 F/σ, 3F/σ] = [√2, 3]
```

The bounds keep each bar's stress under σ. Stadler and Dauer's survey (1993) and Cheng and Li's
paper (1999), the problem's sources, couldn't be read: the definition, constants and bounds are
those of Costa and Fernandes (2009, 8th World Congress on Structural and Multidisciplinary
Optimization, problem (4-truss)), who restate it after Stadler and Dauer with the displacement as a
constraint. Tanabe and Ishibuchi's restatement (2020, problem RE2-4-1) has √x₃ where the volume has
√2 x₃, which is dimensionally wrong and moves the front.

The optimal front, derived from the definition: x₃ adds to the volume and takes from the
displacement's third term, so both objectives are least at x₃ = √2. The rest is a convex problem,
whose optimal solutions minimize `f₁ + λ f₂` for some λ > 0: x₁ = √(λ/20000) and x₂ = x₄ =
√(λ/10000), each clamped to its bounds. That gives three pieces: x₁ = 1, x₂ = √2 and x₄ from 1 to √2
(volumes 1400 to 1482.84); then x₂ = x₄ = √2 x₁ from √2 to 3 (to 2697.06); then x₂ = x₄ = 3 and x₁
from 3/√2 to 3 (to 3048.53). genoxide's `FourBarTruss` gives these pieces as its optimal front.

## What makes it hard

Not much: there are no constraints besides the bounds, and the front is convex. The objectives'
scales differ by five orders of magnitude, and one gene, x₃, must sit on its lower bound in every
optimal solution; the front's kinks, where a gene reaches a bound, need solutions on both sides.

## Representation

A `Real` genome of 4 genes, the cross-sections. The problem is genoxide's
`multi::problems::engineering::FourBarTruss` (`gx.problems.multi_engineering.FourBarTruss` in
Python).

## Algorithm

NSGA-II with a population of 100 for 250 generations, simulated binary crossover (η = 20, at a rate
of 0.9) and polynomial mutation (η = 20, at a rate of 1/4 per gene).

## Output

The first line gives the size of the final front and how many of its solutions are feasible (all
are: the problem has no constraints), the second the range of each objective on it. The third gives
its IGD+ (Ishibuchi et al., 2015) to 2,000 points of the optimal front, and its hypervolume up to
the reference point (1.1, 1.1), as a share of the whole front's, from 100,000 of its points, 0.8891.
Both use objectives scaled to [0, 1] on the front by its ideal point (1400, 0.0027614) and nadir
point (3048.53, 0.04). In Python, `run` evaluates the problem in Rust, so both versions print the
same.

[The project page](https://tachsin.gr/projects/genoxide/examples/four-bar-truss) plays this run
back, over the optimal front.

## Good results

A good front covers all three pieces, from a volume of 1400 to 3048.53. 100 points of the optimal
front, spread evenly along it, give 99.48% of its hypervolume and an IGD+ of 0.0023.

The run's front has 100 solutions, 13 on the first piece, 72 on the second and 15 on the third, from
(1400.00, 0.040000) to (3048.52, 0.002762), none with x₃ above 1.417: an IGD+ of 0.0040 and 99.09%
of the whole front's hypervolume. Over seeds 1 to 20, every run ends the same way: IGD+ from 0.0038
to 0.0043, and 99.04% to 99.16% of the hypervolume.
