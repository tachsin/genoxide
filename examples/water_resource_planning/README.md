---
title: Water resource planning
category: multi-objective
summary: Minimize five costs and losses of a storm drainage system, subject to seven constraints, whose front, derived from the definition, is a surface in five dimensions, with SPEA2.
reference: "Musselman, K. and Talavage, J. (1980). A tradeoff cut approach to multiple objective optimization. Operations Research 28(6): 1424-1435."
reference_url: https://doi.org/10.1287/opre.28.6.1424
optimum: "the image of every design with x₃ = 0.01 and x₁x₂ ≥ 0.00139 / 1.0306; ideal point (63840.28, 40.46, 285346.9, 183749.97, 7.22), nadir point (73450.51, 1350, 2853469.0, 6575303.1, 25000)"
languages: [rust, python]
order: 244
---

# Water resource planning

## The problem

Water resource planning (WATER) designs a storm drainage system: the local detention storage
capacity x₁, the maximum treatment rate x₂ and the maximum allowable overflow rate x₃, under five
costs and losses to minimize and seven constraints:

```text
minimize   f₁ = 106780.37 (x₂ + x₃) + 61704.67              the drainage network's cost
           f₂ = 3000 x₁                                      the storage facility's
           f₃ = 305700 · 2289 x₂ / (0.06 · 2289)^0.65        the treatment facility's
           f₄ = 250 · 2289 exp(−39.75 x₂ + 9.9 x₃ + 2.74)    the expected flood damage
           f₅ = 25 (1.39 / (x₁x₂) + 4940 x₃ − 80)            the expected economic loss from floods
subject to 0.00139/(x₁x₂) + 4.94x₃ − 0.08       ≤ 1
           0.000306/(x₁x₂) + 1.082x₃ − 0.0986   ≤ 1
           12.307/(x₁x₂) + 49408.24x₃ + 4051.02 ≤ 50000
           2.098/(x₁x₂) + 8046.33x₃ − 696.71    ≤ 16000
           2.138/(x₁x₂) + 7883.39x₃ − 705.04    ≤ 10000
           0.417x₁x₂ + 1721.26x₃ − 136.54       ≤ 2000
           0.164/(x₁x₂) + 631.13x₃ − 54.48      ≤ 550
x₁ in [0.01, 0.45], x₂, x₃ in [0.01, 0.1]
```

Musselman and Talavage (1980) and Ray, Tai and Seow (2001) couldn't be read. The definition is the
NSGA-II paper's (Deb, Pratap, Agarwal and Meyarivan, 2002, table V), checked in the published paper
and its preprint, KanGAL report 200001 (table VI), and the same in Jain and Deb's appendix (2014,
part II). The sixth constraint is printed with a product x₁x₂ where the others divide by it, in all
three: as printed it holds everywhere in the bounds, and read as a quotient it would still hold
wherever the seventh does, so the reading doesn't change the problem.

The optimal front, derived from the definition: f₁, f₄ and f₅ grow with x₃, which the other two
don't depend on, and every constraint is easier to meet with a smaller x₃, so the optimal solutions
have x₃ = 0.01. There the constraints come down to a least x₁x₂, the first's 0.00139 / (1.08 −
0.0494) ≈ 0.0013487. No two such solutions dominate each other: f₂ grows with x₁ and f₁ with x₂, so
a solution that dominates another has neither a larger x₁ nor a larger x₂; f₄ falls as x₂ grows, so
it has the same x₂; and then f₅ falls as x₁ grows, so it has the same x₁. The optimal solutions are
thus every (x₁, x₂, 0.01) with x₁x₂ ≥ 0.0013487, and the front is their image, a surface in five
dimensions. genoxide's `WaterResourcePlanning` samples it on a grid over x₁ and x₂ and along its
edge, x₁x₂ = 0.0013487, where f₅ reaches its worst, 25,000.

## What makes it hard

Five objectives, whose scales differ by five orders of magnitude, and a front that is a surface bent
through them by an exponential and a reciprocal. Every solution with x₃ above its bound is dominated
only by the same design with x₃ = 0.01: unless the population holds that design, nothing removes it
from the front.

## Representation

A `Real` genome of 3 genes: x₁, x₂ and x₃. The problem is genoxide's
`multi::problems::engineering::WaterResourcePlanning`
(`gx.problems.multi_engineering.WaterResourcePlanning` in Python), whose fitness is the five
objectives and the total constraint violation. Solutions compare by constrained dominance.

## Algorithm

SPEA2 with a population of 212 for 500 generations, simulated binary crossover (η = 20, at a rate of
0.9) and polynomial mutation (η = 20, at a rate of 1/3 per gene). Its truncation spreads the front
evenly. NSGA-III with the 210 reference directions of Das and Dennis's method with 6 divisions, and
the same population, ends with an IGD+ of 0.038 to 0.046 over seeds 1 to 5, NSGA-II with 0.034 to
0.040.

## Output

The first line gives the size of the final front and how many of its solutions are feasible. The
second gives the largest x₃ among them: the optimal solutions have 0.01. The third gives the front's
IGD+ (Ishibuchi et al., 2015) to 5,067 points of the optimal front, in objectives scaled to [0, 1]
by the front's ideal and nadir points. In Python, `run` evaluates the problem in Rust, so both
versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/water-resource-planning) plays this
run back, in two panels of three objectives each.

## Good results

A good front is feasible, has x₃ at 0.01 or near it, and spreads over the whole surface. 218 points
of the optimal front, on the grid that samples it, have an IGD+ of 0.0231 to it.

The run's front has 212 feasible solutions, with an IGD+ of 0.0299. Some of its solutions still have
x₃ up to 0.0395: dominated by the optimal front, but not by the rest of the run's. Over seeds 1 to
20, SPEA2's IGD+ ranges from 0.0275 to 0.0313.
