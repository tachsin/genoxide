---
title: Viennet 2
category: multi-objective
summary: Minimize three convex quadratic objectives of two variables, whose Pareto front is a small curved triangle far from most of the search space, with NSGA-III and SMS-EMOA.
reference: "Viennet, R., Fonteix, C. and Marc, I. (1996). Multicriteria optimization using a genetic algorithm for determining a Pareto set. International Journal of Systems Science 27(2): 255-260."
reference_url: https://doi.org/10.1080/00207729608929211
optimum: "not known in closed form; hypervolume about 0.7744 (reference point (4.3697, −16.4242, −11.9584))"
languages: [rust, python]
order: 163
family: Viennet
---

# Viennet 2

## The problem

Viennet, Fonteix and Marc (1996) posed three problems with two variables and three objectives. This
is the second, VNT2. genoxide's `Viennet2` uses the definition and bounds that Van Veldhuizen
restates (1999, PhD thesis AFIT/DS/ENG/99-01, Air Force Institute of Technology, table B.1; its
table 5.3 has wider bounds); genoxide hasn't yet checked them against the original paper.

Over x₁ and x₂ in [−4, 4], all three objectives are minimized:

```text
f₁ = (x₁ − 2)²/2 + (x₂ + 1)²/13 + 3
f₂ = (x₁ + x₂ − 3)²/36 + (−x₁ + x₂ + 2)²/8 − 17
f₃ = (x₁ + 2x₂ − 1)²/175 + (2x₂ − x₁)²/17 − 13
```

Each objective is a convex quadratic, a bowl whose contours are ellipses. The three bowls are
stretched in different directions, and their minima are at different points:

| objective | minimum | at | the other two there |
|---|---|---|---|
| f₁ | 3 | (2, −1) | f₂ = −16.7639, f₃ = −12.0531 |
| f₂ | −17 | (2.5, 0.5) | f₁ = 3.2981, f₃ = −12.8319 |
| f₃ | −13 | (0.5, 0.25) | f₁ = 4.2452, f₂ = −16.4766 |

Because the objectives are convex, every optimal solution minimizes a weighted sum
w₁f₁ + w₂f₂ + w₃f₃ for some weights w ≥ 0, and every such minimum is optimal. For each choice of
weights, the minimum solves a 2 × 2 linear system. The optimal solutions form a curved triangle with
the three minima as corners, and its edges are the trade-offs between two objectives at a time. The
Pareto front is its image, a curved triangle in the space of the objectives with the corners in the
table's rows. genoxide gives no closed form for it: `optimal_front` is `None`.

The ideal point, the best value of each objective on the front, is (3, −17, −13). The nadir point,
the worst, is (883/208, −2109/128, −35858/2975) ≈ (4.2452, −16.4766, −12.0531): f₁ and f₂ are
worst at f₃'s minimum, and f₃ at f₁'s. genoxide's `ideal_point` and `nadir_point` give both, and
the example computes its reference point from them.

## What makes it hard

The front is a surface in three objectives, as for VNT1, and it is small. Across the box, f₁ reaches
22.9, f₂ −4.25 and f₃ −4.39. On the front, f₁ varies by 1.25, f₂ by 0.52 and f₃ by 0.95: 4% to 11%
of their ranges over the box. The initial population's non-dominated solutions, only 6 in the
NSGA-III run, are far from it.

The optimal solutions fill a thin, curved triangle of area about 0.35 in a box of area 64: less than
1% of the box. Its edges bend: the edge between f₁'s and f₃'s minima, (2, −1) and (0.5, 0.25),
rises to x₂ = 0.63 on the way. A point just off this thin region isn't optimal, but only points of
an even thinner region beat it in all three objectives. If the population holds none of them, it
stays on the population's front.

## Representation

A `Real` genome of 2 genes in [−4, 4]: the point x. The problem is genoxide's `Viennet2`, whose
fitness is the three objectives. In Python, `run` evaluates it in Rust.

## Algorithm

Two algorithms, each with a population of 92 for 50 generations, 4,692 evaluations, with simulated
binary crossover with η = 30, as Deb and Jain use, and polynomial mutation with η = 20 at a rate of
0.5 per gene, one of the two genes per child on average.

- NSGA-III (Deb and Jain, 2014, IEEE Transactions on Evolutionary Computation 18(4): 577-601), with
  the 91 reference directions of Das and Dennis's method (1998, SIAM Journal on Optimization 8(3):
  631-657) with 12 divisions, as Deb and Jain use for 3 objectives, and a crossover rate of 1. Like
  NSGA-II, it ranks solutions into non-dominated fronts, and parents and children compete for the
  next population. Instead of crowding distance, it spreads the front along reference directions.
  It first normalizes the objectives by the best values and the extreme points it has found, which
  matters here: the objectives' ranges on the front differ from each other, and from their ranges
  over the box, by large factors. Each solution joins the direction nearest to it, and directions
  with few members get more. genoxide's docs recommend it for three or more objectives.
- SMS-EMOA (Beume, Naujoks and Emmerich, 2007, European Journal of Operational Research 181(3):
  1653-1669), in genoxide's generational form: 92 children a generation, and genoxide's default
  crossover rate of 0.9. From the last front that fits only in part, it removes the solutions that
  add the least hypervolume, one at a time.

SMS-EMOA is there for the target, a hypervolume within 1% of the whole front's (see Good results). A
solution just off the thin region of optimal solutions adds little hypervolume, since the optimal
ones next to it cover most of what it dominates; SMS-EMOA drops such solutions first, while NSGA-III
keeps whichever solution is nearest to each direction.

NSGA-III's front needs about 7 generations: then all 92 solutions are non-dominated, and the
hypervolume is 0.765. After that, it stays between 0.763 and 0.767; a run of 400 generations ends at
0.765. SMS-EMOA's hypervolume reaches 0.768 at generation 6 and keeps rising slowly, to 0.7700 at
generation 10 and 0.7716 at 50.

## Output

A line per algorithm: how many solutions are on its final front, its hypervolume, and the
hypervolume as a share of the whole front's. The last line gives the whole front's hypervolume.

The hypervolume is the volume that the front dominates, up to a reference point. Larger is better.
The reference point here is (4.3697, −16.4242, −11.9584): the nadir point plus a tenth of each
objective's range from the ideal point, so that the extreme solutions count too, rounded to 4
decimals.

The whole front has no closed form, and its hypervolume is about 0.7744: a 4,001 × 4,001 grid of the
variables gives 0.77436, and the minima of 80,601 weighted sums, from Das and Dennis's weights with
400 divisions, give 0.77433. A reference front from a much longer run agrees: NSGA-III with 1,891
reference directions (60 divisions) and as many solutions, for 1,000 generations, ends with a front
of 1,891 solutions and a hypervolume of 0.7732, 99.8% of 0.7744. In Python, `run` evaluates the
problem in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/viennet2) plays the SMS-EMOA run
back.

## Good results

The target is a hypervolume of at least 99% of the whole front's, 0.7666. No finite set of solutions
reaches 0.7744. For comparison, the minima of 91 weighted sums, with Das and Dennis's weights with
12 divisions, have a hypervolume of 0.7686, 99.3%. They are optimal but not spread evenly on the
front.

NSGA-III's front has 92 solutions and a hypervolume of 0.7657, 98.9% of the whole front's: just
short of the target. Checked against 45,451 optimal solutions, the minima of weighted sums with 300
divisions, 45 of its solutions are optimal: none of those beats them. The other 47 lie next to the
thin region of optimal solutions, at most 0.14 from it: no other solution of the population beats
them.

SMS-EMOA's front has 92 solutions and a hypervolume of 0.7716, 99.6% of the whole front's: within
the target, and above the 91 weighted-sum minima. 76 of its solutions are optimal, and the other 16
lie at most 0.038 from the region.

Over seeds 1 to 20, SMS-EMOA reaches the target in every run, with hypervolumes from 0.7715 to
0.7717, 99.6% to 99.7% of the whole front's. NSGA-III reaches it in 1 run, with hypervolumes from
0.7629 to 0.7670, 98.5% to 99.0%.
