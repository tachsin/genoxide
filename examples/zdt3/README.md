---
title: ZDT3
category: multi-objective
summary: Minimize two conflicting objectives over 30 variables, with a Pareto front in five disconnected pieces, with NSGA-II.
reference: "Zitzler, E., Deb, K. and Thiele, L. (2000). Comparison of multiobjective evolutionary algorithms: empirical results. Evolutionary Computation 8(2): 173-195."
reference_url: https://doi.org/10.1162/106365600568202
optimum: "five pieces of f₂ = 1 − √f₁ − f₁ sin(10π f₁); hypervolume 1.3318 (reference point (1.1, 1.1))"
languages: [rust, python]
order: 82
---

# ZDT3

## The problem

Zitzler, Deb and Thiele (2000) built six test problems with two objectives from one scheme: f₁
depends on the first variable, a function g on the others, and f₂ on both. ZDT3 is the third. It
has 30 variables in [0, 1], and minimizes both objectives:

```text
f₁ = x₁
g  = 1 + 9 (x₂ + … + x₃₀) / 29
f₂ = g (1 − √(f₁ / g) − (f₁ / g) sin(10π f₁))
```

It is ZDT1 with a sine term added to f₂. The best solutions have g = 1, that is x₂ = … = x₃₀ = 0,
where f₂ = 1 − √f₁ − f₁ sin(10π f₁). That curve goes down and up again as f₁ grows: the sine
makes five waves. Where the curve rises, a solution is beaten by one with a smaller f₁ and a smaller
f₂. So only five pieces of it are optimal, the Pareto front:

| Piece | f₁ | f₂ |
|---|---|---|
| 1 | 0 to 0.0830 | 1 to 0.6697 |
| 2 | 0.1822 to 0.2578 | 0.6697 to 0.2422 |
| 3 | 0.4093 to 0.4539 | 0.2422 to −0.1242 |
| 4 | 0.6184 to 0.6525 | −0.1242 to −0.4583 |
| 5 | 0.8233 to 0.8518 | −0.4583 to −0.7734 |

Each piece ends at a local minimum of the curve. The next piece starts where the curve, after its
rise, comes back down to that value. f₂ is negative on much of the front: the problem doesn't
require the objectives to be positive.

## What makes it hard

A disconnected front. Zitzler, Deb and Thiele built ZDT3 to test how an algorithm handles
"discreteness": the front is in pieces, though the variables are not. Between the pieces, no
solution is optimal. An algorithm has to keep a separate group of solutions on each piece, or it
loses a piece, and the trade-offs it holds.

The pieces are unequal. The first is 0.083 wide in f₁, the last only 0.029, and the last holds the
smallest f₂. An algorithm that spreads its solutions by distance in the objectives has to share
them out among pieces of different lengths.

## Representation

A `Real` genome of 30 genes in [0, 1]: the vector x. The problem is genoxide's `Zdt3`, whose
fitness is the pair (f₁, f₂). In Python, `run` evaluates it in Rust, so both versions print the
same.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation
6(2): 182-197), as the ZDT1 example runs it. It ranks solutions by non-dominated sorting: the first
front is the solutions that no other solution beats in both objectives, the second front those
beaten only by the first, and so on. Within a front, it prefers solutions in less crowded regions
(crowding distance). Parents and children compete for the next population, so it keeps the best
solutions found so far.

Crowding distance suits a front in pieces. It measures the gap between a solution's two neighbors
on the front, and the solutions at the ends of each piece have a neighbor across a gap. So they
count as uncrowded, and the ends of every piece are kept.

- a population of 100, for 250 generations, as in the NSGA-II paper;
- simulated binary crossover with η = 15, at genoxide's default rate of 0.9;
- polynomial mutation with η = 20, at a rate of 1/30 per gene, one gene per child on average.

## Output

The first line gives the size of the final front. The second gives how many of its solutions are
on each piece: with an f₁ within 0.001 of the piece's range.

The third gives its IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 500 points of
the optimal front, spread over the pieces in proportion to their widths. IGD+ averages, over those
500 points, the distance to the nearest point of the found front, counting only the objectives in
which the found point is worse. 0 means that the found front covers the optimal one. Smaller is
better.

The fourth gives the front's hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on
Evolutionary Computation 3(4): 257-271): the area it dominates, up to the reference point
(1.1, 1.1). Larger is better. For the whole front, it is 1.3318, found numerically from 2 million
points of the front.

[The project page](https://tachsin.gr/projects/genoxide/examples/zdt3) plays this run back.

## Good results

A good front has 100 solutions on all five pieces, an IGD+ near 0 and a hypervolume near 1.3318.
No set of 100 points reaches that hypervolume: 100 points of the optimal front, spread over the
pieces in proportion to their widths, give 1.3291 and an IGD+ of 0.0011.

The run starts with 9 solutions on its front, 4 of them between the pieces. The front reaches all
five pieces by generation 12, and first has 100 solutions at generation 68. Until about generation
200, one to four of them are still just past the end of a piece: a solution there is beaten only
by one at the end of the piece, with a g as small as its own. The run ends with 100 solutions, 21,
26, 21, 18 and 14 on the five pieces, an IGD+ of 0.0020 and a hypervolume of 1.3274, 99.7% of the
whole front's.

The pieces get solutions roughly in proportion to their length, measured as crowding distance
measures it, with each objective scaled by its range on the front. By that measure, the second
piece, which falls the most in f₂, by 0.43, is the longest, and it gets the most solutions.

On seeds 1 to 5, NSGA-II ends between 1.3273 and 1.3279, with an IGD+ of 0.0020 to 0.0022. With the
same settings, SPEA2 ends between 1.3270 and 1.3277, and SMS-EMOA, which keeps the solutions that
add the most hypervolume, between 1.3287 and 1.3289, with an IGD+ of about 0.0014. MOEA/D, with a
weight vector per solution, does worse here: the weight vectors that point into a gap all have
their best solution at the end of a piece, so several of them hold the same solution. It ends with
79 to 85 solutions on its front and an IGD+ of about 0.0038.
