---
title: Schwefel 2.26
category: continuous
summary: Minimize a deceptive 30-dimensional function whose global minimum lies near a corner of the box, far from the next best.
reference: "Schwefel, H.-P. (1981). Numerical Optimization of Computer Models. Wiley. Problem 2.26."
reference_url: ""
optimum: "−12569.4866 in 30 dimensions (every gene 420.9687)"
languages: [rust, python]
order: 52
family: Schwefel
trace_note: "Recorded from another run: L-SHADE in 2 dimensions, so that the population can be drawn on the function's contour."
---

# Schwefel 2.26

## The problem

Schwefel's problem 2.26 is a sum of one term per gene:

```text
f(x) = −Σ xᵢ sin √|xᵢ|,   each xᵢ in [−500, 500]
```

In Schwefel's book (1981, problem 2.26, the translation of the German edition of 1977), it has one
variable and no bounds, and so no finite minimum. The sum over n variables on [−500, 500] is
Mühlenbein, Schomisch and Born's (1991, F7), restated by Yao, Liu and Lin (1999, f8). genoxide
checked its definition against Schwefel's book (issue #168). This is the form without an offset:
some papers add 418.9829 n, which moves the minimum near 0.

Each term, −x sin √|x|, is minimized alone, so the minimum is where every gene minimizes its term:
at xᵢ = 420.9687, with −418.9829 per gene. Here n = 30, and the minimum is −12569.4866.

## What makes it hard

In one dimension, the term has seven local minima inside the box, and an eighth at the bound −500.
They alternate between the two sides of 0 and get deeper away from it:

| x | −302.52 | −124.83 | −25.88 | 5.24 | 65.55 | 203.81 | 420.97 |
|---|---|---|---|---|---|---|---|
| value | −300.54 | −122.88 | −24.08 | −3.95 | −63.64 | −201.84 | −418.98 |

The best is at 420.97, near the upper bound. The second best, −302.52, is on the other side of the
box, 723 away. In 30 dimensions there are 8³⁰ ≈ 1.2 × 10²⁷ local minima. The second best have 29
genes at 420.97 and one at −302.52: only 118.44 above the minimum, and 723 away from it.

The function is deceptive: its shape over large distances doesn't point to the minimum. Rastrigin's
function is a bowl with ripples, and a search that follows the bowl gets near its minimum. Here, the
ripples grow with |x| on both sides of 0. Averaged over windows 250 wide, about the width of a
ripple, the term stays between −60 and 60, with no trend towards 420. A search that follows the
average of its good points, rather than each gene's best basin, settles with many genes in the
wrong basins.

What helps is that the function is separable: each gene can be improved alone, whatever the others
are.

## Representation

A `Real` genome of 30 genes, each in [−500, 500]: the point x itself. The fitness is f(x), to
minimize. The function is genoxide's `problems::Schwefel2_26`, which brings its bounds and its
minimum. 30 is the dimension of Yao, Liu and Lin's comparison.

## Algorithm

Three algorithms, each with a budget of 10,000 evaluations per dimension, 300,000 in all, and a
target 1e-8 above the minimum: L-SHADE, which reaches it, and, for contrast, CMA-ES and a particle
swarm, which don't.

L-SHADE (Tanabe and Fukunaga, 2014, IEEE CEC 2014: 1658-1665) is a differential evolution. For
each parent, it makes a mutant point from the parent and the differences between other points. The
trial point takes each gene from the mutant with a probability CR, and the others from the parent.
L-SHADE adapts CR from the trials that succeeded. With a small CR, a trial changes a few genes and
keeps the rest: on a separable function, a gene can jump to a better basin without the others
losing theirs. Its population starts at 18 times the number of genes, 540, and shrinks linearly to
4 over the budget. genoxide's `De::l_shade` takes L-SHADE's settings, so the example only gives it
the budget.

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) samples a population
from a normal distribution, and adapts its mean, step size and covariance matrix. It uses genoxide's
defaults, with IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776): a run that has
converged starts again from a random point with twice the population. genoxide's docs recommend
IPOP for multimodal functions.

Particle swarm optimization (Kennedy and Eberhart, 1995, Proceedings of ICNN'95: 1942-1948) with 40
particles on a ring: each follows the best of itself and its two neighbors, so that good points
spread slowly and the swarm explores for longer. It uses Clerc and Kennedy's constriction
coefficients (2002, IEEE Transactions on Evolutionary Computation 6(1): 58-73).

## Output

The first line gives the minimum and the budget. Then one line per algorithm, L-SHADE's first and
then the two contrasts: the best value it found, to 2 decimals, its error (the best value minus the
minimum, to two significant digits), and how many of its 30 genes are within 1 of 420.97, in the
basin of the global minimum. In Python, `run` evaluates the function in Rust, so both versions
print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/schwefel-2-26) plays back another
run: L-SHADE on Schwefel 2.26 in 2 dimensions, so that the population can be drawn on the
function's contour.

## Good results

The minimum is −12569.49, with all 30 genes at 420.97. L-SHADE reaches it: its error is 8.6e-9,
below the target of 1e-8, after about 222,000 evaluations. CMA-ES ends at −8423.89, with 12 genes at
420.97, and PSO at −8573.03, with 11: both about 4,000 above the minimum.

The difference isn't the seed's. With seeds 1 to 20, L-SHADE reaches the minimum every time, after
at most 223,000 evaluations. With seeds 1 to 10, CMA-ES with IPOP restarts ends between −7694 and
−9727, and PSO on a ring between −8451 and −9326. In 10 dimensions it's the same: CMA-ES and PSO
stay far from the minimum, and a differential evolution comes within 1e-8 of it.

In runs not shown here, SHADE with F = 0.5 and a fixed CR, without restarts, with seeds 1 to 3,
reached the minimum every time with CR = 0.1, and ended 700 to 2,000 above it with CR = 0.9: a
small CR is what uses the separability.
