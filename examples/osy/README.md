---
title: OSY (Osyczka and Kundu)
category: multi-objective
summary: Minimize two objectives of six variables under six constraints with NSGA-II; each of the five pieces of the optimal front lies on a different set of active constraints.
reference: "Osyczka, A. and Kundu, S. (1995). A new method to solve generalized multicriteria optimization problems using the simple genetic algorithm. Structural Optimization 10(2): 94-99."
reference_url: https://doi.org/10.1007/BF01743536
optimum: "a front in five pieces, from (−274, 76) to (−42, 4); hypervolume 16546.1 (reference point (−20, 85))"
languages: [rust, python]
order: 124
---

# OSY (Osyczka and Kundu)

## The problem

Osyczka and Kundu (1995) test a genetic algorithm for multicriteria optimization on this problem
with two objectives, six variables and six constraints:

```text
minimize   f₁ = −[25 (x₁ − 2)² + (x₂ − 2)² + (x₃ − 1)² + (x₄ − 4)² + (x₅ − 1)²]
           f₂ = x₁² + x₂² + x₃² + x₄² + x₅² + x₆²
subject to C1: x₁ + x₂ ≥ 2           C2: x₁ + x₂ ≤ 6
           C3: x₂ − x₁ ≤ 2           C4: x₁ − 3x₂ ≤ 2
           C5: (x₃ − 3)² + x₄ ≤ 4    C6: (x₅ − 3)² + x₆ ≥ 4
```

with x₁, x₂ and x₆ in [0, 10], x₃ and x₅ in [1, 5], and x₄ in [0, 6]. Minimizing f₁ means moving
far from the point (2, 2, 1, 4, 1), in a distance that weighs x₁ 25 times as much as the others.
Minimizing f₂ means staying close to the origin. The two pull apart.

genoxide restates the problem as Deb, Pratap and Meyarivan (2001, Constrained test problems for
multi-objective evolutionary optimization, EMO 2001, LNCS 1993: 284-298, eq. 3) do; it hasn't
checked it against the original paper yet.

The constraints split the variables into three groups:

- C1 to C4 keep (x₁, x₂) in a quadrilateral with the corners (0, 2), (2, 0), (5, 1) and (2, 4).
- C5 bounds x₄ by a parabola in x₃: x₄ ≤ 4 − (x₃ − 3)².
- C6 needs x₅ near an end of its range, or a positive x₆: with x₆ = 0, only x₅ = 1 or x₅ = 5
  satisfy it.

On the front, x₄ = x₆ = 0. x₄ = 0 is both farthest from 4 and nearest to 0, the best for both
objectives. x₆ only costs f₂; its only use is to let x₅ into the middle of its range, which never
pays. So x₅ is 1 or 5: 5 lowers f₁ by 16 and raises f₂ by 24. x₃ trades (x₃ − 1)² in f₁ against
x₃² in f₂. And (x₁, x₂) goes to the corner or edge of the quadrilateral that trades 25 (x₁ − 2)²
against x₁² + x₂² best.

The front, derived from the definition, has five pieces. Each lies on a different set of active
constraints:

| Piece | x₁ | x₂ | x₃ | x₅ | Active | f from | to |
|---|---|---|---|---|---|---|---|
| 1 | 5 | 1 | 5 to 1 | 5 | C2, C4, C6 | (−274, 76) | (−258, 52) |
| 2 | 5 | 1 | 5 to 1 | 1 | C2, C4, C6 | (−258, 52) | (−242, 28) |
| 3 | 5 to 4.0565 | (x₁ − 2)/3 | 1 | 1 | C4, C6 | (−242, 28) | (−123.46, 18.93) |
| 4 | 0 | 2 | 3.7317 to 1 | 1 | C1, C3, C6 | (−123.46, 18.93) | (−116, 6) |
| 5 | 0 to 1 | 2 − x₁ | 1 | 1 | C1, C6 | (−116, 6) | (−42, 4) |

The first three pieces are at x₁ = 5 or near it, as far right of x₁ = 2 as the quadrilateral
allows; the last two at x₁ = 1 or less, to its left. The third and fourth meet where both reach the
same objectives, (−123.46, 18.93). In the objective space, the pieces join end to end, and the
front is one connected line with four kinks; in the space of the variables, they are far apart.
Deb, Pratap and Meyarivan's table 1 lists the same five pieces.

## What makes it hard

The constraints. About 3% of the box is feasible: 6 of the run's first 100 random solutions are.
Six variables must be right at once: x₄ and x₆ at 0, x₅ at an end of its range, and (x₁, x₂) on a
corner or an edge of the quadrilateral.

Each piece lies on a different combination of active constraints, and the pieces are far apart in
the variables. Pieces 3 and 4 give the same objectives at the point where they meet, but one has
x₁ = 4.06 and the other x₁ = 0. Pieces 1 and 2 differ in x₅, 5 against 1, and with x₆ = 0 the
values between are infeasible. A population that finds one piece doesn't reach the others by small
steps.

The objectives span 232 and 72, and the front's slope changes at each kink: pieces 1, 2 and 4 are
steep, 3 and 5 nearly flat. The fifth runs over 74 units of f₁ and only 2 of f₂.

## Representation

A `Real` genome of 6 genes with the bounds above: the point x. The problem is genoxide's `Osy`,
whose fitness is the two objectives and the total constraint violation: the sum of how far x breaks
each constraint, 0 when it is feasible.

Solutions compare by constrained dominance, the rule of the NSGA-II paper (Deb, Pratap, Agarwal and
Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2): 182-197). A feasible solution
beats an infeasible one. Of two infeasible ones, the smaller violation wins. Of two feasible ones,
the usual Pareto dominance decides. Infeasible solutions thus lead the search towards the feasible
region.

## Algorithm

NSGA-II, which ranks solutions into non-dominated fronts and prefers the less crowded ones within a
front, with parents and children competing for the next population. The settings are those of the
NSGA-II paper:

- a population of 100, for 250 generations;
- simulated binary crossover with η = 20, at genoxide's default rate of 0.9;
- polynomial mutation with η = 20, at a rate of 1/n per gene for n genes: 1/6, one of the six
  genes per child on average.

## Output

The first line gives the size of the final front, and how many of its solutions are feasible.

The second counts the solutions along each of the five pieces of the optimal front, told apart by
f₁: the pieces meet at f₁ = −258, −242, −123.46 and −116.

The third gives the front's IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) to 500
points of the optimal front. IGD+ averages, over those 500 points, the distance to the nearest
point of the found front, counting only the objectives in which the found point is worse. 0 means
that the found front covers the optimal one. Smaller is better.

The fourth gives the front's hypervolume: the area it dominates, up to the reference point
(−20, 85), a little beyond the worst value of each objective on the front. Larger is better. For
the whole optimal front, computed from 2,000,000 of its points, it is 16546.1. In Python, `run`
evaluates the problem in Rust, so both versions print the same.

The run's `trace.json` has one field more than a front's trace: `feasible`, the share of the
population that is feasible in each recorded generation.

[The project page](https://tachsin.gr/projects/genoxide/examples/osy) plays this run back.

## Good results

A good front is all feasible, has solutions along all five pieces, and has an IGD+ near 0 and a
hypervolume near 16546.1. 100 points of the optimal front, spread evenly along it, give a
hypervolume of 16477.5 and an IGD+ of 0.17.

The run's population is all feasible by generation 3, but it finds the pieces one after another.
At generation 16, its front lies along pieces 3 to 5, from f₁ = −235 to −96. It reaches piece 2 at
generation 20 and piece 1 at generation 28. It comes within 2 of the end of the front, (−274, 76),
where x₃ and x₅ are both 5, only at generation 64. The hypervolume rises with each piece found, to
15934 at generation 48, 16195 at generation 64 and 16378 at generation 96, and slowly after that.

The final front has 100 solutions, 23, 24, 31, 12 and 10 along the five pieces, an IGD+ of 0.40 and
a hypervolume of 16427.8, 99.3% of the whole front's. It puts more solutions than the evenly
spread points (11, 11, 46, 4 and 28) on the steep pieces and fewer on the flat ones: NSGA-II
measures crowding with each objective scaled to its range, and pieces 1 and 2 each span a third of
f₂'s.
