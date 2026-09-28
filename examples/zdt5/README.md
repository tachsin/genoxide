---
title: ZDT5
category: multi-objective
summary: Minimize two conflicting objectives over a string of 80 bits, whose Pareto front is 31 points behind deceptive fronts, with NSGA-II.
reference: "Zitzler, E., Deb, K. and Thiele, L. (2000). Comparison of multiobjective evolutionary algorithms: empirical results. Evolutionary Computation 8(2): 173-195."
reference_url: https://doi.org/10.1162/106365600568202
optimum: "31 points f₂ = 10 / f₁ for f₁ = 1, 2, …, 31; hypervolume 323.15 (reference point (34.1, 11))"
languages: [rust, python]
order: 114
family: ZDT
---

# ZDT5

## The problem

Zitzler, Deb and Thiele (2000) built six test problems with two objectives from one scheme: f₁
depends on the first part of the genome, a function g on the rest, and f₂ on both. ZDT5, their
eq. 11 (p. 178), is the only one on bit strings. The genome has 80 bits, in 11 substrings: x₁ of 30
bits, then x₂ to x₁₁ of 5 bits each. With u(xᵢ) the number of ones in a substring, both objectives
are minimized:

```text
f₁ = 1 + u(x₁)
g  = v(u(x₂)) + … + v(u(x₁₁)),   v(u) = 2 + u for u < 5,  v(5) = 1
f₂ = g / f₁
```

f₁ runs from 1 to 31. Each substring after x₁ adds 1 to g if it's all ones, and 2 to 6 otherwise: 2
with no ones, 6 with four. So g is 10, its smallest, when all ten substrings are all ones. The
Pareto front is those solutions: f₂ = 10 / f₁ for the 31 values of f₁, 31 separate points from
(1, 10) to (31, 0.3226). With x₁ holding 9 ones and the rest all ones, the solution is on the
front at (10, 1).

## What makes it hard

Each 5-bit substring is deceptive. From any substring that isn't all ones, removing a one lowers v
by 1, and adding one raises it by 1, until the fifth. So every small change leads to all zeros, v =
2, the substring's attractor. All ones, v = 1, is better, but it's isolated: from all zeros it takes
five flips at once, and each flip on the way is worse.

A solution with k substrings all ones and the others all zeros has g = 2 (10 − k) + k = 20 − k. For
each k there is a front, f₂ = (20 − k) / f₁: the deceptive fronts. The nearest, with one substring
all zeros, has g = 11, and the farthest, every substring at its attractor, g = 20. Zitzler, Deb and
Thiele plot their results against that last one (p. 186). genoxide's docs note that bit-flip
mutation and crossover are drawn to the attractors, and a run usually ends on a deceptive front.

So the search can't build an all-ones substring. It can only keep the ones it has. A random
string has each substring all ones with probability 1/32. That is about 3 per substring position
in a population of 100, and about 31 in a population of 1000. They have to be kept, and combined in
one genome, while selection drives the other substrings to their attractors. A position whose
all-ones substrings all die out is lost for good.

## Representation

A `Binary` genome of 80 bits: x₁ and the ten substrings, in order. The problem is genoxide's
`Zdt5`, whose fitness is the pair (f₁, f₂). In Python, `run` evaluates it in Rust, so both versions
print the same.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation
6(2): 182-197). It sorts solutions into non-dominated fronts, and within a front prefers solutions
with a larger crowding distance, a measure of the gap between their neighbors. Parents and
children compete for the next population, and a child that copies a member of the population is
dropped and bred again.

Bit-flip mutation flips each bit with probability 1/80, one bit per child on average. Crossover, at
genoxide's default rate of 0.9, is the choice that matters. The example runs NSGA-II three times,
each for 250 generations:

- Uniform crossover and a population of 100. Uniform crossover takes each bit from either parent.
  A substring that is all ones in one parent and all zeros in the other stays whole in a child with
  probability 1/32, so it breaks the substrings that the search can't rebuild.
- Two-point crossover and a population of 100. Two-point crossover exchanges one segment of the
  string between two random cuts. The substrings are contiguous, so it cuts at most two of them,
  and passes the others whole. It combines all-ones substrings from two parents.
- Two-point crossover and a population of 1000. The larger population starts with more all-ones
  substrings, and keeps them long enough to combine them.

## Output

A line per run, and one for the whole front. g is f₁ f₂, the same for the whole front of a run when
it has settled; 10 is the optimal front. Then how many of the 31 optimal points the front has, its
IGD+ and its hypervolume.

IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) is measured to the 31 points of the
optimal front. It averages, over those points, the distance to the nearest point of the found
front, counting only the objectives in which the found point is worse. 0 means that the found front
covers the optimal one. Smaller is better.

The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4):
257-271) is the area that the front dominates, up to the reference point (34.1, 11), 1.1 times the
nadir point (31, 10). Larger is better. The front is 31 points, so a run can reach its hypervolume,
323.15, exactly.

A front has many genomes on each point: x₁ holds its ones anywhere. The plot shows each point once.

[The project page](https://tachsin.gr/projects/genoxide/examples/zdt5) plays this run back.

## Good results

A good front is all 31 points, with g = 10, an IGD+ of 0 and a hypervolume of 323.15. Only the
large population with two-point crossover gets there in this run.

With uniform crossover and 100 individuals, the front ends at g = 18: two substrings all ones, the
other eight at their attractors. The initial population has all-ones substrings in eight of the ten
positions, and loses all but two by generation 31. The front settles at g = 18 from generation 109,
with an IGD+ of 0.6717 and a hypervolume of 297.33.

With two-point crossover and 100 individuals, it ends at g = 15, with five substrings all ones.
Five positions have none left after generation 10. From generation 17, one genome carries all five
of the others, and the front settles at g = 15 from generation 38, with an IGD+ of 0.4491 and a
hypervolume of 306.68.

With two-point crossover and 1000 individuals, every position keeps its all-ones substrings. At
generation 21, a genome has all ten, and the front first reaches g = 10. By generation 34, the
whole front is at g = 10, with 22 of the points. The last points, near the ends of the front, need
x₁ nearly all zeros or all ones, and mutation fills them in one at a time: the front has all 31
from generation 145.

Neither run with 100 individuals changes after it settles. Run for 2,500 generations, with the
250,100 evaluations of the large population, they still end at g = 18 and g = 15. Once every
genome has a substring at its attractor, only five flips at once make it all ones.

Over seeds 1 to 20, uniform crossover ends at g = 17 to 20 with 100 individuals, and at g = 19 or
20 with 1000: more individuals don't help when crossover breaks the substrings. Two-point crossover
with 100 individuals ends at g = 12 to 17, one-point crossover at 12 to 17. With 1000, two-point
crossover reaches g = 10 in all 20 runs, with 30 or 31 points, and one-point crossover in 18. With
300 individuals, two-point crossover reaches g = 10 in 3 runs, and with 500 in 15.
