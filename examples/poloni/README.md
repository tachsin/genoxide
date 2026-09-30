---
title: Poloni
category: multi-objective
summary: Minimize two objectives over two angles with NSGA-II, on a disconnected front that isn't known in closed form.
reference: "Poloni, C., Giurgevich, A., Onesti, L. and Pediroda, V. (2000). Hybridization of a multi-objective genetic algorithm, a neural network and a classical optimizer for a complex design problem in fluid dynamics. Computer Methods in Applied Mechanics and Engineering 186(2-4): 403-420."
reference_url: https://doi.org/10.1016/S0045-7825(99)00394-1
optimum: "not known in closed form; a fine grid gives hypervolume 444.57 (reference point (18.4, 27.5))"
languages: [rust, python]
order: 140
---

# Poloni

## The problem

Poloni, Giurgevich, Onesti and Pediroda (2000) use a problem with two variables, x₁ and x₂, each in
[−π, π], and two objectives, both minimized:

```text
f₁ = 1 + (A₁ − B₁)² + (A₂ − B₂)²
f₂ = (x₁ + 3)² + (x₂ + 1)²

A₁ = 0.5 sin 1 − 2 cos 1 + sin 2 − 1.5 cos 2
A₂ = 1.5 sin 1 − cos 1 + 2 sin 2 − 0.5 cos 2
B₁ = 0.5 sin x₁ − 2 cos x₁ + sin x₂ − 1.5 cos x₂
B₂ = 1.5 sin x₁ − cos x₁ + 2 sin x₂ − 0.5 cos x₂
```

B is a point that moves with x, and A is B at x = (1, 2), about (0.874, 2.749). f₁ is 1 plus the
squared distance from B to A: 1 at (1, 2). f₂ is the squared distance from x to (−3, −1): 0 there.
The two minima are far apart, so the objectives conflict. The best trade-offs, the Pareto front,
run from (1, 25), at x = (1, 2), to (16.77, 0), at x = (−3, −1).

The problem first appeared in Poloni et al. (1996, ECCOMAS '96, Wiley: 258-264) and Poloni (1997,
in Genetic Algorithms in Engineering and Computer Science, Wiley: 397-414), which, as Van Veldhuizen
(1999, PhD thesis, Air Force Institute of Technology, table B.1) notes, print it mistyped. The
definition and bounds here are as the NSGA-II paper (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE
Transactions on Evolutionary Computation 6(2): 182-197, table I) restates them, minimized. Van
Veldhuizen restates it as the maximization of the negated objectives, and Rigoni and Poles (2005,
Dagstuhl Seminar Proceedings 04461) minimize it with the same constants. genoxide hasn't yet
checked these restatements against the originals.

## What makes it hard

The front isn't known in closed form. There is no formula for the optimal solutions, so there is no
exact target, and genoxide's `optimal_front` gives none. On a fine grid over the box, the front has
two pieces:

- a short one, from (1, 25) to about (2.07, 20.88), whose solutions run from (1, 2) to about
  (0.77, 1.55);
- a long one, from about (2.07, 3.14) to (16.77, 0), whose solutions run along the bound
  x₁ = −π, from x₂ ≈ 0.76 down to about −0.93, then leave it for (−3, −1).

At f₁ ≈ 2.07, the best f₂ drops from 20.9 to 3.1: the solutions in between are all dominated. The
two pieces' solutions lie far apart, near (1, 2) and along the left edge of the box, so the
population has to keep two separate groups. The short piece is easy to lose: it has about a fifth
of the front's length in objective space, and all its solutions are close to (1, 2).

f₁ is multimodal. The sines and cosines make it rise and fall across the box: it goes from 1 to
about 61.6, and only 6% of the box has f₁ below 2. f₁ is 1 at a second point, near (2.02, 0.73),
where f₂ = 28.2: (1, 2) beats that solution. Much of the long piece lies on a bound, where the
search has to push genes to the edge of their range.

## Representation

A `Real` genome of 2 genes in [−π, π]: the vector x. The problem is genoxide's `Poloni`, whose
fitness is the pair (f₁, f₂). In Python, `run` evaluates it in Rust, so both versions print the
same.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002), with the settings of the paper that restates
this problem. It ranks solutions by non-dominated sorting: the first front is the solutions that no
other solution beats in both objectives, the second front those beaten only by the first, and so on.
Within a front, it prefers solutions in less crowded regions (crowding distance). Parents and
children compete for the next population, so it keeps the best solutions found so far, on both
pieces.

- a population of 100, for 250 generations;
- simulated binary crossover with η = 15, at genoxide's default rate of 0.9;
- polynomial mutation with η = 20, at a rate of 1/2 per gene, one gene per child on average.

## Output

The first line gives the size of the final front, and how many of its solutions are on each piece:
on the first, the short one, f₂ > 12; on the second, the long one, f₂ < 12.

The second gives the front's hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on
Evolutionary Computation 3(4): 257-271): the area it dominates, up to a reference point. Larger is
better. The reference point is (18.4, 27.5): the nadir point, the front's worst point (16.77, 25),
plus a tenth of the front's range from the ideal point, (1, 0), rounded up to a tenth. genoxide's
`ideal_point` and `nadir_point` give both. The front isn't known, so neither is its hypervolume.
For comparison, the example prints that of the non-dominated points of a grid of 4001 × 4001
points over the box, 444.57. It is a lower bound: the whole front dominates at least as much.
Finer grids give more, up to about 444.59. The front isn't known, so there is no IGD+ either.

[The project page](https://tachsin.gr/projects/genoxide/examples/poloni) plays this run back.

## Good results

A good front covers both pieces, with solutions spread over each, and a hypervolume near 444.59.
No set of 100 points reaches it. 100 points of the grid's front, spread along its two pieces by
their lengths in objective space, the way genoxide spreads the known disconnected fronts, give
444.12, with 22 of them on the short piece.

The run starts with 8 solutions on its front, and a hypervolume of 412.78. It has 100 solutions
after 8 generations, and from about generation 30 on its hypervolume stays near 444.1. It ends with
100 solutions, 22 on the short piece and 78 on the long one, and a hypervolume of 444.12: that of
the 100 grid points, and 99.9% of the grid's.

On seeds 1 to 5, NSGA-II ends between 444.07 and 444.12. SPEA2, with the same settings, ends
between 444.16 and 444.20, and SMS-EMOA, which keeps the solutions that add the most hypervolume,
at 444.27. The front's solutions come close to the bound x₁ = −π: 66 of them have x₁ below −3.13,
though none is exactly on it.
