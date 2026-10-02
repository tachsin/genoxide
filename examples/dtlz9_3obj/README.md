---
title: DTLZ9 with 3 objectives
category: multi-objective
summary: Minimize three objectives over 30 variables subject to 2 constraints, whose Pareto front is a curve in a thin region, with NSGA-II and SMS-EMOA.
reference: "Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2001). Scalable Test Problems for Evolutionary Multi-Objective Optimization. TIK-Report 112, Computer Engineering and Networks Laboratory, ETH Zürich."
reference_url: https://sop.tik.ee.ethz.ch/publicationListFiles/dtlz2001a.pdf
optimum: "the curve f₁ = f₂ = cos θ, f₃ = sin θ, θ in [0, π/2]; ideal point (0, 0, 0), nadir point (1, 1, 1); hypervolume 0.2697 (normalized objectives, reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 180
family: DTLZ
tab: DTLZ9
---

# DTLZ9 with 3 objectives

## The problem

Deb, Thiele, Laumanns and Zitzler's report (2001) builds its scalable test problems three ways; the last, the constraint surface approach, gives DTLZ8 and DTLZ9: each objective is a function of its own block of variables, so that the objectives can take values independently, and constraints on the objectives cut the front out of their space. (The report's DTLZ8 is the 2002 conference paper's DTLZ7, and its DTLZ9 isn't in the paper; genoxide follows the report.)

DTLZ9 (section 8.9, eq. 29), with 3 objectives and 30 variables in [0, 1]:

```text
minimize   fⱼ = Σ xᵢ^0.1 over the j-th block of 10 variables,  j = 1, 2, 3
subject to f₃² + f₁² − 1 ≥ 0
           f₃² + f₂² − 1 ≥ 0
```

The blocks are DTLZ8's (see its page). Unlike DTLZ8's, the objectives are sums, from 0 to 10.

The front is the curve f₁ = f₂ = cos θ, f₃ = sin θ, θ in [0, π/2], where both constraints meet: for a given f₃ in [0, 1], each constraint asks fⱼ ≥ √(1 − f₃²) alone. With f₃ and either other objective it's a quarter of the unit circle. Its ideal point is (0, 0, 0) and its nadir point (1, 1, 1). The report describes the front but doesn't write it out; genoxide derives it from the constraints, and its tests check it against random feasible solutions.

## What makes it hard

The power 0.1 first: on the front each block sums to at most 1 over 10 variables, so each variable is about (fⱼ/10)¹⁰, below 10⁻¹⁰, or exactly 0, and a random solution has its objectives near the top of their range: "the density of solutions gets thinner towards the Pareto-optimal region". Then the curve: as on DTLZ8's line, a solution with f₁ ≠ f₂ next to it is only weakly dominated, and a population spreads over the surfaces around the curve (the report's figures 30 and 31).

## Representation

A `Real` genome of 30 genes in [0, 1], the report's n = 10M. The problem is genoxide's `Dtlz9::<3>`, whose fitness is the three objectives and the total constraint violation, 0 when it is feasible; in Python, `gx.problems.Dtlz9()`. Solutions compare by constrained dominance: a feasible solution beats an infeasible one, of two infeasible ones the smaller violation wins, and of two feasible ones Pareto dominance decides.

## Algorithm

Two runs, with a population of 100, simulated binary crossover with η = 20 and polynomial mutation with η = 20 at a rate of 1/30 per gene:

- NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2): 182-197), crossover at a rate of 0.9, for 500 generations, as long as the report's runs (its figures 30 and 31);
- SMS-EMOA (Beume, Naujoks and Emmerich, 2007, European Journal of Operational Research 181(3): 1653-1669), which keeps the solutions that add the most hypervolume, with as many children a generation as the population, for 20,000 generations, 2,000,000 evaluations. A solution that only a point of the front weakly dominates adds almost no hypervolume next to it, and is the first to go.

## Output

A line per run: the size of its final front, how many of its solutions the problem finds feasible, their IGD+ and hypervolume, and the hypervolume as a share of that of a sample of the optimal front with at least as many points as the population. Then the hypervolumes of the whole front and of the sample.

The indicators use the objectives normalized by the front's ideal and nadir points, so that the front spans [0, 1] in each. IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over 2,000 points of the optimal front from genoxide's `optimal_front`, the distance to the nearest feasible solution of the found front, counting only the objectives in which the solution is worse. Smaller is better; a front of as many points as the population can't cover the 2,000 exactly, and the sample shows what it can. The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the volume the feasible solutions dominate up to the reference point (1.1, 1.1, 1.1). Larger is better; the whole front's is computed from 3,000 of its points, and the sample's is what a front of that many points reaches. In Python, `run` evaluates the problem in Rust, so both versions print the same.

The page plays the runs back side by side, each population as the problem scores it: its feasible non-dominated solutions, its infeasible ones (hollow, at most 100 a frame), and the optimal front, sampled.

[The project page](https://tachsin.gr/projects/genoxide/examples/dtlz9-3obj) plays these runs back.

## Good results

The target: every solution feasible, and 99% of the hypervolume of the sample of 100 points of the front, about what a front of 100 solutions spread like it has.

NSGA-II's run with seed 1 ends with 100 solutions, 100 feasible, IGD+ 3.4675, hypervolume 0.0000, 0.0% of the sample's; SMS-EMOA's with 100 solutions, 100 feasible, IGD+ 0.0024, hypervolume 0.2661, 99.9% of the sample's. Over seeds 1 to 20, NSGA-II ends with IGD+ from 3.3579 to 4.8193 and 0.0% to 0.0% of the sample's hypervolume, no run reaching the target, and SMS-EMOA with IGD+ from 0.0024 to 0.0026 and 99.8% to 99.9% of the sample's hypervolume, every run reaching the target. SMS-EMOA's IGD+ is the sample's own: its 100 solutions lie on the curve, spread along it. The report's NSGA-II and SPEA2, after 500 generations, "could not cover the entire range of the circle", with many solutions away from it.
