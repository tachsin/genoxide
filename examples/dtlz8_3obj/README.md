---
title: DTLZ8 with 3 objectives
category: multi-objective
summary: Minimize three objectives over 30 variables subject to 3 constraints, whose Pareto front is a line and a triangle, with NSGA-II and SMS-EMOA.
reference: "Deb, K., Thiele, L., Laumanns, M. and Zitzler, E. (2001). Scalable Test Problems for Evolutionary Multi-Objective Optimization. TIK-Report 112, Computer Engineering and Networks Laboratory, ETH Zürich."
reference_url: https://sop.tik.ee.ethz.ch/publicationListFiles/dtlz2001a.pdf
optimum: "the line f₁ = f₂ = t, f₃ = 1 − 4t, t in [0, 1/6], and the triangle 2f₃ + f₁ + f₂ = 1 with f₁, f₂ ≥ (1 − f₃)/4; ideal point (0, 0, 0), nadir point (3/4, 3/4, 1); hypervolume 0.9724 (normalized objectives, reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 179
family: DTLZ
tab: DTLZ8
---

# DTLZ8 with 3 objectives

## The problem

Deb, Thiele, Laumanns and Zitzler's report (2001) builds its scalable test problems three ways; the last, the constraint surface approach, gives DTLZ8 and DTLZ9: each objective is a function of its own block of variables, so that the objectives can take values independently, and constraints on the objectives cut the front out of their space. (The report's DTLZ8 is the 2002 conference paper's DTLZ7, and its DTLZ9 isn't in the paper; genoxide follows the report.)

DTLZ8 (section 8.8, eq. 28), with 3 objectives and 30 variables in [0, 1]:

```text
minimize   fⱼ = (1/10) Σ xᵢ over the j-th block of 10 variables,  j = 1, 2, 3
subject to f₃ + 4f₁ − 1 ≥ 0
           f₃ + 4f₂ − 1 ≥ 0
           2f₃ + f₁ + f₂ − 1 ≥ 0
```

The report prints the blocks from ⌊(j − 1) n/M⌋ to ⌊j n/M⌋, which with variables counted from 1 would start at a variable x₀ and let neighbouring blocks share one: genoxide reads them as ⌊(j − 1) n/M⌋ + 1 to ⌊j n/M⌋, blocks of 10. (With more objectives, the last constraint takes the least sum of two of the first M − 1 objectives.)

The report's front is "a combination of a straight line and a hyper-plane": the line where the first two constraints meet, f₁ = f₂ = t and f₃ = 1 − 4t for t in [0, 1/6], from (0, 0, 1) to (1/6, 1/6, 1/3), and from its end the part of the plane 2f₃ + f₁ + f₂ = 1 that the first two constraints allow, the triangle from (1/6, 1/6, 1/3) to (1/4, 3/4, 0) and (3/4, 1/4, 0). Its ideal point is (0, 0, 0) and its nadir point (3/4, 3/4, 1). The report describes the front but doesn't write it out; genoxide derives it from the constraints, and its tests check it against random feasible solutions.

## What makes it hard

The line and the surfaces next to it. A solution beside the line, with f₁ ≠ f₂ and f₃ just above 1 − 4 min(f₁, f₂), is only weakly dominated by the line's point: Pareto dominance keeps it, and a population fills the surfaces around the line rather than the line itself, what the report shows in its figures 28 and 29 and discusses in its figure 32. The triangle is easier: it's a plane, and every point of it is reached with its blocks at their means.

## Representation

A `Real` genome of 30 genes in [0, 1], the report's n = 10M. The problem is genoxide's `Dtlz8::<3>`, whose fitness is the three objectives and the total constraint violation, 0 when it is feasible; in Python, `gx.problems.Dtlz8()`. Solutions compare by constrained dominance: a feasible solution beats an infeasible one, of two infeasible ones the smaller violation wins, and of two feasible ones Pareto dominance decides.

## Algorithm

Two runs, with a population of 100, simulated binary crossover with η = 20 and polynomial mutation with η = 20 at a rate of 1/30 per gene:

- NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2): 182-197), crossover at a rate of 0.9, for 500 generations, as long as the report's runs (its figures 28 and 29);
- SMS-EMOA (Beume, Naujoks and Emmerich, 2007, European Journal of Operational Research 181(3): 1653-1669), which keeps the solutions that add the most hypervolume, with as many children a generation as the population, for 20,000 generations, 2,000,000 evaluations. A solution that only a point of the front weakly dominates adds almost no hypervolume next to it, and is the first to go.

## Output

A line per run: the size of its final front, how many of its solutions the problem finds feasible, their IGD+ and hypervolume, and the hypervolume as a share of that of a sample of the optimal front with at least as many points as the population. Then the hypervolumes of the whole front and of the sample.

The indicators use the objectives normalized by the front's ideal and nadir points, so that the front spans [0, 1] in each. IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over 2,000 points of the optimal front from genoxide's `optimal_front`, the distance to the nearest feasible solution of the found front, counting only the objectives in which the solution is worse. Smaller is better; a front of as many points as the population can't cover the 2,000 exactly, and the sample shows what it can. The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the volume the feasible solutions dominate up to the reference point (1.1, 1.1, 1.1). Larger is better; the whole front's is computed from 3,000 of its points, and the sample's is what a front of that many points reaches. In Python, `run` evaluates the problem in Rust, so both versions print the same.

The page plays the runs back side by side, each population as the problem scores it: its feasible non-dominated solutions, its infeasible ones (hollow, at most 100 a frame), and the optimal front, sampled.

[The project page](https://tachsin.gr/projects/genoxide/examples/dtlz8-3obj) plays these runs back.

## Good results

The target: every solution feasible, and 99% of the hypervolume of the sample of 109 points of the front, about what a front of 100 solutions spread like it has.

NSGA-II's run with seed 1 ends with 100 solutions, 100 feasible, IGD+ 0.0355, hypervolume 0.8892, 93.6% of the sample's; SMS-EMOA's with 100 solutions, 100 feasible, IGD+ 0.0254, hypervolume 0.9462, 99.5% of the sample's. Over seeds 1 to 20, NSGA-II ends with IGD+ from 0.0332 to 0.0432 and 88.6% to 95.3% of the sample's hypervolume, no run reaching the target, and SMS-EMOA with IGD+ from 0.0249 to 0.0281 and 99.5% to 100.7% of the sample's hypervolume, every run reaching the target. The IGD+ stays above the sample's own, about 0.015: the front has a line and a triangle, and the hypervolume rewards the triangle more.
