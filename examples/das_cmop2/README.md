---
title: DAS-CMOP2
category: multi-objective
summary: Minimize two objectives over 30 variables subject to 11 constraints of adjustable difficulty, whose Pareto front is pieces of a convex curve and of ellipses, with MOEA/D-DE, which reaches it in every run of 20, and NSGA-II, which doesn't, as the paper's doesn't.
reference: "Fan, Z., Li, W., Cai, X., Li, H., Wei, C., Zhang, Q., Deb, K. and Goodman, E. (2020). Difficulty adjustable and scalable constrained multiobjective test problem toolkit. Evolutionary Computation 28(3): 339-378."
reference_url: https://doi.org/10.1162/evco_a_00259
optimum: "for the difficulty triplet (0, 0.5, 0.5), the curve f₂ = 1.5 − √(f₁ − 0.5) and a piece of an ellipse; ideal point (0.5, 0.5), nadir point (1.5, 1.5); hypervolume 0.8612 (normalized objectives, reference point (1.1, 1.1))"
languages: [rust, python]
order: 208
family: DAS-CMOP
---

# DAS-CMOP2

## The problem

Fan et al. (2020) built a toolkit of constrained test problems whose difficulty is set by three numbers, a triplet (η, ζ, γ) in [0, 1]³, one for each kind of difficulty: η for diversity, ζ for feasibility and γ for convergence. Each kind comes from a type of constraint. Type I constraints cut the front into pieces, narrower as η grows. A type II constraint asks the distance from the unconstrained front to be in a band, narrower as ζ grows. Type III constraints are infeasible regions in the objective space that block the way to the front, larger as γ grows. The paper suggests nine problems built this way, DAS-CMOP1 to DAS-CMOP9, and runs each with sixteen triplets (its table 3).

DAS-CMOP2 (table 2) has two objectives over 30 variables in [0, 1]:

```text
f₁ = x₁ + g
f₂ = 1 − √x₁ + g
g = Σⱼ₌₂³⁰ (xⱼ − sin(0.5πx₁))²
subject to
  sin(20πx₁) − b ≥ 0,           b = 2η − 1                  (type I)
  (e − g)(g − 0.5) ≥ 0,         e = 0.5 − ln ζ              (type II)
  ((f₁ − p_k) cos θ − (f₂ − q_k) sin θ)²/0.3
    + ((f₁ − p_k) sin θ + (f₂ − q_k) cos θ)²/1.2 ≥ r,  r = γ/2  (type III, k = 1, …, 9)
θ = −π/4, (p_k, q_k) = (0, 1.5), (1, 0.5), (0, 2.5), (1, 1.5), (2, 0.5), (0, 3.5), (1, 2.5), (2, 1.5), (3, 0.5)
```

Both objectives are minimized. At g = 0 the unconstrained front is convex. The distance function g is 0 where every xⱼ equals sin(0.5πx₁): the variables are linked, each other variable's best value depending on x₁. The example uses the triplet with which the paper's figure 6 plots DAS-CMOP2, (0, 0.5, 0.5). Then the type I constraint has no effect (b = −1); g must be between 0.5 and 0.5 + ln 2 ≈ 1.193, which moves the front out along the diagonal by 0.5; and the nine ellipses, each 45° from the axes, have semi-axes √(0.3 r) ≈ 0.27 along the diagonal and √(1.2 r) ≈ 0.55 across it.

The front depends on the triplet, and the paper samples it rather than writing it down; genoxide samples it in the same way, as the first feasible point of each of 20,000 rays α(x₁) + g (1, 1) that the type I constraint allows, non-dominated. With this triplet it is one piece from (0.5, 1.5) to (1.5, 0.5): the shifted curve, but for f₁ from 0.839 to 1.248, where the curve runs through the ellipse centered at (1, 0.5) and the front follows that ellipse's boundary, where the rays of the curve's points leave it. Its ideal point is (0.5, 0.5) and its nadir point (1.5, 1.5). The authors published sampled fronts for the sixteen triplets with their code, which genoxide's agree with (the docs of `multi::problems::DasCmop1` say how they were compared).

## What makes it hard

The linked variables first. A solution on the front at x₁ has every other variable at sin(0.5πx₁): to move along the front, all 29 of them have to move together, by the same amount. Simulated binary crossover and polynomial mutation change each variable on its own, so a population that has settled on one value of x₁ stays there: any other x₁ costs 29 (Δ sin)² in g. Differential evolution moves them together: its steps are differences between solutions of the population, along the front once the population lies near it. The paper's MOEA/D-CDP uses it, and so does MOEA/D-DE here. Then the constraints: the band of g between 0.5 and e is feasible, a thin shell above the front, and the ellipses block parts of it.

## Representation

A `Real` genome of 30 genes in [0, 1]. The problem is genoxide's `DasCmop2`, whose fitness is the two objectives and the total constraint violation, 0 when it is feasible; the triplet is a `Difficulty`, `Difficulty::standard(k)` for the paper's sixteen. In Python, `gx.problems.DasCmop2()`, with `difficulty=(η, ζ, γ)` or the number of one of the sixteen, which `run` evaluates in Rust, so both versions print the same.

Solutions compare by constrained dominance, the rule of the NSGA-II paper and the constraint handling that the paper calls CDP. A feasible solution beats an infeasible one. Of two infeasible ones, the smaller violation wins. Of two feasible ones, Pareto dominance decides.

## Algorithm

Two runs, each with the paper's population of 300 for 1,000 generations, 300,000 evaluations (section 7.1), and polynomial mutation with η = 20 at a rate of 1/30 per gene:

- MOEA/D with differential evolution, MOEA/D-DE (Li and Zhang, 2009, IEEE Transactions on Evolutionary Computation 13(2): 284-302): a subproblem for each of 300 weight vectors spread evenly on the line w₁ + w₂ = 1, each the weighted Tchebycheff distance to the ideal point, with the paper's 30 neighbours (0.1 N) and at most 2 replacements per child; a child is its subproblem's solution moved by F = 0.5 times the difference of two parents, every gene (CR = 1), the parents from the neighbourhood with probability δ = 0.2;
- NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2): 182-197) with the paper's settings, simulated binary crossover with η = 20 at a rate of 0.9, as the contrast.

Both compare solutions by constrained dominance. Differential evolution moves the linked variables together, which simulated binary crossover can't.

δ = 0.2, where the paper and Li and Zhang use 0.9, keeps the population spread. While no solution is feasible, constrained dominance ranks solutions by their violation alone, and that pulls the whole population to one stretch of x₁: with seed 1 and δ = 0.9, 80% of the population has x₁ between 0.31 and 0.43 after 10 generations. Parents from the whole population, most of the time, spread it again once it is feasible; parents from the neighbourhood, nine times in ten, mostly can't, as the neighbourhood's solutions are all at that x₁. Over seeds 1 to 20, δ = 0.9 reaches the target in 13 runs of 20 (IGD+ from 0.0013 to 0.1224 and 75.2% to 99.9% of the sample's hypervolume).

## Output

A line per run: the size of its final front, how many of its solutions are feasible, the front's IGD+ and hypervolume, and the hypervolume as a share of that of a sample of 300 points of the optimal front, as many as the population. Then the hypervolumes of the whole optimal front and of the sample.

Both indicators use the objectives normalized by the front's ideal and nadir points, so that the front spans [0, 1] in each. IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over 500 points of the optimal front from genoxide's `optimal_front`, the distance to the nearest point of the found front, counting only the objectives in which the found point is worse. Smaller is better, and 0 means the found front covers the optimal one. The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the area the front dominates up to the reference point (1.1, 1.1). Larger is better; the whole front's is computed from 20,000 of its points, and the sample's is what a front of 300 points reaches. Only feasible solutions count.

The page plays both runs back over the grey feasible region, sampled from genomes with x₁ evenly spread and every distance variable at one value, swept; hollow points are infeasible solutions of the populations (at most 100 a frame), and the line is the optimal front, in its pieces.

[The project page](https://tachsin.gr/projects/genoxide/examples/das-cmop2) plays this run back.

## Good results

The target: every solution feasible, and 99% of the hypervolume of the sample of 300 points of the front, about what a front of 300 solutions spread like it has.

MOEA/D-DE's run with seed 1 ends with 266 solutions, 266 feasible, IGD+ 0.0017, hypervolume 0.8577, 99.8% of the sample's; NSGA-II's with 253 solutions, 253 feasible, IGD+ 0.1432, hypervolume 0.6155, 71.6% of the sample's. Over seeds 1 to 20, MOEA/D-DE ends with IGD+ from 0.0016 to 0.0018 and 99.7% to 99.8% of the sample's hypervolume, every run reaching the target, and NSGA-II with IGD+ from 0.1276 to 0.1589 and 68.9% to 74.2% of the sample's hypervolume, no run reaching it: NSGA-II converges to a short stretch of the front, or of the band above it, and stays there, for the reasons above.

On DAS-CMOP2 the paper's NSGA-II-CDP ends with a mean IGD of 0.335 with the first triplet, (0.25, 0, 0), and 0.216 with (0.5, 0.5, 0.5) (table 4), where MOEA/D-CDP, with differential evolution, has 0.00135 with the first.
