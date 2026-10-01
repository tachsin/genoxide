---
title: DAS-CMOP4
category: multi-objective
summary: Minimize two objectives over 30 variables subject to 11 constraints of adjustable difficulty, whose Pareto front is pieces of a concave curve, with NSGA-II.
reference: "Fan, Z., Li, W., Cai, X., Li, H., Wei, C., Zhang, Q., Deb, K. and Goodman, E. (2020). Difficulty adjustable and scalable constrained multiobjective test problem toolkit. Evolutionary Computation 28(3): 339-378."
reference_url: https://doi.org/10.1162/evco_a_00259
optimum: "for the difficulty triplet (0.5, 0.5, 0.5), six pieces of the curve f₂ = 1.5 − (f₁ − 0.5)²; ideal point (0.5, 0.5975), nadir point (1.45, 1.5); hypervolume 0.4476 (normalized objectives, reference point (1.1, 1.1))"
languages: [rust, python]
order: 190
family: DAS-CMOP
---

# DAS-CMOP4

## The problem

Fan et al. (2020) built a toolkit of constrained test problems whose difficulty is set by three numbers, a triplet (η, ζ, γ) in [0, 1]³, one for each kind of difficulty: η for diversity, ζ for feasibility and γ for convergence. Each kind comes from a type of constraint. Type I constraints cut the front into pieces, narrower as η grows. A type II constraint asks the distance from the unconstrained front to be in a band, narrower as ζ grows. Type III constraints are infeasible regions in the objective space that block the way to the front, larger as γ grows. The paper suggests nine problems built this way, DAS-CMOP1 to DAS-CMOP9, and runs each with sixteen triplets (its table 3).

DAS-CMOP4 (table 2) has two objectives over 30 variables in [0, 1]:

```text
f₁ = x₁ + g
f₂ = 1 − x₁² + g
g = 29 + Σⱼ₌₂³⁰ ((xⱼ − 0.5)² − cos(20π(xⱼ − 0.5)))
subject to
  sin(20πx₁) − b ≥ 0,           b = 2η − 1                  (type I)
  (e − g)(g − 0.5) ≥ 0,         e = 0.5 − ln ζ              (type II)
  ((f₁ − p_k) cos θ − (f₂ − q_k) sin θ)²/0.3
    + ((f₁ − p_k) sin θ + (f₂ − q_k) cos θ)²/1.2 ≥ r,  r = γ/2  (type III, k = 1, …, 9)
θ = −π/4, (p_k, q_k) = (0, 1.5), (1, 0.5), (0, 2.5), (1, 1.5), (2, 0.5), (0, 3.5), (1, 2.5), (2, 1.5), (3, 0.5)
```

Both objectives are minimized. At g = 0 the unconstrained front is concave. The distance function g is 0 with every xⱼ at 0.5, and has 11²⁹ − 1 local minima, the cosine's, as DTLZ1's. The example uses the triplet with which the paper's figure 6 plots DAS-CMOP4, (0.5, 0.5, 0.5). Then b = 0: sin(20πx₁) ≥ 0 keeps x₁ in [0, 0.05], [0.1, 0.15], …, half of it; g must be between 0.5 and 0.5 + ln 2 ≈ 1.193, which moves the front out along the diagonal by 0.5; and the nine ellipses, each 45° from the axes, have semi-axes √(0.3 r) ≈ 0.27 along the diagonal and √(1.2 r) ≈ 0.55 across it.

The front depends on the triplet, and the paper samples it rather than writing it down; genoxide samples it in the same way, as the first feasible point of each of 20,000 rays α(x₁) + g (1, 1) that the type I constraint allows, non-dominated. With this triplet it is six pieces of the shifted concave curve, f₁ from 0.5 to 0.55, 0.6 to 0.65, 1.1424 to 1.15, 1.2 to 1.25, 1.3 to 1.35 and 1.4 to 1.45: the type I constraint cuts x₁ into ten intervals of width 0.05, and the ellipse centered at (1, 1.5) takes out the middle ones. Its ideal point is (0.5, 0.5975) and its nadir point (1.45, 1.5). The authors published sampled fronts for the sixteen triplets with their code, which genoxide's agree with (the docs of `multi::problems::DasCmop1` say how they were compared).

## What makes it hard

The multimodal g first: its local minima put local fronts parallel to the true one, and the type II band, g between 0.5 and e, is itself a target among them: a population that reaches the band at a local minimum of g must leave the band to improve, through infeasible solutions. Then the type I constraint, which cuts the front into short pieces that a population must spread over, and the ellipses, which block the way to some of them: a population that converges behind one stays there, on the far side, with constraint dominance, which never accepts an infeasible step.

## Representation

A `Real` genome of 30 genes in [0, 1]. The problem is genoxide's `DasCmop4`, whose fitness is the two objectives and the total constraint violation, 0 when it is feasible; the triplet is a `Difficulty`, `Difficulty::standard(k)` for the paper's sixteen. In Python, `gx.problems.DasCmop4()`, with `difficulty=(η, ζ, γ)` or the number of one of the sixteen, which `run` evaluates in Rust, so both versions print the same.

Solutions compare by constrained dominance, the rule of the NSGA-II paper and the constraint handling that the paper calls CDP. A feasible solution beats an infeasible one. Of two infeasible ones, the smaller violation wins. Of two feasible ones, Pareto dominance decides.

## Algorithm

NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary Computation 6(2): 182-197) with the paper's settings: a population of 300 for 1,000 generations, 300,000 evaluations (section 7.1), simulated binary crossover with η = 20 at a rate of 0.9, and polynomial mutation at a rate of 1/30 per gene, twice:

- with η = 20 for the mutation, the paper's;
- with η = 5, whose steps are larger: the median step is about 3% of the range with η = 20 and about 11% with η = 5.

The larger steps of η = 5 keep the population spread over x₁ while it converges, so that it reaches every piece.

## Output

A line per run: the size of its final front, how many of its solutions are feasible, and the front's IGD+ and hypervolume. Then the hypervolume of the whole optimal front.

Both indicators use the objectives normalized by the front's ideal and nadir points, so that the front spans [0, 1] in each. IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) averages, over 500 points of the optimal front from genoxide's `optimal_front`, the distance to the nearest point of the found front, counting only the objectives in which the found point is worse. Smaller is better, and 0 means the found front covers the optimal one. The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4): 257-271) is the area the front dominates up to the reference point (1.1, 1.1). Larger is better; the whole front's is computed from 20,000 of its points. Only feasible solutions count.

The page plays both runs back over the grey feasible region, sampled from genomes with x₁ evenly spread and every distance variable at one value, swept; hollow points are infeasible solutions of the populations (at most 100 a frame), and the line is the optimal front, in its pieces.

[The project page](https://tachsin.gr/projects/genoxide/examples/das-cmop4) plays this run back.

## Good results

The target: an IGD+ of at most 0.01 in normalized objectives, with every solution feasible.

With the paper's settings, the run with seed 1 ends with 300 solutions, 300 feasible, IGD+ 0.0155, hypervolume 0.4201. With η = 5, 300 solutions, 300 feasible, IGD+ 0.0005, hypervolume 0.4459. Over seeds 1 to 20, with η = 20, 12 runs reach the target; the median IGD+ is 0.0006, the worst 0.2721; with η = 5, every run reaches the target, with IGD+ from 0.0002 to 0.0014.

The paper's NSGA-II-CDP ends with a mean IGD of 0.111 on DAS-CMOP4 with this triplet (its table 4, number 8), with a standard deviation of 0.157: some runs reach the front and some don't, as here with η = 20; MOEA/D-CDP has 0.204.
