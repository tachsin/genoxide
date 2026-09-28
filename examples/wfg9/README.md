---
title: WFG9
category: multi-objective
summary: Minimize two objectives whose concave front lies behind 24 biased, deceptive, multimodal and non-separable parameters, with NSGA-II and MOEA/D.
reference: "Huband, S., Hingston, P., Barone, L. and While, L. (2006). A review of multiobjective test problems and a scalable test problem toolkit. IEEE Transactions on Evolutionary Computation 10(5): 477-506."
reference_url: https://doi.org/10.1109/TEVC.2005.861417
optimum: "the quarter ellipse (f₁/2)² + (f₂/4)² = 1; hypervolume 3.3968 (reference point (2.2, 4.4))"
languages: [rust, python]
order: 134
family: WFG
---

# WFG9

## The problem

Huband, Hingston, Barone and While (2006), of the Walking Fish Group, built a toolkit for test
problems with any number of objectives, and nine problems from it, WFG1 to WFG9. Each problem
passes its variables through a chain of transformations down to a few values, then turns those
into the objectives. WFG9 is the example of the paper's table XIII. genoxide's `Wfg9` follows
the paper's table XIV, with the transformations of its table XI and the shapes of its table X;
the optimal solutions are those of its section VIII-A. It was checked against the paper as
published, against its first version (Huband, Barone, While and Hingston, 2005, EMO 2005, LNCS
3410: 280-295, the authors' corrected version), and against the authors' C++ toolkit, version
2006.03.28, whose values genoxide's tests match.

This example uses 2 objectives and the recommended sizes: k = 4 position parameters and l = 20
distance parameters, 24 variables zᵢ in [0, 2i]. The problem first divides each by its upper
bound, yᵢ = zᵢ / 2i, then:

1. biases every parameter but the last by the mean of the parameters after it:
   yᵢ ← b_param(yᵢ, mean(yᵢ₊₁, …, y₂₄)) for i = 1, …, 23;
2. shifts the position parameters with a deceptive shift, yᵢ ← s_decept(yᵢ, 0.35, 0.001, 0.05)
   for i = 1, …, 4, and the distance parameters with a multimodal one,
   yᵢ ← s_multi(yᵢ, 30, 95, 0.35) for i = 5, …, 24;
3. reduces each group with the non-separable reduction r_nonsep: the position
   x₁ = r_nonsep(y₁, …, y₄), and the distance x₂ = r_nonsep(y₅, …, y₂₄).

Both objectives are minimized:

```text
f₁ = x₂ + 2 sin(x₁ π/2)
f₂ = x₂ + 4 cos(x₁ π/2)
```

The Pareto front is where the distance x₂ is 0, and there (f₁/2)² + (f₂/4)² = 1: a quarter
ellipse from (0, 4) to (2, 0), the same front as WFG4 to WFG8. The ideal point is (0, 0) and the
nadir point (2, 4). genoxide's docs give the optimal distance parameters as the toolkit computes
them: z₂₄ = 0.35 × 48, then, from z₂₃ back to z₅,

```text
zᵢ = 2i × 0.35^(1 / (0.02 + 1.96 u)),   u = mean(yᵢ₊₁, …, y₂₄)
```

They depend only on the parameters after them, so they are the same for every position on the
front: y₂₃ = 0.226, y₂₂ = 0.166, and so on down to y₅ = 0.00076. With every distance parameter at
0.35 × 2i instead, where WFG4 to WFG7 have their optimum, the distance is 0.0103.

## What makes it hard

WFG9 has four difficulties at once, which WFG4 to WFG8 have one or two at a time.

- **A parameter-dependent bias,** as in WFG7 and WFG8. b_param raises yᵢ to a power set by u, the
  mean of the parameters after it: 0.02 + 1.96 u for u up to 0.5, and 1 + 49 (2u − 1) above, up
  to 50. (Table XI writes it with the constants A = 0.98/49.98, B = 0.02 and C = 50.)
- **Deceptive position parameters,** as in WFG5. s_decept(y, 0.35, 0.001, 0.05) is 0 at 0.35,
  at the bottom of a basin of width 0.002 whose edges are at 1. Outside the basin it falls
  linearly to two wide deceptive minima of 0.05, at 0 and at 1.
- **Multimodal distance parameters,** as in WFG4, whose s_multi has A = 30 and 60 local minima;
  WFG9's has B = 95 in place of 10. It is 0 at 0.35 and 1 at both 0 and 1.
- **Non-separable reductions,** as in WFG6. r_nonsep adds each value and its differences from the
  others in its group, and scales the sum into [0, 1]. Equal values v give 0.4 v for the 4
  position parameters and v / 10.5 for the 20 distance parameters; the largest results need values
  that differ.

Together they make traps. Two show in the runs below.

The ends of the front sit behind the deceptive basin. Outside it, s_decept gives values from 0.05
to 1. x₁ = 0 needs all four shifted position values at 0, so all four biased position parameters
in the basin; at the deceptive minima, 0.05 each, x₁ is 0.4 × 0.05 = 0.02. x₁ = 1 needs two of
the values at 0 and two at 1; with 0.05 in place of the 0s, x₁ reaches only 0.97. Without the
basin, the front is cut to x₁ from 0.02 to 0.97.

The distance has a trap of its own at 2/21 ≈ 0.0952: all 20 shifted distance values at 1, s_multi's
largest value. From there, lowering any one of them to v saves 1 − v in the sum but adds 1 − v to
38 of its differences, so the distance gets worse. Only moving all of them together helps. The bias
keeps the search there: when the mean of the parameters after yᵢ is above 0.5, the power is above
1, up to 50, and sends most values of yᵢ close to 0, where s_multi is 1.

## Representation

A `Real` genome of 24 genes, the i-th in [0, 2i]: the vector z. The problem is genoxide's
`Wfg9::<2>::default()`, whose fitness is the pair (f₁, f₂). In Python,
`gx.problems.Wfg9(2)` gives the same problem, and `run` evaluates it in Rust, so both versions
print the same.

## Algorithm

Two algorithms, each for 1,000 generations; the example reports each front after 250 generations
and after 1,000. Both use simulated binary crossover with η = 15 and polynomial mutation with
η = 20 at a rate of 1/24 per gene, one gene per child on average.

- NSGA-II (Deb, Pratap, Agarwal and Meyarivan, 2002, IEEE Transactions on Evolutionary
  Computation 6(2): 182-197), with the settings of the ZDT examples: a population of 100 and a
  crossover rate of 0.9. It sorts solutions into non-dominated fronts, and within a front prefers
  solutions with a larger crowding distance, a measure of the gap between their neighbors.
- MOEA/D (Zhang and Li, 2007, IEEE Transactions on Evolutionary Computation 11(6): 712-731). It
  splits the problem into 101 single-objective subproblems, one per weight vector, with the
  weights (0, 1), (0.01, 0.99), …, (1, 0). Each subproblem keeps one solution and gets one child a
  generation, bred from its 20 nearest neighbors (with probability 0.9) and always recombined, at
  genoxide's default crossover rate for MOEA/D, 1. A child replaces at most 2 neighbors that it
  improves on. Each subproblem scores a solution by penalty-based boundary intersection (PBI),
  with θ = 5: the distance along its weight vector from the ideal point, plus 5 times the distance
  from that line.

MOEA/D is here because, of the algorithms tried, it leaves the distance trap most often; see Good
results. The weight vectors are spaced 0.01 apart, 101 of them rather than 100: the Python version
passes its weights to Rust as JSON text, and with 100 weight vectors, 1/99 apart, the Python and
Rust runs differed. With values like 0.01 and 0.99, they match.

## Output

Two lines per algorithm and budget: the size of the front, its IGD+ and its hypervolume, and then
how far its solutions are from the true front. The last line gives the whole front's hypervolume.

The distance of a point (f₁, f₂) is the d for which (f₁ − d, f₂ − d) lies on the front: the
smaller root of ((f₁ − d)/2)² + ((f₂ − d)/4)² = 1. Since WFG adds the distance x₂ to both
objectives, d is the solution's x₂. The example prints the smallest, the largest and the mean
over the front.

IGD+ (Ishibuchi et al., 2015, EMO 2015, LNCS 9019: 110-125) is measured to 500 points of the
optimal front, from genoxide's `optimal_front`. It averages, over those 500 points, the distance
to the nearest point of the found front, counting only the objectives in which the found point is
worse. 0 means that the found front covers the optimal one. Smaller is better.

The hypervolume (Zitzler and Thiele, 1999, IEEE Transactions on Evolutionary Computation 3(4):
257-271) is the area that the front dominates, up to a reference point. Larger is better. The
reference point is (2.2, 4.4), 1.1 times the nadir point (2, 4), as the ZDT examples use 1.1
times theirs. For the whole front, the hypervolume is the box up to the reference point less the
quarter ellipse under the front: 2.2 × 4.4 − 2π = 3.3968.

[The project page](https://tachsin.gr/projects/genoxide/examples/wfg9) plays this run back.

## Good results

A good front would have 100 solutions spread from (0, 4) to (2, 0), distances near 0, an IGD+ near
0 and a hypervolume near 3.3968; the 100 points of `optimal_front(100)` give 3.3610 and an IGD+
of 0.0051. Neither run gets there.

NSGA-II falls into the distance trap early. By generation 32 its whole front is 0.097 to 0.110
from the true one, and it stays there: after 250 generations the distances are 0.0956 to 0.1058,
after 1,000 0.0953 to 0.0984, a copy of the true front moved up by about 2/21 in both
objectives. Its IGD+ is 0.1278 and then 0.1267, its hypervolume 2.6914 and then 2.6979. In its
final front, y₂₄ is within 10⁻⁴ of its upper bound 1, the other distance parameters are raised to
powers from 6.5 to 50, and they come out of the bias below 0.002, where s_multi is above 0.98.

MOEA/D starts no better: after 250 generations its front is 0.069 to 0.105 from the true one,
with an IGD+ of 0.1294 and a hypervolume of 2.6845. Its first points leave the trap at about
generation 200, and the rest follow slowly. After 1,000, most of its front, from x₁ = 0.2 on, is
0.045 to 0.052 from the true one, with an IGD+ of 0.0819 and a hypervolume of 2.9394; the part
below x₁ = 0.1 is still at 0.097. Its front has 78 and then 97 solutions, of the 101 that MOEA/D
keeps, one per subproblem.

Both fronts span x₁ from 0.02 to 0.97 throughout: neither finds the deceptive basin, and both
ends of the front are missing, with f₁ at least 0.063 and f₂ at least 0.19 above the distance.

On seeds 1 to 10, the runs split. After 1,000 generations, a run either leaves the trap, with a
mean distance below 0.05, or stays at 0.095:

| algorithm | leave the trap | IGD+ after 1,000 generations |
|---|---|---|
| NSGA-II | 3 of 10 | 0.0186 to 0.1279 |
| SMS-EMOA | 4 of 10 | 0.0164 to 0.1263 |
| SPEA2 | 0 of 10 | 0.1263 to 0.1272 |
| MOEA/D, Tchebycheff | 7 of 10 | 0.0196 to 0.1268 |
| MOEA/D, PBI | 9 of 10 | 0.0354 to 0.0819 |

SMS-EMOA and SPEA2 ran with NSGA-II's settings, and both MOEA/Ds with the 101 weight vectors of
this example. MOEA/D with PBI leaves the trap on all 10 seeds after 2,000 generations; on seed 1,
this example's, it is the slowest. The runs of NSGA-II that leave the trap end closer to the
front than MOEA/D with PBI, with IGD+ from 0.0186 to 0.0255: PBI is the more reliable, not the
more precise. Changing NSGA-II's operators on seeds 1 to 3 doesn't help: polynomial mutation with
η = 5 or at a rate of 4/24, and simulated binary crossover with η = 5, stay in the trap on all
three seeds, and blend crossover (α = 0.5) leaves it on one. Arithmetic crossover, which moves
every gene of a child towards the other parent at once, leaves it on all three but stops at mean
distances of 0.034 to 0.042.
