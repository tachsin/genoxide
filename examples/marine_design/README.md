---
title: Conceptual marine design
category: multi-objective
summary: Design the Panamax bulk carrier that carries cargo the cheapest, with the lightest ship and the most cargo a year, subject to nine constraints, with SMS-EMOA.
reference: "Parsons, M. G. and Scott, R. L. (2004). Formulation of multicriterion design optimization problems for solution with scalar numerical optimization methods. Journal of Ship Research 48(1): 61-76."
reference_url: https://doi.org/10.5957/jsr.2004.48.1.61
optimum: "not known in closed form; ideal point (8.376894, 5240.3356, −700,552.76); genoxide's reference front has a hypervolume of 0.8624 in objectives scaled by the ideal point and its estimated nadir point (11.0662, 12,435.8, −386,500) (reference point (1.1, 1.1, 1.1))"
languages: [rust, python]
order: 229
---

# Conceptual marine design

## The problem

Parsons and Scott (2004) design a bulk carrier at the concept stage, with a parametric model after
Sen and Yang (1998): six genes, the ship's length L, beam B, depth D and draft T, in m, its block
coefficient C_B and its speed V_k, in knots, give its displacement, the power it needs, its steel,
outfit and machinery weights, the cargo it carries a voyage, the round trips it makes a year, and
what it costs to build and run. The three criteria are the transportation cost, the light ship
weight, which stands for the cost of building it, and the annual cargo, which is maximized:

```text
minimize   f₁ = annual cost / annual cargo                         the transportation cost, in £/t
           f₂ = W_s + W_o + W_m                                    the light ship weight, in t
           f₃ = −annual cargo                                      in t a year
where      Δ = 1.025 L B T C_B,   V = 0.5144 V_k,   F_n = V / √(9.8065 L)
           P = Δ^(2/3) V_k³ / (a + b F_n),  a = 4977.06 C_B² − 8105.61 C_B + 4456.51,
                                            b = −10847.2 C_B² + 12817 C_B − 6960.32
           W_s = 0.034 L^1.7 B^0.7 D^0.4 C_B^0.5,  W_o = L^0.8 B^0.6 D^0.3 C_B^0.1,
           W_m = 0.17 P^0.9,  DWT = Δ − (W_s + W_o + W_m),
           daily consumption C = 0.19 P · 24 / 1000 + 0.2
           sea days = 5000 / (24 V_k),   cargo deadweight = DWT − C (sea days + 5) − 2 DWT^0.5
           port days = 2 (cargo deadweight / 8000 + 0.5),  RTPA = 350 / (sea days + port days)
           annual cost = 0.2 · 1.3 (2000 W_s^0.85 + 3500 W_o + 2400 P^0.8) + 40000 DWT^0.3
                         + (1.05 C · sea days · 100 + 6.3 DWT^0.8) RTPA
           annual cargo = cargo deadweight · RTPA
subject to L/B ≥ 6, L/D ≤ 15, L/T ≤ 19, T ≤ 0.45 DWT^0.31, T ≤ 0.7 D + 0.7,
           25,000 ≤ DWT ≤ 500,000, F_n ≤ 0.32,
           GM_T = 0.53 T + (0.085 C_B − 0.002) B² / (T C_B) − (1 + 0.52 D) ≥ 0.07 B
L in [150, 274.32], B in [20, 32.31], D in [10, 25], T in [8, 11.71], C_B in [0.63, 0.75],
V_k in [14, 18]
```

The model is the paper's appendix, and the limits its Panamax case 2: the Panama Canal's locks bound
L, B and T, and the least deadweight is 25,000 t. The lower bounds of L, B, D and T and the upper
bound of D are genoxide's, as the paper gives none, and hold every feasible design. The paper's
designs give its printed criteria with this model; Tanabe and Ishibuchi's restatement (2020, problem
RE4-6-2) differs, with a fourth objective, a least deadweight of 3000 t, narrower bounds and, in its
code, sea days of (5000/24) V_k. The problem is genoxide's `MarineDesign` in
`multi::problems::engineering`.

The front isn't known. Its ideal point, (8.376894, 5240.3356, −700,552.76), is the paper's three
single-criterion designs (its table 4) computed again: the cheapest transport at 14 knots, the
lightest ship at the least deadweight, and the most cargo at 18 knots. genoxide's reference front,
the non-dominated designs of about 11,300 runs of SHADE, each minimizing an achievement scalarizing
function along one of Das and Dennis's directions, and of eighty runs of SMS-EMOA and NSGA-III of
500 or 2,000 generations, each design then improved by SHADE looking for one that dominates it, has
15,461 points, worst values (11.0662, 12,435.8, −386,500), the estimated nadir point, and a
hypervolume of 0.8624 in the scaled objectives below: a lower bound on the whole front's.

## What makes it hard

The optima lie on the constraints and the bounds: 88% of the reference front's designs have the
largest block coefficient, 68% the largest draft, and each single-criterion design meets two or
more constraints and several bounds at once (the lightest ship has L/B = 6, a draft at both of its
limits and a deadweight of exactly 25,000 t). The objectives span very different scales, from 8 £/t to 700,000 t
a year. And the front's worst transportation cost, 11.07 £/t, isn't at any of the three single-
criterion designs, whose worst is 10.29: it belongs to a fast ship of middling size (L ≈ 176 m at 18
knots), lighter than the largest and carrying more than the lightest, which the extremes alone
don't show.

## Representation

A `Real` genome of 6 genes, the design variables. The problem is genoxide's
`multi::problems::engineering::MarineDesign` (`gx.problems.multi_engineering.MarineDesign` in
Python), whose fitness is the three objectives and the total constraint violation. Solutions compare
by constrained dominance.

## Algorithm

SMS-EMOA, which keeps the solutions that add the most hypervolume, with a population of 92 for 500
generations, simulated binary crossover (η = 15, at a rate of 0.9) and polynomial mutation (η = 20,
at a rate of 1/6 per gene). With the same operators, NSGA-III with 91 directions ends with less
hypervolume over seeds 1 to 20, 89.36% to 92.15% of the reference front's, SPEA2 89.67% to 91.83%,
and NSGA-II 84.26% to 88.83%.

## Output

The first line gives the size of the final front and how many of its solutions are feasible, the
second the range of each objective on it, the annual cargo as a positive number. The third gives
its hypervolume up to the reference point (1.1, 1.1, 1.1), in objectives scaled to [0, 1] by the
ideal point and the estimated nadir point, as a share of the reference front's. In Python, `run`
evaluates the problem in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/marine-design) plays this run back.

## Good results

A good front is feasible and spread over the whole surface, from the cheapest transport to the
lightest ship and the most cargo. 92 points of the reference front, chosen one by one for the most
hypervolume, give 96.19% of its hypervolume.

The run's front has 92 feasible solutions, with transportation costs from 8.3776 £/t, light ship
weights from 5346.7 t and annual cargoes up to 700,544 t, close to the three minima, and 92.57% of
the reference front's hypervolume, 96.2% of what 92 chosen points reach. Its transportation costs
go up to 11.6209, beyond the reference front's worst: a solution there is dominated by the reference
front, though not by the run's own. Over seeds 1 to 20, every run ends between 92.09% and 93.26%.
