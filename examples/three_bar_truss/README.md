---
title: Three-bar truss
category: constrained
summary: The least volume of a planar truss of three bars whose stresses stay within the allowed stress, with a minimum known in closed form.
reference: "Nowacki, H. (1974). Optimization in pre-contract ship design. In Computer Applications in the Automation of Shipyard Operation and Ship Design, North-Holland: 327-338."
reference_url: ""
optimum: "263.895843 (volume, cm³)"
languages: [rust, python]
order: 74
---

# Three-bar truss

## The problem

Three bars hang from a rigid support and meet at one joint below it, where a load acts. The middle
bar is vertical, of length l = 100 cm. The two outer bars leave it at 45°, so each is √2 l long.
The truss is symmetric: both outer bars have the same cross-section. There are two design
variables, in cm²: the cross-section A₁ of each outer bar and A₂ of the middle one, genoxide's x₁
and x₂. The goal is the least volume of material:

```text
(2√2 A₁ + A₂) l
```

The load is P = 2 kN, and no bar's stress may exceed σ = 2 kN/cm². Three constraints, written
g(x) ≤ 0:

```text
g1 = (√2 A₁ + A₂) / (√2 A₁² + 2 A₁A₂) P − σ      the stress in the first outer bar
g2 = A₂ / (√2 A₁² + 2 A₁A₂) P − σ                the stress in the second outer bar
g3 = P / (A₁ + √2 A₂) − σ                        the stress in the middle bar
```

These are the stresses of a load that pulls the joint down and to one side, at 45°: it stretches
the first outer bar and the middle one, and compresses the second outer bar. Thinner bars carry the
same load at a higher stress, so the volume and the stresses pull against each other. The stresses
are undefined at A₁ = 0, where the outer bars vanish: there, the fitness is invalid or its
violation infinite.

The problem comes from Nowacki (1974) and Ray and Saini (2001, Engineering Optimization 33(6):
735-748). genoxide's `ThreeBarTruss` uses the definition and bounds that Yang and Gandomi restate
(2012, Engineering Computations 29(5): 464-483, eqs. 14-17): A₁ and A₂ in [0, 1].

The minimum is derived from the definition. With g1 active, A₂ = √2 A₁ (1 − A₁) / (2A₁ − 1), and
the volume is √2 l (3u/4 + 1 + 1/(4u)) with u = 2A₁ − 1, smallest at u = 1/√3. That gives A₁ =
(3 + √3)/6 ≈ 0.788675, A₂ = 1/√6 ≈ 0.408248 and a volume of 100 (√2 + √6/2) ≈ 263.8958434 cm³:
a proven optimum, not only a best known value.

## What makes it hard

Not much, and that makes it a useful check. It has two variables, a smooth volume and a feasible
region that covers about 22 % of the box, estimated from 2 million random designs. An algorithm
that fails here likely mishandles its constraints.

The one difficulty is the boundary. The volume falls toward smaller cross-sections, so the minimum
lies on the boundary of g1, the stress of the most loaded bar, and the search has to approach a
curved boundary it can't cross without becoming infeasible. Near the minimum, the volume barely
changes along that curve: moving A₁ by 1e-5 along it changes the volume by 7e-8, a relative
3e-10.

The exact minimum is also a limit of floating-point arithmetic: at A₁ = (3 + √3)/6 and A₂ = 1/√6
computed in `f64`, g1 can come out a rounding error, 1e-16, above 0, and the design infeasible.
genoxide's `optimum()` gives that design grown by a unit in the last place until it's feasible.

## Representation

A `Real` genome of 2 genes, (A₁, A₂), in [0, 1]². The fitness is the volume and the total
constraint violation, the sum of max(0, g) over the three constraints, 0 for a feasible design.
genoxide compares fitnesses with Deb's feasibility rules (Deb, 2000, Computer Methods in Applied
Mechanics and Engineering 186: 311-338): a feasible design beats an infeasible one, two feasible
ones compare by volume, and two infeasible ones by violation. The rules need no penalty weights.

## Algorithm

SHADE (Tanabe and Fukunaga, 2013, IEEE CEC 2013: 71-78), a differential evolution that adapts its
scale factor and crossover rate from successful trials, with genoxide's defaults: its published
population of 100, and a restart after 200 generations without progress. The run stops once its
best design is feasible and within 1e-10 of the minimum, relative to it, or after 50,000
evaluations.

## Output

The first line gives the volume of the best design, the evaluations the run took, and the minimum.
The second gives its constraint violation; 0 means it's feasible. The third gives the design: A₁
and A₂, in cm². The last names the constraints at their limit, within 1e-6 of 0. In Python, `run`
evaluates the problem in Rust, so both versions print the same.

The plot shows the two cross-sections on their ranges, and each constraint's value g: satisfied with
its slack, active (within 1e-6 of its limit) or violated. The best of the first 100 random designs
is feasible, at a volume of 274.1. The median volume of that population, 201.1, is below it: the
smallest designs are infeasible, too thin for the load. The best volume is within 0.1 of the minimum
after about 1,700 evaluations, and within 0.01 after about 3,500; the rest of the run settles the
last digits, while the design slides along the boundary of g1. g1 is active from about 6,500
evaluations on; g2 and g3 keep a slack of 1.46 and 0.54 kN/cm².

[The project page](https://tachsin.gr/projects/genoxide/examples/three-bar-truss) plays this run back.

## Good results

The run reaches the minimum, 263.895843, within a relative 1e-10, after 13,300 evaluations, with no
violation: A₁ = 0.788676 and A₂ = 0.408246, against the exact 0.788675 and 0.408248. g1 is at its
limit: the first outer bar carries exactly the allowed stress. The second outer bar carries 0.54
kN/cm² and the middle bar 1.46 kN/cm², both well below 2.

Since the minimum is proven, a volume below 263.8958434 in this formulation comes from an
infeasible design, one whose stresses exceed the limit, perhaps only by a rounding error.
