---
title: Double pole balancing without velocities
category: neuroevolution
summary: Evolve a recurrent neural network that balances two poles on a cart seeing only their angles and the cart's position, by CMA-ES, and passes Gruau's generalization test.
reference: "Gruau, F., Whitley, D. and Pyeatt, L. (1996). A comparison between cellular encoding and direct encoding for genetic neural networks. Genetic Programming 1996: 81-89. On Wieland's (1991) double pole, with the settings of Gomez, F., Schmidhuber, J. and Miikkulainen, R. (2008). Accelerated neural evolution through cooperatively coevolved synapses. JMLR 9: 937-965."
reference_url: https://www.jmlr.org/papers/v9/gomez08a.html
optimum: "Balanced for 100,000 steps, and for 1000 steps from at least 200 of 625 other starts (Gruau et al.'s success criteria)"
languages: [rust]
order: 253
family: pole balancing
tab: Two poles, no velocities
---

# Double pole balancing without velocities

## The problem

The [double pole](../double_pole/) again, two poles side by side on a cart, with the same system and
settings, but the controller sees only three of the six state variables: the cart's position and
the two poles' angles. Without the velocities, it can't tell a pole that is passing through the
vertical from one at rest there; it has to infer them from what it saw before, so it needs memory.
Gruau, Whitley and Pyeatt (1996) introduced this version as the hardest of the pole balancing
tasks, and it remains the standard test of neuroevolution of recurrent networks (Stanley and
Miikkulainen 2002; Igel 2003; Gomez et al. 2008).

| Setting | Value |
|---|---|
| Cart, poles, friction, force | as the double pole's: 1 kg; 1 m and 0.1 kg, 0.1 m and 0.01 kg; up to 10 N every 0.02 s |
| Observed | the cart's position and both poles' angles, scaled to about [−1, 1] |
| Start | the long pole at 4° from vertical, the rest 0 |
| Failure | either pole beyond 36°, or the cart beyond ±2.4 m |
| Fitness | Gruau et al.'s damping fitness over 1000 steps |
| Success | 100,000 steps from the start, and 1000 steps from at least 200 of 625 other starts |

Gruau et al. made two further changes to rule out controllers that keep the poles up by jiggling
the cart back and forth, which needs no velocities:

- **The damping fitness**, `0.1 f₁ + 0.9 f₂` over an episode of 1000 steps: `f₁ = t / 1000` for
  the `t` steps the poles stayed up, and `f₂ = 0.75 / Σ (|x| + |ẋ| + |θ₁| + |θ̇₁|)` over the
  last 100 of them (0 if `t < 100`), which rewards bringing the cart and the long pole to rest.
- **The generalization test**: a network that balances the poles for 100,000 steps must also
  balance them for 1000 steps from at least 200 of 625 starts, which give the cart's position and
  velocity and the long pole's angle and angular velocity each of 5 values across ±2.16 m,
  ±1.35 m/s, ±3.6° and ±8.6°/s (Igel 2003 gives the values).

The task is `genoxide::problems::control::DoublePole::without_velocities()`, with
`damping_fitness`, `generalization` and `solved`, integrated as the double pole by fourth-order
Runge-Kutta with portable `sin` and `cos`, the same bits on every platform.

There's no Python version: the Python package has no networks or control tasks yet.

## What makes it hard

The controller must estimate velocities from positions, and the fitness sees only 1000 steps
(20 s) while success takes 100,000 and a test from 625 other starts that the search never sees.
A search can climb the damping fitness to networks that hold the poles for 1000 steps, calmly,
and fail soon after, or that balance from the one start but from no other. Igel (2003) found that
CMA-ES does exactly that: it converges on such a network, its step size shrinks, and nothing in
the fitness pulls it towards networks that generalize.

## Representation

An Elman network (`nn::Elman`, Elman 1990, Finding structure in time, Cognitive Science 14(2):
179-211): 3 inputs, 3 hidden tanh units, each receiving the inputs and the three hidden outputs of
the previous step, and a tanh output, the force in units of 10 N. Without biases, as for the
double pole, it has 3 × (3 + 3) + 3 = 21 weights, a `Real` genome in [−1, 1] each. The context
starts at 0 in every episode.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) with its default
population, 4 + ⌊3 ln 21⌋ = 13, and initial step size, 0.3 of each gene's range, maximizing the
damping fitness, with IPOP restarts. As Igel (2003) did against the convergence above, its step
size is bounded below (`CmaesBuilder::min_step`), at 0.05 of each gene's range, a sixth of the
initial one: the search keeps exploring around the networks that balance for 1000 steps until one
passes the tests.

After each generation, the generation's best network, if it balanced the 1000 steps, runs the
tests: 100,000 steps from the start, and the 625 starts of the generalization test. The run stops
when one passes both, or after 100,000 evaluations; the tests aren't counted as evaluations, as
in the papers.

## Output

The first line gives the evaluations and generations until a network passed the tests, the
second how many of the 625 starts it balanced, the third its damping fitness. The fourth gives how
far the cart and the two poles went from the middle and the vertical over the 100,000 steps, and
the fifth the network's weights.

[The project page](https://tachsin.gr/projects/genoxide/examples/double-pole-no-velocities) plays
this run back.

## Good results

The goal is the task's success criteria. The run of `output.txt` meets them after 1,677
evaluations, in 128 generations; its network balances from 201 of the 625 starts.

Over seeds 1 to 100, all 100 runs solved the task, after 3,043 evaluations on average (a median of
2,405, at most 9,698), their networks balancing from 247 of the 625 starts on average.
Without the bound on the step size, 13 of 20 runs I tried solved it within 30,000 evaluations: the
others converged on networks that balance for about 2,000 steps.

Gomez et al. (2008, table 4) list the average evaluations to solve this task with the damping
fitness: 3,416 for CoSyNE, 6,061 for CMA-ES (from Igel 2003, a recurrent network of 3 hidden units,
its step size bounded at half the initial one), 6,929 for NEAT (with populations of 16), 26,342 for
ESP, 87,623 for CNE, 451,612 for SANE, 840,000 for cellular encoding (Gruau et al.'s single run) and
1,232,296 for random weight guessing. Stanley and Miikkulainen (2002) report 33,184 for NEAT with
populations of 1000, and generalization scores of 286 for NEAT, 289 for ESP and 300 for cellular
encoding; Igel reports 250 for CMA-ES. The setups differ in the network (Igel's is fully recurrent,
its output fed back too), the start (Igel started the long pole at 4.5°) and the sum of the damping
fitness (Stanley and Miikkulainen and Gomez et al. print a sum from `t − 100` to `t`, 101 steps;
here, as in Igel, 100).