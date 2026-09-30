---
title: Cart-pole
category: neuroevolution
summary: Evolve the weights of a neural network that balances a pole on a cart for 100,000 steps, by CMA-ES.
reference: "Barto, A. G., Sutton, R. S. and Anderson, C. W. (1983). Neuronlike adaptive elements that can solve difficult learning control problems. IEEE Transactions on Systems, Man, and Cybernetics 13(5): 834-846. Equations corrected by Florian, R. V. (2007). Correct equations for the dynamics of the cart-pole system. Technical report, Coneural, Romania."
reference_url: https://doi.org/10.1109/TSMC.1983.6313077
optimum: "Balanced for 100,000 steps of 0.02 s (the success criterion of Gomez et al. 2008)"
languages: [rust]
order: 251
family: pole balancing
tab: One pole
---

# Cart-pole

## The problem

A pole is hinged to a cart that runs on a track, and falls unless the cart moves under it. A
controller pushes the cart left or right to keep the pole upright and the cart on the track. Barto,
Sutton and Anderson (1983) made it the standard test of learning control, and it has been used to
compare reinforcement learning and neuroevolution methods since.

The system here is theirs, with the equations of motion corrected by Florian (2007), and the
settings of Gomez, Schmidhuber and Miikkulainen (2008, Accelerated neural evolution through
cooperatively coevolved synapses, JMLR 9: 937-965), who compare many methods on it:

| Setting | Value |
|---|---|
| Cart | 1 kg, on a track from −2.4 m to 2.4 m |
| Pole | 0.1 kg, 1 m long (a half-length of 0.5 m) |
| Friction | 0.0005 of the cart on the track, 0.000002 N m s at the hinge |
| Force | up to 10 N either way, held for each step of 0.02 s |
| Start | the pole at 4° from vertical, the cart at rest in the middle |
| Failure | the pole beyond 12°, or the cart beyond the track's ends |
| Success | 100,000 steps without failing, over 33 minutes of simulated time |

The equations are integrated by fourth-order Runge-Kutta in two steps of 0.01 s per step, with
genoxide's portable `math::sin_cos`, so a run is the same bits on every platform
(`genoxide::problems::control::CartPole`).

There's no Python version: the Python package has no networks or control tasks yet.

## What makes it hard

Not much, for a method that searches the weights of a network: Gomez et al. (2008) found that
random weight guessing solves it in 199 attempts on average. It is the baseline of the pole
balancing family, before the double pole.

What is hard is the length of the test: a controller has to keep the system stable for 100,000
steps, not just to catch the pole once. Most random networks drop the pole within a few dozen
steps, and some keep it up for thousands of steps while the cart drifts slowly to the end of the
track.

## Representation

A multilayer perceptron (`nn::Mlp`) of 4 inputs, 8 hidden tanh units and a tanh output, without
biases: Igel's (2003, Neuroevolution for reinforcement learning using evolution strategies, CEC
2003: 2588-2595) network for this task, 40 weights, a `Real` genome in [−1, 1] each
(`Mlp::representation`). The inputs are the cart's position and velocity and the pole's angle and
angular velocity, scaled to about [−1, 1]; the output, in (−1, 1), is the force in units of 10 N.
Without biases the network is an odd function of the state, as the task is symmetric: Igel found
that biases slow the search.

The fitness is the number of steps the network balances the pole, up to 100,000, maximized.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), the method genoxide
recommends for continuous problems of up to a few hundred genes, with its defaults: a population
of 4 + ⌊3 ln 40⌋ = 15 and a step size of 0.3 of each gene's range, and IPOP restarts in case a run
converges without a solution. The run stops at 100,000 steps, or after 100,000 evaluations.

## Output

The first line gives the steps the best network balanced, the evaluations and the generations it
took. The second gives how far the cart and the pole went from the middle and the vertical over
the 100,000 steps, and the third the network's weights.

[The project page](https://tachsin.gr/projects/genoxide/examples/cart-pole) plays this run back.

## Good results

The goal is the task's success criterion, 100,000 steps. The run of `output.txt` reaches it after
30 evaluations, in 1 generation, and its network keeps the cart within 9 cm of the middle and the
pole within 4° over the 100,000 steps.

Over seeds 1 to 100, all 100 runs solved the task, after 45 evaluations on average (a median of 30,
at most 165). The counts are of whole generations of 15: the engine evaluates a generation before
it checks the stop.

Gomez et al. (2008, table 1) list the average evaluations of 50 runs for this task: 98 for CoSyNE,
199 for random weight guessing, 283 for CMA-ES (from Igel 2003), 289 for ESP, 302 for SANE, 352 for
CNE and 743 for NEAT, and thousands for the value-function methods. The setups differ in details
that matter at these small numbers: Igel's CMA-ES started the pole upright, without friction, and
turned the network's output into a push of ±10 N at random with the output's probability, where
here the force is the output itself and the pole starts at 4°.
