---
title: Double pole balancing
category: neuroevolution
summary: Evolve the weights of a neural network that balances two poles of different lengths on one cart for 100,000 steps, by CMA-ES.
reference: "Wieland, A. P. (1991). Evolving neural network controllers for unstable systems. IJCNN 1991, vol. 2: 667-673. Settings of Gomez, F., Schmidhuber, J. and Miikkulainen, R. (2008). Accelerated neural evolution through cooperatively coevolved synapses. JMLR 9: 937-965."
reference_url: https://doi.org/10.1109/IJCNN.1991.155416
optimum: "Balanced for 100,000 steps of 0.02 s (the success criterion of Gomez et al. 2008)"
languages: [rust]
order: 252
family: pole balancing
tab: Two poles
---

# Double pole balancing

## The problem

Two poles stand side by side on one cart, a long one and a short one, and a controller pushes the
cart to keep both upright and the cart on the track. Wieland (1991) introduced it as a harder
successor of the [cart-pole](../cart_pole/): the poles respond differently to the same push, and a
single force has to catch both. It became the standard benchmark of neuroevolution: Stanley and
Miikkulainen (2002) introduced NEAT on it, and Gomez, Schmidhuber and Miikkulainen (2008) compare
many methods on it.

The system is Wieland's, with the friction corrected as Florian (2007) derives it for the
cart-pole, and the settings of Gomez et al. (2008):

| Setting | Value |
|---|---|
| Cart | 1 kg, on a track from −2.4 m to 2.4 m |
| Poles | 0.1 kg and 1 m long; 0.01 kg and 0.1 m long |
| Friction | 0.0005 of the cart on the track, 0.000002 N m s at each hinge |
| Force | from −10 N to 10 N, held for each step of 0.02 s |
| Start | the long pole at 4° from vertical, the short one upright, the cart at rest in the middle |
| Failure | either pole beyond 36°, or the cart beyond the track's ends |
| Success | 100,000 steps without failing, over 33 minutes of simulated time |

The equations are integrated by fourth-order Runge-Kutta in two steps of 0.01 s per step, with
genoxide's portable `math::sin_cos`, so a run is the same bits on every platform
(`genoxide::problems::control::DoublePole`).

There's no Python version: the Python package has no networks or control tasks yet.

## What makes it hard

The short pole is a tenth of the long one's length and falls about three times as fast (the time a
pole takes to fall grows with the square root of its length), and both stand on the same cart: a
push that saves one can topple the other. A controller has to hold both near vertical and bring the
cart back to the middle, a six-variable unstable system, for 100,000 steps. Random weight guessing,
which solves the cart-pole in a few hundred attempts, needs about 474,000 here (Gomez et al. 2008).

## Representation

A multilayer perceptron (`nn::Mlp`) of 6 inputs, 6 hidden tanh units and a tanh output, without
biases: the network with which Igel (2003, Neuroevolution for reinforcement learning using
evolution strategies, CEC 2003: 2588-2595) got his best results on this task, 42 weights, a `Real`
genome in [−1, 1] each. The inputs are the cart's position and velocity and each pole's angle and
angular velocity, scaled to about [−1, 1]; the output, in (−1, 1), is the force in units of 10 N.
Without biases the network is an odd function of the state, as the task is symmetric; Igel found
that with biases CMA-ES needs about three times the evaluations, and here it's 13 times as many.

The fitness is the number of steps the network balances both poles, up to 100,000, maximized.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195) with its defaults: a
population of 4 + ⌊3 ln 42⌋ = 15 and a step size of 0.3 of each gene's range, and IPOP restarts in
case a run converges without a solution. The run stops at 100,000 steps, or after 100,000
evaluations.

## Output

The first line gives the steps the best network balanced, the evaluations and the generations it
took. The second gives how far the cart and the two poles went from the middle and the vertical
over the 100,000 steps, and the third the network's weights.

[The project page](https://tachsin.gr/projects/genoxide/examples/double-pole) plays this run back.

## Good results

The goal is the task's success criterion, 100,000 steps. The run of `output.txt` reaches it after
585 evaluations, in 38 generations; its network keeps the cart within 0.67 m of the middle, the
long pole within 5.4° and the short one within 10.9°.

Over seeds 1 to 100, all 100 runs solved the task, after 683 evaluations on average (a median of
675, at most 1,575). The counts are of whole generations of 15: the engine evaluates a generation
before it checks the stop. With biases (49 weights), 99 of 100 runs solved it, after 8,711
evaluations on average.

Gomez et al. (2008, table 3) list the average evaluations to solve this task: 895 for CMA-ES (from
Igel 2003, with this network), 954 for CoSyNE, 3,600 for NEAT, 3,800 for ESP, 12,600 for SANE,
22,100 for CNE, 307,200 for evolutionary programming and 474,329 for random weight guessing; of the
value-function methods only Q-MLP solved it, in 10,582. Igel's runs started the long pole at 1°,
Stanley and Miikkulainen's (NEAT) too; Gomez et al. start it at 4°, as here.
