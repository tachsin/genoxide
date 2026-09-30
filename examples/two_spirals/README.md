---
title: Two spirals
category: neuroevolution
summary: Evolve the 2,545 weights of a neural network that tells two interleaved spirals apart, by OpenAI's evolution strategy.
reference: "Lang, K. J. and Witbrock, M. J. (1988). Learning to tell two spirals apart. Proceedings of the 1988 Connectionist Models Summer School: 52-59. The method: Salimans, T., Ho, J., Chen, X., Sidor, S. and Sutskever, I. (2017). Evolution strategies as a scalable alternative to reinforcement learning. arXiv:1703.03864."
reference_url: https://arxiv.org/abs/1703.03864
optimum: "All 194 points classified"
languages: [rust]
order: 254
---

# Two spirals

## The problem

Two spirals wind around the origin three times, one inside the other: each point of the first
has its mirror image through the origin on the second. A classifier gets the 194 points, 97 on
each spiral, and has to tell which spiral each is on. Lang and Witbrock (1988) posed it as a
benchmark for training neural networks, and it has been one since, for learning rules and for
neuroevolution alike.

The points are Lang and Witbrock's: point `i` of the first spiral, for `i` from 0 to 96, is at the
angle `i π / 16` and the radius `6.5 (104 − i) / 104`, at `(r sin θ, r cos θ)`, and the second
spiral's point is at `(−x, −y)`. Here they're scaled by 1/6.5, to [−1, 1]², and computed with
genoxide's portable `math::sin` and `math::cos`, so they're the same bits on every platform.

There's no Python version: the Python package has no networks or `OpenEs` yet.

## What makes it hard

The two classes are interleaved all the way around, so no simple boundary separates them: the
boundary a network learns has to be a spiral too, and fit between points that are close together
near the center. Lang and Witbrock needed a network with shortcut connections, and a
single hidden layer of the usual kind trained by backpropagation often fails to learn it.

For an evolutionary method there's a second difficulty: the network needs thousands of weights.
CMA-ES, genoxide's method for continuous problems of up to a few hundred genes, keeps a covariance
matrix of the genes, 2,545² = 6.5 million entries here, and learns it too slowly at this size.

## Representation

A multilayer perceptron (`nn::Mlp`) of 2 inputs, two hidden layers of 48 tanh units and a tanh
output, with biases: 2,545 weights, a `Real` genome in [−3, 3] each (`Mlp::representation`). Its
output, in (−1, 1), is positive for the first spiral and negative for the second.

The fitness is the mean squared error of the outputs to the targets 1 and −1, minimized. The run
stops when the generation's best network classifies all 194 points, or after 1,000,000
evaluations.

## Algorithm

OpenAI's evolution strategy (`OpenEs`, Salimans et al. 2017), the baseline of neuroevolution at
scale. It keeps no population, only the mean of a search distribution, a network, and moves it
along an estimate of the gradient of the fitness:

- Each generation, it draws 50 perturbations `ε`, standard normal, and asks for the networks
  `mean + σ ε` and `mean − σ ε` (mirrored sampling, Brockhoff et al. 2010): 100 samples.
- It replaces their fitness by their ranks, spread evenly over [−0.5, 0.5], the best at 0.5.
- The gradient estimate is the sum of the perturbations weighted by their samples' ranks, and Adam
  (Kingma and Ba, 2015) moves the mean along it.
- Its cost per sample is linear in the number of weights, without a covariance matrix.

The settings: σ 0.01 of each weight's range (0.06), Adam's learning rate 0.01 of the range, the
mean evaluated each generation too (`evaluate_mean`, often the best network), and small initial
weights, uniform in [−0.25, 0.25]: large ones saturate the tanh units. The samples are drawn in
parallel (`parallel_breeding`), a random stream per pair, and evaluated in parallel: the results
are the same on any number of threads.

## Output

The first line gives the evaluations and generations until a network classified all the points,
and the second its mean squared error. Then comes a map of the network's decision over
[−1, 1]², a character per cell: `#` where it says the first spiral, `.` where it says the second,
and the points as `A` (first spiral) and `B` (second). Every `A` lies in the `#` region and every
`B` in the `.` region, along two spiral bands.

[The project page](https://tachsin.gr/projects/genoxide/examples/two-spirals) plays the run back:
the network's decision over the plane as it learns.

## Good results

The goal is every point classified. The run of `output.txt` reaches it after 135,744 evaluations,
in 1,343 generations.

RESULTS
