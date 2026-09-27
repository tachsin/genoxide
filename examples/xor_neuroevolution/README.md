---
title: XOR neuroevolution
category: neuroevolution
summary: Evolve the weights of a 2-2-1 neural network until it computes XOR.
reference: "Stanley, K. O. and Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. Evolutionary Computation 10(2): 99-127."
reference_url: https://doi.org/10.1162/106365602320169811
optimum: "0 (squared error)"
languages: [rust, python]
order: 110
---

# XOR neuroevolution

## The problem

XOR, exclusive or, is 1 when exactly one of its two inputs is 1:

| a | b | a xor b |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

No straight line separates the two 1s from the two 0s in the plane of the inputs, so a single layer
of threshold units can't compute XOR (Minsky and Papert, 1969, Perceptrons, MIT Press). A network
needs a hidden layer. That makes XOR a classic first test of neuroevolution: NEAT (Stanley and
Miikkulainen, 2002) used it to check that it evolves the hidden structure it needs.

Here the structure is fixed, and only the weights evolve: 2 inputs, 2 hidden units and 1 output
unit, each unit a sigmoid 1 / (1 + e⁻ᶻ) of a weighted sum of its inputs plus a bias.

## What makes it hard

The weights are 9 continuous numbers: two weights and a bias per hidden unit, and two weights and a
bias for the output. The fitness is the sum of the squared errors over the four cases.

A sigmoid never reaches 0 or 1, so the error is never exactly 0: it only approaches 0 as the weights
grow. With large weights the sigmoids saturate, and the error barely changes when a weight does,
which leaves wide flat regions. A network that outputs 0.5 for every input has an error of 1.
Swapping the two hidden units gives the same network, so every solution has a twin.

## Representation

A `Real` genome of 9 genes in [−10, 10]: the first three are the first hidden unit's two weights and
bias, the next three the second's, and the last three the output unit's. The network is decoded from
the genome for every evaluation.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), which genoxide's docs
recommend for continuous problems of up to a few hundred genes. It samples a population from a
normal distribution, and adapts its mean, step size and covariance matrix to the steps that worked.
It uses genoxide's defaults: a population of 4 + ⌊3 ln 9⌋ = 10, and a step size of 0.3 of each
gene's range. With IPOP restarts (Auger and Hansen, 2005), a run that has converged, for example on
a flat region, starts again from a random point with twice the population.

The run stops when the error is below 0.01, or after 20,000 evaluations.

## Output

The first line gives the squared error of the best network and the evaluations it took. The next
four lines give, for each input pair, the expected output and the network's output, to 3 decimals.

[The project page](https://tachsin.gr/projects/genoxide/examples/xor-neuroevolution) plays this run back.

## Good results

The error can't reach 0, so the example aims at 0.01. An error below 0.01 puts each output within
0.1 of its target, on the right side of 0.5. The run gets there after about 840 evaluations, with
outputs of about 0.05 for the 0s and 0.94 to 0.96 for the 1s.
