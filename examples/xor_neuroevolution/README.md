---
title: XOR neuroevolution
category: neuroevolution
summary: Evolve the weights of a 2-2-1 neural network until it computes XOR.
reference: "Stanley, K. O. and Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. Evolutionary Computation 10(2): 99-127."
reference_url: https://doi.org/10.1162/106365602320169811
optimum: "0 (squared error)"
languages: [rust, python]
order: 100
---

# XOR neuroevolution

XOR is the classic first test of neuroevolution, used for example by NEAT: it isn't linearly
separable, so a network needs a hidden layer to compute it. The example fixes the topology, 2
inputs, 2 sigmoid hidden units and a sigmoid output, and evolves its 9 weights and biases in
[−10, 10]. The fitness is the sum of the squared errors over the four input pairs, minimized by
CMA-ES with IPOP restarts until it is below 0.01 or after 20,000 evaluations. It prints the error
and the network's output for each input pair.
