---
title: XOR by NEAT
category: neuroevolution
summary: Evolve a network's structure and weights until it computes XOR, from networks without hidden nodes, by NEAT.
reference: "Stanley, K. O. and Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. Evolutionary Computation 10(2): 99-127."
reference_url: https://doi.org/10.1162/106365602320169811
optimum: "Every output on the right side of 0.5 (the paper's success criterion); a fitness of 16 is a perfect network"
languages: [rust]
order: 250.5
family: XOR
tab: NEAT
---

# XOR by NEAT

## The problem

Find a neural network that computes XOR: an output of 1 when exactly one of its two inputs is 1,
and 0 otherwise. XOR isn't linearly separable, so a network without hidden nodes can't compute it;
the [fixed network](../xor_neuroevolution/) tab gives the network two hidden units and evolves only
its weights. Here the structure evolves too, by NEAT (NeuroEvolution of Augmenting Topologies,
Stanley and Miikkulainen 2002), starting from networks with no hidden nodes at all: the two inputs
and a bias connected straight to the output. XOR is the paper's first experiment, a check that NEAT
finds the structure a problem needs.

The fitness is the paper's, (4 − Σ|error|)², the error summed over the four cases: 16 for a perfect
network. The problem is solved at the paper's criterion: every output on the right side of 0.5.

There's no Python version: the Python package has no NEAT yet.

## What makes it hard

The search has to add a hidden node and connect it before the fitness can reward it: a new node
first changes the network's function little (it splits a connection, with a weight of 1 in and the
old weight out), and a network with a new node is usually worse until its weights are tuned. NEAT
protects such networks by speciation: networks are grouped by how different their genes are, and
compete mainly within their species, whose offspring depend on its members' mean fitness.

## Representation

A network (`neat::Network`): node genes (the inputs, the bias, the output and hidden nodes, each
with the paper's steepened sigmoid 1 / (1 + e^(−4.9x)) on the output and the hidden nodes) and
connection genes, each with an innovation number that records when that connection first appeared
in the run. Crossover aligns two networks' genes by these numbers. Networks stay feed-forward; each
is evaluated through `Network::feed_forward`, compiled once into the order of its nodes, with
genoxide's `math` functions, so the outputs are the same bits on every platform.

## Algorithm

NEAT (`neat::Neat`) with the paper's settings: 150 networks, speciated by the compatibility distance
δ = E/N + D/N + 0.4 W̄ (excess and disjoint genes, and the mean weight difference of the matching
ones) with a threshold of 3. Each species gets offspring in proportion to its members' shared
fitness (the paper's: the raw fitness divided by the species' size). A quarter of the offspring come
from mutation alone, the rest from crossover within the species (between species with probability
0.001); weights are mutated in 80% of the offspring, a new node is added with probability 0.03 and a
new connection with 0.05. The champion of each species of more than five networks is kept, and a
species that doesn't improve for 15 generations stops reproducing.

Where the paper gives no value: new weights are normal with deviation 1 and the parents of each
species are its best 20%, as in neat-python, and weight perturbations are normal with deviation 1
(with neat-python's 0.5, 3 of 100 runs failed within 1000 generations).

## Output

The first line gives the setting, then the generation in which a network solved XOR and the
evaluations it took, the network (its hidden nodes and its enabled connections, from node to node
with their weights: `in0` and `in1` the inputs, `h` the hidden nodes by id) and its outputs on the
four cases.

[The project page](https://tachsin.gr/projects/genoxide/examples/xor-neat) plays the run back:
the best network's output over [0, 1]² in each generation.

## Good results

The run of `output.txt` solved XOR in generation 68, after 9,539 evaluations, with a network of 3
hidden nodes and 12 enabled connections. Its outputs are on the right side of 0.5, two of them
barely: the criterion is the paper's, and later generations push them apart.

Over 100 runs (seeds 1 to 100), all 100 solved it, in 39 generations in the median and 47 on
average (245 at most), after 6,719 evaluations on average, with 2.95 hidden nodes and 10.7 enabled
connections on average. The paper reports 32 generations and 4,755 evaluations on average, no
failure in 100 runs, and 2.35 hidden nodes: genoxide's runs take about 1.4 times as long and grow
slightly larger networks. The details the paper leaves out, such as the size of weight perturbations
and which members breed, can account for the difference.
