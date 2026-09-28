---
title: Neuroevolution on the GPU
category: engine
summary: Evaluate a whole generation of neural networks at once on the GPU, with wgpu.
reference: null
reference_url: null
optimum: null
languages: [rust]
order: 160
trace_note: "Recorded from another run like the one below: its times and values differ."
---

# Neuroevolution on the GPU

## The problem

This example is about the engine: it evaluates a whole generation on the GPU. The task is to fit a
small neural network to a function of two variables, sin(2x) cos(y), sampled at 4,096 points of a 64
× 64 grid over [−2, 2]². The network has 2 inputs, 16 hidden tanh units and a linear output:

```text
output = c + Σ vⱼ tanh(aⱼ x + bⱼ y + dⱼ)    over the 16 hidden units j
```

The fitness is the mean squared error between the network's output and the function, over the 4,096
samples.

There's no Python version: the example is a Rust crate of its own, with wgpu. In Python,
`batch=True` hands a whole generation to the fitness function, which can pass it to any GPU library.

## What makes it hard

The cost of the fitness. One evaluation runs the network on 4,096 samples. A generation of 512
networks takes about 2 million network evaluations. The work is the same for every network and every
sample, and independent: that suits a GPU, which runs thousands of such computations at once.

## Representation

A `Real` genome of 65 genes in [−3, 3]: per hidden unit, its two input weights and its bias (48
genes), then the 16 output weights, then the output bias. The network is decoded from the genome for
every evaluation, on the CPU or on the GPU.

## Algorithm

A genetic algorithm with a population of 512, tournaments of size 3, simulated binary crossover with
η = 15, and polynomial mutation with η = 20 at a rate of 2/65 per gene, two genes per child on
average. It runs for 300 generations, twice:

- on the CPU, with `Engine::parallel(true)`: rayon spreads the generation's evaluations over every
  core, in double precision;
- on the GPU, through `Batch`, which scores a whole generation in one call. The call uploads the
  weights, dispatches a WGSL compute shader with [wgpu](https://wgpu.rs/), and downloads the 512
  errors. Each genome gets a workgroup of 256 threads; each thread takes a share of the samples, and
  the workgroup adds their errors up. The GPU computes in single precision.

The example is a crate of its own, since it depends on wgpu, and is Rust only. Run it with:

```sh
cargo run --release --manifest-path examples/gpu/Cargo.toml
```

## Output

The first line names the GPU; without one, the example says so and stops. The second gives the size
of the task. Then there is a line per run: its time, and the best network's error. The GPU line also
gives that network's error recomputed on the CPU in double precision, which shows how much single
precision changed it.

The times depend on the machine: each run prints other numbers.

[The project page](https://tachsin.gr/projects/genoxide/examples/gpu) plays back another run, recorded the same way: its times and values differ from
the ones above.

## Good results

The function isn't known to be exactly representable by this network, so there is no known optimum.
For scale, the samples have a variance of 0.177: a network that outputs 0 everywhere has that error.
In one run, both versions reached an error of 0.0115, which leaves 6.5% of the variance unexplained.
On that machine, with an NVIDIA GeForce RTX 4060 and 20 CPU threads, the GPU run took 1.10 s and the
CPU run 5.53 s.
