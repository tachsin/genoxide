---
title: Neuroevolution on the GPU
category: engine
summary: Evaluate a whole generation of neural networks at once on the GPU, with wgpu.
reference: null
reference_url: null
optimum: "none known; the best fit found by gradient descent: 4.4e-9 (mean squared error)"
languages: [rust]
order: 270
trace_note: "The CMA-ES run of output.txt, on an NVIDIA GeForce RTX 4060."
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

The search. The 65 weights interact: an output weight only matters through its hidden unit's input
weights and bias, and swapping two hidden units gives the same network. A good fit takes hundreds of
thousands of evaluations.

## Representation

A `Real` genome of 65 genes in [−3, 3]: per hidden unit, its two input weights and its bias (48
genes), then the 16 output weights, then the output bias. The network is decoded from the genome for
every evaluation, on the CPU or on the GPU.

## Algorithm

The main run is CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), which
genoxide's guide recommends for continuous problems of up to a few hundred genes whose genes
interact. It samples a population from a normal distribution, and adapts its mean, step size and
covariance matrix to the steps that worked, which suits the correlated weights of a network. It has
a population of 128 instead of the default 4 + ⌊3 ln 65⌋ = 16, since a batch of 16 networks leaves
most of the GPU idle: with 16, seed 1 took about as many evaluations, 249,000, but 9.5 s, most of
them spent on the round trips of small batches. The initial step size is
genoxide's default, 0.3 of each gene's range. IPOP restarts (Auger and Hansen, 2005) start again
from a random point with twice the population if a run converges before the target. The run stops
at an error of 10⁻⁵, or after 2,000,000 evaluations.

Every generation goes to the GPU through `Batch`, which scores a whole generation in one call. The
call uploads the weights, dispatches a WGSL compute shader with [wgpu](https://wgpu.rs/), and
downloads the errors. Each genome gets a workgroup of 256 threads; each thread takes a share of the
samples, and the workgroup adds their errors up. The GPU computes in single precision.

Then, as a contrast, a genetic algorithm runs 300 generations of 512 networks, with tournaments of
size 3, simulated binary crossover with η = 15, and polynomial mutation with η = 20 at a rate of
2/65 per gene, two genes per child on average. It runs twice, with the same seed:

- on the GPU, through the same `Batch`;
- on the CPU, with `Engine::parallel(true)`: rayon spreads the generation's evaluations over every
  core, in double precision.

The two runs compare the speed of the evaluations; the genetic algorithm isn't meant to fit the
network well.

The example is a crate of its own, since it depends on wgpu, and is Rust only. Run it with:

```sh
cargo run --release --manifest-path examples/gpu/Cargo.toml
```

## Output

The first line names the GPU; without one, the example says so and stops. The second gives the size
of the task. Then there is a line per run: its time, its evaluations, and the best network's error.
Under the CMA-ES line, that network's error is recomputed on the CPU in double precision, which
shows how much single precision changed it.

CI only compiles this example, since it has no GPU: `output.txt` comes from a run on an NVIDIA
GeForce RTX 4060, with 20 CPU threads. The times depend on the machine, and change from run to run.

[The project page](https://tachsin.gr/projects/genoxide/examples/gpu) plays back the CMA-ES run of
`output.txt`.

## Good results

The function isn't known to be exactly representable by this network, so its best fit isn't known.
For scale, the samples have a variance of 0.177: a network that outputs 0 everywhere has that error.
Gradient descent, which genoxide doesn't do, gives a reference: L-BFGS-B with the weights in the
same [−3, 3], from 6 random starts and up to 100,000 iterations each, reached errors of 4.4 × 10⁻⁹
to 1.2 × 10⁻⁶, the best 4.4 × 10⁻⁹, in several minutes each.

CMA-ES gets to 10⁻⁵, about 0.006% of the variance: in the run of `output.txt`, after 296,704
evaluations and 2.90 s, with the same error in double precision. Over seeds 1 to 20 on the RTX 4060,
every run got there, after 243,000 to 640,000 evaluations (289,000 at the median), in 2.3 to 6
seconds. Getting further is slow: two runs aimed at 10⁻⁶ got there after 575,000 and 1,600,000
evaluations, 7 and 18 seconds. The best fits of gradient descent, a thousand times smaller than
10⁻⁶, are out of reach of CMA-ES in seconds: it doesn't reach the best fit known.

The genetic algorithm stops at 0.0115 after its 300 generations, over a thousand times the error of
CMA-ES with half the evaluations. On the GPU it took 0.55 s, against 4.35 s on every CPU core, eight
times as long.
