---
title: Neuroevolution on the GPU
category: engine
summary: Evaluate a whole generation of neural networks at once on the GPU, with wgpu.
reference: null
reference_url: null
optimum: null
languages: [rust]
order: 120
---

# Neuroevolution on the GPU

This example evolves the 65 weights of a small neural network (2 inputs, 16 hidden tanh units, 1
output) to fit a function sampled at 4,096 points: about 2 million network evaluations per
generation. Through `Batch`, each generation goes to the GPU at once, with a WGSL compute shader
and [wgpu](https://wgpu.rs/): one upload of the weights, one dispatch, one download of the errors.
It runs the same search on every CPU core and on the GPU, and prints the time and error of both.
It is Rust only, and a crate of its own with wgpu as a dependency (the code is in `src/main.rs`):
run it with `cargo run --release --manifest-path examples/gpu/Cargo.toml`.
