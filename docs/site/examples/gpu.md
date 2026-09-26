# GPU

This example evolves the 65 weights of a small neural network (2 inputs, 16 hidden tanh units,
1 output) to fit a function sampled at 4,096 points: about 2 million network evaluations per
generation. Through `Batch`, each generation goes to the GPU at once, with a WGSL compute shader
and [wgpu](https://wgpu.rs/): one upload of the weights, one dispatch, one download of the errors.
It runs the same search on every CPU core and on the GPU, and prints the time and error of both.

This example is Rust only: the Python package doesn't have it.

The example is a crate of its own, with wgpu as a dependency:
[examples/gpu](https://github.com/tachsin/genoxide/tree/main/examples/gpu), with the code in
[src/main.rs](https://github.com/tachsin/genoxide/blob/main/examples/gpu/src/main.rs). To run it:

```sh
cargo run --release --manifest-path examples/gpu/Cargo.toml
```
