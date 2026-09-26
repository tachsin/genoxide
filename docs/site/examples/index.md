# Examples

Each example runs as it is: the code on these pages is the code in the repository, which CI
compiles and runs. Choosing Rust or Python on one page selects it on every page.

- [OneMax](one-max.md): a binary genome and a genetic algorithm
- [Knapsack](knapsack.md): a constraint
- [N-Queens](n-queens.md): a permutation genome
- [Rastrigin](rastrigin.md): a real-valued function to minimize
- [ZDT1](zdt1.md): two objectives with NSGA-II
- [Asynchronous evaluation](asynchronous.md): a fitness function that takes a varying time (Rust)
- [GPU](gpu.md): a generation evaluated at once on the GPU (Rust)

Run a Rust example with `cargo run --release --example <name>` from the root of the repository,
and a Python one with `python python/examples/<name>.py` after `pip install genoxide`.
