# genoxide

genoxide is a library for evolutionary computation, written in Rust and available in Python.
It has genetic algorithms, evolution strategies, CMA-ES, differential evolution, particle swarms,
local search, and multi-objective algorithms such as NSGA-II. Runs evaluate in parallel, in
batches or asynchronously, and a seed makes them reproducible. In Python, fitness functions can
take a whole generation as a numpy array.

## Install

=== "Rust"

    ```sh
    cargo add genoxide
    ```

=== "Python"

    ```sh
    pip install genoxide
    ```

## Links

- [Examples](examples/index.md), in Rust and Python
- [Python API](api/python/): the reference of the Python package
- [Rust API](https://docs.rs/genoxide) on docs.rs
- [GitHub](https://github.com/tachsin/genoxide)
- [Benchmarks](https://github.com/tachsin/genoxide/tree/main/docs/benchmarks): rules, methodology and results
