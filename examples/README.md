# Examples

Each example is a folder with the same program in Rust (`main.rs`) and Python (`main.py`), and a
`README.md` that describes the problem, cites its source and gives the known optimum. The
README's YAML front matter (`title`, `category`, `summary`, `reference`, `reference_url`,
`optimum`, `languages`, `order`) is what the [docs site](https://tachsin.github.io/genoxide/)
builds its example pages from. CI runs every example in both languages.

Both versions use the same algorithm, settings and seed, and print the same output. The exception
is a fitness function in numpy with functions such as cosine, whose last bit can differ from
Rust's: the runs then drift apart. The test problems of `genoxide.problems` are evaluated in Rust
in both languages, so their runs don't.

Run a Rust example from the root of the repository:

```sh
cargo run --release --example tsp_berlin52
```

and a Python one after `pip install genoxide`:

```sh
python examples/tsp_berlin52/main.py
```

| Example | Category | Languages |
|---|---|---|
| [OneMax](one_max/) | binary | Rust, Python |
| [0/1 knapsack](knapsack/) | constrained | Rust, Python |
| [N-Queens](n_queens/) | permutation | Rust, Python |
| [Travelling salesman (berlin52)](tsp_berlin52/) | permutation | Rust, Python |
| [Job shop scheduling (ft06)](jobshop_ft06/) | permutation | Rust, Python |
| [Rastrigin function](rastrigin/) | continuous | Rust, Python |
| [Function suite](function_suite/) | continuous | Rust, Python |
| [Himmelblau's function](himmelblau/) | continuous | Rust, Python |
| [Pressure vessel design](pressure_vessel/) | constrained | Rust, Python |
| [ZDT1](zdt1/) | multi-objective | Rust, Python |
| [DTLZ2 with 3 objectives](dtlz2_3obj/) | multi-objective | Rust, Python |
| [XOR neuroevolution](xor_neuroevolution/) | neuroevolution | Rust, Python |
| [Asynchronous evaluation](asynchronous/) | engine | Rust |
| [Neuroevolution on the GPU](gpu/) | engine | Rust |

The GPU example is a crate of its own, with wgpu as a dependency:
`cargo run --release --manifest-path examples/gpu/Cargo.toml`.

## Adding an example

Add a folder `examples/<name>/` with `main.rs`, `main.py` (unless the Python package lacks a
feature it needs, which its README says) and a `README.md` with the front matter, and a row to the
table above. Cargo finds `main.rs` as the example `<name>`, CI runs every `main.rs` and `main.py`,
and the docs site makes a page for it, whose build fails if the table lacks the example. The
categories are binary, permutation, integer, continuous, multi-objective, constrained,
neuroevolution and engine.
