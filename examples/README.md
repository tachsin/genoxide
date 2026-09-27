# Examples

Each example is a folder with the same program in Rust (`main.rs`) and Python (`main.py`), and a
`README.md` that describes the problem, cites its source and gives the known optimum. The
README's YAML front matter (`title`, `category`, `summary`, `reference`, `reference_url`,
`optimum`, `languages`, `order`) is what the [docs site](https://tachsin.github.io/genoxide/)
builds its example pages from. CI runs every example in both languages.

Both versions use the same algorithm, settings and seed, and print the same output. The exception
is a fitness function in numpy with functions such as cosine, whose last bit can differ from
Rust's: the runs then drift apart. The test problems of `genoxide.problems` are evaluated in Rust
in both languages, so on one platform their runs don't. They call the platform's `sin`, `cos` and
`exp`, though, so between operating systems a long run can drift too: `output.txt` is the Linux
output.

Run a Rust example from the root of the repository:

```sh
cargo run --release --example tsp_berlin52
```

and a Python one after `pip install genoxide`:

```sh
python examples/tsp_berlin52/main.py
```

| Example | Category | Languages | Interactive run |
|---|---|---|---|
| [OneMax](one_max/) | binary | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/one-max) |
| [0/1 knapsack](knapsack/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/knapsack) |
| [N-Queens](n_queens/) | permutation | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/n-queens) |
| [Travelling salesman (berlin52)](tsp_berlin52/) | permutation | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/tsp-berlin52) |
| [Job shop scheduling (ft06)](jobshop_ft06/) | permutation | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/jobshop-ft06) |
| [Rastrigin function](rastrigin/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/rastrigin) |
| [Function suite](function_suite/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/function-suite) |
| [Himmelblau's function](himmelblau/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/himmelblau) |
| [Branin, Goldstein-Price and the six-hump camel](minima_2d/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/minima-2d) |
| [Pressure vessel design](pressure_vessel/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/pressure-vessel) |
| [Welded beam design](welded_beam/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/welded-beam) |
| [Gear train design](gear_train/) | integer | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/gear-train) |
| [ZDT1](zdt1/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/zdt1) |
| [Classic two-objective fronts](classic_fronts/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/classic-fronts) |
| [BNH, a constrained two-objective problem](bnh/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/bnh) |
| [Constrained two-objective fronts](constrained_fronts/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/constrained-fronts) |
| [Kursawe's disconnected front](kursawe/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/kursawe) |
| [DTLZ2 with 3 objectives](dtlz2_3obj/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/dtlz2-3obj) |
| [XOR neuroevolution](xor_neuroevolution/) | neuroevolution | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/xor-neuroevolution) |
| [Asynchronous evaluation](asynchronous/) | engine | Rust | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/asynchronous) |
| [Neuroevolution on the GPU](gpu/) | engine | Rust | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/gpu) |

The GPU example is a crate of its own, with wgpu as a dependency:
`cargo run --release --manifest-path examples/gpu/Cargo.toml`.

## Adding an example

Add a folder `examples/<name>/` with `main.rs`, `main.py` (unless the Python package lacks a
feature it needs, which its README says) and a `README.md` with the front matter, and a row to the
table above. Cargo finds `main.rs` as the example `<name>`, CI runs every `main.rs` and `main.py`,
and the docs site makes a page for it, whose build fails if the table lacks the example. The
categories are binary, permutation, integer, continuous, multi-objective, constrained,
neuroevolution and engine.
