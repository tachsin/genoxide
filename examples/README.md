# Examples

Each example is a folder with the same program in Rust (`main.rs`) and Python (`main.py`), and a
`README.md` that describes the problem, cites its source and gives the known optimum. The
README's YAML front matter (`title`, `category`, `summary`, `reference`, `reference_url`,
`optimum`, `languages`, `order`) is what the [docs site](https://tachsin.github.io/genoxide/)
builds its example pages from. CI runs every example in both languages. The problems of one paper
(ZDT, DTLZ, WFG, CEC 2006...) share an optional `family`, all in one category: the
[project page](https://tachsin.gr/projects/genoxide/examples) shows them as tabs of one another,
labelled by the optional `tab` (the title when absent), and the docs site's navigation in a section
of their own.

Both versions use the same algorithm, settings and seed, and print the same output. The exception
is a fitness function in numpy with functions such as cosine, whose last bit can differ from
Rust's: the runs then drift apart. The test problems of `genoxide.problems` are evaluated in Rust
in both languages, with genoxide's portable [`math`](https://docs.rs/genoxide/latest/genoxide/math/),
so their runs don't, on any platform.

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
| [Sphere](sphere/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/sphere) |
| [Axis-parallel ellipsoid](axis_parallel_ellipsoid/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/axis-parallel-ellipsoid) |
| [Schwefel 1.2](schwefel_1_2/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/schwefel-1-2) |
| [Zakharov](zakharov/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/zakharov) |
| [Rosenbrock](rosenbrock/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/rosenbrock) |
| [Levy](levy/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/levy) |
| [Styblinski-Tang](styblinski_tang/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/styblinski-tang) |
| [Michalewicz](michalewicz/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/michalewicz) |
| [Ackley](ackley/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/ackley) |
| [Rastrigin function](rastrigin/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/rastrigin) |
| [Griewank](griewank/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/griewank) |
| [Schwefel 2.26](schwefel_2_26/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/schwefel-2-26) |
| [Function suite](function_suite/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/function-suite) |
| [Himmelblau's function](himmelblau/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/himmelblau) |
| [Branin](branin/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/branin) |
| [Goldstein-Price](goldstein_price/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/goldstein-price) |
| [Six-hump camel](six_hump_camel/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/six-hump-camel) |
| [Pressure vessel design](pressure_vessel/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/pressure-vessel) |
| [Welded beam design](welded_beam/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/welded-beam) |
| [Tension/compression spring](tension_compression_spring/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/tension-compression-spring) |
| [Speed reducer](speed_reducer/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/speed-reducer) |
| [Three-bar truss](three_bar_truss/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/three-bar-truss) |
| [Cantilever beam](cantilever_beam/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cantilever-beam) |
| [Car side impact](car_side_impact/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/car-side-impact) |
| [CEC 2006 g01](cec2006_g01/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g01) |
| [CEC 2006 g02](cec2006_g02/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g02) |
| [CEC 2006 g03](cec2006_g03/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g03) |
| [CEC 2006 g04](cec2006_g04/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g04) |
| [CEC 2006 g05](cec2006_g05/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g05) |
| [CEC 2006 g06](cec2006_g06/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g06) |
| [CEC 2006 g07](cec2006_g07/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g07) |
| [CEC 2006 g08](cec2006_g08/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g08) |
| [CEC 2006 g09](cec2006_g09/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g09) |
| [CEC 2006 g10](cec2006_g10/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g10) |
| [CEC 2006 g11](cec2006_g11/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g11) |
| [CEC 2006 g12](cec2006_g12/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g12) |
| [CEC 2006 g13](cec2006_g13/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g13) |
| [CEC 2006 g14](cec2006_g14/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g14) |
| [CEC 2006 g15](cec2006_g15/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g15) |
| [CEC 2006 g16](cec2006_g16/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g16) |
| [CEC 2006 g17](cec2006_g17/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g17) |
| [CEC 2006 g18](cec2006_g18/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g18) |
| [Gear train design](gear_train/) | integer | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/gear-train) |
| [ZDT1](zdt1/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/zdt1) |
| [ZDT2](zdt2/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/zdt2) |
| [ZDT3](zdt3/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/zdt3) |
| [ZDT4](zdt4/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/zdt4) |
| [ZDT5](zdt5/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/zdt5) |
| [ZDT6](zdt6/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/zdt6) |
| [Schaffer 1](schaffer1/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/schaffer1) |
| [Schaffer 2](schaffer2/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/schaffer2) |
| [Fonseca-Fleming](fonseca_fleming/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/fonseca-fleming) |
| [Kursawe's disconnected front](kursawe/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/kursawe) |
| [Poloni](poloni/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/poloni) |
| [BNH, a constrained two-objective problem](bnh/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/bnh) |
| [SRN (Srinivas and Deb)](srn/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/srn) |
| [TNK (Tanaka)](tnk/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/tnk) |
| [OSY (Osyczka and Kundu)](osy/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/osy) |
| [CONSTR](constr/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/constr) |
| [WFG1](wfg1/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/wfg1) |
| [WFG2](wfg2/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/wfg2) |
| [WFG3](wfg3/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/wfg3) |
| [WFG4](wfg4/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/wfg4) |
| [WFG5](wfg5/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/wfg5) |
| [WFG6](wfg6/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/wfg6) |
| [WFG7](wfg7/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/wfg7) |
| [WFG8](wfg8/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/wfg8) |
| [WFG9](wfg9/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/wfg9) |
| [DTLZ1 with 3 objectives](dtlz1_3obj/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/dtlz1-3obj) |
| [DTLZ2 with 3 objectives](dtlz2_3obj/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/dtlz2-3obj) |
| [DTLZ3 with 3 objectives](dtlz3_3obj/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/dtlz3-3obj) |
| [DTLZ4 with 3 objectives](dtlz4_3obj/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/dtlz4-3obj) |
| [DTLZ5 with 3 objectives](dtlz5_3obj/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/dtlz5-3obj) |
| [DTLZ6 with 3 objectives](dtlz6_3obj/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/dtlz6-3obj) |
| [DTLZ7 with 3 objectives](dtlz7_3obj/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/dtlz7-3obj) |
| [Viennet 1](viennet1/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/viennet1) |
| [Viennet 2](viennet2/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/viennet2) |
| [Viennet 3](viennet3/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/viennet3) |
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
