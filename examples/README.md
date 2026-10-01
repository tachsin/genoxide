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
| [N-Queens 8×8](n_queens_8/) | permutation | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/n-queens-8) |
| [N-Queens 16×16](n_queens_16/) | permutation | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/n-queens-16) |
| [N-Queens 32×32](n_queens_32/) | permutation | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/n-queens-32) |
| [N-Queens 64×64](n_queens/) | permutation | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/n-queens) |
| [N-Queens 128×128](n_queens_128/) | permutation | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/n-queens-128) |
| [Travelling salesman (berlin52)](tsp_berlin52/) | permutation | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/tsp-berlin52) |
| [Job shop scheduling (ft06)](jobshop_ft06/) | permutation | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/jobshop-ft06) |
| [Sphere](sphere/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/sphere) |
| [Axis-parallel ellipsoid](axis_parallel_ellipsoid/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/axis-parallel-ellipsoid) |
| [Schwefel 1.2](schwefel_1_2/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/schwefel-1-2) |
| [Schwefel 2.21](schwefel_2_21/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/schwefel-2-21) |
| [Schwefel 2.22](schwefel_2_22/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/schwefel-2-22) |
| [Zakharov](zakharov/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/zakharov) |
| [Trid](trid/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/trid) |
| [Dixon-Price](dixon_price/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/dixon-price) |
| [Powell](powell/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/powell) |
| [Rosenbrock](rosenbrock/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/rosenbrock) |
| [Levy](levy/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/levy) |
| [Styblinski-Tang](styblinski_tang/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/styblinski-tang) |
| [Michalewicz](michalewicz/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/michalewicz) |
| [Ackley](ackley/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/ackley) |
| [Rastrigin function](rastrigin/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/rastrigin) |
| [Griewank](griewank/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/griewank) |
| [Schwefel 2.26](schwefel_2_26/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/schwefel-2-26) |
| [Himmelblau's function](himmelblau/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/himmelblau) |
| [Branin](branin/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/branin) |
| [Goldstein-Price](goldstein_price/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/goldstein-price) |
| [Six-hump camel](six_hump_camel/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/six-hump-camel) |
| [Three-hump camel](three_hump_camel/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/three-hump-camel) |
| [Beale](beale/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/beale) |
| [Booth](booth/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/booth) |
| [Matyas](matyas/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/matyas) |
| [Bohachevsky 1](bohachevsky1/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/bohachevsky1) |
| [Bohachevsky 2](bohachevsky2/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/bohachevsky2) |
| [Bohachevsky 3](bohachevsky3/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/bohachevsky3) |
| [Hartmann 3-D](hartmann3/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/hartmann3) |
| [Hartmann 6-D](hartmann6/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/hartmann6) |
| [Shekel 5](shekel5/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/shekel5) |
| [Shekel 7](shekel7/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/shekel7) |
| [Shekel 10](shekel10/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/shekel10) |
| [Shekel's foxholes](shekel_foxholes/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/shekel-foxholes) |
| [Langermann](langermann/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/langermann) |
| [Kowalik](kowalik/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/kowalik) |
| [Easom](easom/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/easom) |
| [Eggholder](eggholder/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/eggholder) |
| [Schaffer F6](schaffer_f6/) | continuous | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/schaffer-f6) |
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
| [CEC 2006 g19](cec2006_g19/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g19) |
| [CEC 2006 g20](cec2006_g20/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g20) |
| [CEC 2006 g21](cec2006_g21/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g21) |
| [CEC 2006 g22](cec2006_g22/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g22) |
| [CEC 2006 g23](cec2006_g23/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g23) |
| [CEC 2006 g24](cec2006_g24/) | constrained | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cec2006-g24) |
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
| [CTP1](ctp1/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/ctp1) |
| [CTP2](ctp2/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/ctp2) |
| [CTP3](ctp3/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/ctp3) |
| [CTP4](ctp4/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/ctp4) |
| [CTP5](ctp5/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/ctp5) |
| [CTP6](ctp6/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/ctp6) |
| [CTP7](ctp7/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/ctp7) |
| [CTP8](ctp8/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/ctp8) |
| [C1-DTLZ1 with 3 objectives](c1_dtlz1_3obj/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/c1-dtlz1-3obj) |
| [C1-DTLZ3 with 3 objectives](c1_dtlz3_3obj/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/c1-dtlz3-3obj) |
| [C2-DTLZ2 with 3 objectives](c2_dtlz2_3obj/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/c2-dtlz2-3obj) |
| [Convex C2-DTLZ2 with 3 objectives](convex_c2_dtlz2_3obj/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/convex-c2-dtlz2-3obj) |
| [C3-DTLZ1 with 3 objectives](c3_dtlz1_3obj/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/c3-dtlz1-3obj) |
| [C3-DTLZ4 with 3 objectives](c3_dtlz4_3obj/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/c3-dtlz4-3obj) |
| [Convex DTLZ2 with 3 objectives](convex_dtlz2/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/convex-dtlz2) |
| [Scaled DTLZ1 with 3 objectives](scaled_dtlz1/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/scaled-dtlz1) |
| [Scaled DTLZ2 with 3 objectives](scaled_dtlz2/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/scaled-dtlz2) |
| [Inverted DTLZ1 with 3 objectives](inverted_dtlz1/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/inverted-dtlz1) |
| [MW1](mw1/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/mw1) |
| [MW2](mw2/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/mw2) |
| [MW3](mw3/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/mw3) |
| [MW4](mw4/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/mw4) |
| [MW5](mw5/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/mw5) |
| [MW6](mw6/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/mw6) |
| [MW7](mw7/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/mw7) |
| [MW8](mw8/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/mw8) |
| [MW9](mw9/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/mw9) |
| [MW10](mw10/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/mw10) |
| [MW11](mw11/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/mw11) |
| [MW12](mw12/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/mw12) |
| [MW13](mw13/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/mw13) |
| [MW14](mw14/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/mw14) |
| [Two-bar truss](two_bar_truss/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/two-bar-truss) |
| [Welded beam, two objectives](welded_beam_2obj/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/welded-beam-2obj) |
| [Disc brake](disc_brake/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/disc-brake) |
| [Speed reducer, two objectives](speed_reducer_2obj/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/speed-reducer-2obj) |
| [Four-bar truss](four_bar_truss/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/four-bar-truss) |
| [Car side impact, three objectives](car_side_impact_3obj/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/car-side-impact-3obj) |
| [Rocket injector](rocket_injector/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/rocket-injector) |
| [Vehicle crashworthiness](vehicle_crashworthiness/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/vehicle-crashworthiness) |
| [Water resource planning](water_resource_planning/) | multi-objective | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/water-resource-planning) |
| [XOR neuroevolution](xor_neuroevolution/) | neuroevolution | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/xor-neuroevolution) |
| [XOR by NEAT](xor_neat/) | neuroevolution | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/xor-neat) |
| [Cart-pole](cart_pole/) | neuroevolution | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/cart-pole) |
| [Double pole balancing](double_pole/) | neuroevolution | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/double-pole) |
| [Double pole balancing without velocities](double_pole_no_velocities/) | neuroevolution | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/double-pole-no-velocities) |
| [Two spirals](two_spirals/) | neuroevolution | Rust, Python | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/two-spirals) |
| [Koza's quartic](koza_quartic/) | genetic programming | Rust | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/koza-quartic) |
| [Koza's 11-multiplexer](multiplexer_11/) | genetic programming | Rust | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/multiplexer-11) |
| [\|x\| by strongly typed GP](abs_typed/) | genetic programming | Rust | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/abs-typed) |
| [Nguyen-1](nguyen_1/) | genetic programming | Rust | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/nguyen-1) |
| [Nguyen-5](nguyen_5/) | genetic programming | Rust | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/nguyen-5) |
| [Nguyen-9](nguyen_9/) | genetic programming | Rust | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/nguyen-9) |
| [Nguyen-1 to 12](nguyen_all/) | genetic programming | Rust | |
| [Accuracy against size](accuracy_and_size/) | genetic programming | Rust | [tachsin.gr](https://tachsin.gr/projects/genoxide/examples/accuracy-and-size) |
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
