# Plan: genetic programming and neuroevolution

A working plan for milestone 0.11, removed when the work is done. Nothing in it is implemented yet.
genoxide evolves fixed-length genomes (bits, integers, reals, permutations) with GAs, evolution
strategies, CMA-ES, DE, PSO, local search and the multi-objective algorithms. This plan adds
programs and networks as genomes: tree-based genetic programming (strongly typed), symbolic
regression, NEAT, and neuroevolution of fixed networks with evolution strategies; and it closes the
Python package's gaps that ROADMAP.md lists for 0.11. It decides the representations, designs the
API on genoxide's traits, sets how everything is tested, and orders the work in batches.

**Rules.**

- As in [optimization-plan.md](optimization-plan.md): every method is implemented from its paper
  or book, and its docs cite them. Nothing is copied from another library: no code, no text, no
  default values taken on trust, no expected test values. Other libraries appear in this plan to
  learn what users expect (section 1.2) and later in the benchmark suite, not as sources.
- The guarantees of AGENTS.md hold for everything new: a seed gives the same results to the bit on
  every platform and thread count; errors, not panics; validated builders; ask / tell at the core;
  every example reaches its optimum.
- **Status of the citations:** every reference was recorded from the literature while planning,
  not re-read (**U** in the problem plan's terms). The settings quoted from papers (NEAT's
  coefficients, the pole-balancing constants, the benchmark sampling rules) are marked "to check"
  where it matters. Before a batch starts, the primary sources of its methods and problems are read
  and each entry is marked verified or corrected here.
- The methods are general. Docs and examples motivate them with generic cases (model discovery
  from data, controllers, classifiers), not with an application domain.

**Contents.**

1. [Scope](#1-scope)
2. [Tree genetic programming](#2-tree-genetic-programming)
3. [Symbolic regression](#3-symbolic-regression)
4. [NEAT](#4-neat)
5. [Neuroevolution with evolution strategies](#5-neuroevolution-with-evolution-strategies)
6. [Python](#6-python)
7. [Reproducibility and the guarantees](#7-reproducibility-and-the-guarantees)
8. [Tests and validation](#8-tests-and-validation)
9. [Examples](#9-examples)
10. [Benchmarks](#10-benchmarks)
11. [Order of work](#11-order-of-work)
12. [Open questions](#12-open-questions)

## 1. Scope

### 1.1 What genoxide has, and what's missing

| Area | genoxide today | Missing for 0.11 |
|---|---|---|
| Genomes | `Bits`, `Integers`, `Reals`, `Order`, `AdaptiveReals`: fixed length | Variable-length, structured genomes: trees, networks |
| Operators | `Select`, `Crossover<R>`, `Mutate<R>`, generic over the representation | Tree operators; selection that sees size (parsimony) |
| Neuroevolution | The `xor_neuroevolution` example (a 2-2-1 network's 9 weights, CMA-ES) and the `gpu` example; the network decoded by hand in the fitness function | A network module, control tasks, NEAT, an ES made for many weights |
| Python | `Ga`, `De`, `Cmaes`, `Pso`, `LocalSearch`, NSGA-II, NSGA-III, SPEA2, MOEA/D, SMS-EMOA; a stateless `run(config)` | `Es` (and `AdaptiveReal`), `Islands`, checkpoints, the asynchronous engine, memetic and initial genomes; a genome copied into a new numpy array per call; GP and NEAT |

What fits already, and shapes the design: `Ga` is generic over `R: Representation` and never
assumes a fixed length (only DE's L-SHADE reads `genome_len`), so a tree representation with its
own `Crossover` and `Mutate` runs on `Ga`, `SteadyGa`, `AsyncEngine`, `Islands` of GAs, NSGA-II
and the other multi-objective algorithms, and `LocalSearch`, with checkpoints, parallel
evaluation, parallel breeding, `control` and observers, without new engine code. `Select::select`
receives the whole `Population<G>`, and `Genome::len()` is the number of genes, so a selection
operator can see each individual's size without an API change (section 2.6). A child equal to a
parent inherits its fitness, which for trees is a comparison of two flat arrays.

### 1.2 Prior art

What each does that matters here, from their docs and papers (U); nothing is taken over as code or
defaults.

| Library | Relevant design | What to learn |
|---|---|---|
| DEAP `gp` (Fortin, F.-A., De Rainville, F.-M., Gardner, M.-A., Parizeau, M. and Gagné, C. (2012). DEAP: evolutionary algorithms made easy. *JMLR* 13: 2171-2175) | `PrimitiveTree`: a flat list in prefix order; `PrimitiveSetTyped` for STGP; ephemeral constants; grow, full, half-and-half; one-point (subtree) crossover, a leaf-biased variant, uniform, node-replacement, insert, shrink and ephemeral mutations; `staticLimit`; double tournament. Trees are compiled to Python source and `eval`ed | The flat prefix list works for every operator; typed sets as data; the Python compile step is its bottleneck |
| gplearn (T. Stephens, github.com/trevorstephens/gplearn) | Prefix lists; a named function set with protected division, log and sqrt; subtree, hoist and point mutation; a parsimony coefficient, optionally set by covariance (Poli and McPhee 2008); each node evaluated on all samples at once with numpy | Vectorized evaluation over the data; hoist mutation as the bloat-reducing move; users expect `fit(X, y)` and a readable expression |
| Karoo GP (Staats, K., Pantridge, E., Cavaglia, M., Milovanov, I. and Aniyan, A. (2017). TensorFlow enabled genetic programming. GECCO 2017 companion: 1872-1879. doi:10.1145/3067695.3084216) | Evaluation of whole datasets through TensorFlow | Again: evaluate over all points at once |
| Operon (Burlacu, B., Kronberger, G. and Kommenda, M. (2020). Operon C++: an efficient genetic programming framework for symbolic regression. GECCO 2020 companion: 1562-1570. doi:10.1145/3377929.3398099) | Postfix arrays with subtree lengths, batched vectorized evaluation, nonlinear least squares for constants | The performance reference for symbolic regression; constant optimization pays off once a least-squares solver exists |
| PySR / SymbolicRegression.jl (Cranmer, M. (2023). Interpretable machine learning for science with PySR and SymbolicRegression.jl. arXiv:2305.01582) | Many populations, a Pareto front of loss against complexity, constant optimization, simplification | Accuracy against size as two objectives: genoxide's NSGA-II gives it for free (section 3.5) |
| PonyGE2 (Fenton, M., McDermott, J., Fagan, D., Forstenlechner, S., Hemberg, E. and O'Neill, M. (2017). PonyGE2: grammatical evolution in Python. GECCO 2017 companion: 1194-1201. doi:10.1145/3067695.3082469); GE: Ryan, C., Collins, J. J. and O'Neill, M. (1998). Grammatical evolution: evolving programs for an arbitrary language. EuroGP 1998, LNCS 1391: 83-96. doi:10.1007/BFb0055930 | Integer genomes mapped through a BNF grammar | Out of 0.11: GE is an `Integer` genome plus a decoder, which users can write today; a grammar module later if asked for (section 1.3) |
| ECJ (Luke, S. (2017). ECJ then and now. GECCO 2017 companion: 1223-1230. doi:10.1145/3067695.3082467) | STGP with type constraints and sets, ERCs, ADFs, Koza's builders, PTC2, the bloat methods of Luke and Panait | The reference for typed generation tables and bloat control as selection |
| TPOT (Olson, R. S., Bartley, N., Urbanowicz, R. J. and Moore, J. H. (2016). Evaluation of a tree-based pipeline optimization tool for automating data science. GECCO 2016: 485-492. doi:10.1145/2908812.2908918) | Strongly typed tree GP (on DEAP) over pipelines of steps, with NSGA-II on accuracy and size | Typed GP's main use beyond formulas: trees of typed components, whose semantics are the user's |
| neat-python (A. McIntyre et al., github.com/CodeReclaimers/neat-python) | Connection genes keyed by (input node, output node) instead of innovation numbers; a bias, activation and aggregation per node; feed-forward, recurrent and CTRNN networks; a config file | Structural keys make innovation numbers a deterministic function of structure; its defaults differ from the paper (section 4.6) |
| SharpNEAT (C. Green, github.com/colgreen/sharpneat) | Acyclic networks compiled to flat arrays in layer order; cyclic networks run for a fixed number of steps; complexity regulation | Compile once per evaluation; keep feed-forward and recurrent activation apart |
| HyperNEAT (Stanley, K. O., D'Ambrosio, D. B. and Gauci, J. (2009). A hypercube-based encoding for evolving large-scale neural networks. *Artificial Life* 15(2): 185-212. doi:10.1162/artl.2009.15.2.15202), CPPNs (Stanley, K. O. (2007). Compositional pattern producing networks. *GPEM* 8(2): 131-162. doi:10.1007/s10710-007-9028-8) | NEAT evolves a CPPN that paints the weights of a large substrate | Out of 0.11; a node gene carries its activation function so CPPNs remain possible (section 4.1) |
| EvoTorch (Toklu, N. E., Atkinson, T., Micka, V., Liskowski, P. and Srivastava, R. K. (2023). EvoTorch: scalable evolutionary computation in Python. arXiv:2302.12600), evosax (Lange, R. T. (2023). evosax: JAX-based evolution strategies. GECCO 2023 companion. arXiv:2212.04180) | Distribution-based ES (PGPE, SNES, XNES, CMA-ES, OpenAI's ES) for network weights, with antithetic sampling, rank-based shaping and gradient optimizers | What neuroevolution with ES needs beyond CMA-ES: an ES whose cost per sample is O(n) with a gradient-style update (section 5) |
| radiate (github.com/pkalivas/radiate, Rust; already in the benchmark suite) | Tree and graph codecs, GP and NEAT-like graph evolution | The one Rust library with both; a candidate for matched GP and NEAT scenarios (section 10) |
| NEAT crates on crates.io (rustneat and others) | Small NEAT implementations | Nothing in Rust combines typed GP, NEAT and reproducibility; their APIs to be checked before batch N1 |

### 1.3 In and out of 0.11

| Item | Decision |
|---|---|
| Tree GP, strongly typed (monomorphic types), ERCs, grow / full / ramped half-and-half | In (G1) |
| Subtree crossover (Koza's 90/10), subtree, point, hoist, shrink and constant mutation | In (G1, G2) |
| One-point crossover (Poli, R. and Langdon, W. B. (1998). Schema theory for genetic programming with one-point crossover and point mutation. *Evolutionary Computation* 6(3): 231-252. doi:10.1162/evco.1998.6.3.231) | Optional in G2: of interest for schema theory, rarely used for search |
| Bloat control: static depth and size limits, lexicographic parsimony, double tournament, Tarpeian | In (G2) |
| Symbolic regression: a math primitive set, vectorized RMSE, linear scaling, the benchmark problems | In (G3, G4) |
| Constant optimization by least squares | Later: needs Levenberg-Marquardt (optimization plan, batch D1, 0.15) |
| Automatically defined functions (Koza, J. R. (1994). *Genetic Programming II.* MIT Press) | Later: a tree per function in one genome; after the single-tree API settles |
| Polymorphic (generic) types of Montana's STGP | Later: monomorphic types cover formulas, typed pipelines and control programs |
| PTC2 (Luke, S. (2000). Two fast tree-creation algorithms for genetic programming. *IEEE TEVC* 4(3): 274-283. doi:10.1109/4235.873237) | Optional in G2: tree sizes by distribution, useful against the depth bias of ramped half-and-half |
| Grammatical evolution, Cartesian GP (Miller, J. F. and Thomson, P. (2000). Cartesian genetic programming. EuroGP 2000, LNCS 1802: 121-132. doi:10.1007/978-3-540-46239-2_9), linear GP (Brameier, M. and Banzhaf, W. (2007). *Linear Genetic Programming.* Springer. doi:10.1007/978-0-387-31030-5) | Out: each is an `Integer` genome and a decoder that users can write today; revisit after 0.11 |
| Geometric semantic GP (Moraglio, A., Krawiec, K. and Johnson, C. G. (2012). Geometric semantic genetic programming. PPSN XII, LNCS 7491: 21-31. doi:10.1007/978-3-642-32937-1_3) | Out |
| Lexicase and ε-lexicase selection (Spector, L. (2012). GECCO 2012 companion: 401-408. doi:10.1145/2330784.2330846; La Cava, W., Spector, L. and Danai, K. (2016). Epsilon-lexicase selection for regression. GECCO 2016: 741-748. doi:10.1145/2908812.2908898) | Later: needs per-case errors in selection, which `Fitness` doesn't carry (open question 10) |
| NEAT: genome, innovations, speciation, sharing, feed-forward and recurrent networks | In (N1, N2) |
| HyperNEAT, CPPNs, novelty search | Out: HyperNEAT after NEAT settles; novelty search is in 0.17 (quality-diversity) |
| Neuroevolution with ES: a network module, control tasks, CMA-ES and sep-CMA-ES examples, an OpenAI-style ES | In (E1, E2) |
| Python: `Es`, islands, checkpoints, fewer numpy copies; GP and NEAT | In (P1, P2) |

## 2. Tree genetic programming

References for the whole section: Koza, J. R. (1992). *Genetic Programming: On the Programming of
Computers by Means of Natural Selection.* MIT Press (**Koza**); Poli, R., Langdon, W. B. and
McPhee, N. F. (2008). *A Field Guide to Genetic Programming.* lulu.com, gp-field-guide.org.uk
(**Field Guide**); Montana, D. J. (1995). Strongly typed genetic programming. *Evolutionary
Computation* 3(2): 199-230. doi:10.1162/evco.1995.3.2.199 (**Montana**).

**Verified for G1** (the settings G1 relies on, read in the sources): Koza's depth limit of 17
for trees made by crossover, the initial depth of 6, ramped half-and-half over depths 2 to 6 with
the population divided evenly among the depths and half full, half grow, the root chosen from the
functions, duplicates in the initial population redrawn, and crossover points at functions 90% of
the time and at terminals (not "any point") 10%: Koza (1994), *Statistics and Computing* 4(2):
87-112, and the text of Koza and Rice's patent US 5,343,554 (filed 1992), which restate the book;
the book itself (lent only) wasn't read, so its section numbers aren't cited. A tree of one node
has depth 0 (Koza; Field Guide sec. 2.2, p. 12); Montana counts it as 1. Koza aborts a crossover
that would exceed the depth limit and copies the parents (open question 1 keeps the recommended
alternative). The Field Guide's grow picks a terminal with probability |T| / (|T| + |F|) (Algorithm
2.1), subtree crossover and mutation are in sec. 2.4 and the mutation cookbook in sec. 5.2.2.
Montana (1995, sec. 2.1): types possibilities tables per depth, the second crossover point of the
first point's type, and the parents returned when there's none. Angeline (1996): GP 1996: 21-29,
as cited.

**Verified for G2** (read in the sources unless marked):

- **Koza's 11-multiplexer:** terminals A0 to A2 and D0 to D7, A0 the low-order bit of the address;
  functions AND, OR, NOT and IF of 2, 2, 1 and 3 arguments; all 2048 cases; standardized fitness
  the number of mismatches, hits the number of matches; M = 4000 and 51 generations; a correct
  program of 37 points in generation 9. Koza (1990), STAN-CS-90-1314, sec. 4.1.1; Koza (1994) as
  above. Correction: the patent that restates the multiplexer is US 5,136,686; US 5,343,554 is the
  one on automatically defined functions, which has even parity.
- **Even parity:** functions AND, OR, NAND and NOR (no NOT), terminals D0 to Dk-1, all 2^k cases,
  M = 4000 (8000 for even-5 without ADFs). Koza (1994, sec. 6.1); US 5,343,554.
- **Montana (1995):** the first crossover point uniform over all nodes, the second uniform over the
  nodes of its type; IF-THEN-ELSE-INT: BOOLEAN, INTEGER, INTEGER → INTEGER (fig. 4); no typed
  comparison in the paper's tables, and sec. 1.2 notes that Koza avoids predicates such as
  LESS-THAN that return Booleans, which the `abs_typed` example has.
- **Lexicographic parsimony pressure** (Luke and Panait 2002, GECCO): a tournament of 2, the fitter
  wins, then the smaller, then a random one. Correction: the bucketed variants are direct bucketing
  (b buckets) and **ratio bucketing** (the worst 1/r of the rest into each bucket, individuals equal
  to a bucket's best into it), not "ranked"; `LexicographicTournament::ratio_buckets(ratio)` is
  ratio bucketing, with each bucket's count rounded up (the paper doesn't say how it rounds). Luke
  and Panait (2006) found plain lexicographic parsimony failing on symbolic regression, and ratio
  1/2 the bucketing to use.
- **Double tournament** (Luke and Panait 2002, PPSN): fitness tournament size S_f, parsimony size
  S_p in [1, 2], two individuals in the parsimony tournament, the smaller winning with probability
  S_p / 2, ties at random, and a do-fitness-first switch of no significant effect; with the depth
  limit, S_p from 1.2 to 1.6 was as fit with trees often half the size. Luke and Panait (2006):
  F = 7 and D = 1.4 consistently the best; their sec. 8 ran fitness first, their sec. 11.2
  recommendation says do-fitness-first false. genoxide keeps fitness first as the default and
  `size_first()` as the option.
- **Tarpeian** (Poli 2003): **only the abstract read**: the fitness of a fixed proportion of the
  offspring of above-average length zeroed. The Field Guide (p. 106) adds that they're never
  executed; Luke and Panait (2006, sec. 4) give the probability as W, with W = 0.3 consistently
  good. Not verified: the paper's name for the rate, and when it computes the average.
- **Field Guide sec. 5.2.2:** point mutation replaces a node by a primitive of the same arity,
  per node, and leaves a node without one; hoist copies a random subtree, which is smaller;
  **shrink replaces a random subtree by a random terminal** (Angeline 1996, as cited: the plan's
  "one of its own subtrees" was another operator), done here at function nodes so the tree always
  shrinks; constants mutated by Gaussian noise, each change a separate mutation (Schoenauer et al.
  1996, as cited, not read).
- **Hoist's origin:** Kinnear (1994, WCCI) uses hoist and defers its definition to Kinnear (1993),
  Evolving a sort: lessons in genetic programming (ICNN 1993), read: a copy of the subtree at a
  **function** point becomes the new individual. genoxide hoists at function nodes below the root,
  not leaves.
- **One-point crossover** (Poli and Langdon 1997, 1998): **only the abstracts read**; the Field
  Guide (sec. 5.3) defines the common region by the same arity from the roots. The uniform choice
  of the point in the region is genoxide's; with types, the region follows functions with the same
  argument types.

### 2.1 Representation

**Decision: a flat array of nodes in prefix order**, one `Vec<Node>` per tree, as DEAP, gplearn
and the Field Guide's TinyGP do; not a pointer tree of `Box`es.

- **Subtrees are contiguous.** A subtree is the range from its root to the end found by one arity
  scan; subtree crossover is two splices, mutation one splice, hoist a copy of a range.
- **The `Genome` bounds come for free.** `Clone`, `Eq` and `Hash` of a `Vec<Node>` are one memcpy,
  one slice comparison, one slice hash, never recursive. A `Box` tree's derived `Clone`, `Drop`,
  `PartialEq`, `Hash` and serde are recursive: a deep enough tree overflows the stack, which is an
  abort, not an error; the depth limit bounds it for trees made by the operators, but not for a
  genome given by the user or deserialized.
- **Serde is a sequence**, as for `Reals`; checkpoints store it compactly.
- **Evaluation needs no compile step.** Read backwards, a prefix array is a postfix program whose
  children come off the stack in order: a stack machine runs the genome directly.
- Prefix over postfix: both keep subtrees contiguous; prefix reads top-down for printing and for
  interpreters that walk the tree from the root (section 2.4), and matches the libraries users know.
  Subtree lengths per node (as Operon stores) are not kept: a scan is O(subtree), and the
  measurements below don't justify the bookkeeping.

**Measurements** (a scratch prototype outside the repository: 1000 random trees of 45 to 55 nodes,
mean 50.0, over {+, −, ×, analytic quotient, sin, cos, x, constant}; 100 points in [−1, 1]; the sum
of squared errors against a target; sin and cos from `genoxide::math` unless noted; Intel Core
Ultra 7 265K, Windows 11, rustc 1.98.1, release, best of 5 runs, not pinned):

| Evaluation of one tree on all points | 100 points | 1000 points |
|---|---|---|
| `Box` tree, recursive, point by point | 9.84 µs | 92.9 µs |
| Prefix array, recursive, point by point | 11.0 µs | 103.8 µs |
| Prefix array read backwards (stack machine), point by point | 9.38 µs | 88.4 µs |
| `Box` tree compiled to postfix per evaluation, then a stack machine | 9.85 µs | 90.6 µs |
| **Prefix array read backwards, vectorized: a column of all points per stack entry** | **3.04 µs** | **26.9 µs** |
| The same with the platform's sin and cos | 2.90 µs | 24.6 µs |
| Arithmetic only (sin and cos removed, 40.2 nodes): `Box` recursive / vectorized | 6.07 / 0.94 µs (6.4×) | 55.8 / 7.5 µs (7.4×) |

| Operation, 50-node trees | Prefix array | `Box` tree |
|---|---|---|
| Subtree crossover | 149 ns (two scans, two splices) | 752 ns (count, walk, swap) |
| Clone | 39 ns | 1797 ns |
| Hash / equality | 91 ns / about 75 ns | recursive |
| Depth (one scan) | 283 ns | |
| Memory per node | 16 bytes, one allocation per tree | 32 bytes, one allocation per child |

- **Vectorized evaluation decides the speed**, as Operon, gplearn and Karoo GP found: 3.1 to 3.6×
  faster than the point-by-point interpreters with transcendental functions (3.3 to 3.9× on 1000
  points), 6.4× on arithmetic (7.4× on 1000 points). It is what the symbolic regression fitness uses (section 3). The
  vectorized results equalled the point-by-point ones to the bit on 1000 of 1000 trees: the same
  operations per point in the same order, and sums in a fixed order.
- **Point by point, the layout hardly matters** (9.4 to 11.0 µs): compiling a `Box` tree to
  bytecode per evaluation costs as much as it saves. The generic scalar interpreter (the stack
  machine) is for problems whose values aren't columns of numbers.
- `genoxide::math` costs 5 to 9% over the platform's sin and cos here, the price of portability.
- Breeding is negligible next to evaluation either way; the flat array is 5× faster at crossover and
  46× at cloning, which `Ga` does for every elite and inherited copy.

**Nodes.**

```rust
/// A node of a tree: a primitive of its set, by position, or a constant.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Node {
    /// The primitive at this position in the tree's `PrimitiveSet`: a function or a terminal.
    Primitive(u32),
    /// An ephemeral random constant of a type of the set, compared and hashed by its bits.
    Constant { ty: Type, value: f64 },
}

/// A tree in prefix order: each function is followed by its children's subtrees, in order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tree {
    nodes: Vec<Node>,
}

impl Genome for Tree {
    /// The number of nodes: what size limits and parsimony pressure count.
    fn len(&self) -> usize {
        self.nodes.len()
    }
}
```

Nodes refer to primitives by position, so `Tree` isn't generic, and Python and the CLI use the
same type. The set (section 2.2) maps positions to the user's primitives, arities and types.
`Node` is 16 bytes (a `u32` or a `u16` type beside the tag, and an `f64`). Constants are `f64`
whatever their type; an integer or boolean constant is an integral `f64` (open question 3).

**`Representation::genome_len`** is documented as "the number of genes of every genome"; for a
tree it returns the size limit, and its doc becomes "or the most genes, for a variable-length
genome" (a doc change, not breaking). Nothing but L-SHADE's default size reads it.

### 2.2 Strong typing and primitive sets

**Decision: types are names, primitives are the user's `Copy` values, and the set holds their
signatures as data.** Montana's STGP with monomorphic types: every function has argument types and
a return type, every terminal and constant a type, and a tree is valid when each child's type is
its parent's argument type and the root's type is the set's root type. Untyped GP is the case of
one type.

```rust
use genoxide::gp::{Constants, Gp, Init, PrimitiveSet, Tree};
use genoxide::prelude::*;

// the primitives: any Copy type, usually an enum; its meaning stays in the fitness function
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
enum Op {
    Add,
    Mul,
    Less,
    If,
    X,
}

fn main() -> genoxide::Result<()> {
    let mut set = PrimitiveSet::builder();
    let real = set.new_type("real");
    let boolean = set.new_type("bool");
    set.function("add", Op::Add, [real, real], real)
        .function("mul", Op::Mul, [real, real], real)
        .function("less", Op::Less, [real, real], boolean)
        .function("if", Op::If, [boolean, real, real], real)
        .terminal("x", Op::X, real)
        .constants(real, Constants::uniform(-1.0..=1.0)?); // ephemeral random constants
    let set = set.build(real)?; // trees return a `real`

    let gp = Gp::builder(set)
        .init(Init::RampedHalfAndHalf { depths: 2..=6 }) // Koza's, the default
        .max_depth(17) // Koza's, the default
        .max_size(1024) // the default (open question 2)
        .build()?;
    let tree: Tree = gp.primitives().parse("if(less(x, 0.5), mul(x, x), add(x, 1.0))")?;
    gp.validate(&tree)?;
    println!("{}", tree.display(gp.primitives())); // the same text
    Ok(())
}
```

- **Why names and handles, not `std::any::TypeId` or a type parameter per Rust type.** `TypeId`
  isn't stable between builds, so it can't go into a checkpoint, and Python has no Rust types.
  A type per Rust type would need a value enum anyway to put children on one stack. `Type` is a
  small index (`u16`) returned by `new_type`, so a signature is plain data: serializable,
  comparable, and the same from Rust, Python and a TOML file.
- **Why the user's `Copy` values, not closures in the set.** A representation with closures can't
  implement `Serialize` and `Deserialize`, which `Ga`'s checkpoints require of `R`, and can't be
  compared or described. The primitive's meaning belongs to the fitness function, which `match`es
  on `Op`: inlined, checked for exhaustiveness by the compiler, no dynamic calls. Python and the CLI
  use `P = u32` positions with a table of functions on their side (section 6.4).
- **Validation by `build()`** (errors naming the setting, never panics): names unique; argument
  and return types declared; at most 2^24 primitives and 2^16 types; every type reachable from the
  root type has a terminal or constant, or a function whose arguments do, within the depth limit;
  a set whose root type can't make a tree within the depth limit is an `InvalidSetting` naming the
  types that can't be completed. Montana's precomputed table, `possible[type][depth]` (a tree of
  that type fits in that depth), is built once and drives generation and mutation.
- **Ephemeral random constants** (Koza): `Constants::uniform(range)`, `Constants::integers(range)`
  (integral values) or `Constants::choice([...])`, per type; a node draws its value once when
  created and keeps it, and only constant mutation changes it.
- `Gp<P>` is the representation: the set, the limits and the initialization. `Gp<P>: Serialize`
  when `P: Serialize`; deserializing checks the set as `build` does.

### 2.3 Initialization

| Method | Reference | Design notes |
|---|---|---|
| Full | Koza ch. 6 | Functions until the depth, then terminals; with types, a function of the wanted type only where `possible` allows completing it, else a terminal |
| Grow | Koza ch. 6 | Any primitive of the wanted type that fits the remaining depth; a terminal at the depth |
| Ramped half-and-half (default, depths 2 to 6) | Koza ch. 6 | `random_genome` draws a depth uniformly and full or grow with probability ½ (Koza divides the population evenly; the draw is that in expectation). `gp.ramped_half_and_half(n, rng)` gives Koza's exact division and redraws duplicates (small trees repeat often), for `initial_genomes` |
| PTC2 (optional) | Luke (2000) | Trees of a size drawn from a given distribution |

### 2.4 Evaluation

Three ways, all on the same `Tree` and set, none allocating per call once the workspace has grown:

```rust
impl Tree {
    /// Bottom-up with a stack of `V` (the prefix order read backwards). For any value type:
    /// numbers, a user enum of typed values, booleans.
    pub fn evaluate<P: Copy, V>(
        &self,
        set: &PrimitiveSet<P>,
        stack: &mut Vec<V>,
        primitive: impl FnMut(P, &[V]) -> V, // the children's values, in order
        constant: impl FnMut(f64) -> V,
    ) -> V;

    /// Bottom-up on columns of `points` values: each primitive fills its output column from its
    /// children's. The fast path for data (section 2.1).
    pub fn evaluate_columns<'w, P: Copy>(
        &self,
        set: &PrimitiveSet<P>,
        workspace: &'w mut Columns,
        primitive: impl FnMut(P, &[&[f64]], &mut [f64]),
    ) -> &'w [f64];

    /// The root, to walk the tree top-down: for primitives that choose which children to run
    /// (a conditional with side effects, a program that moves an agent).
    pub fn root(&self) -> Subtree<'_>; // .primitive(), .constant(), .children()
}
```

`Columns` is a pool of buffers owned by the caller (a thread-local one inside genoxide's own
fitness functions, so `parallel(true)` works). Panics are documented (`# Panics`) for a tree
evaluated with a set it doesn't belong to, as `Bits::set` documents an index out of bounds;
`gp.validate(&tree)` checks it first.

### 2.5 Operators

All implement `Crossover<Gp<P>>` or `Mutate<Gp<P>>`, so they plug into every algorithm that takes
operators. The representation passed to each call carries the set and the limits, so every
operator keeps trees typed and within the limits.

| Operator | Reference | Design notes |
|---|---|---|
| `SubtreeCrossover::new()`, `with_internal_rate(p)?` | Koza ch. 6; leaf selection: Angeline, P. J. (1996). An investigation into the sensitivity of genetic programming to the frequency of leaf selection during subtree crossover. GP 1996: 21-29 | A point in the first parent, a function node with probability 0.9 (Koza) when it has one; a point of the same type in the second, with the same bias, among those whose exchange keeps both children within the limits; if none, another first point (a few tries), then the parents unchanged (open question 1). Children always differ from their parents unless the exchanged subtrees are equal |
| `OnePointCrossover` (optional) | Poli and Langdon (1998) | Points in the common region of the two shapes |
| `SubtreeMutation::new()`, `with_max_depth(d)?` | Koza; Field Guide sec. 5.2 | A random point replaced by a grown subtree of its type (depth up to 4 by default, within the limits); redrawn while equal to the old subtree (up to 100 times, as local search's neighbors) |
| `PointMutation::per_node(rate)?`, `count(n)?` | Field Guide sec. 5.2 | A node replaced by another primitive of the same signature (arity, argument and return types); a constant by a new draw; nodes with no alternative are never picked, so a picked node always changes |
| `HoistMutation` | Kinnear, K. E. Jr. (1994). Fitness landscapes and difficulty in genetic programming. IEEE WCCI 1994: 142-147. doi:10.1109/ICEC.1994.350026; Field Guide sec. 5.2 | The tree replaced by one of its subtrees of the root's type; the result is always smaller |
| `ShrinkMutation` | Field Guide sec. 5.2, after Angeline (1996) (verified: a subtree replaced by a random terminal) | A function's subtree replaced by a random terminal or constant of its type |
| `ConstantMutation::gaussian(sigma)?` | Field Guide sec. 5.2 | A constant perturbed by N(0, σ²), σ relative to its type's range; the cheap constant tuning until least squares arrives |
| `gp::Mutations` | | One of several tree mutations per call, by weight: `Mutations::new().subtree(0.6).point(0.2).hoist(0.1).shrink(0.1)`, validated to a positive total |

Property tests for each (CONTRIBUTING.md): typed, within the limits, the set's nodes only, no
no-op mutations; subtree crossover conserves the two parents' multiset of nodes.

### 2.6 Bloat control

References: Luke, S. and Panait, L. (2006). A comparison of bloat control methods for genetic
programming. *Evolutionary Computation* 14(3): 309-344. doi:10.1162/evco.2006.14.3.309 (**Luke and
Panait 2006**); Silva, S. and Costa, E. (2009). Dynamic limits for bloat control in genetic
programming and a review of past and current bloat theories. *GPEM* 10(2): 141-179.
doi:10.1007/s10710-008-9075-9.

| Method | Reference | Where it lives |
|---|---|---|
| Static depth and size limits (default: depth 17, size 1024) | Koza (depth 17) | `Gp` settings, kept by every operator (section 2.5) |
| Lexicographic parsimony pressure: `LexicographicTournament::new(k)?` | Luke, S. and Panait, L. (2002). Lexicographic parsimony pressure. GECCO 2002: 829-836 | A `Select`: a tournament where equal fitness goes to the smaller genome; the paper's ratio bucketing for continuous fitness as `.ratio_buckets(r)?` (verified, section 2) |
| Double tournament: `DoubleTournament::new(fitness_size, parsimony)?` | Luke, S. and Panait, L. (2002). Fighting bloat with nonparametric parsimony pressure. PPSN VII, LNCS 2439: 411-421. doi:10.1007/3-540-45712-7_40 | A `Select`: winners of fitness tournaments meet in size tournaments of two, the smaller winning with probability D / 2 (1 ≤ D ≤ 2), or the order reversed with `.size_first()` |
| Tarpeian: `Tarpeian::new(select, rate)?` | Poli, R. (2003). A simple but theoretically-motivated method to control bloat in genetic programming. EuroGP 2003, LNCS 2610: 204-217. doi:10.1007/3-540-36599-0_19 | A `Select` wrapper: once per call, genomes larger than the population's mean size count as invalid with probability `rate`. It keeps the paper's selection effect; it doesn't save their evaluation, which would need the algorithm (documented) |
| Covariant parsimony pressure | Poli, R. and McPhee, N. F. (2008). Parsimony pressure made easy. GECCO 2008: 1267-1274. doi:10.1145/1389095.1389340 | Later: a penalty whose coefficient is set per generation, through `control` and `reevaluate`; an example if asked for |
| Accuracy and size as two objectives | Smits, G. and Kotanchek, M. (2005). Pareto-front exploitation in symbolic regression. GPTP II: 283-299 (to check) | No new code: NSGA-II with `[error, size]` (section 3.5) |
| Size-fair crossover, operator equalisation, dynamic limits | Langdon, W. B. (2000). Size fair and homologous tree crossovers for tree genetic programming. *GPEM* 1(1/2): 95-119. doi:10.1023/A:1010024515191; Silva and Costa (2009) | Not planned for 0.11 |

**Parsimony through `Select` without an API change.** `Select::select<G: Genome>` gets the
population, and every individual's `genome().len()`: the node count of a tree (the connection
count of a NEAT network). The new selections are generic over any genome and reduce to their plain
versions on fixed-length genomes. Nothing changes for existing operators.

**Defaults.** The limits are the representation's defaults. Selection has no default in `Ga`;
AGENTS.md's GP row recommends `DoubleTournament::new(7, 1.4)` with the static limits, the
combination Luke and Panait 2006 found among the best (verified, section 2).

## 3. Symbolic regression

### 3.1 Primitives

`gp::regression::Math`, a `Copy` enum for `PrimitiveSet<Math>`, with the vectorized
implementation of each, all through `genoxide::math` and IEEE operations:

| Primitive | Decision |
|---|---|
| `Add`, `Sub`, `Mul`, `Neg`, `Square`, `Cube` | Plain |
| `Div` | IEEE division: a zero denominator gives ±inf or NaN, and a tree with a non-finite prediction is invalid (not protected, so no silent discontinuities) |
| `Aq` (analytic quotient, a / √(1 + b²)) | The recommended division: Ni, J., Drieberg, R. H. and Rockett, P. I. (2013). The use of an analytic quotient operator in genetic programming. *IEEE TEVC* 17(1): 146-152. doi:10.1109/TEVC.2012.2195319: smooth, defined everywhere, better generalization than protected division |
| `ProtectedDiv`, `ProtectedLog`, `ProtectedSqrt` | Koza's protected versions (1 for a zero denominator; ln\|x\|; √\|x\|), for replicating papers whose function sets use them (Nguyen's set: to check) |
| `Sin`, `Cos`, `Exp`, `Log`, `Sqrt`, `Tanh`, `Abs` | `genoxide::math`; invalid where undefined |
| `Variable(u16)`, constants | Terminals: a column of the data; ERCs of the set |

Interval arithmetic (Keijzer 2003, below), which rejects a tree whose output is undefined anywhere
in the input box, not only at the sample points: later, as a validity check in the fitness.

### 3.2 Fitness

- **RMSE by default**, MSE and MAE as options; minimized. A prediction that isn't finite at any
  training point makes the tree invalid (`None`), as genoxide treats NaN.
- **Linear scaling on by default** (Keijzer, M. (2003). Improving symbolic regression with interval
  arithmetic and linear scaling. EuroGP 2003, LNCS 2610: 70-82. doi:10.1007/3-540-36599-0_7;
  Keijzer, M. (2004). Scaled symbolic regression. *GPEM* 5(3): 259-269.
  doi:10.1023/B:GENP.0000030195.77571.f9): the error of a + b·f(x) with a and b by least squares
  (b = cov(y, f) / var(f), a = ȳ − b f̄; b = 0 for a constant f), so GP searches for the shape and
  not for the scale. The scaled expression is what the result prints. Sums in index order, so the
  same bits everywhere.
- `Regression::new(dataset)`: a `FitnessFunction<Tree>` evaluating with `evaluate_columns` on a
  thread-local workspace; `Evaluated` extras give a and b. A held-out test set is part of the
  dataset, reported, never used by the search.
- **Constants:** ERCs and `ConstantMutation` in 0.11. Least-squares fitting of all constants
  (Kommenda, M., Burlacu, B., Kronberger, G. and Affenzeller, M. (2020). Parameter identification
  for symbolic regression using nonlinear least squares. *GPEM* 21(3): 471-501.
  doi:10.1007/s10710-019-09371-3) when Levenberg-Marquardt exists (0.15); its derivatives need the
  tree's gradient with respect to its constants, by the forward-mode dual numbers of the
  optimization plan (D1) or a reverse sweep over the prefix array.

### 3.3 Test problems

The standard sets and their critique: McDermott, J., White, D. R., Luke, S., Manzoni, L.,
Castelli, M., Vanneschi, L., Jaśkowski, W., Krawiec, K., Harper, R., De Jong, K. and O'Reilly,
U.-M. (2012). Genetic programming needs better benchmarks. GECCO 2012: 791-798.
doi:10.1145/2330163.2330273; White, D. R., McDermott, J., Castelli, M., Manzoni, L., Goldman, B.
W., Kronberger, G., Jaśkowski, W., O'Reilly, U.-M. and Luke, S. (2013). Better GP benchmarks:
community survey results and proposals. *GPEM* 14(1): 3-29. doi:10.1007/s10710-012-9177-2. They
found GP papers mostly tested on toy problems (Koza's quartic above all), with inconsistent function
sets, sampling and budgets, and proposed retiring some and a set of harder ones (the lists to check;
among them, as recalled, Keijzer-6, Korns-12, Vladislavleva-4, Nguyen-7 and Pagie-1). The later
reference point for comparing methods is real and synthetic data: La Cava, W., Orzechowski, P.,
Burlacu, B., de França, F. O., Virgolin, M., Jin, Y., Kommenda, M. and Moore, J. H. (2021).
Contemporary symbolic regression methods and their relative performance. NeurIPS 2021 Datasets and
Benchmarks. arXiv:2107.14351.

genoxide follows that critique: each problem is defined with **its paper's target, sampling of
training and test points, and function set**, and says which it is. Sampled points come from a
fixed seed of genoxide's portable stream, so the data are the same everywhere; the problem's
docs say that another library's random points differ.

| Set | Reference | Problems | Notes |
|---|---|---|---|
| Koza | Koza ch. 7, 10 | Quartic x⁴ + x³ + x² + x; also x⁵ − 2x³ + x, x⁶ − 2x⁴ + x² (Koza 1994) | The tutorial problem; flagged as a toy by the critique, and its docs say so |
| Nguyen | Uy, N. Q., Hoai, N. X., O'Neill, M., McKay, R. I. and Galván-López, E. (2011). Semantically-based crossover in genetic programming: application to real-valued symbolic regression. *GPEM* 12(2): 91-119. doi:10.1007/s10710-010-9121-2 | Nguyen-1 to 12: polynomials, sin and cos, logs, 2-D | 20 random points in [−1, 1] (1-D), 100 in [0, 1]² (2-D), to check; exact recovery is realistic |
| Keijzer | Keijzer (2003) | Keijzer-1 to 15 | Grids and random samples per problem; some (the harmonic sum, Keijzer-6) have no exact form in the set |
| Vladislavleva | Vladislavleva, E. J., Smits, G. F. and den Hertog, D. (2009). Order of nonlinearity as a complexity measure for models generated by symbolic regression via Pareto genetic programming. *IEEE TEVC* 13(2): 333-349. doi:10.1109/TEVC.2008.926486 | Vladislavleva-1 to 8 | Test ranges wider than training ranges: extrapolation |
| Korns | Korns, M. F. (2011). Accuracy in symbolic regression. GPTP IX: 129-151. doi:10.1007/978-1-4614-1770-5_8 | Korns-1 to 15 | 5 variables, 10,000 points, constants that GP must find: after constant optimization |
| Pagie | Pagie, L. and Hogeweg, P. (1997). Evolutionary consequences of coevolving targets. *Evolutionary Computation* 5(4): 401-418. doi:10.1162/evco.1997.5.4.401 | Pagie-1 | 2-D, hard to recover exactly |
| Boolean (typed and untyped GP beyond numbers) | Koza ch. 7 | 11-multiplexer, even-parity 3 to 5 | Also flagged by the critique; kept as GP's classic Boolean tests, with exact optima |

`gp::regression::problems` gives each as a type with `dataset()` (training and test), `primitives()`
(its paper's set), `target(&x)` and `reference()`; `all()` lists them. Only problems whose example
reaches the optimum (exact recovery, section 9) are added in 0.11; the ones that need constant
fitting (Korns, most of Vladislavleva) wait for 0.15, so every problem keeps its example.

### 3.4 Usage

```rust
use genoxide::gp::regression::{self, Regression};
use genoxide::gp::{self, Gp, SubtreeCrossover};
use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    let problem = regression::problems::Nguyen7::new(); // ln(x + 1) + ln(x² + 1)
    let gp = Gp::builder(problem.primitives().clone()).build()?; // its paper's set, the default limits
    let ga = Ga::builder(gp)
        .population_size(500)
        .select(DoubleTournament::new(7, 1.4)?)
        .crossover(SubtreeCrossover::new())
        .mutate(gp::Mutations::new().subtree(0.7).point(0.2).hoist(0.1))
        .mutation_rate(0.2)
        .minimize()
        .seed(1)
        .build()?;
    let outcome = Engine::new(ga, Regression::new(problem.dataset()).linear_scaling(true))
        .stop_when(Stop::target(1e-10).or(Stop::generations(200)))
        .parallel(true)
        .run()?;
    println!("{}", outcome.best_genome().display(problem.primitives()));
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    Ok(())
}
```

### 3.5 Accuracy against size

`Nsga2::builder(gp, [Minimize, Minimize])` with a fitness `[rmse, tree.len() as f64]` gives the
Pareto front of error against size, the approach of Pareto GP (Smits and Kotanchek 2005,
Vladislavleva et al. 2009) and PySR; it works as soon as the tree operators exist, and gets an
example in G4.

## 4. NEAT

Reference: Stanley, K. O. and Miikkulainen, R. (2002). Evolving neural networks through augmenting
topologies. *Evolutionary Computation* 10(2): 99-127. doi:10.1162/106365602320169811 (**the NEAT
paper**; its sections and parameter values are to check before batch N1).

### 4.1 Genome

```rust
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Network {
    nodes: Vec<NodeGene>,             // sorted by id: inputs, bias, outputs, then hidden
    connections: Vec<ConnectionGene>, // sorted by innovation
}

pub struct NodeGene { id: u32, kind: NodeKind, activation: Activation } // Input, Bias, Output, Hidden
pub struct ConnectionGene { innovation: u32, from: u32, to: u32, weight: f64, enabled: bool }
```

- Weights compared and hashed by their bits, as `Reals`; `len()` is the number of connection genes.
- The paper's bias is an input node fixed at 1, followed here; a per-node activation field (the
  steepened sigmoid 1 / (1 + e^(−4.9x)) of the paper by default; tanh, ReLU, identity, and later
  the CPPN functions) costs a byte and keeps HyperNEAT possible.
- At most 2^24 nodes and connections, errors beyond.

### 4.2 Innovation numbers

The paper numbers each new structural gene globally, and gives the same number to the same
mutation within one generation, so crossover aligns genes by history.

**Decision: a registry per run, keyed by structure, inside the `Neat` algorithm.**

- `BTreeMap<(from, to), innovation>` for connections and `BTreeMap<(split innovation, k), node>`
  for nodes (k counts how often one genome's line split the same connection, after re-enabling).
  The same structural mutation gets the same number for the whole run, not only within a
  generation: a superset of the paper's rule (identical mutations in one generation still share a
  number) that also merges identical mutations of different generations, as neat-python's
  (from, to) keys do. Numbers are assigned in child order, after breeding, so seeded runs are the
  same; with parallel breeding (later), children record their structural mutations and a
  sequential pass numbers them in child order.
- The registry is algorithm state: in checkpoints, a `BTreeMap` (ordered, no hash order), a few
  thousand entries per thousand generations.
- **Islands:** each island's registry would give different numbers to the same structure, so
  migrants wouldn't align. `Neat` doesn't implement `Migrate` in 0.11 (open question 7).

### 4.3 Speciation, fitness sharing and reproduction

As the paper: compatibility distance δ = c₁E/N + c₂D/N + c₃W̄ (excess, disjoint, the mean weight
difference of matching genes; N the larger genome's gene count, 1 below 20 genes, to check);
species represented by a random member of the previous generation's species; a genome joins the
first species within δ_t, in species order, else founds one; explicit fitness sharing; offspring
per species in proportion to its share; the champion of each species of more than five copied
unchanged; species that haven't improved for 15 generations stop reproducing; crossover aligning
genes by innovation, disjoint and excess genes from the fitter parent, a gene disabled in either
parent disabled with probability 0.75; mutations: weights (perturbed, or replaced with probability
0.1), add connection, add node (splitting a connection: in-weight 1, out-weight the old one).

genoxide specifics:

- **Any objective and invalid fitness.** The paper's sharing assumes a non-negative, maximized
  fitness. Here, shared fitness uses each individual's fitness normalized in the generation,
  (f − worst) / (best − worst) in the objective's direction, invalid as 0; the raw form as
  `Sharing::Raw`, which needs maximization and non-negative scores (`Error::InvalidFitness`
  otherwise) and replicates the paper exactly (open question 8).
- **Exact counts.** Offspring counts by the largest-remainder method, ties to the lower species
  id, so the population size stays exact and deterministic.
- **An `Algorithm`:** ask the generation's new genomes (the initial population, then offspring;
  unchanged champions keep their fitness), tell, speciate, reproduce. `Reevaluate` rescores the
  population. `population()` is the current generation; `species()` for observers.
- **Recurrent or not:** `.feed_forward(true)` (default) never adds a connection that closes a
  cycle; `false` allows recurrent connections and self-loops, needed for the pole balancing without
  velocities.

### 4.4 Network evaluation

- `network.feed_forward() -> Result<FeedForward>`: the enabled connections compiled once per
  evaluation into a flat, topologically ordered program (as SharpNEAT does); an `Err` for a cycle.
  `activate(&inputs, &mut outputs)`, with no allocation after the first call.
- `network.recurrent() -> Recurrent`: every node updated synchronously from the previous step's
  values, one step per `activate`, with `reset()`; the paper's discrete-time recurrent activation
  (to check against its pole-balancing section).
- Activation functions through `genoxide::math` (`exp`, `tanh`), so outputs are the same bits
  everywhere; sums over incoming connections in innovation order.

### 4.5 Usage

```rust
use genoxide::neat::{Neat, Network};
use genoxide::prelude::*;

const CASES: [([f64; 2], f64); 4] = [([0.0, 0.0], 0.0), ([0.0, 1.0], 1.0), ([1.0, 0.0], 1.0), ([1.0, 1.0], 0.0)];

fn main() -> genoxide::Result<()> {
    let neat = Neat::builder(2, 1) // inputs, outputs; a bias input is added
        .population_size(150)
        .seed(1)
        .build()?;
    // the paper's fitness: (4 − Σ|error|)², maximized
    let xor = |network: &Network| {
        let mut net = network.feed_forward().expect("feed-forward by construction");
        let mut output = [0.0];
        let error: f64 = CASES
            .iter()
            .map(|(input, target)| {
                net.activate(input, &mut output);
                (output[0] - target).abs()
            })
            .sum();
        (4.0 - error).powi(2)
    };
    let outcome = Engine::new(neat, xor)
        // Σ|error| ≤ 0.0125; the paper's success criterion replaces it in N1 (to check)
        .stop_when(Stop::target(15.9).or(Stop::generations(300)))
        .run()?;
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    Ok(())
}
```

### 4.6 Settings: the paper and neat-python

Defaults follow the paper. neat-python's are listed so users moving from it know what differs
(from its XOR example's configuration, as recalled; to check against its current release).

| Setting | The NEAT paper | neat-python | genoxide default |
|---|---|---|---|
| Population | 150 (1000 for the double pole without velocities) | 150 | 150 |
| Compatibility c₁, c₂, c₃, δ_t | 1.0, 1.0, 0.4, 3.0 (c₃ = 3.0, δ_t = 4.0 for the large population) | disjoint and excess alike 1.0, weights 0.5, threshold 3.0; node genes counted too | the paper's |
| Weight mutation | 80% of offspring; 90% perturbed, 10% replaced | rate 0.8, power 0.5, replace 0.1; a bias per node mutated too | the paper's; the perturbation's size to check |
| Add node, add connection | 0.03, 0.05 (0.3 in the large population) | 0.2, 0.5, plus deletions 0.2, 0.5 | the paper's; no deletions |
| Offspring by mutation only | 25% | | 25% |
| Interspecies mating | 0.001 | | 0.001 |
| Disabled if disabled in a parent | 0.75 | inherited as is, enabled with mutation rate 0.01 | 0.75 |
| Elitism | species champions of species > 5 | 2 per species, species elitism 2, survival threshold 0.2 | the paper's |
| Stagnation | 15 generations | 20 | 15 |
| Activation | 1 / (1 + e^(−4.9x)) | sigmoid of 5x, clamped | the paper's |
| Initial networks | inputs fully connected to outputs | configurable, full by default in the example | full, `.initial(Initial::Unconnected)` optional |

## 5. Neuroevolution with evolution strategies

### 5.1 What exists, and what's needed

Fixed-topology networks whose weights are a `Real` genome need no new representation: the existing
`xor_neuroevolution` example does it with CMA-ES. What's missing is (1) a network module, so users
don't decode weights by hand; (2) control tasks written in Rust, without dependencies; (3) an ES
built for many weights. CMA-ES is the method for up to a few hundred weights, and it solves the
pole-balancing tasks with fixed networks (Igel, C. (2003). Neuroevolution for reinforcement learning
using evolution strategies. CEC 2003: 2588-2595. doi:10.1109/CEC.2003.1299414;
Heidrich-Meisner, V. and Igel, C. (2009). Neuroevolution strategies for episodic reinforcement
learning. *J. Algorithms* 64(4): 152-168. doi:10.1016/j.jalgor.2009.04.002); sep-CMA-ES (Ros, R.
and Hansen, N. (2008). A simple modification in CMA-ES achieving linear time and space
complexity. PPSN X, LNCS 5199: 296-305. doi:10.1007/978-3-540-87700-4_30) up to thousands.

**Decision: add a small `OpenEs` algorithm** (Salimans, T., Ho, J., Chen, X., Sidor, S. and
Sutskever, I. (2017). Evolution strategies as a scalable alternative to reinforcement learning.
arXiv:1703.03864), in batch E2, after the module and examples of E1. Reasons: it is the baseline
of neuroevolution at scale (tens of thousands of weights and more), where CMA-ES's covariance can't
be afforded and sep-CMA-ES's learning rates slow down; it's O(n) per sample with a fixed σ and a
gradient step, so it's a few hundred lines on the existing `Algorithm` trait; and its parts
(antithetic sampling, rank shaping, Adam) are small and well defined. Name to settle (open question
9); `OpenEs` follows evosax's.

| Part | Reference | Design notes |
|---|---|---|
| Antithetic (mirrored) sampling | Salimans et al. (2017); Brockhoff, D., Auger, A., Hansen, N., Arnold, D. V. and Hohm, T. (2010). Mirrored sampling and sequential selection for evolution strategies. PPSN XI, LNCS 6238: 11-21. doi:10.1007/978-3-642-15844-5_2 | ε and −ε per pair; population size even, validated |
| Centered-rank fitness shaping | Salimans et al. (2017); Wierstra, D., Schaul, T., Glasmachers, T., Sun, Y., Peters, J. and Schmidhuber, J. (2014). Natural evolution strategies. *JMLR* 15: 949-980 | Ranks mapped to [−0.5, 0.5], ties by position; invalid ranks last |
| Update | Salimans et al. (2017); Kingma, D. P. and Ba, J. (2015). Adam: a method for stochastic optimization. ICLR 2015. arXiv:1412.6980 | `Optimizer::Sgd { learning_rate, momentum }` or `Adam { .. }` on the gradient estimate; weight decay; the Adam step shared with the optimization plan's batch D1 |
| Mean evaluation | | Optional: evaluate the mean each generation (it isn't among the samples), counted as an evaluation |
| Parallel breeding | As `Es` | Perturbations on derived streams per pair; the same results on any thread count |
| SNES (optional) | Schaul, T., Glasmachers, T. and Schmidhuber, J. (2011). High dimensions and heavy tails for natural evolution strategies. GECCO 2011: 845-852. doi:10.1145/2001576.2001692 | A σ per weight; only if E2's example shows a need |

### 5.2 Network module

`genoxide::nn`: `Mlp::new([4, 16, 1], Activation::Tanh)?` (layer sizes; the output activation a
setting), `mlp.parameters()`, `mlp.representation(-5.0..=5.0)?` (a `Real::uniform` of the right
length), `mlp.with(&weights).forward(&input, &mut output)`, and a small recurrent (Elman) network
for tasks with hidden state. Plain loops in a fixed order and `genoxide::math`, no dependency, no
allocation per forward pass. Not a training framework (ROADMAP.md, not planned).

### 5.3 Control tasks

`genoxide::problems::control`: environments driven by a `Policy` (a trait implemented by `nn`'s
networks and NEAT's compiled networks, and by closures), integrated with RK4 (step and constants
from the papers, checked in E1 below), `genoxide::math::sin` and `cos`, so episodes are the same bits
everywhere.

| Task | Reference | Success criterion (checked) |
|---|---|---|
| Cart-pole | Barto, A. G., Sutton, R. S. and Anderson, C. W. (1983). Neuronlike adaptive elements that can solve difficult learning control problems. *IEEE Trans. SMC* 13(5): 834-846. doi:10.1109/TSMC.1983.6313077; corrected equations: Florian, R. V. (2007). Correct equations for the dynamics of the cart-pole system. Technical report, Center for Cognitive and Neural Studies, Romania | Balanced for 100,000 steps |
| Double pole, with velocities | Wieland, A. P. (1991). Evolving neural network controllers for unstable systems. IJCNN 1991, vol. 2: 667-673. doi:10.1109/IJCNN.1991.155416; the settings of Gomez, F., Schmidhuber, J. and Miikkulainen, R. (2008). Accelerated neural evolution through cooperatively coevolved synapses. *JMLR* 9: 937-965 | Balanced for 100,000 steps |
| Double pole, without velocities | As above; damping fitness: Gruau, F., Whitley, D. and Pyeatt, L. (1996). A comparison between cellular encoding and direct encoding for genetic neural networks. GP 1996: 81-89 | 100,000 steps, and the generalization test: at least 200 of 625 starting states balanced for 1000 steps |

Each has `run(&mut policy, steps) -> u32` (steps balanced), the damping fitness where the papers
use it, and the generalization test. Gomez et al. (2008) tabulate evaluations to success for many
methods (NEAT, CMA-ES, CoSyNE, ...): the reference points for the examples' READMEs, not tests.

**Checked in E1** against Florian (2007), Stanley and Miikkulainen (2002), Igel (2003) and Gomez et
al. (2008) (Wieland 1991 and Gruau et al. 1996 through their restatements; Barto et al. 1983
through Florian):

- **Verified:** RK4 with 0.01 s, two integration steps per 0.02 s control step; the masses,
  half-lengths (0.5 m, 0.05 m), friction coefficients (0.0005, 0.000002), track (±2.4 m), failure
  angles (12°, 36°), a force of up to 10 N at least 10/256 N in size; 100,000 steps to succeed;
  the damping fitness `0.1 f₁ + 0.9 f₂` over 1000 steps; the generalization test's 625 starts
  (Igel gives the ranges: ±2.16 m, ±1.35 m/s, ±3.6°, ±8.6°/s) and its threshold of 200.
- **Corrected:** Florian's equation 21 (and 19) as printed has `cos θ − μc sgn(N ẋ)` in its
  denominator; substituting his equation 18 into 16 gives `cos θ − μc sgn(N ẋ) sin θ`, which the
  implementation uses and a test checks against Newton's laws solved directly. For two poles,
  genoxide applies his correction (friction `μc N sgn(N ẋ)`, positive `g`) to Wieland's equations,
  solving for `ẍ` and `N` together.
- **Differs between the papers**, genoxide follows Gomez et al. (2008): the long pole's initial
  angle is 4° (Stanley and Miikkulainen and Igel: 1° with velocities; Igel: 4.5° without); the
  damping sum runs over the last 100 states (Igel), where Stanley and Miikkulainen and Gomez et
  al. print `t − 100` to `t`, 101. The papers scale the inputs to [−1, 1] without giving the
  ranges: genoxide divides by 2.4 m, the failure angle and 2 (velocities).
- **Added:** `CmaesBuilder::min_step`, a lower bound on σ, as Igel used for the task without
  velocities; without it CMA-ES solved 13 of 20 seeds within 30,000 evaluations, with it 100 of
  100 (section 9, E1's examples).

## 6. Python

### 6.1 What Python lacks today

From `python/genoxide/__init__.py` and `python/src/config.rs`: the package describes a run as JSON
and runs it to completion in Rust (`run(config, fitness, ...)`); it has `Ga`, `De`, `Cmaes`,
`Pso`, `LocalSearch` and the five multi-objective algorithms. It lacks `Es` (and `AdaptiveReal`
with `SelfAdaptiveMutation`), `Islands`, checkpoints, `AsyncEngine`, `memetic`, `initial_genomes`
and a hall of fame. ROADMAP.md's 0.11 names three: the ES, islands and checkpoints (batch P1).

### 6.2 Zero-copy numpy genomes, concretely

Today each call of a non-batch fitness function gets a new 1-D array (`PyArray1::from_slice`: an
allocation and a copy per evaluation), and a batch gets one 2-D array per generation, gathered from
the genomes into a `Vec` then moved into numpy without a copy. A view of a genome's own memory
would need `PyArray::borrow_from_array`, which is `unsafe` and ruled out by `#![forbid(unsafe_code)]`
(the Python crate forbids it too).

Measured (numpy 2.5.3, the machine above): a Python call costs 42 ns, a new 30-gene array 106 ns,
`a.sum()` 312 ns. The allocation is a quarter of the smallest numpy fitness; the copy (8 bytes per
gene) is not measurable against it.

**Decision: no new array per call, without unsafe code.**

- **Reused buffers.** The engine keeps the array it passed last time and, if its reference count
  (`Bound::get_refcnt`, safe) shows that Python kept no reference, writes the next genome into it
  (a memcpy) instead of allocating. If the user kept it (appended it to a list), a new array is made,
  so what the user holds never changes. The same for the batch matrix, reused per generation when
  its shape matches. A fitness function that writes into its array changes only that buffer, as
  today.
- **Moves, not copies, out of Rust:** results and snapshots hand over genome memory with
  `into_pyarray` where the genome isn't kept (`Reals::into_vec`); bits stay packed until read
  (#324).
- **Into Rust** (initial genomes, a genome returned by a control), a copy is required and cheap.
- The docs say "no allocation per call", not "zero-copy": what the user gets is the performance of
  a view without its aliasing.

### 6.3 `Es`, islands and checkpoints

- `gx.Es(genome, parents, offspring, recombination, selection, step_sizes, parallel_breeding,
  objective, seed)`, as Rust's builder, with `gx.AdaptiveReal` and `gx.SelfAdaptiveMutation` for a
  GA running an ES.
- `gx.Islands([ga_or_de, ...], topology="ring", interval=10, migrants=2)` with `run(...)` like any
  algorithm; `RunningIslands` for `control`, giving each island's `Running`.
- Checkpoints: `algorithm.run(..., checkpoint="run.ckpt", checkpoint_every=100)` saves from inside
  the run (the Rust type follows from the JSON description, so loading needs no alias);
  `algorithm.run(..., resume="run.ckpt")` continues, with an error for a checkpoint of another
  description or version. Tested as in Rust: a resumed run equals an uninterrupted one.

### 6.4 GP and NEAT in Python (batch P2)

```python
import genoxide as gx
import numpy as np

x = np.linspace(-1.0, 1.0, 20)
tree = gx.Tree(functions=["add", "sub", "mul", "aq", "sin", "cos"], variables=1,
               constants=(-1.0, 1.0), max_depth=17)
ga = gx.Ga(tree, population_size=500, select=gx.DoubleTournament(7, 1.4),
           crossover=gx.SubtreeCrossover(), mutation=gx.SubtreeMutation(), mutation_rate=0.2,
           objective="minimize", seed=1)
result = ga.run(gx.gp.Regression(x[:, None], x**4 + x**3 + x**2 + x), target=1e-10,
                generations=200)          # evaluated in Rust, vectorized
print(result.genome)                      # an expression, e.g. "add(mul(x0, ...), ...)"

# typed, with a user primitive on numpy columns
tree = gx.Tree(types=["real", "bool"], root="real", functions=[
    gx.Function("add", ["real", "real"], "real"),               # built in
    gx.Function("less", ["real", "real"], "bool", np.less),     # user: columns in, a column out
    gx.Function("if", ["bool", "real", "real"], "real", np.where),
], variables=1)

neat = gx.Neat(inputs=2, outputs=1, population_size=150, seed=1)
result = neat.run(lambda net: (4 - sum(abs(net.activate(i)[0] - t) for i, t in cases)) ** 2,
                  target=15.9, generations=300)
```

- Built-in primitives run in Rust; a user primitive is a numpy function called once per node on
  whole columns. As built (P2): the set is `gx.gp.PrimitiveSetBuilder` (`new_type`, `function`,
  `terminal`, `constants`, `build`), data only, so it checkpoints, its primitives one more variant
  of the package's primitive enum (no new Rust instantiations); the functions are given at
  evaluation, `tree.evaluate(x, {"less": np.less, "if": np.where, ...})`, or the user walks
  `tree.nodes()`. Measured: 50 numpy additions on 100-point arrays take 13.5 µs, against 3.0 µs for
  the whole tree in Rust; calling Python per node and point would take 172 µs. So user primitives
  are per column, never per point.
- A tree reaches a Python fitness function as a `gx.TreeGenome` (Rust-backed: `evaluate(X)`,
  `str()`, `nodes`), a network as `gx.Network` (`activate`, `feed_forward()`), not as numpy arrays:
  the Python crate's `Genes` trait gains a sibling for non-array genomes.
- `gx.problems.regression` and `gx.problems.control` evaluated in Rust, as the other problems are.
- The CLI gets `tree` genomes with the built-in regression fitness on a CSV file later, not in 0.11.

## 7. Reproducibility and the guarantees

- **Seeds and threads.** Tree operators and generation draw only from the stream they're given, so
  `Ga`'s parallel breeding (a stream per pair) gives the same trees on any number of threads.
  NEAT's innovation and node numbers are assigned in child order (section 4.2); species are kept in
  a `Vec` by id; registries are `BTreeMap`s. No `HashMap` iteration anywhere in the new code.
- **Platforms.** Every primitive of `regression::Math`, every activation function and the control
  tasks' dynamics use `genoxide::math` and IEEE operations; no `mul_add`; sums (RMSE, linear
  scaling's moments, a node's weighted inputs) in a fixed order, never a rayon reduction. The
  vectorized evaluator does per point exactly what the scalar one does (checked to the bit in the
  prototype, and by a test). CI's examples on Linux, macOS and Windows print enough digits to catch
  a drift.
- **Errors, not panics.** Primitive sets that can't make a tree within the limits, limits of 0 or
  above 2^24, empty type lists, a population or species setting out of range, a feed-forward
  network with a cycle: all errors. Documented panics only for a tree evaluated with a set it
  doesn't belong to (as for an index out of bounds). Deep trees never recurse in genoxide's code:
  clone, equality, hashing, serde and the stack-machine evaluators are loops; `Subtree` walks are
  the user's recursion, bounded by the depth limit.
- **Checkpoints.** `Tree`, `Gp<P>` (with `P: Serialize`), `Network`, `Neat`'s state (population,
  species, registry, stream) and `OpenEs`'s (mean, optimizer moments) implement serde; a resumed
  run equals an uninterrupted one, tested as for the other algorithms.
- **No wasted work.** Mutations always change the tree (section 2.5); copies inherit fitness;
  crossover within the limits instead of producing over-limit children to discard.

## 8. Tests and validation

Reference values come from the papers and from derivations written in the tests (rule 3.3 of the
problem plan). DEAP and neat-python runs are comparisons of behavior (success rates, sizes) for
the READMEs and later benchmarks, not expected values.

| Feature | Tests |
|---|---|
| Primitive sets | Build errors for each invalid case; Montana's `possible` table by hand on a small typed set; parse and display round-trip |
| Generation | Property tests: typed, within depth and size; full trees have every leaf at the depth (where types allow); grow's leaves at most at the depth; ramped half-and-half's depths and methods uniform (chi-squared on 10,000 trees); `ramped_half_and_half` gives no duplicates |
| Operators | Property tests (CONTRIBUTING.md): typed, within limits, no no-op mutations, crossover conserves the parents' nodes; the 90/10 internal-point frequency within binomial bounds; hoist always shrinks |
| Evaluation | The three evaluators agree to the bit on random trees; hand-computed values; non-finite predictions make the fitness invalid |
| Bloat control | Double tournament's size-winner probability D / 2 exactly (a two-individual population); lexicographic ties to the smaller; Tarpeian's marking rate; a statistical test that mean tree size after 50 generations on the quartic is smaller with double tournament than with plain tournament (over 20 seeds) |
| Linear scaling | Exact recovery of a and b for f = (y − a) / b; b = 0 for constant trees; the scaled error is the least-squares minimum |
| Regression problems | Each target against values derived by hand; sampling ranges and counts as in the papers; the dataset the same bits on every platform |
| NEAT genome | The paper's crossover alignment example (its figure of matching, disjoint and excess genes, to check) reproduced; δ by hand; add-node keeps the network's function for a linear activation; `feed_forward` never creates a cycle (property test); registry numbers deterministic and shared across a generation |
| NEAT algorithm | Exact population size with offspring rounding; stagnant species removed after 15 generations; XOR solved in 100 of 100 seeds (the paper reports no failures, 32 generations and 4755 evaluations on average, to check), run in the slow CI job; the mean evaluations reported in the README, not asserted |
| Networks | `FeedForward` and `Recurrent` against hand computation; `Mlp` against a naive implementation |
| Control tasks | A step of each against the equations evaluated by hand (Florian's corrections); energy-like invariants over a frictionless run; the success criteria as the papers define them |
| `OpenEs` | Centered ranks exact; with antithetic pairs on a sphere, the gradient estimate's mean over many generations matches the smoothed gradient (2μ for the sphere) within its standard error; Adam and SGD steps against hand computation |
| Protocol and guarantees | For `Neat` and `OpenEs`: ask twice gives the same genomes; tell errors leave the state unchanged; checkpoint resume equals an uninterrupted run; `Reevaluate` draws no random number; seeded runs identical on 1 and 8 threads |
| Performance | gungraun instruction counts for subtree crossover, the vectorized evaluator and NEAT's reproduction step, as for the other hot paths |

## 9. Examples

Every example follows the repository's rules: its own folder with `main.rs`, `main.py`,
`README.md` (front matter; the problem, what makes it hard, representation, algorithm, output,
good results), `output.txt`, and `trace.json` with `trace.rs` / `trace.py` for the site's player;
one problem per page (the problems of one paper share a `family` and appear as tabs); **every
example reaches its optimum**, and failures appear only as labelled contrasts. New plot kinds, to
settle with the site: the best tree drawn and its curve over the data (GP), the network and a
species stack plot (NEAT), the cart and poles animated (control).

The optimum of a regression problem is exact recovery: an RMSE at the level of rounding (at most
1e-10 of the target's standard deviation) on the training and the test points, with the expression
printed. For Boolean problems, all cases correct.

| Batch | Examples (category) |
|---|---|
| G1 | `koza_quartic`: the quartic, the classic first GP problem, labelled a toy by the critique (`gp`) |
| G2 | `multiplexer_11`: Koza's Boolean problem, all 2048 cases; `abs_typed`: a typed set (`less`, `if`) recovering \|x\| exactly, the STGP example |
| G3 | The Nguyen problems that reach exact recovery, a page each (`family: nguyen`); `accuracy_and_size`: NSGA-II on a Nguyen problem, the front of error against size |
| G4 | Keijzer and Vladislavleva problems that reach exact recovery with linear scaling and ERCs, a page each; the others wait for constant fitting (0.15) |
| E1 | `cart_pole`, `double_pole`, `double_pole_no_velocities` (`family: pole balancing`, `neuroevolution`): fixed networks by CMA-ES |
| E2 | `two_spirals` (Lang, K. J. and Witbrock, M. J. (1988). Learning to tell two spirals apart. Proc. 1988 Connectionist Models Summer School: 52-59): an `Mlp` of a few thousand weights by `OpenEs`, 100% of the points classified, with sep-CMA-ES as the contrast if it's slower (or another task, chosen by measurement in E2) |
| N1 | `xor_neat`, in a `family: xor` with the existing `xor_neuroevolution` (fixed topology), the paper's success criterion |
| N2 | NEAT added to the three pole-balancing pages (as the Rastrigin page shows two methods) |

**G4's outcome (2026-09-30).** With linear scaling, Keijzer's set (add, mul, inv, neg, sqrt) and
normal constants N(0, 5), and the Nguyen pages' search with constant mutation added, 20 runs each:
Keijzer-12 recovered in 3 (3 again with 500 generations) and Keijzer-14 in 4 on both the training
points and the test grid of step 0.01 across [−3, 3]². Keijzer-14 fit its training points in 15 of the 20 runs, but 11 of those built
their constants as x · (1/x), which isn't finite at x = 0 on the test grid. Keijzer-15 and Pagie-1
weren't recovered in 5 runs each; Keijzer-8 (√x) and Vladislavleva-6 (6 sin x cos y) were, from
the first generations, too easily for pages of their own. No problem of G4 reached exact recovery
reliably, so, by the rule of section 3.3, none was added: they wait for constant fitting in 0.15,
with `Constants::normal` and `Math::Inv`, which G4 added for them.

Per batch, AGENTS.md gains the rows and a template (a GP and a NEAT program, doctested),
`docs/features.md` and README's "What's in it" the features, the Python README the classes.

## 10. Benchmarks

Later, under the suite's rules (rule 6: one matched method per problem, each library's own
implementation set to a written definition), once the features have settled:

- **GP:** a matched tree GP on a symbolic regression problem (ramped half-and-half 2 to 6, subtree
  crossover with or without the 90/10 bias, subtree mutation, depth limit 17, tournament size 7,
  RMSE without linear scaling), against DEAP's `gp` and radiate's trees. gplearn, Operon and PySR
  run different algorithms and would fail the matching. Measures: evaluations per second (where
  the vectorized evaluator shows) and time to exact recovery.
- **NEAT:** XOR and the double pole with velocities against neat-python and radiate's graphs, with
  the paper's settings written as the definition (neat-python's structural keys and per-node biases
  are differences the definition has to allow or exclude, which rule 6 settles first).
- **ES neuroevolution:** a fixed network on the double pole, CMA-ES against pycma (already in the
  suite for Rosenbrock), if a matched control task can be written for both.

## 11. Order of work

Each batch is a PR (or two) with its features, tests, Python classes where the batch says so,
docs (module docs citing the sources, AGENTS.md, `docs/features.md`, the READMEs) and examples; a
batch isn't done until every example reaches its optimum on the three platforms.

| Batch | Contents | Depends on | Examples |
|---|---|---|---|
| P1 | Python: `Es`, `AdaptiveReal`, `SelfAdaptiveMutation`, `Islands`, checkpoints (`checkpoint=`, `resume=`), reused numpy buffers | | none new: the Python versions of existing examples use them where they fit |
| G1 | `gp`: `Node`, `Tree`, `PrimitiveSet` (typed, ERCs), `Gp` (limits, grow, full, ramped half-and-half), parse and display, the three evaluators, `SubtreeCrossover`, `SubtreeMutation`, serde | | `koza_quartic` |
| G2 | Point, hoist, shrink, constant mutation, `gp::Mutations`; `LexicographicTournament`, `DoubleTournament`, `Tarpeian`; one-point crossover and PTC2 (optional); Boolean problems | G1 | `multiplexer_11`, `abs_typed` |
| G3 | `gp::regression`: `Math`, `Dataset`, `Regression` (RMSE, linear scaling, thread-local columns); Koza and Nguyen problems | G2 | the Nguyen pages, `accuracy_and_size` |
| G4 | Keijzer, Vladislavleva (and Pagie) problems that reach exact recovery | G3 | a page each |
| E1 | `nn` (`Mlp`, Elman), `problems::control` (cart-pole, double pole with and without velocities), `Policy` | | the three pole-balancing pages with CMA-ES |
| N1 | `neat`: `Network`, registry, `Neat` (speciation, sharing, reproduction), `FeedForward` | G2's selections not needed | `xor_neat` |
| N2 | Recurrent networks and activation; NEAT on the control tasks | N1, E1 | NEAT on the pole-balancing pages |
| E2 | `OpenEs` (antithetic sampling, centered ranks, SGD and Adam, parallel breeding); SNES if needed | E1 | `two_spirals` |
| P2 | Python: `gx.Tree`, the tree operators and selections, `gx.gp.Regression`, user numpy primitives, `gx.Neat`, `gx.Network`, `gx.OpenEs`, `gx.problems.regression` and `.control` | G3, N2, E2 | the Python versions of all the above |

**Recommendation and rationale.** P1 is independent and small: it can go first or alongside G1.
GP first among the rest (G1 to G3): it is the larger and more-asked-for half, and its batches each
end usable. E1 before N1's control work because the tasks and `Policy` serve both CMA-ES and NEAT,
and E1 is small. N1 and N2 next; E2 last among the Rust batches, as it adds a method rather than
closing a gap. P2 closes the milestone. If 0.11 grows too long, the cut is after G3, E1 and N1:
N2, E2, G4 and P2 move to a 0.11.x or into 0.12 (open question 11).

## 12. Open questions

Decided (2026-09-29): every recommendation below, as written. The questions stay for the record.

1. **Crossover and the limits.** Choose the second point among those that keep both children
   within the limits (recommended: no child is wasted, and the variation stays close to Koza's), or
   Koza's rule, where an over-limit child is replaced by its parent (DEAP's `staticLimit`; exact
   replication of papers that use it)? Both could exist, the first as the default.
2. **Default size limit.** 1024 nodes as a memory guard beside Koza's depth 17 (recommended), or
   no size limit by default (depth 17 alone allows 2^18 − 1 nodes in a binary tree)?
3. **Constant types.** `f64` constants for every type (recommended: one node size, one serde form,
   integral values for integer and Boolean types), or a generic constant type on `Tree` (a second
   type parameter everywhere, and Python can't choose it)?
4. **Linear scaling by default** in `Regression` (recommended: Keijzer's evidence, and exact
   recovery examples print the scaled expression), or off by default for comparability with papers
   that don't use it?
5. **Division.** No protected division in the default set, `Aq` recommended (recommended), or
   Koza's protected division by default as DEAP and gplearn do?
6. **Evaluation API names.** `evaluate`, `evaluate_columns`, `root`; `Columns` or `Workspace`.
7. **NEAT islands.** No `Migrate` for `Neat` in 0.11 (recommended), or structural node ids
   (derived from the split connection's endpoints, recursively) that make numbers the same on every
   island without a shared registry, at the cost of wider ids?
8. **Fitness sharing.** Normalized per generation for any objective (recommended default), with
   `Sharing::Raw` for exact replication; or raw only, with NEAT restricted to non-negative
   maximization?
9. **`OpenEs`.** Include it in 0.11 (recommended, batch E2) and under which name (`OpenEs` as
   evosax, `SalimansEs`, `Nes`)? Or leave neuroevolution with ES to CMA-ES and sep-CMA-ES plus the
   network module?
10. **Per-case errors.** Lexicase selection and some parsimony methods need a vector of errors per
    individual in selection. `Fitness` carries a score and a violation; `Evaluated` extras never
    reach the algorithm by design. Design it after 0.11 (recommended), as an optional per-case
    vector in the extras path of the optimization plan's section 2.3?
11. **Milestone cut.** Keep all batches in 0.11 (recommended while they go as planned), or release
    0.11 after P1, G1-G3, E1 and N1, and the rest as 0.11.x patch releases (new features, no
    changed results)?
12. **Grammatical evolution.** Out (recommended: an `Integer` genome and a decoder users can
    write), or a `grammar` module with PonyGE2-style BNF in a later milestone?
13. **Where things live.** `genoxide::gp` and `gp::regression` (recommended), `genoxide::neat`,
    `genoxide::nn`, `genoxide::problems::control`, `algorithm::open_es`; the new selections in
    `operator::select`. Or `problems::regression` for the regression problems, beside the others?
