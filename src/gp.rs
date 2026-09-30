//! Tree genetic programming, strongly typed: programs and formulas as genomes.
//!
//! A genetic program is a tree of primitives: functions (`add`, `if`) whose children are their
//! arguments, and terminals (inputs such as `x`) and constants at the leaves (Koza 1992). Evolved
//! by a [`Ga`](crate::Ga) with tree operators, it finds formulas that fit data (symbolic
//! regression), classifiers, or controllers, as readable expressions.
//!
//! | Part | What it is |
//! |---|---|
//! | [`PrimitiveSet`] | The functions, terminals and [`Constants`] with their [`Type`]s: the user's `Copy` values, e.g. an enum, whose meaning the fitness function gives |
//! | [`Tree`] | The genome: [`Node`]s in a flat array in prefix order; [`parse`](PrimitiveSet::parse) and [`display`](Tree::display) as `add(x, mul(x, 0.5))` |
//! | [`Gp`] | The representation: a set, a depth limit (17) and a size limit (1024), and the initialization, [`Init`] (ramped half-and-half, depths 2 to 6) |
//! | [`SubtreeCrossover`] | Exchanges subtrees of the same type, at function nodes 90% of the time, keeping both children within the limits |
//! | [`SubtreeMutation`] | Replaces a subtree by a new one grown in its place |
//! | [`PointMutation`], [`HoistMutation`], [`ShrinkMutation`], [`ConstantMutation`] | Replace nodes by others of the same signature, the tree by one of its subtrees, a subtree by a leaf, move a constant |
//! | [`Mutations`] | A mix of the mutations, one per call, by weight |
//! | [`OnePointCrossover`] | Exchanges subtrees at a point of the two trees' common region (Poli and Langdon 1998) |
//! | [`boolean`] | Koza's multiplexer and even-parity problems |
//! | [`regression`] | Symbolic regression: mathematical primitives, datasets, the error after linear scaling, and test problems |
//!
//! Against bloat, the growth of trees without better fitness, [`operator::select`](crate::operator::select)
//! has [`DoubleTournament`](crate::operator::DoubleTournament),
//! [`LexicographicTournament`](crate::operator::LexicographicTournament) and
//! [`Tarpeian`](crate::operator::Tarpeian), which see a tree's size as its
//! [`len`](crate::genome::Genome::len); hoist and shrink mutation make trees smaller.
//!
//! Three ways to evaluate a tree, none allocating once its workspace has grown:
//!
//! - [`Tree::evaluate`]: bottom-up on a stack of values of any type, e.g. at one point of data.
//! - [`Tree::evaluate_columns`]: on columns of all the points of the data at once, several times
//!   faster for data, with the same results to the bit.
//! - [`Tree::root`]: the root, to walk the tree from the top with [`Subtree::children`], for
//!   interpreters that choose which children to run.
//!
//! ```
//! use genoxide::gp::{Gp, PrimitiveSet, SubtreeCrossover, SubtreeMutation, Tree};
//! use genoxide::prelude::*;
//!
//! #[derive(Clone, Copy, Debug)]
//! enum Op {
//!     Add,
//!     Mul,
//!     X,
//! }
//!
//! // x² + x at 10 points, found exactly
//! let xs: Vec<f64> = (0..10).map(|i| f64::from(i) / 5.0 - 1.0).collect();
//! let mut set = PrimitiveSet::builder();
//! let real = set.new_type("real");
//! set.function("add", Op::Add, [real, real], real)
//!     .function("mul", Op::Mul, [real, real], real)
//!     .terminal("x", Op::X, real);
//! let gp = Gp::builder(set.build(real)?).build()?;
//! let set = gp.primitives().clone();
//! let error = |tree: &Tree| {
//!     let mut stack = Vec::new();
//!     xs.iter()
//!         .map(|&x| {
//!             let value = tree.evaluate(&set, &mut stack, |op, args: &[f64]| match op {
//!                 Op::Add => args[0] + args[1],
//!                 Op::Mul => args[0] * args[1],
//!                 Op::X => x,
//!             }, |_, c| c);
//!             (value - (x * x + x)).abs()
//!         })
//!         .sum::<f64>()
//! };
//! let ga = Ga::builder(gp)
//!     .population_size(100)
//!     .select(Tournament::new(3)?)
//!     .crossover(SubtreeCrossover::new())
//!     .mutate(SubtreeMutation::new())
//!     .mutation_rate(0.1)
//!     .minimize()
//!     .seed(1)
//!     .build()?;
//! let outcome = Engine::new(ga, error)
//!     .stop_when(Stop::target(0.0).or(Stop::generations(100)))
//!     .run()?;
//! assert_eq!(outcome.best_fitness(), Fitness::new(0.0));
//! println!("{}", outcome.best_genome().display(&set));
//! # Ok::<(), genoxide::Error>(())
//! ```
//!
//! **Why a flat array.** A subtree is a contiguous range of nodes, so crossover is two splices
//! and mutation one; cloning, comparing, hashing and serializing a tree are loops over its
//! nodes, never recursive, so no tree is too deep for them; and a prefix array read backwards is
//! a postfix program that a stack machine runs directly, with no compile step.
//!
//! **Types** (Montana 1995): every function has argument types and a return type, every
//! terminal and constant a type, and every tree genoxide makes is well typed: generation,
//! crossover and mutation choose only primitives and subtrees of the type wanted where they
//! go. Untyped genetic programming is the case of one type. A set's
//! [`build`](PrimitiveSetBuilder::build) computes the fewest nodes of a tree of each type at each
//! depth (Montana's table of the types possible at each depth, with sizes), which lets
//! generation always complete a tree within both limits, and rejects a set whose types can't
//! make a tree.
//!
//! **Limits.** Koza's depth limit of 17 (the root at depth 0) bounds trees made by crossover; a
//! size limit of 1024 nodes bounds memory, since depth 17 alone allows 2^18 − 1 nodes with
//! binary functions. Instead of making an over-limit child and replacing it by a parent, as Koza
//! does, subtree crossover chooses its second point among those that keep both children within
//! the limits, so no child is wasted.
//!
//! **Reproducible.** Generation and the operators draw only from the stream they're given, so
//! seeded runs are the same on any platform and number of threads, with parallel breeding too.
//!
//! References:
//!
//! - Koza, J. R. (1992). *Genetic Programming: On the Programming of Computers by Means of
//!   Natural Selection.* MIT Press. Trees of functions and terminals, ephemeral random constants,
//!   the full, grow and ramped half-and-half methods, subtree crossover with its 90% bias to
//!   function nodes, the depth limit of 17, mutation.
//! - Koza, J. R. (1994). Genetic programming as a means for programming computers by natural
//!   selection. *Statistics and Computing* 4(2): 87-112. doi:10.1007/BF00175355. Restates the
//!   book's defaults: initial depth 6, depth 17 after crossover, crossover points at function
//!   nodes 90% of the time and at terminals 10%.
//! - Montana, D. J. (1995). Strongly typed genetic programming. *Evolutionary Computation* 3(2):
//!   199-230. doi:10.1162/evco.1995.3.2.199. Types, and generating typed trees within a depth
//!   from a table of the types possible at each depth.
//! - Poli, R., Langdon, W. B. and McPhee, N. F. (2008). *A Field Guide to Genetic Programming.*
//!   lulu.com, <http://www.gp-field-guide.org.uk>. Prefix arrays; subtree mutation; point, hoist
//!   and shrink mutation (sec. 5.2.2, shrink after Angeline 1996, as cited there); mutation of
//!   constants by Gaussian noise (after Schoenauer et al. 1996, as cited there); one-point
//!   crossover's common region (sec. 5.3).
//! - Angeline, P. J. (1996). An investigation into the sensitivity of genetic programming to the
//!   frequency of leaf selection during subtree crossover. *Genetic Programming 1996*: 21-29.
//!   The internal-point rate as a setting.
//! - Kinnear, K. E. Jr. (1993). Evolving a sort: lessons in genetic programming. *IEEE
//!   International Conference on Neural Networks 1993*. Hoist mutation: a copy of the subtree of
//!   a function node becomes the new individual. (The Field Guide cites Kinnear 1994, *IEEE World
//!   Congress on Computational Intelligence 1994*: 142-147, doi:10.1109/ICEC.1994.350026, which
//!   uses hoist and refers to the 1993 paper for its definition.)
//! - Poli, R. and Langdon, W. B. (1998). Schema theory for genetic programming with one-point
//!   crossover and point mutation. *Evolutionary Computation* 6(3): 231-252.
//!   doi:10.1162/evco.1998.6.3.231. One-point crossover.

pub mod boolean;
mod evaluate;
mod mutations;
mod operators;
mod primitives;
pub mod regression;
mod representation;
mod tree;

pub use evaluate::Columns;
pub use mutations::{
    ConstantMutation, HoistMutation, Mutations, MutationsBuilder, PointMutation, ShrinkMutation,
    TreeMutation,
};
pub use operators::{OnePointCrossover, SubtreeCrossover, SubtreeMutation};
pub use primitives::{Constants, Primitive, PrimitiveSet, PrimitiveSetBuilder, Type};
pub use representation::{Gp, GpBuilder, Init};
pub use tree::{Children, Display, Node, Subtree, Tree};
