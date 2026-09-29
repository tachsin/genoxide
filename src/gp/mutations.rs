//! Point, hoist, shrink and constant mutation, and a mix of tree mutations.

use super::operators::SubtreeMutation;
use super::primitives::{Constants, PrimitiveSet};
use super::representation::Gp;
use super::tree::{Node, Tree, subtree_end};
use crate::operator::Mutate;
use crate::operator::mutate::{Mode, changed_gene, reflect};
use crate::{Error, Result, StreamRng};
use std::fmt::Debug;

/// Point mutation (Poli, Langdon and McPhee 2008, sec. 5.2.2): nodes replaced by other
/// primitives of the same signature, each node with a probability, or `n` nodes.
///
/// A function is replaced by another function with the same argument types and return type, so
/// its children stay; a leaf by another terminal of its type or, if its type has
/// [`Constants`], a new constant (a constant by one of a different value). Nodes with no
/// replacement are never picked, so a picked node always changes, and the tree keeps its shape,
/// types and limits.
///
/// A per-node mutation changes no node with probability `(1 − rate)^n`; in a genetic algorithm,
/// such a child is a copy that inherits its parent's fitness without an evaluation.
///
/// ```
/// use genoxide::gp::{Gp, PointMutation, PrimitiveSet};
/// use genoxide::prelude::*;
///
/// #[derive(Clone, Copy, Debug)]
/// enum Op {
///     Add,
///     Mul,
///     X,
/// }
///
/// let mut set = PrimitiveSet::builder();
/// let real = set.new_type("real");
/// set.function("add", Op::Add, [real, real], real)
///     .function("mul", Op::Mul, [real, real], real)
///     .terminal("x", Op::X, real);
/// let gp = Gp::builder(set.build(real)?).build()?;
/// let mut tree = gp.primitives().parse("add(x, mul(x, x))")?;
/// PointMutation::count(1)?.mutate(&gp, &mut tree, &mut StreamRng::seed_from_u64(1));
/// // `x` has no replacement: one of the functions changed
/// let text = tree.display(gp.primitives()).to_string();
/// assert!(text == "mul(x, mul(x, x))" || text == "add(x, add(x, x))");
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PointMutation {
    mode: Mode,
}

impl PointMutation {
    /// Replaces each node that has a replacement independently with probability `rate` (greater
    /// than 0 and at most 1).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] (setting `point_mutation_rate`) for a rate outside (0, 1].
    pub fn per_node(rate: f64) -> Result<Self> {
        Ok(Self {
            mode: Mode::per_gene("point_mutation_rate", rate)?,
        })
    }

    /// Replaces `count` distinct random nodes that have a replacement, `count` at least 1; all of
    /// them if there are fewer.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] (setting `point_mutation_count`) for a count of 0.
    pub fn count(count: usize) -> Result<Self> {
        Ok(Self {
            mode: Mode::count("point_mutation_count", count)?,
        })
    }

    // mutates the tree; false if no node has a replacement (then the tree is unchanged)
    pub(crate) fn apply<P: Copy>(&self, gp: &Gp<P>, tree: &mut Tree, rng: &mut StreamRng) -> bool {
        let set = gp.primitives();
        let candidates: Vec<usize> = tree
            .nodes()
            .iter()
            .enumerate()
            .filter(|(_, node)| replacements(set, node) > 0)
            .map(|(position, _)| position)
            .collect();
        if candidates.is_empty() {
            return false;
        }
        let nodes = tree.nodes_mut();
        self.mode.apply(
            candidates.len(),
            |candidate| candidates[candidate],
            rng,
            |position, rng| nodes[position] = replace(set, nodes[position], rng),
        );
        true
    }
}

impl<P: Copy + Debug + Send + Sync> Mutate<Gp<P>> for PointMutation {
    fn mutate(&self, gp: &Gp<P>, tree: &mut Tree, rng: &mut StreamRng) {
        self.apply(gp, tree, rng);
    }
}

// the number of nodes that can replace `node`: the other primitives of its signature, and for a
// leaf of a type with constants, a new constant (of another value, for a constant)
fn replacements<P: Copy>(set: &PrimitiveSet<P>, node: &Node) -> usize {
    match *node {
        Node::Primitive(index) => {
            let peers = set.same_signature(index).len() - 1;
            let constant = set.arity(index) == 0 && set.constants(set.node_type(node)).is_some();
            peers + usize::from(constant)
        }
        Node::Constant { ty, value } => {
            let constant = set.constants(ty).is_some_and(|c| c.has_other(value));
            set.terminals_of(ty).len() + usize::from(constant)
        }
    }
}

// one of the replacements of `node`, uniformly
fn replace<P: Copy>(set: &PrimitiveSet<P>, node: Node, rng: &mut StreamRng) -> Node {
    let choice = rng.below(replacements(set, &node));
    let ty = set.node_type(&node);
    let constants = || set.constants(ty).expect("constants");
    match node {
        Node::Primitive(index) => {
            let peers = set.same_signature(index);
            if choice + 1 < peers.len() {
                // the peers other than `index`, in order
                let own = peers.binary_search(&index).expect("its own signature");
                Node::Primitive(peers[choice + usize::from(choice >= own)])
            } else {
                Node::Constant {
                    ty,
                    value: constants().sample(rng),
                }
            }
        }
        Node::Constant { value, .. } => {
            let terminals = set.terminals_of(ty);
            if choice < terminals.len() {
                Node::Primitive(terminals[choice])
            } else {
                Node::Constant {
                    ty,
                    value: constants().other(value, rng),
                }
            }
        }
    }
}

/// Hoist mutation (Kinnear 1993; Poli, Langdon and McPhee 2008, sec. 5.2.2): the tree is
/// replaced by the subtree of one of its function nodes, chosen uniformly among those of the
/// root's type other than the root.
///
/// The child is always smaller than its parent, which makes hoist a move against bloat. A tree
/// without such a function node (a single node, or a function of leaves) is left unchanged.
///
/// ```
/// use genoxide::gp::{Gp, HoistMutation, PrimitiveSet};
/// use genoxide::prelude::*;
///
/// #[derive(Clone, Copy, Debug)]
/// enum Op {
///     Add,
///     X,
/// }
///
/// let mut set = PrimitiveSet::builder();
/// let real = set.new_type("real");
/// set.function("add", Op::Add, [real, real], real).terminal("x", Op::X, real);
/// let gp = Gp::builder(set.build(real)?).build()?;
/// let mut tree = gp.primitives().parse("add(x, add(x, x))")?;
/// HoistMutation.mutate(&gp, &mut tree, &mut StreamRng::seed_from_u64(1));
/// assert_eq!(tree, gp.primitives().parse("add(x, x)")?); // the only function below the root
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct HoistMutation;

impl HoistMutation {
    // mutates the tree; false if it has no function node of the root's type below the root (then
    // unchanged)
    pub(crate) fn apply<P: Copy>(&self, gp: &Gp<P>, tree: &mut Tree, rng: &mut StreamRng) -> bool {
        let set = gp.primitives();
        let Some(root) = tree.nodes().first() else {
            return false;
        };
        let ty = set.node_type(root);
        let hoistable = |node: &Node| set.node_arity(node) > 0 && set.node_type(node) == ty;
        let count = tree.nodes()[1..].iter().filter(|n| hoistable(n)).count();
        if count == 0 {
            return false;
        }
        let k = rng.below(count);
        let (start, _) = tree.nodes()[1..]
            .iter()
            .enumerate()
            .filter(|(_, n)| hoistable(n))
            .nth(k)
            .expect("a subtree");
        let start = start + 1;
        let end = subtree_end(set, tree.nodes(), start);
        let nodes = tree.nodes_mut();
        nodes.truncate(end);
        nodes.drain(..start);
        true
    }
}

impl<P: Copy + Debug + Send + Sync> Mutate<Gp<P>> for HoistMutation {
    fn mutate(&self, gp: &Gp<P>, tree: &mut Tree, rng: &mut StreamRng) {
        self.apply(gp, tree, rng);
    }
}

/// Shrink mutation (Poli, Langdon and McPhee 2008, sec. 5.2.2, after Angeline 1996): a subtree
/// is replaced by a random terminal of its type: one of the type's terminals, or a new constant
/// if the type has [`Constants`], each choice with the same probability.
///
/// The subtree is chosen uniformly among those whose root is a function of a type with leaves,
/// so the child is always smaller than its parent. A tree without such a subtree is left
/// unchanged.
///
/// ```
/// use genoxide::gp::{Gp, PrimitiveSet, ShrinkMutation};
/// use genoxide::prelude::*;
///
/// #[derive(Clone, Copy, Debug)]
/// enum Op {
///     Add,
///     X,
/// }
///
/// let mut set = PrimitiveSet::builder();
/// let real = set.new_type("real");
/// set.function("add", Op::Add, [real, real], real).terminal("x", Op::X, real);
/// let gp = Gp::builder(set.build(real)?).build()?;
/// let mut tree = gp.primitives().parse("add(x, add(x, x))")?;
/// ShrinkMutation.mutate(&gp, &mut tree, &mut StreamRng::seed_from_u64(1));
/// let text = tree.display(gp.primitives()).to_string();
/// assert!(text == "x" || text == "add(x, x)");
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ShrinkMutation;

impl ShrinkMutation {
    // mutates the tree; false if no function's type has leaves (then unchanged)
    pub(crate) fn apply<P: Copy>(&self, gp: &Gp<P>, tree: &mut Tree, rng: &mut StreamRng) -> bool {
        let set = gp.primitives();
        let leaves = |node: &Node| {
            let ty = set.node_type(node);
            set.terminals_of(ty).len() + usize::from(set.constants(ty).is_some())
        };
        let shrinkable = |node: &Node| set.node_arity(node) > 0 && leaves(node) > 0;
        let count = tree.nodes().iter().filter(|n| shrinkable(n)).count();
        if count == 0 {
            return false;
        }
        let k = rng.below(count);
        let (start, node) = tree
            .nodes()
            .iter()
            .enumerate()
            .filter(|(_, n)| shrinkable(n))
            .nth(k)
            .expect("a subtree");
        let ty = set.node_type(node);
        let choice = rng.below(leaves(node));
        let terminals = set.terminals_of(ty);
        let leaf = if choice < terminals.len() {
            Node::Primitive(terminals[choice])
        } else {
            Node::Constant {
                ty,
                value: set.constants(ty).expect("constants").sample(rng),
            }
        };
        let end = subtree_end(set, tree.nodes(), start);
        tree.nodes_mut().splice(start..end, [leaf]);
        true
    }
}

impl<P: Copy + Debug + Send + Sync> Mutate<Gp<P>> for ShrinkMutation {
    fn mutate(&self, gp: &Gp<P>, tree: &mut Tree, rng: &mut StreamRng) {
        self.apply(gp, tree, rng);
    }
}

/// Constant mutation (Poli, Langdon and McPhee 2008, sec. 5.2.2, after Schoenauer et al. 1996):
/// one constant, chosen uniformly, is perturbed by normal noise of standard deviation `sigma`
/// times the width of its type's range, mirrored at the ends of the range. Each call changes one
/// constant, as each change is a separate mutation there.
///
/// It applies to constants of [`Constants::uniform`] and [`Constants::integers`] with more than
/// one value (integers are rounded); the constants of a [`Constants::choice`] have no range, and
/// [`PointMutation`] draws them anew. The constant always changes: the noise is drawn again
/// while it leaves the value as it is (for integers, while it rounds to the same value), up to 64
/// times, then the value is drawn uniformly from the others. A tree without such a constant is
/// left unchanged. The cheap tuning of constants: it keeps the tree's shape and moves a constant
/// a little.
///
/// ```
/// use genoxide::gp::{Constants, ConstantMutation, Gp, PrimitiveSet};
/// use genoxide::prelude::*;
///
/// #[derive(Clone, Copy, Debug)]
/// enum Op {
///     Mul,
///     X,
/// }
///
/// let mut set = PrimitiveSet::builder();
/// let real = set.new_type("real");
/// set.function("mul", Op::Mul, [real, real], real)
///     .terminal("x", Op::X, real)
///     .constants(real, Constants::uniform(-1.0..=1.0)?);
/// let gp = Gp::builder(set.build(real)?).build()?;
/// let mut tree = gp.primitives().parse("mul(x, 0.5)")?;
/// ConstantMutation::gaussian(0.05)?.mutate(&gp, &mut tree, &mut StreamRng::seed_from_u64(1));
/// assert_ne!(tree, gp.primitives().parse("mul(x, 0.5)")?);
/// assert!(gp.validate(&tree).is_ok()); // still within [-1, 1]
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ConstantMutation {
    sigma: f64,
}

impl ConstantMutation {
    /// Normal noise of standard deviation `sigma` (positive and finite) times the width of the
    /// constant's range, e.g. 0.1.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] (setting `constant_sigma`) for a `sigma` that isn't positive
    /// and finite.
    pub fn gaussian(sigma: f64) -> Result<Self> {
        if !(sigma > 0.0 && sigma.is_finite()) {
            return Err(Error::InvalidSetting {
                setting: "constant_sigma",
                reason: format!("must be positive and finite, got {sigma}"),
            });
        }
        Ok(Self { sigma })
    }

    /// The standard deviation of the noise, as a fraction of a constant's range.
    pub fn sigma(&self) -> f64 {
        self.sigma
    }

    // mutates the tree; false if it has no constant with a range (then unchanged)
    pub(crate) fn apply<P: Copy>(&self, gp: &Gp<P>, tree: &mut Tree, rng: &mut StreamRng) -> bool {
        let set = gp.primitives();
        let ranged = |node: &Node| match *node {
            Node::Constant { ty, .. } => {
                matches!(
                    set.constants(ty),
                    Some(Constants::Uniform { low, high }) if low < high
                ) || matches!(
                    set.constants(ty),
                    Some(Constants::Integers { low, high }) if low < high
                )
            }
            Node::Primitive(_) => false,
        };
        let count = tree.nodes().iter().filter(|n| ranged(n)).count();
        if count == 0 {
            return false;
        }
        let k = rng.below(count);
        let (position, _) = tree
            .nodes()
            .iter()
            .enumerate()
            .filter(|(_, n)| ranged(n))
            .nth(k)
            .expect("a constant");
        let Node::Constant { ty, value } = tree.nodes()[position] else {
            unreachable!("a constant");
        };
        let sigma = self.sigma;
        let value = match *set.constants(ty).expect("constants") {
            Constants::Uniform { low, high } => {
                let range = low..=high;
                let scale = sigma * (high - low);
                changed_gene(&range, value, rng, |rng| {
                    reflect(value + scale * rng.normal(), &range)
                })
            }
            Constants::Integers { low, high } => {
                let range = low as f64..=high as f64;
                let scale = sigma * (high - low) as f64;
                let mut changed = None;
                for _ in 0..64 {
                    let proposal = reflect(value + scale * rng.normal(), &range).round();
                    if proposal != value && range.contains(&proposal) {
                        changed = Some(proposal);
                        break;
                    }
                }
                changed.unwrap_or_else(|| {
                    crate::genome::integer::random_other_in(&(low..=high), value as i64, rng) as f64
                })
            }
            Constants::Choice(_) => unreachable!("a constant with a range"),
        };
        tree.nodes_mut()[position] = Node::Constant { ty, value };
        true
    }
}

impl<P: Copy + Debug + Send + Sync> Mutate<Gp<P>> for ConstantMutation {
    fn mutate(&self, gp: &Gp<P>, tree: &mut Tree, rng: &mut StreamRng) {
        self.apply(gp, tree, rng);
    }
}

/// One of the tree mutations, for [`Mutations`].
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TreeMutation {
    /// [`SubtreeMutation`].
    Subtree(SubtreeMutation),
    /// [`PointMutation`].
    Point(PointMutation),
    /// [`HoistMutation`].
    Hoist(HoistMutation),
    /// [`ShrinkMutation`].
    Shrink(ShrinkMutation),
    /// [`ConstantMutation`].
    Constant(ConstantMutation),
}

impl TreeMutation {
    // mutates the tree; false if this mutation can't change it (then unchanged)
    fn apply<P: Copy>(&self, gp: &Gp<P>, tree: &mut Tree, rng: &mut StreamRng) -> bool {
        match self {
            TreeMutation::Subtree(mutation) => mutation.apply(gp, tree, rng),
            TreeMutation::Point(mutation) => mutation.apply(gp, tree, rng),
            TreeMutation::Hoist(mutation) => mutation.apply(gp, tree, rng),
            TreeMutation::Shrink(mutation) => mutation.apply(gp, tree, rng),
            TreeMutation::Constant(mutation) => mutation.apply(gp, tree, rng),
        }
    }
}

impl<P: Copy + Debug + Send + Sync> Mutate<Gp<P>> for TreeMutation {
    fn mutate(&self, gp: &Gp<P>, tree: &mut Tree, rng: &mut StreamRng) {
        self.apply(gp, tree, rng);
    }
}

macro_rules! tree_mutation {
    ($($variant:ident($mutation:ty)),*) => {
        $(
            impl From<$mutation> for TreeMutation {
                fn from(mutation: $mutation) -> Self {
                    TreeMutation::$variant(mutation)
                }
            }
        )*
    };
}

tree_mutation!(
    Subtree(SubtreeMutation),
    Point(PointMutation),
    Hoist(HoistMutation),
    Shrink(ShrinkMutation),
    Constant(ConstantMutation)
);

/// A mix of tree mutations: each call applies one of them, chosen by weight.
///
/// If the chosen mutation can't change the tree (hoist of a single node, constant mutation of a
/// tree without constants), another is chosen by weight among the rest, so a tree changes
/// whenever one of the mutations can change it. The weights needn't add up to 1.
///
/// ```
/// use genoxide::gp::{Gp, Mutations, PrimitiveSet, SubtreeCrossover};
/// use genoxide::prelude::*;
///
/// #[derive(Clone, Copy, Debug)]
/// enum Op {
///     Add,
///     Mul,
///     X,
/// }
///
/// let mut set = PrimitiveSet::builder();
/// let real = set.new_type("real");
/// set.function("add", Op::Add, [real, real], real)
///     .function("mul", Op::Mul, [real, real], real)
///     .terminal("x", Op::X, real);
/// let gp = Gp::builder(set.build(real)?).build()?;
/// let ga = Ga::builder(gp)
///     .population_size(100)
///     .select(DoubleTournament::new(7, 1.4)?)
///     .crossover(SubtreeCrossover::new())
///     .mutate(Mutations::builder().subtree(0.6).point(0.2).hoist(0.1).shrink(0.1).build()?)
///     .mutation_rate(0.2)
///     .build()?;
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct Mutations {
    mutations: Vec<TreeMutation>,
    weights: Vec<f64>,
}

/// Builds [`Mutations`]: see there.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MutationsBuilder {
    mutations: Vec<TreeMutation>,
    weights: Vec<f64>,
}

impl Mutations {
    /// A builder: add mutations with their weights, then [`build`](MutationsBuilder::build).
    pub fn builder() -> MutationsBuilder {
        MutationsBuilder::default()
    }

    /// The mutations, in the order they were added.
    pub fn mutations(&self) -> &[TreeMutation] {
        &self.mutations
    }

    /// Their weights, in the same order.
    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    // mutates the tree with one of the mutations, by weight, trying the others by weight while
    // the chosen one can't change the tree; false if none can
    fn apply<P: Copy>(&self, gp: &Gp<P>, tree: &mut Tree, rng: &mut StreamRng) -> bool {
        // the mutations that couldn't change the tree, allocated at the first
        let mut tried: Vec<bool> = Vec::new();
        let mut left: f64 = self.weights.iter().sum();
        loop {
            let point = rng.unit_f64() * left;
            let mut sum = 0.0;
            let mut chosen = None;
            for (index, &weight) in self.weights.iter().enumerate() {
                if weight == 0.0 || tried.get(index).copied().unwrap_or(false) {
                    continue;
                }
                sum += weight;
                chosen = Some(index);
                if point < sum {
                    break;
                }
            }
            let Some(index) = chosen else {
                return false;
            };
            if self.mutations[index].apply(gp, tree, rng) {
                return true;
            }
            tried.resize(self.weights.len(), false);
            tried[index] = true;
            // what's left, added up again rather than subtracted, so no rounding remains
            left = self
                .weights
                .iter()
                .zip(&tried)
                .filter(|&(_, &tried)| !tried)
                .map(|(weight, _)| weight)
                .sum();
            if left <= 0.0 {
                return false;
            }
        }
    }
}

impl<P: Copy + Debug + Send + Sync> Mutate<Gp<P>> for Mutations {
    fn mutate(&self, gp: &Gp<P>, tree: &mut Tree, rng: &mut StreamRng) {
        self.apply(gp, tree, rng);
    }
}

impl MutationsBuilder {
    /// Adds [`SubtreeMutation::new`] with `weight`.
    pub fn subtree(self, weight: f64) -> Self {
        self.with(weight, SubtreeMutation::new())
    }

    /// Adds [`PointMutation::count`] of one node with `weight`.
    pub fn point(self, weight: f64) -> Self {
        self.with(
            weight,
            PointMutation {
                mode: Mode::Count(1),
            },
        )
    }

    /// Adds [`HoistMutation`] with `weight`.
    pub fn hoist(self, weight: f64) -> Self {
        self.with(weight, HoistMutation)
    }

    /// Adds [`ShrinkMutation`] with `weight`.
    pub fn shrink(self, weight: f64) -> Self {
        self.with(weight, ShrinkMutation)
    }

    /// Adds any tree mutation with its settings, e.g.
    /// `.with(0.1, ConstantMutation::gaussian(0.1)?)`.
    pub fn with(mut self, weight: f64, mutation: impl Into<TreeMutation>) -> Self {
        self.mutations.push(mutation.into());
        self.weights.push(weight);
        self
    }

    /// The mix.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] (setting `mutations`) for a weight that is negative or not
    /// finite, a total weight that isn't positive and finite, or more than 2^24 mutations.
    pub fn build(self) -> Result<Mutations> {
        let invalid = |reason: String| {
            Err(Error::InvalidSetting {
                setting: "mutations",
                reason,
            })
        };
        crate::operator::check_size("mutations", self.weights.len())?;
        if let Some(weight) = self
            .weights
            .iter()
            .find(|weight| !(weight.is_finite() && **weight >= 0.0))
        {
            return invalid(format!(
                "weights must be finite and at least 0, got {weight}"
            ));
        }
        let total: f64 = self.weights.iter().sum();
        if !(total > 0.0 && total.is_finite()) {
            return invalid(format!(
                "the weights must add up to a positive, finite total, got {total}"
            ));
        }
        Ok(Mutations {
            mutations: self.mutations,
            weights: self.weights,
        })
    }
}

// deserialized like `build`
#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Mutations {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(rename = "Mutations")]
        struct Raw {
            mutations: Vec<TreeMutation>,
            weights: Vec<f64>,
        }
        let raw = Raw::deserialize(deserializer)?;
        if raw.mutations.len() != raw.weights.len() {
            return Err(serde::de::Error::custom(
                "as many weights as mutations are needed",
            ));
        }
        MutationsBuilder {
            mutations: raw.mutations,
            weights: raw.weights,
        }
        .build()
        .map_err(serde::de::Error::custom)
    }
}
