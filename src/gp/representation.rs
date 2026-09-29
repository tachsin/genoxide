//! The representation of genetic programs: a primitive set, limits and an initialization.

use super::primitives::{INFINITE, PrimitiveSet, Type};
use super::tree::{Node, Tree};
use crate::genome::Representation;
use crate::operator::check_size;
use crate::rng::Chance;
use crate::{Error, Result, StreamRng};
use std::collections::HashSet;
use std::fmt::Debug;
use std::ops::RangeInclusive;

/// How [`Gp`] makes random trees (Koza 1992, ch. 6), of a depth drawn uniformly from `depths`.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Init {
    /// Functions down to the depth, then terminals: every leaf at the depth where the types
    /// and the size limit allow.
    Full {
        /// The depths.
        depths: RangeInclusive<usize>,
    },
    /// Any primitive of the wanted type that fits the remaining depth: trees of every shape up
    /// to the depth.
    Grow {
        /// The depths.
        depths: RangeInclusive<usize>,
    },
    /// Full or grow, each with probability ½ (Koza divides the population evenly among the
    /// depths and the two methods; [`Gp::ramped_half_and_half`] does exactly that).
    RampedHalfAndHalf {
        /// The depths.
        depths: RangeInclusive<usize>,
    },
}

impl Init {
    /// The depths of the trees.
    pub fn depths(&self) -> &RangeInclusive<usize> {
        match self {
            Init::Full { depths } | Init::Grow { depths } | Init::RampedHalfAndHalf { depths } => {
                depths
            }
        }
    }
}

impl Default for Init {
    /// Ramped half-and-half with depths 2 to 6 (Koza 1992).
    fn default() -> Self {
        Init::RampedHalfAndHalf { depths: 2..=6 }
    }
}

// how a tree is grown
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Method {
    Full,
    Grow,
}

/// Trees of a genetic program ([`Tree`]): a strongly typed [`PrimitiveSet`], limits on depth
/// and size, and how random trees are made.
///
/// - **Depth limit** (default 17, Koza 1992): the root is at depth 0, so a tree of one node has
///   depth 0.
/// - **Size limit** (default 1024 nodes): a memory guard beside the depth limit, which alone
///   allows trees of 2^18 − 1 nodes with binary functions.
/// - **Initialization** (default [`Init::RampedHalfAndHalf`] with depths 2 to 6, Koza 1992).
///
/// Every tree it makes, and every tree the [`gp`](crate::gp) operators make from its trees, is
/// typed and within the limits.
///
/// ```
/// use genoxide::gp::{Gp, Init, PrimitiveSet};
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
/// let gp = Gp::builder(set.build(real)?)
///     .init(Init::RampedHalfAndHalf { depths: 2..=6 }) // the default
///     .max_depth(17) // the default
///     .max_size(1024) // the default
///     .build()?;
/// let tree = gp.random_genome(&mut StreamRng::seed_from_u64(1));
/// assert!(gp.validate(&tree).is_ok());
/// assert!((2..=6).contains(&tree.depth(gp.primitives())));
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(bound(serialize = "P: serde::Serialize")))]
pub struct Gp<P> {
    primitives: PrimitiveSet<P>,
    init: Init,
    max_depth: usize,
    max_size: usize,
}

/// Builds a [`Gp`]: see there.
#[derive(Clone, Debug)]
pub struct GpBuilder<P> {
    primitives: PrimitiveSet<P>,
    init: Init,
    max_depth: usize,
    max_size: usize,
}

impl<P: Copy> Gp<P> {
    /// A builder of the trees of `primitives`, with the default limits and initialization.
    pub fn builder(primitives: PrimitiveSet<P>) -> GpBuilder<P> {
        GpBuilder {
            primitives,
            init: Init::default(),
            max_depth: 17,
            max_size: 1024,
        }
    }

    /// The primitive set.
    pub fn primitives(&self) -> &PrimitiveSet<P> {
        &self.primitives
    }

    /// How random trees are made.
    pub fn init(&self) -> &Init {
        &self.init
    }

    /// The largest depth of a tree.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }

    /// The most nodes of a tree.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// A tree made by the full method: functions down to `depth`, then terminals, where the
    /// types and the size limit allow. The root is a function when one fits (Koza 1992).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] (setting `depth`) for a depth above the depth limit, or at
    /// which no tree of the root type fits the size limit.
    pub fn full(&self, depth: usize, rng: &mut StreamRng) -> Result<Tree> {
        self.check_depth(depth)?;
        Ok(self.make(depth, Method::Full, rng))
    }

    /// A tree made by the grow method: any primitive of the wanted type that fits the remaining
    /// depth, down to `depth` at most. The root is a function when one fits (Koza 1992).
    ///
    /// # Errors
    ///
    /// As [`full`](Gp::full).
    pub fn grow(&self, depth: usize, rng: &mut StreamRng) -> Result<Tree> {
        self.check_depth(depth)?;
        Ok(self.make(depth, Method::Grow, rng))
    }

    /// `count` trees by ramped half-and-half, as Koza (1992) divides a population: the same
    /// number of trees of each depth of [`init`](Gp::init)'s range (the first depths get one more
    /// when `count` isn't a multiple), half of each by full and half by grow, each tree drawn
    /// again (up to 100 times) while it equals one made before. For
    /// [`initial_genomes`](crate::algorithm::ga::GaBuilder::initial_genomes).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] (setting `count`) for a count above 2^24.
    pub fn ramped_half_and_half(&self, count: usize, rng: &mut StreamRng) -> Result<Vec<Tree>> {
        check_size("count", count)?;
        let depths = self.init.depths();
        let levels = depths.end() - depths.start() + 1;
        let mut seen = HashSet::with_capacity(count);
        let mut trees = Vec::with_capacity(count);
        for index in 0..count {
            let depth = depths.start() + index % levels;
            let method = if (index / levels).is_multiple_of(2) {
                Method::Full
            } else {
                Method::Grow
            };
            let mut tree = self.make(depth, method, rng);
            for _ in 0..100 {
                if !seen.contains(&tree) {
                    break;
                }
                tree = self.make(depth, method, rng);
            }
            seen.insert(tree.clone());
            trees.push(tree);
        }
        Ok(trees)
    }

    // the depth and method of a random tree
    pub(crate) fn draw(&self, rng: &mut StreamRng) -> (usize, Method) {
        let depths = self.init.depths();
        let depth = depths.start() + rng.below(depths.end() - depths.start() + 1);
        let method = match self.init {
            Init::Full { .. } => Method::Full,
            Init::Grow { .. } => Method::Grow,
            Init::RampedHalfAndHalf { .. } => {
                if rng.chance(Chance::Half) {
                    Method::Full
                } else {
                    Method::Grow
                }
            }
        };
        (depth, method)
    }

    fn check_depth(&self, depth: usize) -> Result<()> {
        if depth > self.max_depth {
            return Err(Error::InvalidSetting {
                setting: "depth",
                reason: format!("must be at most max_depth, {}, got {depth}", self.max_depth),
            });
        }
        let root = self.primitives.root();
        let depth = self.effective_depth(depth);
        if self.primitives.min_size(root, depth) as usize > self.max_size {
            return Err(Error::InvalidSetting {
                setting: "depth",
                reason: format!(
                    "no tree of depth at most {depth} fits in max_size, {}",
                    self.max_size
                ),
            });
        }
        Ok(())
    }

    // the depth of a tree made for `depth`: at least the smallest depth of the root type
    fn effective_depth(&self, depth: usize) -> usize {
        let root = self.primitives.root();
        depth.max(self.primitives.min_depth(root).unwrap_or(0))
    }

    // a tree of the root type made by `method`, for a checked depth
    fn make(&self, depth: usize, method: Method, rng: &mut StreamRng) -> Tree {
        let depth = self.effective_depth(depth);
        let mut nodes = Vec::new();
        let root = self.primitives.root();
        generate(
            &self.primitives,
            root,
            depth,
            self.max_size,
            method,
            true,
            rng,
            &mut nodes,
        );
        Tree::from(nodes)
    }
}

impl<P: Copy> GpBuilder<P> {
    /// How random trees are made: [`Init::RampedHalfAndHalf`] with depths 2 to 6 by default
    /// (Koza 1992).
    pub fn init(mut self, init: Init) -> Self {
        self.init = init;
        self
    }

    /// The largest depth of a tree, the root at depth 0: 17 by default (Koza 1992).
    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// The most nodes of a tree: 1024 by default.
    pub fn max_size(mut self, max_size: usize) -> Self {
        self.max_size = max_size;
        self
    }

    /// The representation.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for a depth limit above 2^24 (`max_depth`), a size limit of 0
    /// or above 2^24 (`max_size`), a root type whose trees don't fit both limits, or
    /// initialization depths that are empty, above the depth limit, or whose trees don't fit the
    /// size limit (`init`).
    pub fn build(self) -> Result<Gp<P>> {
        let max_depth = check_size("max_depth", self.max_depth)?;
        let max_size = check_size("max_size", self.max_size)?;
        if max_size == 0 {
            return Err(Error::InvalidSetting {
                setting: "max_size",
                reason: "a tree has at least 1 node".to_string(),
            });
        }
        let set = &self.primitives;
        let root = set.root();
        if set.min_size(root, max_depth) as usize > max_size {
            return Err(Error::InvalidSetting {
                setting: "max_size",
                reason: format!(
                    "no tree of the root type `{}` has depth at most {max_depth} and at most \
                     {max_size} nodes",
                    set.type_name(root)
                ),
            });
        }
        let gp = Gp {
            primitives: self.primitives,
            init: self.init,
            max_depth,
            max_size,
        };
        let depths = gp.init.depths().clone();
        if depths.is_empty() || *depths.end() > max_depth {
            return Err(Error::InvalidSetting {
                setting: "init",
                reason: format!(
                    "the depths must be a non-empty range up to max_depth, {max_depth}, got \
                     {depths:?}"
                ),
            });
        }
        // the fewest nodes of a tree only fall with the depth: the smallest depth decides
        gp.check_depth(*depths.start())
            .map_err(|error| match error {
                Error::InvalidSetting { reason, .. } => Error::InvalidSetting {
                    setting: "init",
                    reason,
                },
                other => other,
            })?;
        Ok(gp)
    }
}

impl<P: Copy + Debug + Send + Sync> Representation for Gp<P> {
    type Genome = Tree;

    /// The size limit: the most nodes of a tree.
    fn genome_len(&self) -> usize {
        self.max_size
    }

    /// A tree made by [`init`](Gp::init): a depth drawn uniformly from its range, and for
    /// ramped half-and-half, full or grow with probability ½.
    fn random_genome(&self, rng: &mut StreamRng) -> Tree {
        let (depth, method) = self.draw(rng);
        self.make(depth, method, rng)
    }

    /// Checks that the tree is a tree of the set, typed, of the root type, with constants of
    /// types that have constants and among their values, and within the limits.
    fn validate(&self, tree: &Tree) -> Result<()> {
        let invalid = |reason: String| Err(Error::InvalidGenome { reason });
        let set = &self.primitives;
        let nodes = tree.nodes();
        if nodes.is_empty() {
            return invalid("a tree has at least 1 node".to_string());
        }
        if nodes.len() > self.max_size {
            return invalid(format!(
                "{} nodes, above max_size, {}",
                nodes.len(),
                self.max_size
            ));
        }
        // the types expected by the nodes still to come, the next on top
        let mut expected = vec![set.root()];
        for (position, node) in nodes.iter().enumerate() {
            let Some(wanted) = expected.pop() else {
                return invalid(format!("nodes after the end of the tree, from {position}"));
            };
            let ty = match *node {
                Node::Primitive(index) => {
                    let Some(primitive) = set.primitives().get(index as usize) else {
                        return invalid(format!(
                            "node {position} is primitive {index}, which isn't in the set"
                        ));
                    };
                    expected.extend(primitive.args().iter().rev());
                    primitive.returns()
                }
                Node::Constant { ty, value } => {
                    match set.constants(ty) {
                        Some(constants) if constants.contains(value) => {}
                        _ => {
                            return invalid(format!(
                                "node {position} is the constant {value:?}, which isn't a \
                                 constant of its type"
                            ));
                        }
                    }
                    ty
                }
            };
            if ty != wanted {
                return invalid(format!(
                    "node {position} returns `{}` where `{}` is expected",
                    type_name(set, ty),
                    type_name(set, wanted)
                ));
            }
        }
        if !expected.is_empty() {
            return invalid(format!("{} arguments are missing", expected.len()));
        }
        let depth = tree.depth(set);
        if depth > self.max_depth {
            return invalid(format!(
                "depth {depth}, above max_depth, {}",
                self.max_depth
            ));
        }
        Ok(())
    }
}

fn type_name<P: Copy>(set: &PrimitiveSet<P>, ty: Type) -> String {
    if ty.index() < set.type_count() {
        set.type_name(ty).to_string()
    } else {
        format!("{ty:?}")
    }
}

// Appends to `out` a tree of type `ty`, of depth at most `depth` and at most `budget` nodes, made
// by `method`; with `root_function`, the root is a function when one fits (Koza 1992). Needs
// `set.min_size(ty, depth) <= budget`. Without recursion: a stack of the argument slots still to
// fill, and the fewest nodes they need, so every choice leaves room to complete the tree.
#[allow(clippy::too_many_arguments)]
pub(crate) fn generate<P: Copy>(
    set: &PrimitiveSet<P>,
    ty: Type,
    depth: usize,
    budget: usize,
    method: Method,
    root_function: bool,
    rng: &mut StreamRng,
    out: &mut Vec<Node>,
) {
    debug_assert!(set.min_size(ty, depth) as usize <= budget);
    let limit = out.len() + budget;
    let mut slots = vec![(ty, depth)];
    // the fewest nodes of the open slots
    let mut pending = set.min_size(ty, depth) as usize;
    let mut first = true;
    while let Some((ty, depth)) = slots.pop() {
        pending -= set.min_size(ty, depth) as usize;
        let available = limit - out.len() - pending;
        // the fewest nodes of a function's tree, if it fits in the depth
        let cost = |function: u32| -> usize {
            let mut size = 1usize;
            for &arg in set.primitives()[function as usize].args() {
                match set.min_size(arg, depth - 1) {
                    INFINITE => return usize::MAX,
                    min => size = size.saturating_add(min as usize),
                }
            }
            size
        };
        let fits = |function: &&u32| depth > 0 && cost(**function) <= available;
        let functions = set.functions_of(ty);
        let fitting = functions.iter().filter(fits).count();
        let terminals = set.terminals_of(ty);
        let constants = set.constants(ty);
        let leaves = terminals.len() + usize::from(constants.is_some());
        let functions_only =
            fitting > 0 && (method == Method::Full || (first && root_function) || leaves == 0);
        let choices = if functions_only {
            fitting
        } else {
            leaves + fitting
        };
        debug_assert!(choices > 0, "a tree of {ty:?} fits");
        let mut choice = rng.below(choices);
        if !functions_only {
            if choice < terminals.len() {
                out.push(Node::Primitive(terminals[choice]));
                first = false;
                continue;
            }
            if choice < leaves {
                let value = constants.expect("constants").sample(rng);
                out.push(Node::Constant { ty, value });
                first = false;
                continue;
            }
            choice -= leaves;
        }
        let function = *functions
            .iter()
            .filter(fits)
            .nth(choice)
            .expect("a fitting function");
        out.push(Node::Primitive(function));
        for &arg in set.primitives()[function as usize].args().iter().rev() {
            slots.push((arg, depth - 1));
            pending += set.min_size(arg, depth - 1) as usize;
        }
        first = false;
    }
}

// deserialized like `build`
#[cfg(feature = "serde")]
impl<'de, P: Copy + serde::Deserialize<'de>> serde::Deserialize<'de> for Gp<P> {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(rename = "Gp")]
        #[serde(bound(deserialize = "P: Copy + serde::Deserialize<'de>"))]
        struct Raw<P> {
            primitives: PrimitiveSet<P>,
            init: Init,
            max_depth: usize,
            max_size: usize,
        }
        let raw = Raw::<P>::deserialize(deserializer)?;
        Gp::builder(raw.primitives)
            .init(raw.init)
            .max_depth(raw.max_depth)
            .max_size(raw.max_size)
            .build()
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug)]
    enum Op {
        Add,
        X,
    }

    #[test]
    fn ramped_half_and_half_draws_depths_and_methods_uniformly() {
        let mut set = PrimitiveSet::builder();
        let real = set.new_type("real");
        set.function("add", Op::Add, [real, real], real)
            .terminal("x", Op::X, real);
        let gp = Gp::builder(set.build(real).unwrap()).build().unwrap();
        let mut rng = StreamRng::seed_from_u64(1);
        let mut counts = [[0u32; 2]; 5];
        for _ in 0..10_000 {
            let (depth, method) = gp.draw(&mut rng);
            counts[depth - 2][usize::from(method == Method::Full)] += 1;
        }
        // chi-squared with 9 degrees of freedom, 27.88 at p = 0.001
        let chi_squared: f64 = counts
            .iter()
            .flatten()
            .map(|&count| (f64::from(count) - 1000.0).powi(2) / 1000.0)
            .sum();
        assert!(chi_squared < 27.88, "{counts:?}: {chi_squared}");
    }
}
