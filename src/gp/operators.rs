//! Crossover and mutation of trees: every child is typed and within its representation's limits.

use super::primitives::{INFINITE, Type};
use super::representation::{Gp, Method, generate};
use super::tree::{Tree, for_each_depth, subtree_end};
use crate::genome::Genome;
use crate::operator::{Crossover, Mutate, check_probability, check_size};
use crate::rng::Chance;
use crate::{Result, StreamRng};
use std::fmt::Debug;

/// Subtree crossover (Koza 1992): a random subtree of each parent is exchanged with a subtree of
/// the same type of the other.
///
/// - **The first point**, in the first parent, is a function node with probability
///   `internal_rate` (0.9 by default, Koza's), else a leaf (terminal or constant); a tree
///   without function nodes gives a leaf, and one of only a leaf the root.
/// - **The second point**, in the second parent, is of the first point's type, chosen with the
///   same bias among the points whose exchange keeps both children within the depth and size
///   limits, so no child is over a limit and none is wasted. If there's none, the first point is
///   drawn again, up to 10 times, and then the parents stay unchanged.
///
/// Leaf selection (Angeline 1996) is the case `internal_rate` = 0, and a uniform choice of the
/// points is roughly `internal_rate` = the share of function nodes.
///
/// ```
/// use genoxide::gp::{Gp, PrimitiveSet, SubtreeCrossover};
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
/// let mut a = gp.primitives().parse("add(x, mul(x, x))")?;
/// let mut b = gp.primitives().parse("mul(add(x, x), x)")?;
/// SubtreeCrossover::new().crossover(&gp, &mut a, &mut b, &mut StreamRng::seed_from_u64(1));
/// assert!(gp.validate(&a).is_ok() && gp.validate(&b).is_ok());
/// assert_eq!(a.len() + b.len(), 10); // the parents' nodes, exchanged
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SubtreeCrossover {
    internal_rate: f64,
}

impl Default for SubtreeCrossover {
    fn default() -> Self {
        Self::new()
    }
}

impl SubtreeCrossover {
    /// Crossover points at function nodes with probability 0.9 (Koza 1992).
    pub fn new() -> Self {
        Self { internal_rate: 0.9 }
    }

    /// Crossover points at function nodes with probability `rate`, in `[0, 1]`, else at leaves.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`](crate::Error::InvalidSetting) (setting `internal_rate`) for a rate outside `[0, 1]`.
    pub fn with_internal_rate(rate: f64) -> Result<Self> {
        Ok(Self {
            internal_rate: check_probability("internal_rate", rate)?,
        })
    }

    /// The probability of a point at a function node.
    pub fn internal_rate(&self) -> f64 {
        self.internal_rate
    }
}

// what crossover needs of a node
#[derive(Clone, Copy, Debug)]
struct Shape {
    ty: Type,
    internal: bool,
    // from the root, and of its subtree
    depth: u32,
    height: u32,
    size: u32,
}

// the shape of every node of `tree`
fn shapes<P: Copy>(gp: &Gp<P>, tree: &Tree, out: &mut Vec<Shape>) {
    let set = gp.primitives();
    let nodes = tree.nodes();
    out.clear();
    out.extend(nodes.iter().map(|node| Shape {
        ty: set.node_type(node),
        internal: set.node_arity(node) > 0,
        depth: 0,
        height: 0,
        size: 1,
    }));
    for_each_depth(set, nodes, |position, depth| {
        out[position].depth = depth as u32
    });
    // heights and sizes bottom-up: the children's on a stack, the first on top
    let mut stack: Vec<(u32, u32)> = Vec::with_capacity(nodes.len());
    for (position, node) in nodes.iter().enumerate().rev() {
        let arity = set.node_arity(node);
        let (mut height, mut size) = (0, 1);
        for (child_height, child_size) in stack.drain(stack.len() - arity..) {
            height = height.max(child_height + 1);
            size += child_size;
        }
        out[position].height = height;
        out[position].size = size;
        stack.push((height, size));
    }
}

// the position of the `k`-th node, among those that `wanted` accepts
fn nth(shapes: &[Shape], k: usize, wanted: impl Fn(&Shape) -> bool) -> usize {
    shapes
        .iter()
        .enumerate()
        .filter(|(_, shape)| wanted(shape))
        .nth(k)
        .map(|(position, _)| position)
        .expect("a point")
}

// one of the nodes that `allowed` accepts: a function node with probability `internal` (a leaf
// if there's none), else a leaf (a function node if there's none); None if none is allowed
fn point(
    shapes: &[Shape],
    internal: Chance,
    rng: &mut StreamRng,
    allowed: impl Fn(&Shape) -> bool,
) -> Option<usize> {
    let (mut functions, mut leaves) = (0, 0);
    for shape in shapes.iter().filter(|shape| allowed(shape)) {
        if shape.internal {
            functions += 1;
        } else {
            leaves += 1;
        }
    }
    if functions + leaves == 0 {
        return None;
    }
    let wants_function = rng.chance(internal);
    let function = (wants_function && functions > 0) || leaves == 0;
    let count = if function { functions } else { leaves };
    let k = rng.below(count);
    Some(nth(shapes, k, |shape| {
        allowed(shape) && shape.internal == function
    }))
}

// the tries at a first point with a matching second point
const TRIES: usize = 10;

impl<P: Copy + Debug + Send + Sync> Crossover<Gp<P>> for SubtreeCrossover {
    fn crossover(&self, gp: &Gp<P>, a: &mut Tree, b: &mut Tree, rng: &mut StreamRng) {
        if a.is_empty() || b.is_empty() {
            return;
        }
        let internal = Chance::new(self.internal_rate);
        let (mut shapes_a, mut shapes_b) = (Vec::new(), Vec::new());
        shapes(gp, a, &mut shapes_a);
        shapes(gp, b, &mut shapes_b);
        let (max_depth, max_size) = (gp.max_depth(), gp.max_size());
        let (len_a, len_b) = (a.len(), b.len());
        for _ in 0..TRIES {
            let first = point(&shapes_a, internal, rng, |_| true).expect("a node");
            let x = shapes_a[first];
            let fits = |y: &Shape| {
                y.ty == x.ty
                    && (x.depth + y.height) as usize <= max_depth
                    && (y.depth + x.height) as usize <= max_depth
                    && len_a - x.size as usize + y.size as usize <= max_size
                    && len_b - y.size as usize + x.size as usize <= max_size
            };
            let Some(second) = point(&shapes_b, internal, rng, fits) else {
                continue;
            };
            let y = shapes_b[second];
            let from_a = a.nodes()[first..first + x.size as usize].to_vec();
            let from_b = &b.nodes()[second..second + y.size as usize];
            a.nodes_mut()
                .splice(first..first + x.size as usize, from_b.iter().copied());
            b.nodes_mut()
                .splice(second..second + y.size as usize, from_a);
            return;
        }
    }
}

/// Subtree mutation (Koza 1992; Poli, Langdon and McPhee 2008, sec. 2.4): a node chosen
/// uniformly is replaced, with its subtree, by a subtree of the same type grown by the grow
/// method, of depth at most [`max_depth`](SubtreeMutation::max_depth) (4 by default) and within
/// the representation's limits.
///
/// The new subtree is grown again (up to 100 times) while it equals the old one, so the tree
/// changes unless no other subtree fits there. Where the limits leave less room than the depth,
/// the new subtree is smaller; where a type's smallest tree is deeper, it's as deep as needed.
///
/// ```
/// use genoxide::gp::{Gp, Init, PrimitiveSet, SubtreeMutation};
/// use genoxide::prelude::*;
///
/// #[derive(Clone, Copy, Debug)]
/// enum Op {
///     Add,
///     X,
///     Y,
/// }
///
/// let mut set = PrimitiveSet::builder();
/// let real = set.new_type("real");
/// set.function("add", Op::Add, [real, real], real)
///     .terminal("x", Op::X, real)
///     .terminal("y", Op::Y, real);
/// let gp = Gp::builder(set.build(real)?)
///     .max_depth(3)
///     .init(Init::RampedHalfAndHalf { depths: 1..=3 })
///     .build()?;
/// let mut tree = gp.primitives().parse("add(x, add(y, x))")?;
/// let before = tree.clone();
/// SubtreeMutation::new().mutate(&gp, &mut tree, &mut StreamRng::seed_from_u64(1));
/// assert_ne!(tree, before);
/// assert!(gp.validate(&tree).is_ok()); // depth at most 3
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SubtreeMutation {
    max_depth: usize,
}

impl Default for SubtreeMutation {
    fn default() -> Self {
        Self::new()
    }
}

impl SubtreeMutation {
    /// New subtrees of depth at most 4.
    pub fn new() -> Self {
        Self { max_depth: 4 }
    }

    /// New subtrees of depth at most `max_depth`, 0 for a single leaf where the types allow.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`](crate::Error::InvalidSetting) (setting `max_depth`) for a depth above 2^24.
    pub fn with_max_depth(max_depth: usize) -> Result<Self> {
        Ok(Self {
            max_depth: check_size("max_depth", max_depth)?,
        })
    }

    /// The largest depth of a new subtree.
    pub fn max_depth(&self) -> usize {
        self.max_depth
    }
}

impl<P: Copy + Debug + Send + Sync> Mutate<Gp<P>> for SubtreeMutation {
    fn mutate(&self, gp: &Gp<P>, tree: &mut Tree, rng: &mut StreamRng) {
        if tree.is_empty() {
            return;
        }
        let set = gp.primitives();
        let point = rng.below(tree.len());
        let mut depth = 0;
        for_each_depth(set, tree.nodes(), |position, d| {
            if position == point {
                depth = d;
            }
        });
        let end = subtree_end(set, tree.nodes(), point);
        let ty = set.node_type(&tree.nodes()[point]);
        // the room the rest of the tree leaves
        let room = gp.max_depth().saturating_sub(depth);
        let budget = gp.max_size().saturating_sub(tree.len() - (end - point));
        let fewest = |depth: usize| set.min_size(ty, depth);
        if fewest(room) == INFINITE || fewest(room) as usize > budget {
            return;
        }
        let mut depth = self.max_depth.min(room);
        while fewest(depth) as usize > budget {
            depth += 1;
        }
        let mut subtree = Vec::new();
        for _ in 0..100 {
            subtree.clear();
            generate(
                set,
                ty,
                depth,
                budget,
                Method::Grow,
                false,
                rng,
                &mut subtree,
            );
            if subtree[..] != tree.nodes()[point..end] {
                break;
            }
        }
        tree.nodes_mut().splice(point..end, subtree);
    }
}
