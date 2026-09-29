//! Trees: nodes in a flat array, in prefix order.

use super::primitives::{PrimitiveSet, Type};
use crate::genome::Genome;
use std::fmt;
use std::hash::{Hash, Hasher};

/// A node of a [`Tree`]: a primitive of its set, by position, or a constant.
///
/// Constants are compared and hashed by their bits, as [`Reals`](crate::genome::Reals) genes
/// are: `0.0` and `-0.0` are different constants.
#[derive(Clone, Copy, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Node {
    /// The function or terminal at this position in the set's
    /// [`primitives`](PrimitiveSet::primitives).
    Primitive(u32),
    /// An ephemeral random constant of a type of the set.
    Constant {
        /// Its type.
        ty: Type,
        /// Its value.
        value: f64,
    },
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Node::Primitive(a), Node::Primitive(b)) => a == b,
            (Node::Constant { ty: a, value: x }, Node::Constant { ty: b, value: y }) => {
                a == b && x.to_bits() == y.to_bits()
            }
            _ => false,
        }
    }
}

impl Eq for Node {}

impl Hash for Node {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match *self {
            Node::Primitive(index) => {
                state.write_u8(0);
                state.write_u32(index);
            }
            Node::Constant { ty, value } => {
                state.write_u8(1);
                ty.hash(state);
                state.write_u64(value.to_bits());
            }
        }
    }
}

/// A tree of a genetic program, in prefix order: each function is followed by its children's
/// subtrees, in order. A subtree is a contiguous range of nodes.
///
/// A flat array, not a tree of boxes: cloning, comparing, hashing and serializing a tree are
/// loops over its nodes, never recursive, so no tree is too deep for them. A tree only has a
/// meaning with the [`PrimitiveSet`] its primitives come from, which its methods take.
///
/// ```
/// use genoxide::gp::{Node, PrimitiveSet, Tree};
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
/// let set = set.build(real)?;
///
/// // add(x, add(x, x)), node by node
/// let (add, x) = (Node::Primitive(0), Node::Primitive(1));
/// let tree = Tree::from(vec![add, x, add, x, x]);
/// assert_eq!(tree.display(&set).to_string(), "add(x, add(x, x))");
/// assert_eq!(tree, set.parse("add(x, add(x, x))")?);
/// assert_eq!(tree.depth(&set), 2);
///
/// // x + (x + x) at x = 1.5
/// let value = tree.evaluate(
///     &set,
///     &mut Vec::new(),
///     |op, args: &[f64]| match op {
///         Op::Add => args[0] + args[1],
///         Op::X => 1.5,
///     },
///     |_, constant| constant,
/// );
/// assert_eq!(value, 4.5);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Tree {
    nodes: Vec<Node>,
}

impl Clone for Tree {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            nodes: self.nodes.clone(),
        }
    }

    // in the memory of `self`: algorithms copy parents into the genomes they no longer use
    #[inline]
    fn clone_from(&mut self, source: &Self) {
        self.nodes.clone_from(&source.nodes);
    }
}

impl From<Vec<Node>> for Tree {
    /// The tree of these nodes, in prefix order. Check it with
    /// [`Gp::validate`](crate::genome::Representation::validate) or build it with
    /// [`PrimitiveSet::parse`].
    fn from(nodes: Vec<Node>) -> Self {
        Self { nodes }
    }
}

impl Genome for Tree {
    /// The number of nodes: what size limits count.
    fn len(&self) -> usize {
        self.nodes.len()
    }
}

impl Tree {
    /// The nodes, in prefix order.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// The nodes, consuming the tree.
    pub fn into_nodes(self) -> Vec<Node> {
        self.nodes
    }

    pub(crate) fn nodes_mut(&mut self) -> &mut Vec<Node> {
        &mut self.nodes
    }

    /// The depth: the most edges from the root to a leaf, 0 for a tree of one node (Koza's
    /// convention).
    ///
    /// # Panics
    ///
    /// If a primitive isn't one of `set`'s.
    pub fn depth<P: Copy>(&self, set: &PrimitiveSet<P>) -> usize {
        let mut deepest = 0;
        for_each_depth(set, &self.nodes, |_, depth| deepest = deepest.max(depth));
        deepest
    }

    /// The tree as text, e.g. `add(mul(x, x), 0.5)`, which [`PrimitiveSet::parse`] reads back:
    /// each primitive by its name, its arguments in parentheses, and constants as Rust writes an
    /// `f64` with `{:?}` (the shortest text that reads back as the same value).
    pub fn display<'a, P: Copy>(&'a self, set: &'a PrimitiveSet<P>) -> Display<'a, P> {
        Display { tree: self, set }
    }

    /// The root, to walk the tree from the top: for interpreters that decide which children to
    /// run (a conditional with side effects, a program that moves an agent). The walk is the
    /// caller's, bounded by the depth limit.
    ///
    /// # Panics
    ///
    /// If the tree is empty.
    pub fn root<'a, P: Copy>(&'a self, set: &'a PrimitiveSet<P>) -> Subtree<'a, P> {
        assert!(!self.nodes.is_empty(), "an empty tree has no root");
        Subtree {
            nodes: &self.nodes,
            set,
            start: 0,
        }
    }
}

// the end of the subtree starting at `start`: one scan of the arities
#[inline]
pub(crate) fn subtree_end<P: Copy>(set: &PrimitiveSet<P>, nodes: &[Node], start: usize) -> usize {
    let mut open = 1usize;
    let mut position = start;
    while open > 0 {
        open += set.node_arity(&nodes[position]);
        open -= 1;
        position += 1;
    }
    position
}

// calls `visit(position, depth)` for every node, in order, the root at depth 0
pub(crate) fn for_each_depth<P: Copy>(
    set: &PrimitiveSet<P>,
    nodes: &[Node],
    mut visit: impl FnMut(usize, usize),
) {
    // the depths of the nodes still to come, the next on top
    let mut pending = vec![0usize];
    for (position, node) in nodes.iter().enumerate() {
        let depth = pending.pop().expect("a tree of this primitive set");
        visit(position, depth);
        pending.extend(std::iter::repeat_n(depth + 1, set.node_arity(node)));
    }
}

/// A tree written as text, from [`Tree::display`].
pub struct Display<'a, P> {
    tree: &'a Tree,
    set: &'a PrimitiveSet<P>,
}

impl<P: Copy> fmt::Display for Display<'_, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // the arguments still to write of each open function, innermost on top
        let mut open: Vec<usize> = Vec::new();
        for node in &self.tree.nodes {
            let arity = match *node {
                Node::Primitive(index) => match self.set.primitives().get(index as usize) {
                    Some(primitive) => {
                        f.write_str(primitive.name())?;
                        primitive.arity()
                    }
                    // not of this set: shown, not a panic in formatting
                    None => {
                        write!(f, "#{index}")?;
                        0
                    }
                },
                Node::Constant { value, .. } => {
                    write!(f, "{value:?}")?;
                    0
                }
            };
            if arity > 0 {
                f.write_str("(")?;
                open.push(arity);
                continue;
            }
            // a leaf ends every function whose last argument it completes
            while let Some(left) = open.last_mut() {
                *left -= 1;
                if *left > 0 {
                    f.write_str(", ")?;
                    break;
                }
                f.write_str(")")?;
                open.pop();
            }
        }
        Ok(())
    }
}

impl<P: Copy> fmt::Debug for Display<'_, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

/// A subtree of a [`Tree`], from [`Tree::root`] and [`Subtree::children`].
///
/// ```
/// use genoxide::gp::{PrimitiveSet, Subtree};
///
/// #[derive(Clone, Copy, Debug)]
/// enum Op {
///     Add,
///     Neg,
///     X,
/// }
///
/// let mut set = PrimitiveSet::builder();
/// let real = set.new_type("real");
/// set.function("add", Op::Add, [real, real], real)
///     .function("neg", Op::Neg, [real], real)
///     .terminal("x", Op::X, real);
/// let set = set.build(real)?;
///
/// // a recursive interpreter, top-down
/// fn value(subtree: Subtree<'_, Op>, x: f64) -> f64 {
///     let mut children = subtree.children();
///     let mut next = || value(children.next().unwrap(), x);
///     match subtree.primitive() {
///         Some(Op::Add) => next() + next(),
///         Some(Op::Neg) => -next(),
///         Some(Op::X) => x,
///         None => subtree.constant().unwrap(),
///     }
/// }
/// let tree = set.parse("add(x, neg(add(x, x)))")?;
/// assert_eq!(value(tree.root(&set), 2.0), -2.0);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Copy)]
pub struct Subtree<'a, P> {
    nodes: &'a [Node],
    set: &'a PrimitiveSet<P>,
    start: usize,
}

impl<'a, P: Copy> Subtree<'a, P> {
    /// Its root node.
    pub fn node(&self) -> Node {
        self.nodes[self.start]
    }

    /// Its root's primitive, `None` for a constant.
    ///
    /// # Panics
    ///
    /// If the primitive isn't one of the set's.
    pub fn primitive(&self) -> Option<P> {
        match self.node() {
            Node::Primitive(index) => Some(self.set.primitives()[index as usize].value()),
            Node::Constant { .. } => None,
        }
    }

    /// Its root's value, if it's a constant.
    pub fn constant(&self) -> Option<f64> {
        match self.node() {
            Node::Constant { value, .. } => Some(value),
            Node::Primitive(_) => None,
        }
    }

    /// The number of children of its root.
    pub fn arity(&self) -> usize {
        self.set.node_arity(&self.node())
    }

    /// The type it returns.
    pub fn returns(&self) -> Type {
        self.set.node_type(&self.node())
    }

    /// Its nodes, in prefix order.
    pub fn nodes(&self) -> &'a [Node] {
        &self.nodes[self.start..subtree_end(self.set, self.nodes, self.start)]
    }

    /// The subtrees of its root's children, in order.
    pub fn children(&self) -> Children<'a, P> {
        Children {
            nodes: self.nodes,
            set: self.set,
            next: self.start + 1,
            left: self.arity(),
        }
    }
}

impl<P: Copy> fmt::Debug for Subtree<'_, P> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Subtree")
            .field("nodes", &self.nodes())
            .finish()
    }
}

/// The children of a [`Subtree`], in order.
#[derive(Clone)]
pub struct Children<'a, P> {
    nodes: &'a [Node],
    set: &'a PrimitiveSet<P>,
    next: usize,
    left: usize,
}

impl<'a, P: Copy> Iterator for Children<'a, P> {
    type Item = Subtree<'a, P>;

    fn next(&mut self) -> Option<Subtree<'a, P>> {
        if self.left == 0 {
            return None;
        }
        self.left -= 1;
        let start = self.next;
        if self.left > 0 {
            self.next = subtree_end(self.set, self.nodes, start);
        }
        Some(Subtree {
            nodes: self.nodes,
            set: self.set,
            start,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.left, Some(self.left))
    }
}

impl<P: Copy> ExactSizeIterator for Children<'_, P> {}
