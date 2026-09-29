//! Evaluation of trees: bottom-up on a stack of values, or of columns of values for many points
//! at once.

use super::primitives::{PrimitiveSet, Type};
use super::tree::{Node, Tree};

/// The workspace of [`Tree::evaluate_columns`]: a pool of columns of `points` values, reused from
/// one evaluation to the next, so evaluation allocates nothing once it has grown.
///
/// Keep one per thread, e.g. in a `thread_local!`, for a fitness function evaluated in parallel.
#[derive(Clone, Debug, Default)]
pub struct Columns {
    points: usize,
    pool: Vec<Vec<f64>>,
    free: Vec<usize>,
    stack: Vec<usize>,
}

impl Columns {
    /// A workspace for columns of `points` values.
    pub fn new(points: usize) -> Self {
        Self {
            points,
            ..Self::default()
        }
    }

    /// The number of values in a column.
    pub fn points(&self) -> usize {
        self.points
    }
}

// the children's columns given to a primitive, without an allocation for up to this many
const INLINE_ARITY: usize = 4;

impl Tree {
    /// The value of the tree, bottom-up on `stack` (the prefix order read backwards: a stack
    /// machine), for values of any type `V`: numbers, Booleans, or an enum of the set's types.
    ///
    /// `primitive(p, args)` gives the value of the primitive `p` from its children's values
    /// `args`, in order (none for a terminal); `constant(ty, value)` the value of a constant of
    /// type `ty`. `stack` is scratch space, cleared first: reuse it, and evaluation allocates
    /// nothing once it has grown.
    ///
    /// # Panics
    ///
    /// If the tree isn't a tree of `set` (see
    /// [`Gp::validate`](crate::genome::Representation::validate)).
    pub fn evaluate<P: Copy, V>(
        &self,
        set: &PrimitiveSet<P>,
        stack: &mut Vec<V>,
        mut primitive: impl FnMut(P, &[V]) -> V,
        mut constant: impl FnMut(Type, f64) -> V,
    ) -> V {
        stack.clear();
        for node in self.nodes().iter().rev() {
            let value = match *node {
                Node::Constant { ty, value } => constant(ty, value),
                Node::Primitive(index) => {
                    let p = &set.primitives()[index as usize];
                    let start = stack
                        .len()
                        .checked_sub(p.arity())
                        .expect("a tree of this primitive set");
                    // the first child is on top: in order
                    let args = &mut stack[start..];
                    args.reverse();
                    let value = primitive(p.value(), args);
                    stack.truncate(start);
                    value
                }
            };
            stack.push(value);
        }
        assert_eq!(stack.len(), 1, "a tree of this primitive set");
        stack.pop().expect("a value")
    }

    /// The values of the tree at [`points`](Columns::points) points at once, bottom-up on
    /// columns: each primitive fills its output column from its children's columns. For data,
    /// this is several times faster than [`evaluate`](Tree::evaluate) point by point (one call
    /// per node instead of one per node and point, and loops the compiler vectorizes).
    ///
    /// `primitive(p, args, output)` writes every value of `output` from the children's columns
    /// `args`, in order (none for a terminal, which writes its data, e.g. a variable's values).
    /// Constants are columns of their value. The result is the root's column, in `workspace`.
    ///
    /// Doing per point what [`evaluate`](Tree::evaluate)'s `primitive` does, the same operations
    /// in the same order, gives the same values to the bit.
    ///
    /// # Panics
    ///
    /// If the tree isn't a tree of `set` (see
    /// [`Gp::validate`](crate::genome::Representation::validate)).
    pub fn evaluate_columns<'w, P: Copy>(
        &self,
        set: &PrimitiveSet<P>,
        workspace: &'w mut Columns,
        mut primitive: impl FnMut(P, &[&[f64]], &mut [f64]),
    ) -> &'w [f64] {
        let Columns {
            points,
            pool,
            free,
            stack,
        } = workspace;
        stack.clear();
        free.clear();
        free.extend((0..pool.len()).rev());
        for node in self.nodes().iter().rev() {
            let id = free.pop().unwrap_or_else(|| {
                pool.push(vec![0.0; *points]);
                pool.len() - 1
            });
            let mut output = std::mem::take(&mut pool[id]);
            match *node {
                Node::Constant { value, .. } => output.fill(value),
                Node::Primitive(index) => {
                    let p = &set.primitives()[index as usize];
                    let arity = p.arity();
                    let start = stack
                        .len()
                        .checked_sub(arity)
                        .expect("a tree of this primitive set");
                    // the first child is on top
                    let child = |k: usize| pool[stack[stack.len() - 1 - k]].as_slice();
                    if arity <= INLINE_ARITY {
                        let mut args: [&[f64]; INLINE_ARITY] = [&[]; INLINE_ARITY];
                        for (k, arg) in args.iter_mut().enumerate().take(arity) {
                            *arg = child(k);
                        }
                        primitive(p.value(), &args[..arity], &mut output);
                    } else {
                        let args: Vec<&[f64]> = (0..arity).map(child).collect();
                        primitive(p.value(), &args, &mut output);
                    }
                    free.extend(stack.drain(start..));
                }
            }
            pool[id] = output;
            stack.push(id);
        }
        assert_eq!(stack.len(), 1, "a tree of this primitive set");
        &pool[stack[0]]
    }
}
