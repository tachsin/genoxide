//! Primitive sets: the functions, terminals and constants that trees are made of, with their types.

use super::tree::{Node, Tree};
use crate::operator::{MAX_SIZE, check_size};
use crate::{Error, Result, StreamRng};
use std::collections::BTreeMap;
use std::fmt;
use std::ops::RangeInclusive;

/// The type of a value in a tree: a handle returned by
/// [`PrimitiveSetBuilder::new_type`], naming one of its set's types.
///
/// Types are data, not Rust types: a signature is a list of `Type`s, so a set is serializable and
/// the same from Rust, Python or a file. Untyped genetic programming is the case of one type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Type(pub(crate) u16);

impl Type {
    /// The position of the type in its set, in the order of
    /// [`new_type`](PrimitiveSetBuilder::new_type).
    pub fn index(self) -> usize {
        usize::from(self.0)
    }
}

/// Ephemeral random constants of a type (Koza 1992): a constant node draws its value once, when
/// it's created, and keeps it.
///
/// Values are `f64` whatever the type: an integer or Boolean type takes integral values
/// ([`integers`](Constants::integers)), which its primitives convert.
///
/// ```
/// use genoxide::gp::Constants;
///
/// assert!(Constants::uniform(-1.0..=1.0).is_ok());
/// assert!(Constants::integers(-5..=5).is_ok());
/// assert!(Constants::choice([0.5, 1.0, 2.0]).is_ok());
/// assert!(Constants::uniform(1.0..=0.0).is_err());
/// assert!(Constants::choice([]).is_err());
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub enum Constants {
    /// Uniform over `[low, high]`.
    Uniform {
        /// The smallest value.
        low: f64,
        /// The largest value.
        high: f64,
    },
    /// An integer uniform over `[low, high]`, as an `f64`.
    Integers {
        /// The smallest value.
        low: i64,
        /// The largest value.
        high: i64,
    },
    /// One of the values, each with the same probability.
    Choice(Vec<f64>),
}

// the largest integer constant: every integer up to it is an exact `f64`
const MAX_INTEGER: i64 = 1 << 53;

impl Constants {
    /// Values uniform over `range`, whose ends are finite and whose width is finite.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for an empty range (NaN ends included) or one whose width isn't
    /// finite.
    pub fn uniform(range: RangeInclusive<f64>) -> Result<Self> {
        let (low, high) = (*range.start(), *range.end());
        if range.is_empty() || !(high - low).is_finite() {
            return Err(Error::InvalidSetting {
                setting: "constants",
                reason: format!("the range must be non-empty and finite, got {range:?}"),
            });
        }
        Ok(Constants::Uniform { low, high })
    }

    /// Integers uniform over `range`, whose ends are within ±2^53 (so every value is an exact
    /// `f64`).
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for an empty range or an end beyond ±2^53.
    pub fn integers(range: RangeInclusive<i64>) -> Result<Self> {
        let (low, high) = (*range.start(), *range.end());
        if range.is_empty() || low < -MAX_INTEGER || high > MAX_INTEGER {
            return Err(Error::InvalidSetting {
                setting: "constants",
                reason: format!("the range must be non-empty and within ±2^53, got {range:?}"),
            });
        }
        Ok(Constants::Integers { low, high })
    }

    /// One of `values`, at least one and at most 2^24, each finite.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for no values, more than 2^24, or a value that isn't finite.
    pub fn choice(values: impl IntoIterator<Item = f64>) -> Result<Self> {
        let values: Vec<f64> = values.into_iter().collect();
        if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
            return Err(Error::InvalidSetting {
                setting: "constants",
                reason: format!("the values must be finite, and at least one, got {values:?}"),
            });
        }
        check_size("constants", values.len())?;
        Ok(Constants::Choice(values))
    }

    /// A random value.
    pub fn sample(&self, rng: &mut StreamRng) -> f64 {
        match self {
            Constants::Uniform { low, high } => {
                // at most `high`: `low + (high - low) * u` can round up to it, never beyond
                (low + (high - low) * rng.unit_f64()).min(*high)
            }
            Constants::Integers { low, high } => {
                // within ±2^53: the difference and the sum are exact
                let width = (high - low) as u64 + 1;
                (low + rng.below_u64(width) as i64) as f64
            }
            Constants::Choice(values) => values[rng.below(values.len())],
        }
    }

    /// Whether `value` is one of the values.
    pub fn contains(&self, value: f64) -> bool {
        match self {
            Constants::Uniform { low, high } => (*low..=*high).contains(&value),
            Constants::Integers { low, high } => {
                value.fract() == 0.0 && (*low as f64..=*high as f64).contains(&value)
            }
            Constants::Choice(values) => values.iter().any(|v| v.to_bits() == value.to_bits()),
        }
    }

    // checked like the constructors, for deserialized constants
    #[cfg(feature = "serde")]
    fn checked(self) -> Result<Self> {
        match self {
            Constants::Uniform { low, high } => Constants::uniform(low..=high),
            Constants::Integers { low, high } => Constants::integers(low..=high),
            Constants::Choice(values) => Constants::choice(values),
        }
    }
}

/// A function or terminal of a [`PrimitiveSet`]: its name, the user's value `P` that stands for
/// it, the types of its arguments (none for a terminal) and the type it returns.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Primitive<P> {
    name: String,
    value: P,
    args: Vec<Type>,
    returns: Type,
}

impl<P: Copy> Primitive<P> {
    /// The name, as trees are displayed and parsed.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The user's value for it, which the fitness function matches on.
    pub fn value(&self) -> P {
        self.value
    }

    /// The types of the arguments, in order: none for a terminal.
    pub fn args(&self) -> &[Type] {
        &self.args
    }

    /// The number of arguments.
    pub fn arity(&self) -> usize {
        self.args.len()
    }

    /// The type of the value it returns.
    pub fn returns(&self) -> Type {
        self.returns
    }
}

/// Builds a [`PrimitiveSet`]: see there.
#[derive(Clone, Debug)]
pub struct PrimitiveSetBuilder<P> {
    types: Vec<String>,
    primitives: Vec<Primitive<P>>,
    constants: Vec<(Type, Constants)>,
}

impl<P: Copy> PrimitiveSetBuilder<P> {
    /// Declares a type named `name`, and returns its handle. At most 2^16 types.
    pub fn new_type(&mut self, name: impl Into<String>) -> Type {
        // beyond 2^16 types, `build` fails before the truncated handle is used
        let ty = Type(self.types.len() as u16);
        self.types.push(name.into());
        ty
    }

    /// Adds a function `name` with the argument types `args` returning `returns`; `value` stands
    /// for it in the fitness function. A function without arguments is a terminal.
    pub fn function(
        &mut self,
        name: impl Into<String>,
        value: P,
        args: impl IntoIterator<Item = Type>,
        returns: Type,
    ) -> &mut Self {
        self.primitives.push(Primitive {
            name: name.into(),
            value,
            args: args.into_iter().collect(),
            returns,
        });
        self
    }

    /// Adds a terminal `name` of type `returns`, e.g. an input variable.
    pub fn terminal(&mut self, name: impl Into<String>, value: P, returns: Type) -> &mut Self {
        self.function(name, value, [], returns)
    }

    /// Gives the type `ty` ephemeral random constants: in generation, they are one more terminal
    /// of the type.
    pub fn constants(&mut self, ty: Type, constants: Constants) -> &mut Self {
        self.constants.push((ty, constants));
        self
    }

    /// The set, whose trees return the type `root`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] (setting `primitives`) for no type or more than 2^16, more than
    /// 2^24 primitives, a type handle not of this builder, a name that is empty, repeated,
    /// contains whitespace, `(`, `)` or `,`, or reads as a number, constants given twice to a
    /// type, or a type needed by the root type that can't make a tree (no terminal, constant or
    /// function whose arguments can be made).
    pub fn build(self, root: Type) -> Result<PrimitiveSet<P>> {
        PrimitiveSet::from_parts(self.types, self.primitives, self.constants, root)
    }
}

/// The functions, terminals and ephemeral random constants of the trees of a genetic program,
/// strongly typed (Montana 1995).
///
/// Every primitive has argument types and a return type, every constant a type; a tree is valid
/// when each child's type is its parent's argument type and the root's type is the set's
/// [`root`](PrimitiveSet::root) type. The primitives are the user's `Copy` values, usually an
/// enum, whose meaning the fitness function gives by matching on them.
///
/// ```
/// use genoxide::gp::{Constants, PrimitiveSet};
///
/// #[derive(Clone, Copy, Debug, PartialEq)]
/// enum Op {
///     Add,
///     Mul,
///     Less,
///     If,
///     X,
/// }
///
/// let mut set = PrimitiveSet::builder();
/// let real = set.new_type("real");
/// let boolean = set.new_type("bool");
/// set.function("add", Op::Add, [real, real], real)
///     .function("mul", Op::Mul, [real, real], real)
///     .function("less", Op::Less, [real, real], boolean)
///     .function("if", Op::If, [boolean, real, real], real)
///     .terminal("x", Op::X, real)
///     .constants(real, Constants::uniform(-1.0..=1.0)?);
/// let set = set.build(real)?;
///
/// let tree = set.parse("if(less(x, 0.5), mul(x, x), add(x, 1.0))")?;
/// assert_eq!(tree.display(&set).to_string(), "if(less(x, 0.5), mul(x, x), add(x, 1.0))");
/// // a Boolean where a real is expected
/// assert!(set.parse("add(x, less(x, x))").is_err());
/// // `bool` has no terminal, but `less` makes one: the smallest Boolean tree has depth 1
/// assert_eq!(set.min_depth(boolean), Some(1));
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct PrimitiveSet<P> {
    types: Vec<String>,
    primitives: Vec<Primitive<P>>,
    constants: Vec<(Type, Constants)>,
    root: Type,
    #[cfg_attr(feature = "serde", serde(skip))]
    tables: Tables,
}

// what generation, parsing and the operators look up, derived from the set
#[derive(Clone, Debug, Default)]
struct Tables {
    // the terminals and the functions returning each type
    terminals: Vec<Vec<u32>>,
    functions: Vec<Vec<u32>>,
    // the constants of each type, a position in `constants`
    constants: Vec<Option<usize>>,
    // the fewest nodes of a tree of type t of depth at most d, at [min(d, columns - 1) * types
    // + t], or INFINITE: Montana's table of the types possible at each depth, with sizes
    min_size: Vec<u32>,
    columns: usize,
    // the primitives by name
    names: BTreeMap<String, u32>,
}

// no tree of the type fits
pub(crate) const INFINITE: u32 = u32::MAX;

impl<P: Copy> PrimitiveSet<P> {
    /// A builder of a set: declare types with
    /// [`new_type`](PrimitiveSetBuilder::new_type), then add functions, terminals and constants,
    /// and [`build`](PrimitiveSetBuilder::build) with the root type.
    pub fn builder() -> PrimitiveSetBuilder<P> {
        PrimitiveSetBuilder {
            types: Vec::new(),
            primitives: Vec::new(),
            constants: Vec::new(),
        }
    }

    fn from_parts(
        types: Vec<String>,
        primitives: Vec<Primitive<P>>,
        constants: Vec<(Type, Constants)>,
        root: Type,
    ) -> Result<Self> {
        let invalid = |reason: String| {
            Err(Error::InvalidSetting {
                setting: "primitives",
                reason,
            })
        };
        if types.is_empty() || types.len() > 1 << 16 {
            return invalid(format!("a set needs 1 to 2^16 types, got {}", types.len()));
        }
        if primitives.len() > MAX_SIZE {
            return invalid(format!(
                "a set has at most 2^24 primitives, got {}",
                primitives.len()
            ));
        }
        let declared = |ty: Type| ty.index() < types.len();
        if !declared(root) {
            return invalid(format!("the root type {root:?} isn't a type of the set"));
        }
        let mut names = BTreeMap::new();
        for (index, primitive) in primitives.iter().enumerate() {
            if let Some(ty) = primitive
                .args
                .iter()
                .chain([&primitive.returns])
                .find(|&&ty| !declared(ty))
            {
                return invalid(format!(
                    "`{}` has the type {ty:?}, which isn't a type of the set",
                    primitive.name
                ));
            }
            if let Err(reason) = check_name(&primitive.name) {
                return invalid(reason);
            }
            if names.insert(primitive.name.clone(), index as u32).is_some() {
                return invalid(format!("the name `{}` is used twice", primitive.name));
            }
        }
        let mut type_names = BTreeMap::new();
        for name in &types {
            if type_names.insert(name, ()).is_some() {
                return invalid(format!("the type name `{name}` is used twice"));
            }
        }
        let mut constant_of = vec![None; types.len()];
        for (position, (ty, _)) in constants.iter().enumerate() {
            if !declared(*ty) {
                return invalid(format!(
                    "constants of the type {ty:?}, which isn't a type of the set"
                ));
            }
            if constant_of[ty.index()].replace(position).is_some() {
                return invalid(format!(
                    "the type `{}` is given constants twice",
                    types[ty.index()]
                ));
            }
        }
        let mut terminals = vec![Vec::new(); types.len()];
        let mut functions = vec![Vec::new(); types.len()];
        for (index, primitive) in primitives.iter().enumerate() {
            let list = if primitive.args.is_empty() {
                &mut terminals
            } else {
                &mut functions
            };
            list[primitive.returns.index()].push(index as u32);
        }
        let (min_size, columns) = min_sizes(&types, &primitives, &terminals, &constant_of)
            .map_err(|reason| Error::InvalidSetting {
                setting: "primitives",
                reason,
            })?;
        let tables = Tables {
            terminals,
            functions,
            constants: constant_of,
            min_size,
            columns,
            names,
        };
        let set = Self {
            types,
            primitives,
            constants,
            root,
            tables,
        };
        // every type the root needs can make a tree
        let incomplete: Vec<&str> = set
            .needed_types()
            .into_iter()
            .filter(|&ty| set.min_depth(ty).is_none())
            .map(|ty| set.type_name(ty))
            .collect();
        if !incomplete.is_empty() {
            return invalid(format!(
                "no tree can be made of the types {incomplete:?}: each needs a terminal, \
                 constants, or a function whose arguments can be made"
            ));
        }
        Ok(set)
    }

    // the root type and the argument types of the functions returning a needed type, in order
    fn needed_types(&self) -> Vec<Type> {
        let mut needed = vec![false; self.types.len()];
        let mut stack = vec![self.root];
        needed[self.root.index()] = true;
        while let Some(ty) = stack.pop() {
            for &function in &self.tables.functions[ty.index()] {
                for &arg in &self.primitives[function as usize].args {
                    if !needed[arg.index()] {
                        needed[arg.index()] = true;
                        stack.push(arg);
                    }
                }
            }
        }
        (0..self.types.len())
            .filter(|&index| needed[index])
            .map(|index| Type(index as u16))
            .collect()
    }

    /// The type that trees return.
    pub fn root(&self) -> Type {
        self.root
    }

    /// The number of types.
    pub fn type_count(&self) -> usize {
        self.types.len()
    }

    /// The name of `ty`.
    ///
    /// # Panics
    ///
    /// If `ty` isn't a type of this set.
    pub fn type_name(&self, ty: Type) -> &str {
        &self.types[ty.index()]
    }

    /// The functions and terminals, in the order they were added: a
    /// [`Node::Primitive`] is a position in it.
    pub fn primitives(&self) -> &[Primitive<P>] {
        &self.primitives
    }

    /// The position of the primitive named `name`, for a [`Node::Primitive`].
    pub fn position(&self, name: &str) -> Option<u32> {
        self.tables.names.get(name).copied()
    }

    /// The ephemeral random constants of `ty`, if it has any.
    pub fn constants(&self, ty: Type) -> Option<&Constants> {
        let position = (*self.tables.constants.get(ty.index())?)?;
        Some(&self.constants[position].1)
    }

    /// The smallest depth of a tree of type `ty` (0 for a single terminal or constant), `None` if
    /// no tree of it can be made. A tree of `ty` fits in depth `d` (Montana's table of the types
    /// possible at each depth) exactly when `d` is at least this.
    pub fn min_depth(&self, ty: Type) -> Option<usize> {
        (0..self.tables.columns).find(|&depth| self.min_size(ty, depth) != INFINITE)
    }

    /// The fewest nodes of a tree of type `ty` of depth at most `depth`, `None` if there's none.
    pub fn min_nodes(&self, ty: Type, depth: usize) -> Option<usize> {
        match self.min_size(ty, depth) {
            INFINITE => None,
            size => Some(size as usize),
        }
    }

    // the fewest nodes of a tree of `ty` of depth at most `depth`, or INFINITE
    #[inline]
    pub(crate) fn min_size(&self, ty: Type, depth: usize) -> u32 {
        let column = depth.min(self.tables.columns - 1);
        self.tables.min_size[column * self.types.len() + ty.index()]
    }

    #[inline]
    pub(crate) fn terminals_of(&self, ty: Type) -> &[u32] {
        &self.tables.terminals[ty.index()]
    }

    #[inline]
    pub(crate) fn functions_of(&self, ty: Type) -> &[u32] {
        &self.tables.functions[ty.index()]
    }

    #[inline]
    pub(crate) fn arity(&self, index: u32) -> usize {
        self.primitives[index as usize].args.len()
    }

    // the type of a node
    #[inline]
    pub(crate) fn node_type(&self, node: &Node) -> Type {
        match *node {
            Node::Primitive(index) => self.primitives[index as usize].returns,
            Node::Constant { ty, .. } => ty,
        }
    }

    // the number of children of a node
    #[inline]
    pub(crate) fn node_arity(&self, node: &Node) -> usize {
        match *node {
            Node::Primitive(index) => self.arity(index),
            Node::Constant { .. } => 0,
        }
    }

    /// Parses a tree written as [`Tree::display`] writes it, e.g. `add(mul(x, x), 0.5)`: a
    /// primitive's name followed by its arguments in parentheses, separated by commas, or a
    /// number for a constant of the type expected there. Whitespace between tokens is ignored.
    /// The tree is typed, but not checked against a representation's limits:
    /// [`Gp::validate`](crate::genome::Representation::validate) does that.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidGenome`] for text that isn't a tree of this set: an unknown name, a
    /// type that doesn't fit, a constant of a type without constants or outside its values, or
    /// misplaced parentheses and commas.
    pub fn parse(&self, text: &str) -> Result<Tree> {
        Parser::new(self, text).parse()
    }
}

impl<P: PartialEq> PartialEq for PrimitiveSet<P> {
    fn eq(&self, other: &Self) -> bool {
        self.types == other.types
            && self.primitives == other.primitives
            && self.constants == other.constants
            && self.root == other.root
    }
}

// a name that display and parse can tell apart from punctuation and constants
fn check_name(name: &str) -> std::result::Result<(), String> {
    if name.is_empty()
        || name
            .chars()
            .any(|c| c.is_whitespace() || matches!(c, '(' | ')' | ','))
    {
        return Err(format!(
            "the name {name:?} must be non-empty, without whitespace, `(`, `)` or `,`"
        ));
    }
    if name.parse::<f64>().is_ok() {
        return Err(format!(
            "the name {name:?} reads as a number, which is how constants are written"
        ));
    }
    Ok(())
}

// Montana's table of the types possible at each depth, as the fewest nodes of a tree of each type
// of at most each depth: column d from column d - 1, until a column repeats (a smallest tree
// repeats no type along a path, so by `types` columns at the latest). Loops over the columns and
// the primitives, no recursion.
fn min_sizes<P>(
    types: &[String],
    primitives: &[Primitive<P>],
    terminals: &[Vec<u32>],
    constants: &[Option<usize>],
) -> std::result::Result<(Vec<u32>, usize), String> {
    let count = types.len();
    let mut table: Vec<u32> = (0..count)
        .map(|ty| {
            if terminals[ty].is_empty() && constants[ty].is_none() {
                INFINITE
            } else {
                1
            }
        })
        .collect();
    let mut columns = 1;
    loop {
        let previous = &table[(columns - 1) * count..];
        let mut next = previous.to_vec();
        for primitive in primitives.iter().filter(|p| !p.args.is_empty()) {
            let mut size: u32 = 1;
            for arg in &primitive.args {
                size = size.saturating_add(previous[arg.index()]);
            }
            let slot = &mut next[primitive.returns.index()];
            if size < *slot {
                *slot = size;
            }
        }
        if next == previous {
            return Ok((table, columns));
        }
        if (columns + 1) * count > MAX_SIZE {
            return Err(format!(
                "the table of the smallest trees per type and depth would have more than 2^24 \
                 entries ({count} types)"
            ));
        }
        table.extend(next);
        columns += 1;
    }
}

// Parses a tree without recursion: a stack of the functions whose arguments are still open.
struct Parser<'a, P> {
    set: &'a PrimitiveSet<P>,
    text: &'a str,
    position: usize,
}

impl<'a, P: Copy> Parser<'a, P> {
    fn new(set: &'a PrimitiveSet<P>, text: &'a str) -> Self {
        Self {
            set,
            text,
            position: 0,
        }
    }

    fn error<T>(&self, what: impl fmt::Display) -> Result<T> {
        Err(Error::InvalidGenome {
            reason: format!("at byte {} of {:?}: {what}", self.position, self.text),
        })
    }

    fn skip_whitespace(&mut self) {
        let rest = &self.text[self.position..];
        self.position += rest.len() - rest.trim_start().len();
    }

    // the next punctuation character, consumed if it's `expected`
    fn expect(&mut self, expected: char) -> Result<()> {
        self.skip_whitespace();
        if self.text[self.position..].starts_with(expected) {
            self.position += 1;
            Ok(())
        } else {
            self.error(format_args!("expected `{expected}`"))
        }
    }

    // the next name or number
    fn token(&mut self) -> Result<&'a str> {
        self.skip_whitespace();
        let rest = &self.text[self.position..];
        let len = rest
            .find(|c: char| c.is_whitespace() || matches!(c, '(' | ')' | ','))
            .unwrap_or(rest.len());
        if len == 0 {
            return self.error("expected a name or a number");
        }
        self.position += len;
        Ok(&rest[..len])
    }

    fn parse(mut self) -> Result<Tree> {
        let set = self.set;
        let mut nodes = Vec::new();
        // the open functions: their argument types, and the next argument
        let mut open: Vec<(&[Type], usize)> = Vec::new();
        loop {
            let expected = match open.last() {
                Some(&(args, next)) => args[next],
                None => set.root,
            };
            self.skip_whitespace();
            let start = self.position;
            let token = self.token()?;
            if let Some(index) = set.position(token) {
                let primitive = &set.primitives[index as usize];
                if primitive.returns != expected {
                    self.position = start;
                    return self.error(format_args!(
                        "`{token}` returns `{}` where `{}` is expected",
                        set.type_name(primitive.returns),
                        set.type_name(expected)
                    ));
                }
                nodes.push(Node::Primitive(index));
                if !primitive.args.is_empty() {
                    self.expect('(')?;
                    open.push((&primitive.args, 0));
                    continue;
                }
            } else if let Ok(value) = token.parse::<f64>() {
                match set.constants(expected) {
                    None => {
                        self.position = start;
                        return self.error(format_args!(
                            "the constant {token} where `{}`, which has no constants, is expected",
                            set.type_name(expected)
                        ));
                    }
                    Some(constants) if !constants.contains(value) => {
                        self.position = start;
                        return self.error(format_args!(
                            "the constant {token} isn't one of the constants of `{}`",
                            set.type_name(expected)
                        ));
                    }
                    Some(_) => {}
                }
                nodes.push(Node::Constant {
                    ty: expected,
                    value,
                });
            } else {
                self.position = start;
                return self.error(format_args!("`{token}` isn't a primitive of the set"));
            }
            // a subtree is complete: close the functions whose last argument it was
            loop {
                let Some((args, next)) = open.last_mut() else {
                    self.skip_whitespace();
                    if self.position < self.text.len() {
                        return self.error("expected the end of the tree");
                    }
                    return Ok(Tree::from(nodes));
                };
                *next += 1;
                if *next < args.len() {
                    self.expect(',')?;
                    break;
                }
                self.expect(')')?;
                open.pop();
            }
        }
    }
}

// validated like `build`
#[cfg(feature = "serde")]
impl<'de, P: Copy + serde::Deserialize<'de>> serde::Deserialize<'de> for PrimitiveSet<P> {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(rename = "PrimitiveSet")]
        struct Raw<P> {
            types: Vec<String>,
            primitives: Vec<Primitive<P>>,
            constants: Vec<(Type, Constants)>,
            root: Type,
        }
        // the constants are checked by their own deserialization
        let raw = Raw::<P>::deserialize(deserializer)?;
        Self::from_parts(raw.types, raw.primitives, raw.constants, raw.root)
            .map_err(serde::de::Error::custom)
    }
}

// validated like the constructors, by `PrimitiveSet`'s deserialization
#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for Constants {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(rename = "Constants")]
        enum Raw {
            Uniform { low: f64, high: f64 },
            Integers { low: i64, high: i64 },
            Choice(Vec<f64>),
        }
        let constants = match Raw::deserialize(deserializer)? {
            Raw::Uniform { low, high } => Constants::Uniform { low, high },
            Raw::Integers { low, high } => Constants::Integers { low, high },
            Raw::Choice(values) => Constants::Choice(values),
        };
        constants.checked().map_err(serde::de::Error::custom)
    }
}
