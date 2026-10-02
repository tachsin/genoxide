//! Genetic programming: trees of genoxide's built-in primitives, those of symbolic regression
//! (`gp::regression::Math`) and of the Boolean problems (`gp::boolean::Logic`), and of the user's
//! own primitives, evaluated by Python functions, in one primitive type, [`Op`], so that a tree
//! genome is one Rust type whichever set it uses. The primitive set and the trees as Python
//! objects ([`PyPrimitiveSet`], [`PyTree`], [`PyNode`]), the representation `Gp`, the tree
//! operators, and the algorithms that run trees: a GA, islands of GAs and NSGA-II with two
//! objectives.

use crate::config;
use crate::control::{GaSettings, IslandsSettings};
use crate::errors::{setting, setting_named};
use crate::genes::{GenomeContext, PyGenome};
use crate::operators::{RateOrCount, rate_or_count, wrong_crossover, wrong_mutate};
use crate::run::{
    Context, Failure, Returns, build_islands, ga_builder, generational, multi_objective,
};
use crate::snapshot::Kept;
use genoxide::genome::Representation;
use genoxide::gp::boolean::Logic;
use genoxide::gp::regression::Math;
use genoxide::gp::{
    Columns, ConstantMutation, Gp, HoistMutation, Init, Mutations, MutationsBuilder, Node,
    OnePointCrossover, PointMutation, PrimitiveSet, ShrinkMutation, SubtreeCrossover,
    SubtreeMutation, Tree, TreeMutation, Type,
};
use genoxide::operator::{Crossover, Mutate, NoCrossover};
use genoxide::prelude::*;
use numpy::{AllowTypeChange, PyArray1, PyArrayLikeDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyFloat, PyMapping, PyTuple};
use serde::{Deserialize, Serialize};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;

type Result<T> = std::result::Result<T, String>;

/// A primitive of the package's trees: a function or variable of symbolic regression, a
/// function or input of the Boolean problems, or a primitive of the user's own, by its position
/// in its set, which Python functions evaluate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Op {
    Math(Math),
    Logic(Logic),
    User(u32),
}

/// A primitive set of the package.
pub type Set = PrimitiveSet<Op>;

/// `set` with each primitive's value mapped by `map`: the same names, positions, types and
/// constants, so that a tree of one is a tree of the other. `None` if `map` maps a primitive to
/// `None`.
pub fn map_set<P: Copy, Q: Copy>(
    set: &PrimitiveSet<P>,
    map: impl Fn(P) -> Option<Q>,
) -> Option<PrimitiveSet<Q>> {
    let types = type_handles(set.type_count());
    let mut builder = PrimitiveSet::builder();
    for &ty in &types {
        builder.new_type(set.type_name(ty));
    }
    for primitive in set.primitives() {
        builder.function(
            primitive.name(),
            map(primitive.value())?,
            primitive.args().iter().copied(),
            primitive.returns(),
        );
    }
    for &ty in &types {
        if let Some(constants) = set.constants(ty) {
            builder.constants(ty, constants.clone());
        }
    }
    builder.build(set.root()).ok()
}

/// The handles of the first `count` types of a set, in order.
fn type_handles(count: usize) -> Vec<Type> {
    let mut handles = PrimitiveSet::<()>::builder();
    (0..count).map(|_| handles.new_type("")).collect()
}

/// A set of regression's primitives as a set of the package.
pub fn from_math(set: &PrimitiveSet<Math>) -> Set {
    map_set(set, |math| Some(Op::Math(math))).expect("the same set")
}

/// A set of the Boolean problems' primitives as a set of the package.
pub fn from_logic(set: &PrimitiveSet<Logic>) -> Set {
    map_set(set, |logic| Some(Op::Logic(logic))).expect("the same set")
}

/// A set of the package as a set of regression's primitives, if it is one.
pub fn to_math(set: &Set) -> Option<PrimitiveSet<Math>> {
    map_set(set, |op| match op {
        Op::Math(math) => Some(math),
        _ => None,
    })
}

// whether a set has a primitive of the user's own (`user`), or one of genoxide's
fn has_user(set: &Set, user: bool) -> bool {
    set.primitives()
        .iter()
        .any(|primitive| matches!(primitive.value(), Op::User(_)) == user)
}

/// A set of the user's own primitives: the types by name, the functions and terminals as
/// `(name, argument types, return type)`, with types by position in `types`, the constants of
/// types as `(type, JSON description of gx.gp.Constants)`, and the root type.
#[pyfunction]
pub fn user_primitives(
    types: Vec<String>,
    functions: Vec<(String, Vec<usize>, usize)>,
    constants: Vec<(usize, String)>,
    root: usize,
) -> PyResult<PyPrimitiveSet> {
    let mut builder = PrimitiveSet::builder();
    let handles: Vec<Type> = types
        .into_iter()
        .map(|name| builder.new_type(name))
        .collect();
    let ty = |index: usize| {
        handles.get(index).copied().ok_or_else(|| {
            PyValueError::new_err(format!("invalid setting `primitives`: no type {index}"))
        })
    };
    for (position, (name, args, returns)) in functions.into_iter().enumerate() {
        let args = args.into_iter().map(ty).collect::<PyResult<Vec<Type>>>()?;
        let position = u32::try_from(position)
            .map_err(|_| PyValueError::new_err("invalid setting `primitives`: too many"))?;
        builder.function(name, Op::User(position), args, ty(returns)?);
    }
    for (index, description) in constants {
        builder.constants(
            ty(index)?,
            crate::tree_problems::parse_constants(&description)?,
        );
    }
    let set = setting(builder.build(ty(root)?)).map_err(PyValueError::new_err)?;
    Ok(PyPrimitiveSet::new(set))
}

// whether two sets are the same: the same set object, or equal
fn same_set(a: &Arc<Set>, b: &Arc<Set>) -> bool {
    Arc::ptr_eq(a, b) || a == b
}

/// A primitive set of `gx.gp`: functions, terminals and constants of genoxide's built-in
/// primitives.
#[pyclass(frozen, module = "genoxide._genoxide", name = "PrimitiveSet")]
pub struct PyPrimitiveSet {
    pub set: Arc<Set>,
}

impl PyPrimitiveSet {
    pub fn new(set: Set) -> Self {
        Self { set: Arc::new(set) }
    }

    fn names(&self, terminals: bool) -> Vec<String> {
        self.set
            .primitives()
            .iter()
            .filter(|primitive| (primitive.arity() == 0) == terminals)
            .map(|primitive| primitive.name().to_string())
            .collect()
    }
}

#[pymethods]
impl PyPrimitiveSet {
    /// The set of its JSON description, as `_json` gives it.
    #[staticmethod]
    fn _from_json(text: &str) -> PyResult<Self> {
        let set: Set = serde_json::from_str(text)
            .map_err(|error| PyValueError::new_err(format!("invalid primitive set: {error}")))?;
        Ok(Self::new(set))
    }

    /// The set as JSON, for a run's description.
    fn _json(&self) -> PyResult<String> {
        serde_json::to_string(self.set.as_ref())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    /// The names of the functions, in order.
    #[getter]
    fn functions(&self) -> Vec<String> {
        self.names(false)
    }

    /// The names of the terminals (variables or inputs), in order.
    #[getter]
    fn terminals(&self) -> Vec<String> {
        self.names(true)
    }

    /// The names of the types, in order.
    #[getter]
    fn types(&self) -> Vec<String> {
        type_handles(self.set.type_count())
            .into_iter()
            .map(|ty| self.set.type_name(ty).to_string())
            .collect()
    }

    /// The name of the type that trees return.
    #[getter]
    fn root_type(&self) -> String {
        self.set.type_name(self.set.root()).to_string()
    }

    /// The ephemeral random constants of the root type, as text, or None.
    #[getter]
    fn constants(&self) -> Option<String> {
        let root = self.set.root();
        self.set.constants(root).map(constants_text)
    }

    /// The tree written as `display` writes it, e.g. `add(mul(x, x), 0.5)`.
    fn parse(&self, text: &str) -> PyResult<PyTree> {
        let tree = self
            .set
            .parse(text)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(PyTree {
            tree,
            set: Arc::clone(&self.set),
        })
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other
            .cast::<PyPrimitiveSet>()
            .is_ok_and(|other| same_set(&self.set, &other.get().set))
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.names(false).hash(&mut hasher);
        self.names(true).hash(&mut hasher);
        hasher.finish()
    }

    fn __repr__(&self) -> String {
        let constants = match self.constants() {
            Some(constants) => format!(", constants={constants}"),
            None => String::new(),
        };
        let types = if has_user(&self.set, true) {
            format!("types=[{}], ", self.types().join(", "))
        } else {
            String::new()
        };
        format!(
            "PrimitiveSet({types}functions=[{}], terminals=[{}]{constants})",
            self.names(false).join(", "),
            self.names(true).join(", ")
        )
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (String,))> {
        let load = py.get_type::<PyPrimitiveSet>().getattr("_from_json")?;
        Ok((load, (self._json()?,)))
    }
}

// ephemeral random constants as Python writes their constructor
fn constants_text(constants: &genoxide::gp::Constants) -> String {
    use genoxide::gp::Constants;
    match constants {
        Constants::Uniform { low, high } => format!("Constants.uniform({low:?}, {high:?})"),
        Constants::Integers { low, high } => format!("Constants.integers({low}, {high})"),
        Constants::Choice(values) => format!("Constants.choice({values:?})"),
        Constants::Normal { mean, deviation } => {
            format!("Constants.normal({mean:?}, {deviation:?})")
        }
    }
}

/// A node of a tree, as `Tree.nodes` gives it: its kind ("function", "terminal" or "constant"),
/// its name (a constant's value as the tree's text writes it), its number of children, its type
/// and a constant's value (None for a function or terminal).
#[pyclass(
    frozen,
    get_all,
    skip_from_py_object,
    module = "genoxide.gp",
    name = "Node"
)]
pub struct PyNode {
    kind: &'static str,
    name: String,
    arity: usize,
    #[pyo3(name = "type")]
    ty: String,
    value: Option<f64>,
}

#[pymethods]
impl PyNode {
    fn __repr__(&self) -> String {
        format!(
            "Node(kind='{}', name='{}', arity={}, type='{}')",
            self.kind, self.name, self.arity, self.ty
        )
    }
}

/// A tree of `gx.gp`: its nodes and its primitive set.
#[pyclass(frozen, module = "genoxide._genoxide", name = "Tree")]
pub struct PyTree {
    pub tree: Tree,
    pub set: Arc<Set>,
}

impl PyTree {
    /// The tree, if it's a tree of `set`.
    pub fn of(&self, set: &Arc<Set>) -> PyResult<&Tree> {
        if same_set(&self.set, set) {
            Ok(&self.tree)
        } else {
            Err(PyValueError::new_err(
                "the tree is of another primitive set: use a tree of this one",
            ))
        }
    }
}

#[pymethods]
impl PyTree {
    /// The tree of its JSON descriptions, as `_json` gives them.
    #[staticmethod]
    fn _from_json(set: &str, tree: &str) -> PyResult<Self> {
        let set = PyPrimitiveSet::_from_json(set)?.set;
        let tree: Tree = serde_json::from_str(tree)
            .map_err(|error| PyValueError::new_err(format!("invalid tree: {error}")))?;
        // a tree whose nodes aren't of its set would panic when displayed or evaluated: checked
        // by a representation of the set without limits
        let depth = set.min_depth(set.root()).unwrap_or(0);
        let gp = Gp::builder(set.as_ref().clone())
            .max_depth(1 << 24)
            .max_size(1 << 24)
            .init(Init::Grow {
                depths: depth..=depth,
            })
            .build()
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        gp.validate(&tree)
            .map_err(|error| PyValueError::new_err(error.to_string()))?;
        Ok(Self { tree, set })
    }

    /// The tree's nodes as JSON, for a run's description.
    fn _json(&self) -> PyResult<String> {
        serde_json::to_string(&self.tree)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn __len__(&self) -> usize {
        self.tree.len()
    }

    /// The depth: 0 for a single node.
    #[getter]
    fn depth(&self) -> usize {
        self.tree.depth(&self.set)
    }

    /// The primitive set.
    #[getter]
    fn primitives(&self) -> PyPrimitiveSet {
        PyPrimitiveSet {
            set: Arc::clone(&self.set),
        }
    }

    /// The tree as text, e.g. `add(mul(x, x), 0.5)`.
    fn display(&self) -> String {
        self.tree.display(&self.set).to_string()
    }

    fn __str__(&self) -> String {
        self.display()
    }

    /// The nodes in prefix order: each function followed by its children's subtrees, in order.
    fn nodes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let nodes = self.tree.nodes().iter().map(|node| match *node {
            Node::Primitive(index) => {
                let primitive = &self.set.primitives()[index as usize];
                PyNode {
                    kind: if primitive.arity() > 0 {
                        "function"
                    } else {
                        "terminal"
                    },
                    name: primitive.name().to_string(),
                    arity: primitive.arity(),
                    ty: self.set.type_name(primitive.returns()).to_string(),
                    value: None,
                }
            }
            Node::Constant { ty, value } => PyNode {
                kind: "constant",
                name: format!("{value:?}"),
                arity: 0,
                ty: self.set.type_name(ty).to_string(),
                value: Some(value),
            },
        });
        PyTuple::new(py, nodes)
    }

    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let text = pyo3::types::PyString::new(py, &self.display()).repr()?;
        Ok(format!("Tree({text})"))
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        other.cast::<PyTree>().is_ok_and(|other| {
            let other = other.get();
            self.tree == other.tree && same_set(&self.set, &other.set)
        })
    }

    fn __hash__(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.tree.hash(&mut hasher);
        hasher.finish()
    }

    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, (String, String))> {
        let load = py.get_type::<PyTree>().getattr("_from_json")?;
        let set = serde_json::to_string(self.set.as_ref())
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))?;
        Ok((load, (set, self._json()?)))
    }

    /// The tree's values at points, on all of them at once: `x` holds a point per row, a value
    /// per variable (or input) in the order of the set's terminals, or for one variable a 1-D
    /// array of its values. Numbers for regression's primitives, bools for the Boolean ones.
    ///
    /// With `functions`, a tree of the user's own primitives: each function's callable, by
    /// name, called once per node, its children's values before it; the terminals' values from
    /// `x`, a mapping by name or an array as above, a column per terminal.
    #[pyo3(signature = (x, functions = None))]
    fn evaluate<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        functions: Option<&Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Some(functions) = functions {
            if has_user(&self.set, false) {
                return Err(PyValueError::new_err(
                    "functions are for trees of your own primitives: genoxide evaluates its \
                     own, with evaluate(x)",
                ));
            }
            return self.evaluate_user(py, x, functions);
        }
        if has_user(&self.set, true) {
            return Err(PyValueError::new_err(
                "a tree of your own primitives is evaluated with evaluate(x, functions), \
                 functions mapping each function's name to a callable",
            ));
        }
        let x: PyArrayLikeDyn<'py, f64, AllowTypeChange> = x.extract()?;
        let columns = columns(&x)?;
        let variables = self
            .set
            .primitives()
            .iter()
            .filter_map(|primitive| match primitive.value() {
                Op::Math(Math::Variable(k)) | Op::Logic(Logic::Input(k)) => {
                    Some(usize::from(k) + 1)
                }
                _ => None,
            })
            .max()
            .unwrap_or(0);
        if columns.len() < variables {
            return Err(PyValueError::new_err(format!(
                "the tree's set has {variables} variables, but x has {} columns",
                columns.len()
            )));
        }
        let boolean = self
            .set
            .primitives()
            .iter()
            .all(|primitive| matches!(primitive.value(), Op::Logic(_)));
        let points = columns.first().map_or(x.shape()[0], Vec::len);
        let values = py.detach(|| values(&self.tree, &self.set, &columns, points));
        if boolean {
            let bools: Vec<bool> = values.iter().map(|&value| value != 0.0).collect();
            Ok(PyArray1::from_vec(py, bools).into_any())
        } else {
            Ok(PyArray1::from_vec(py, values).into_any())
        }
    }
}

impl PyTree {
    // the value of a tree of the user's primitives, bottom-up as Rust's `Tree::evaluate`: the
    // prefix order read backwards, each function called with its children's values in order
    fn evaluate_user<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'py, PyAny>,
        functions: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let functions = functions.cast::<PyMapping>().map_err(|_| {
            PyValueError::new_err("functions is a mapping of each function's name to a callable")
        })?;
        let primitives = self.set.primitives();
        let terminals = primitives.iter().filter(|p| p.arity() == 0).count();
        // the terminals' values: by name, or the columns of an array, in order
        let mut columns = match x.cast::<PyMapping>() {
            Ok(_) => None,
            Err(_) => {
                let array = py.import("numpy")?.call_method1("asarray", (x,))?;
                let columns: Vec<Bound<'py, PyAny>> = match array.getattr("ndim")?.extract()? {
                    1usize => vec![array],
                    2 => array.getattr("T")?.try_iter()?.collect::<PyResult<_>>()?,
                    _ => {
                        return Err(PyValueError::new_err(
                            "x is a mapping of each terminal's name to its values, a 2-D array \
                             with a column per terminal, or a 1-D array of one terminal",
                        ));
                    }
                };
                if columns.len() != terminals {
                    return Err(PyValueError::new_err(format!(
                        "the tree's set has {terminals} terminals, but x has {} columns",
                        columns.len()
                    )));
                }
                Some(columns.into_iter())
            }
        };
        // each primitive's callable or value
        let mut meanings = Vec::with_capacity(primitives.len());
        for primitive in primitives {
            let name = primitive.name();
            let meaning = if primitive.arity() > 0 {
                functions.get_item(name).map_err(|_| {
                    PyValueError::new_err(format!("functions has no function `{name}`"))
                })?
            } else if let Some(columns) = &mut columns {
                columns.next().expect("a column per terminal")
            } else {
                x.get_item(name).map_err(|_| {
                    PyValueError::new_err(format!("x has no value for the terminal `{name}`"))
                })?
            };
            meanings.push(meaning);
        }
        let mut stack: Vec<Bound<'py, PyAny>> = Vec::new();
        for node in self.tree.nodes().iter().rev() {
            let value = match *node {
                Node::Constant { value, .. } => PyFloat::new(py, value).into_any(),
                Node::Primitive(index) => {
                    let index = index as usize;
                    let arity = primitives[index].arity();
                    if arity == 0 {
                        meanings[index].clone()
                    } else {
                        // the first child is on top
                        let start = stack.len() - arity;
                        let args = PyTuple::new(py, stack.drain(start..).rev())?;
                        meanings[index].call1(args)?
                    }
                }
            };
            stack.push(value);
        }
        Ok(stack.pop().expect("a tree has a root"))
    }
}

// the columns of a point per row, or of a 1-D array of one variable
fn columns(x: &PyArrayLikeDyn<'_, f64, AllowTypeChange>) -> PyResult<Vec<Vec<f64>>> {
    let array = x.as_array();
    match array.ndim() {
        1 => Ok(vec![array.iter().copied().collect()]),
        2 => Ok(array
            .columns()
            .into_iter()
            .map(|column| column.iter().copied().collect())
            .collect()),
        _ => Err(PyValueError::new_err(format!(
            "x is a 2-D array, a point per row, or a 1-D array of one variable, not of shape {:?}",
            array.shape()
        ))),
    }
}

/// The values of a tree at the points of `columns`, a column per variable: regression's
/// primitives as `Regression` computes them, the Boolean ones on 0 and 1, nonzero as true.
pub fn values(tree: &Tree, set: &Set, columns: &[Vec<f64>], points: usize) -> Vec<f64> {
    let mut workspace = Columns::new(points);
    tree.evaluate_columns(set, &mut workspace, |op, args, output| match op {
        Op::Math(math) => math.apply_columns(args, columns, output),
        Op::Logic(logic) => logic_columns(logic, args, columns, output),
        // evaluated by Python: not here
        Op::User(_) => output.fill(f64::NAN),
    })
    .to_vec()
}

fn logic_columns(logic: Logic, args: &[&[f64]], inputs: &[Vec<f64>], output: &mut [f64]) {
    let truth = |value: f64| value != 0.0;
    for (point, out) in output.iter_mut().enumerate() {
        let arg = |k: usize| truth(args[k][point]);
        let value = match logic {
            Logic::And => arg(0) && arg(1),
            Logic::Or => arg(0) || arg(1),
            Logic::Nand => !(arg(0) && arg(1)),
            Logic::Nor => !(arg(0) || arg(1)),
            Logic::Not => !arg(0),
            Logic::If => {
                if arg(0) {
                    arg(1)
                } else {
                    arg(2)
                }
            }
            Logic::Input(input) => truth(inputs[usize::from(input)][point]),
        };
        *out = if value { 1.0 } else { 0.0 };
    }
}

/// Trees kept after a generation: a tuple of `Tree`s when read.
struct KeptTrees {
    trees: Vec<Tree>,
    set: Arc<Set>,
}

impl Kept for KeptTrees {
    fn genomes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let trees = self.trees.iter().map(|tree| PyTree {
            tree: tree.clone(),
            set: Arc::clone(&self.set),
        });
        Ok(PyTuple::new(py, trees)?.into_any())
    }
}

// the set of a run's trees
fn set_of(cx: &GenomeContext) -> PyResult<&Arc<Set>> {
    match cx {
        GenomeContext::Tree(set) => Ok(set),
        _ => Err(PyRuntimeError::new_err("a tree without its primitive set")),
    }
}

impl PyGenome for Tree {
    fn object<'py>(&self, py: Python<'py>, cx: &GenomeContext) -> PyResult<Bound<'py, PyAny>> {
        let tree = PyTree {
            tree: self.clone(),
            set: Arc::clone(set_of(cx)?),
        };
        Ok(Bound::new(py, tree)?.into_any())
    }

    fn batch<'py>(
        py: Python<'py>,
        genomes: &[&Self],
        cx: &GenomeContext,
    ) -> PyResult<Bound<'py, PyAny>> {
        let set = set_of(cx)?;
        let trees = genomes.iter().map(|tree| PyTree {
            tree: (*tree).clone(),
            set: Arc::clone(set),
        });
        Ok(PyTuple::new(py, trees)?.into_any())
    }

    fn keep<'a>(
        genomes: impl ExactSizeIterator<Item = &'a Self>,
        cx: &GenomeContext,
    ) -> Box<dyn Kept>
    where
        Self: 'a,
    {
        let set = match cx {
            GenomeContext::Tree(set) => Arc::clone(set),
            _ => unreachable!("a run of trees has their primitive set"),
        };
        Box::new(KeptTrees {
            trees: genomes.cloned().collect(),
            set,
        })
    }

    fn tree(&self) -> Option<&Tree> {
        Some(self)
    }
}

/// The representation that `gp` describes.
pub fn build_gp(gp: config::Gp) -> Result<Gp<Op>> {
    let init = match gp.init {
        config::Init::Full {
            depths: (low, high),
        } => Init::Full { depths: low..=high },
        config::Init::Grow {
            depths: (low, high),
        } => Init::Grow { depths: low..=high },
        config::Init::RampedHalfAndHalf {
            depths: (low, high),
        } => Init::RampedHalfAndHalf { depths: low..=high },
    };
    let names = [
        ("max_depth", "Gp.max_depth"),
        ("max_size", "Gp.max_size"),
        ("init", "Gp.init"),
    ];
    setting_named(
        Gp::builder(gp.primitives)
            .max_depth(gp.max_depth)
            .max_size(gp.max_size)
            .init(init)
            .build(),
        &names,
    )
}

// the representation of a `gx.gp.Gp`'s description
fn parse_gp(description: &str) -> PyResult<Gp<Op>> {
    let genome: config::Genome = serde_json::from_str(description)
        .map_err(|error| PyValueError::new_err(format!("invalid setting `genome`: {error}")))?;
    match genome {
        config::Genome::Gp(gp) => build_gp(*gp).map_err(PyValueError::new_err),
        _ => Err(PyValueError::new_err("not a Gp")),
    }
}

fn py_tree(gp: &Gp<Op>, set: &Arc<Set>, tree: Tree) -> PyTree {
    debug_assert!(gp.primitives() == set.as_ref());
    PyTree {
        tree,
        set: Arc::clone(set),
    }
}

/// Checks a `gx.gp.Gp`'s description: a ValueError naming the setting if it's wrong.
#[pyfunction]
pub fn gp_check(description: &str) -> PyResult<()> {
    parse_gp(description).map(drop)
}

/// `count` trees of Koza's ramped half-and-half, from `seed`: the representation's depths
/// divided evenly among the depths and the full and grow methods, without duplicates.
#[pyfunction]
pub fn gp_ramped_half_and_half(
    py: Python<'_>,
    description: &str,
    count: usize,
    seed: u64,
) -> PyResult<Vec<PyTree>> {
    let gp = parse_gp(description)?;
    let set = Arc::new(gp.primitives().clone());
    let trees =
        py.detach(|| setting(gp.ramped_half_and_half(count, &mut StreamRng::seed_from_u64(seed))));
    let trees = trees.map_err(PyValueError::new_err)?;
    Ok(trees
        .into_iter()
        .map(|tree| py_tree(&gp, &set, tree))
        .collect())
}

/// A random tree of the representation's initialization, from `seed`.
#[pyfunction]
pub fn gp_random_genome(description: &str, seed: u64) -> PyResult<PyTree> {
    let gp = parse_gp(description)?;
    let set = Arc::new(gp.primitives().clone());
    let tree = gp.random_genome(&mut StreamRng::seed_from_u64(seed));
    Ok(py_tree(&gp, &set, tree))
}

/// Checks that `tree` is a tree of the representation: of its set and within its limits.
#[pyfunction]
pub fn gp_validate(description: &str, tree: &PyTree) -> PyResult<()> {
    let gp = parse_gp(description)?;
    if gp.primitives() != tree.set.as_ref() {
        return Err(PyValueError::new_err(
            "the tree is of another primitive set than the Gp's",
        ));
    }
    gp.validate(&tree.tree)
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// The tree written in `text`, checked against the representation's limits.
#[pyfunction]
pub fn gp_parse(description: &str, text: &str) -> PyResult<PyTree> {
    let gp = parse_gp(description)?;
    let tree = gp
        .primitives()
        .parse(text)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    gp.validate(&tree)
        .map_err(|error| PyValueError::new_err(error.to_string()))?;
    let set = Arc::new(gp.primitives().clone());
    Ok(py_tree(&gp, &set, tree))
}

/// A crossover of trees.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TreeCrossover {
    Subtree(SubtreeCrossover),
    OnePoint(OnePointCrossover),
    None(NoCrossover),
}

impl TreeCrossover {
    pub fn new(crossover: config::Crossover) -> Result<Self> {
        match crossover {
            config::Crossover::Subtree { internal_rate } => {
                Ok(Self::Subtree(match internal_rate {
                    Some(rate) => setting_named(
                        SubtreeCrossover::with_internal_rate(rate),
                        &[("internal_rate", "SubtreeCrossover.internal_rate")],
                    )?,
                    None => SubtreeCrossover::new(),
                }))
            }
            config::Crossover::OnePoint {} => Ok(Self::OnePoint(OnePointCrossover)),
            config::Crossover::None {} => Ok(Self::None(NoCrossover)),
            _ => Err(wrong_crossover(
                crossover,
                "tree",
                "gx.gp.SubtreeCrossover, gx.gp.OnePointCrossover or NoCrossover",
            )),
        }
    }
}

impl Crossover<Gp<Op>> for TreeCrossover {
    fn crossover(&self, gp: &Gp<Op>, a: &mut Tree, b: &mut Tree, rng: &mut StreamRng) {
        match self {
            Self::Subtree(crossover) => crossover.crossover(gp, a, b, rng),
            Self::OnePoint(crossover) => crossover.crossover(gp, a, b, rng),
            Self::None(crossover) => crossover.crossover(gp, a, b, rng),
        }
    }

    fn recombines(&self) -> bool {
        !matches!(self, Self::None(_))
    }
}

/// A mutation of trees: one of them, or a mix chosen by weight.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TreeMutate {
    One(TreeMutation),
    Mix(Mutations),
}

const TREE_MUTATIONS: &str = "gx.gp.SubtreeMutation, gx.gp.PointMutation, gx.gp.HoistMutation, \
                              gx.gp.ShrinkMutation, gx.gp.ConstantMutation or gx.gp.Mutations";

// one tree mutation, of a mix or not
fn tree_mutation(mutate: config::Mutate) -> Result<TreeMutation> {
    Ok(match mutate {
        config::Mutate::Subtree { max_depth } => TreeMutation::Subtree(match max_depth {
            Some(depth) => setting_named(
                SubtreeMutation::with_max_depth(depth),
                &[("max_depth", "SubtreeMutation.max_depth")],
            )?,
            None => SubtreeMutation::new(),
        }),
        config::Mutate::Point { rate, count } => {
            let names = [
                ("point_mutation_rate", "PointMutation.rate"),
                ("point_mutation_count", "PointMutation.count"),
            ];
            TreeMutation::Point(match rate_or_count(rate, count)? {
                RateOrCount::Rate(rate) => setting_named(PointMutation::per_node(rate), &names)?,
                RateOrCount::Count(count) => setting_named(PointMutation::count(count), &names)?,
            })
        }
        config::Mutate::Hoist {} => TreeMutation::Hoist(HoistMutation),
        config::Mutate::Shrink {} => TreeMutation::Shrink(ShrinkMutation),
        config::Mutate::Constant { sigma } => TreeMutation::Constant(setting_named(
            ConstantMutation::gaussian(sigma),
            &[("constant_sigma", "ConstantMutation.sigma")],
        )?),
        mutate => return Err(wrong_mutate(&mutate, "tree", TREE_MUTATIONS)),
    })
}

impl TreeMutate {
    pub fn new(mutate: config::Mutate) -> Result<Self> {
        match mutate {
            config::Mutate::Mutations { mutations } => {
                let mut builder = MutationsBuilder::default();
                for (weight, mutation) in mutations {
                    if let config::Mutate::Mutations { .. } = mutation {
                        return Err(
                            "invalid setting `Mutations.mutations`: a mix of single mutations, \
                             not of other mixes"
                                .to_string(),
                        );
                    }
                    builder = builder.with(weight, tree_mutation(mutation)?);
                }
                let names = [("mutations", "Mutations.mutations")];
                Ok(Self::Mix(setting_named(builder.build(), &names)?))
            }
            mutate => Ok(Self::One(tree_mutation(mutate)?)),
        }
    }
}

impl Mutate<Gp<Op>> for TreeMutate {
    fn mutate(&self, gp: &Gp<Op>, tree: &mut Tree, rng: &mut StreamRng) {
        match self {
            Self::One(mutation) => mutation.mutate(gp, tree, rng),
            Self::Mix(mutations) => mutations.mutate(gp, tree, rng),
        }
    }
}

// the initial trees of a builder's population, checked by its `build`
fn initial_error(error: genoxide::Error) -> String {
    match error {
        genoxide::Error::InvalidGenome { reason } => {
            format!("invalid setting `initial_genomes`: {reason}")
        }
        error => setting::<()>(Err(error)).unwrap_err(),
    }
}

/// Runs a GA, islands of GAs or NSGA-II with two objectives on trees of `gp`.
pub fn tree_algorithm<'py>(
    py: Python<'py>,
    gp: Result<Gp<Op>>,
    algorithm: config::Algorithm,
    context: &Context,
) -> Returns<'py> {
    let gp = gp?;
    if let Some(fitness) = &context.tree {
        fitness.check(gp.primitives(), &context.objectives)?;
    }
    let settings = GaSettings {
        crossover: TreeCrossover::new,
        mutate: TreeMutate::new,
    };
    match algorithm {
        config::Algorithm::Ga(ga) => {
            let ga = build_ga(gp, ga, context)?;
            generational(py, ga, settings, context)
        }
        config::Algorithm::Islands {
            islands,
            topology,
            interval,
            migrants,
            seed,
        } => {
            let islands = islands
                .into_iter()
                .map(|island| match island {
                    config::Algorithm::Ga(ga) => build_ga(gp.clone(), ga, context),
                    _ => Err(Failure::from("the islands of trees are Ga".to_string())),
                })
                .collect::<std::result::Result<Vec<_>, _>>()?;
            let islands = build_islands(islands, topology, interval, migrants, seed)?;
            generational(py, islands, IslandsSettings(settings), context)
        }
        config::Algorithm::Nsga2 {
            population_size,
            seed,
            variation,
            initial_genomes,
        } => {
            let objectives: [Objective; 2] = context.objectives.clone().try_into().map_err(
                |objectives: Vec<_>| {
                    format!(
                        "trees run with Nsga2 of 2 objectives, e.g. the error and the size, not {}",
                        objectives.len()
                    )
                },
            )?;
            let mut builder = Nsga2::builder(gp, objectives)
                .population_size(population_size)
                .crossover(TreeCrossover::new(variation.crossover)?)
                .mutate(TreeMutate::new(variation.mutate)?);
            if let Some(rate) = variation.crossover_rate {
                builder = builder.crossover_rate(rate);
            }
            if let Some(rate) = variation.mutation_rate {
                builder = builder.mutation_rate(rate);
            }
            if let Some(seed) = seed {
                builder = builder.seed(seed);
            }
            if let Some(eliminate) = variation.eliminate_duplicates {
                builder = builder.eliminate_duplicates(eliminate);
            }
            if let Some(trees) = initial_genomes {
                builder = builder.initial_genomes(trees);
            }
            let nsga2 = builder.build().map_err(initial_error)?;
            multi_objective::<_, 2>(py, nsga2, context)
        }
        _ => Err(Failure::from(
            "trees run with Ga, Islands of Ga, or Nsga2 of 2 objectives".to_string(),
        )),
    }
}

type TreeGa = Ga<Gp<Op>, crate::operators::AnySelect, TreeCrossover, TreeMutate>;

// a GA of trees, with its initial trees
fn build_ga(
    gp: Gp<Op>,
    mut ga: config::Ga,
    context: &Context,
) -> std::result::Result<TreeGa, Failure> {
    let initial = ga.initial_genomes.take();
    let mut builder = ga_builder(gp, &ga, context, TreeCrossover::new, TreeMutate::new)?;
    if let Some(trees) = initial {
        builder = builder.initial_genomes(trees);
    }
    Ok(builder.build().map_err(initial_error)?)
}
