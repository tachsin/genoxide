import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, overload

import numpy as np
from typing_extensions import Self

__version__: str

def run(
    config: str,
    fitness: Callable[[Any], Any],
    batch: bool = False,
    parallel: bool = False,
    on_generation: Callable[..., bool] | None = None,
    problem: str | None = None,
    control: Callable[..., None] | None = None,
    checkpoint: str | os.PathLike[str] | None = None,
    checkpoint_every: int | None = None,
    resume: str | os.PathLike[str] | None = None,
    gradient: Callable[[Any], Any] | None = None,
    combined_gradient: bool = False,
    constraints: int = 0,
    on_stage: Callable[[int], Any] | None = None,
    on_stage_finished: Callable[[dict[str, Any], Any], Any] | None = None,
) -> dict[str, Any]: ...

class Running:
    """The running algorithm, for a control: valid during the control's call only."""

    def get(self, name: str) -> str: ...
    def set(self, name: str, value: str) -> None: ...
    def reevaluate(self) -> None: ...
    def model(self) -> GaussianProcess | None: ...
    def acquisition(self, points: np.ndarray) -> np.ndarray: ...
    def feasibility(self, points: np.ndarray) -> np.ndarray: ...

class GaussianProcess:
    """A Gaussian process of ``genoxide::model::gp``, fitted from its description."""

    def __init__(self, description: str, points: np.ndarray, values: np.ndarray) -> None: ...
    def predict(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    def predict_with_gradient(
        self, point: np.ndarray
    ) -> tuple[float, float, np.ndarray, np.ndarray]: ...
    @property
    def hyperparameters(self) -> tuple[float, list[float], float, float]: ...
    @property
    def log_marginal_likelihood(self) -> float: ...
    @property
    def jitter(self) -> float: ...
    @property
    def kernel(self) -> str: ...
    @property
    def genes(self) -> int: ...
    def __len__(self) -> int: ...

class Snapshot:
    """A population (or a front) after a generation, for a progress object: its arrays are made
    when read."""

    def genomes(self) -> Any: ...
    def values(self) -> np.ndarray: ...
    def violations(self) -> np.ndarray: ...

def das_dennis(objectives: int, divisions: int) -> np.ndarray: ...
def problem_info(problem: str) -> dict[str, Any]: ...
def evaluate(problem: str, genomes: np.ndarray) -> Any: ...
def constraints(problem: str, genome: np.ndarray) -> np.ndarray: ...
def optimal_front(problem: str, points: int) -> np.ndarray | None: ...
def design(problem: str, genome: np.ndarray) -> np.ndarray: ...
def problem_names() -> list[str]: ...
def multi_problem_names(objectives: int) -> list[str]: ...
def indicator(
    kind: str, front: np.ndarray, reference: np.ndarray, objectives: list[str]
) -> float: ...

class Network:
    """A network of ``gx.nn``, from its description."""

    def __init__(self, description: str) -> None: ...
    @property
    def parameters(self) -> int: ...
    @property
    def inputs(self) -> int: ...
    @property
    def outputs(self) -> int: ...
    def forward(self, weights: np.ndarray, inputs: np.ndarray) -> np.ndarray: ...
    def policy(self, weights: np.ndarray) -> Policy: ...

class Policy:
    """A network with its weights, a policy of the control tasks that runs in Rust."""

    @property
    def inputs(self) -> int: ...
    @property
    def outputs(self) -> int: ...

class Task:
    """A control task of ``gx.problems.control``, from its description."""

    def __init__(self, description: str) -> None: ...
    @property
    def observations(self) -> int: ...
    def run(self, policy: Any, steps: int) -> int: ...
    def solved(self, policy: Any) -> bool: ...
    def damping_fitness(self, policy: Any) -> float: ...
    def generalization(self, policy: Any) -> int: ...
    def episode(self, policy: Any, steps: int) -> np.ndarray: ...

SUCCESS_STEPS: int
DAMPING_STEPS: int
GENERALIZATION_THRESHOLD: int

def balance_evaluate(description: str, genomes: np.ndarray) -> np.ndarray: ...
def portable_math(function: str, a: np.ndarray, b: np.ndarray | None = None) -> np.ndarray: ...
def random_real(bounds: list[tuple[float, float]], seed: int) -> np.ndarray: ...

class NeatNetwork:
    """A NEAT network: node genes and connection genes. A NEAT run's genome, as
    ``genoxide.neat.Network``."""

    @staticmethod
    def fully_connected(inputs: int, outputs: int, weights: Sequence[float]) -> NeatNetwork:
        """The network of ``inputs`` inputs and the bias, each connected to each of ``outputs``
        outputs with the steepened sigmoid, the weights output by output: the inputs', then the
        bias's. Its innovation numbers are those of a NEAT run's initial networks."""
    @property
    def inputs(self) -> int:
        """The number of inputs, without the bias."""
    @property
    def outputs(self) -> int:
        """The number of outputs."""
    def hidden(self) -> int:
        """The number of hidden nodes."""
    def enabled(self) -> int:
        """The number of enabled connections."""
    def nodes(self) -> tuple[NodeGene, ...]:
        """The node genes, by id: the inputs, the bias, the outputs, then the hidden nodes."""
    def connections(self) -> tuple[ConnectionGene, ...]:
        """The connection genes, by innovation number, enabled or not."""
    def __len__(self) -> int:
        """The number of connection genes, enabled or not."""
    def feed_forward(self) -> FeedForward:
        """The enabled connections compiled into a feed-forward evaluator: a ``ValueError`` if
        they form a cycle."""
    def recurrent(self) -> Recurrent:
        """The enabled connections compiled into a recurrent evaluator: each ``activate`` a step
        of time, every node computed from the previous step's values."""
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...

class NodeGene:
    """A node gene of a NEAT network."""

    @property
    def id(self) -> int:
        """Its id."""
    @property
    def kind(self) -> Literal["input", "bias", "output", "hidden"]:
        """What the node is."""
    @property
    def activation(self) -> str:
        """Its activation function: "identity" for the inputs and the bias."""

class ConnectionGene:
    """A connection gene of a NEAT network."""

    @property
    def innovation(self) -> int:
        """Its innovation number: the same for the same structure throughout a run."""
    @property
    def from_(self) -> int:
        """The id of the node it leaves."""
    @property
    def to(self) -> int:
        """The id of the node it enters."""
    @property
    def weight(self) -> float:
        """Its weight."""
    @property
    def enabled(self) -> bool:
        """Whether it's enabled: a disabled gene is inherited but not expressed."""

class FeedForward:
    """A NEAT network compiled for feed-forward evaluation, from ``Network.feed_forward()``: a
    policy of the control tasks too, run in Rust."""

    @property
    def inputs(self) -> int: ...
    @property
    def outputs(self) -> int: ...
    def activate(self, input: Any, out: np.ndarray | None = None) -> np.ndarray:
        """The outputs for ``input``, a 1-D array of ``inputs`` values, as a 1-D array (written
        into ``out``, a 1-D float64 array, and returned, if given); or for each row of a 2-D
        array, a row each. A node without enabled inputs, and an output nothing reaches, has the
        activation of 0. A ``ValueError`` for another number of inputs."""
    def policy(self, *, scale: float = 1.0, offset: float = 0.0) -> Policy:
        """A policy of the control tasks whose actions are ``scale * output + offset``: with 2 and
        -1, NEAT's sigmoid outputs in (0, 1) as forces in (-1, 1)."""

class Recurrent:
    """A NEAT network compiled for recurrent evaluation, from ``Network.recurrent()``: a step of
    time per ``activate``, until ``reset``; a policy of the control tasks too, run in Rust, from a
    reset network each episode."""

    @property
    def inputs(self) -> int: ...
    @property
    def outputs(self) -> int: ...
    def activate(self, input: Any, out: np.ndarray | None = None) -> np.ndarray:
        """One step: the outputs for ``input``, a 1-D array, as a 1-D array (into ``out`` if
        given); or a step for each row of a 2-D array, in order, a row of outputs each."""
    def reset(self) -> None:
        """Sets every node back to 0, as before the first step."""
    def policy(self, *, scale: float = 1.0, offset: float = 0.0) -> Policy:
        """A policy of the control tasks whose actions are ``scale * output + offset``."""

def neat_network(json: str) -> NeatNetwork: ...

class PrimitiveSet:
    """The functions, terminals and ephemeral random constants of a genetic program's trees,
    strongly typed: of genoxide's built-in primitives, from ``gx.gp.regression.primitives(...)``
    or a problem's ``primitives()``, or of your own, from ``gx.gp.PrimitiveSetBuilder``. Sets
    compare equal by their types, primitives and constants, and pickle."""

    @property
    def types(self) -> list[str]:
        """The names of the types, in order: one for the built-in sets."""
    @property
    def root_type(self) -> str:
        """The name of the type that trees return."""

    @property
    def functions(self) -> list[str]:
        """The names of the functions, in order."""
    @property
    def terminals(self) -> list[str]:
        """The names of the terminals (the variables or inputs), in order."""
    @property
    def constants(self) -> str | None:
        """The ephemeral random constants of the root type, e.g.
        ``"Constants.normal(0.0, 5.0)"``, or None."""
    def parse(self, text: str) -> Tree:
        """The tree written in ``text`` as ``str(tree)`` writes it, e.g. ``add(mul(x, x), 0.5)``:
        a primitive's name and its arguments in parentheses, separated by commas, or a number
        for a constant. A ``ValueError`` for text that isn't a tree of the set. The limits of a
        ``gx.gp.Gp`` aren't checked: ``gp.parse`` does that."""
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    @staticmethod
    def _from_json(text: str) -> PrimitiveSet: ...
    def _json(self) -> str: ...

class Tree:
    """A tree of a genetic program, in Rust: its nodes in prefix order and its primitive set. A
    run's fitness function gets one, and its result's ``best_genome`` is one. It's read-only;
    trees compare equal by their nodes and set, hash, and pickle."""

    def __len__(self) -> int:
        """The number of nodes: what ``max_size`` counts, and the size that
        ``DoubleTournament``, ``LexicographicTournament`` and ``Tarpeian`` see."""
    @property
    def depth(self) -> int:
        """The depth, the root at depth 0: 0 for a single node."""
    @property
    def primitives(self) -> PrimitiveSet:
        """The tree's primitive set."""
    def display(self) -> str:
        """The tree as text, e.g. ``add(mul(x, x), 0.5)``, as ``str(tree)``: the set's
        ``parse`` reads it back."""
    @overload
    def evaluate(self, x: Any) -> np.ndarray:
        """The tree's values at points, on all of them at once in Rust, without the GIL: ``x``
        holds a point per row, a value per variable (or input) in the order of the set's
        terminals, or for one variable a 1-D array of its values. A ``float64`` array for
        regression's primitives, computed as ``Regression`` computes them (before linear
        scaling); a ``bool`` array for the Boolean problems' (an input is true where it isn't
        0). A ``ValueError`` for fewer columns than variables."""
    @overload
    def evaluate(
        self, x: Mapping[str, Any] | Any, functions: Mapping[str, Callable[..., Any]]
    ) -> Any:
        """The value of a tree of your own primitives (``gx.gp.PrimitiveSetBuilder``), with
        what they mean: ``functions`` maps each function's name to a callable, called with its
        children's values, in order, and returning the node's value; ``x`` gives the terminals'
        values, a mapping by name, or an array: a column per terminal, in the order of the
        set's terminals (a 1-D array for one terminal). A constant's value is a ``float``.

        Each function is called once per node, its children before it (bottom-up, as Rust's
        ``Tree::evaluate``: the prefix order read backwards), on whatever the values are. With
        numpy columns and numpy functions (``{"add": np.add, "if": np.where, ...}``), a tree is
        evaluated on all its points with one call per node, and constants broadcast; with
        numbers and plain functions, at one point. Returns the root's value. A ``ValueError``
        for a function or terminal missing, or a tree of genoxide's built-in primitives; an
        exception of a function propagates."""
    def nodes(self) -> tuple[Node, ...]:
        """The nodes in prefix order, each function followed by its children's subtrees, in
        order: for your own interpreter, e.g. recursive, a node's children starting right after
        it and each taking its subtree's nodes."""
    def __eq__(self, other: object) -> bool: ...
    def __hash__(self) -> int: ...
    @staticmethod
    def _from_json(set: str, tree: str) -> Tree: ...
    def _json(self) -> str: ...

class Node:
    """A node of a tree, from ``Tree.nodes()``."""

    @property
    def kind(self) -> Literal["function", "terminal", "constant"]:
        """What the node is."""
    @property
    def name(self) -> str:
        """The primitive's name, or a constant's value as the tree's text writes it."""
    @property
    def arity(self) -> int:
        """The number of children: 0 for a terminal or a constant."""
    @property
    def type(self) -> str:
        """The name of the node's type, the one its value has."""
    @property
    def value(self) -> float | None:
        """A constant's value; None for a function or terminal."""

def user_primitives(
    types: list[str],
    functions: list[tuple[str, list[int], int]],
    constants: list[tuple[int, str]],
    root: int,
) -> PrimitiveSet: ...
def gp_check(description: str) -> None: ...
def gp_ramped_half_and_half(description: str, count: int, seed: int) -> list[Tree]: ...
def gp_random_genome(description: str, seed: int) -> Tree: ...
def gp_validate(description: str, tree: Tree) -> None: ...
def gp_parse(description: str, text: str) -> Tree: ...
def regression_primitives(
    functions: list[str], variables: list[str], constants: str | None = None
) -> PrimitiveSet: ...

class Sample:
    """Points and their targets, for symbolic regression: ``Sample(x, y)``, ``x`` a point per row
    with a value per variable (or a 1-D array of one variable), ``y`` a target per point; all
    finite. A ``ValueError`` for no points or variables, more than 2^16 variables, a number of
    targets other than the points', or a value that isn't finite."""

    def __init__(self, x: Any, y: Any) -> None: ...
    @property
    def x(self) -> np.ndarray:
        """The points, a row each with a value per variable."""
    @property
    def y(self) -> np.ndarray:
        """The targets."""
    @property
    def variables(self) -> int:
        """The number of variables."""
    @property
    def points(self) -> int:
        """The number of points."""
    def deviation(self) -> float:
        """The standard deviation of the targets (divided by the number of points): the scale of
        an error, exact recovery being an error of at most 1e-10 of it."""
    def __len__(self) -> int:
        """The number of points."""

class Dataset:
    """The data of a regression: the ``training`` sample that the search fits, and a ``test``
    sample it never sees, to measure how the result generalizes. A ``ValueError`` for a test
    sample of other variables."""

    def __init__(self, training: Sample, test: Sample | None = None) -> None: ...
    @property
    def training(self) -> Sample:
        """The training sample."""
    @property
    def test(self) -> Sample | None:
        """The test sample, or None."""
    @property
    def variables(self) -> int:
        """The number of variables."""

class Regression:
    """The fitness function of symbolic regression, evaluated in Rust: the error of a tree's
    predictions on the training sample of ``dataset``, minimized; None (an invalid tree) if a
    value isn't finite.

    ``primitives`` is a set of ``gx.gp.regression``'s functions, whose variables are the
    dataset's columns. ``metric`` is "rmse" (the root mean squared error, the default), "mse" or
    "mae"; with ``linear_scaling`` (the default), the error of ``a + b f(x)``, with ``a`` and
    ``b`` fitted by least squares on the training sample. A tree is evaluated on all the points
    at once, without the GIL. ``regression(tree)`` is ``regression.evaluate(tree)``. Each method
    taking a tree raises a ``ValueError`` for a tree of another primitive set."""

    def __init__(
        self,
        primitives: PrimitiveSet,
        dataset: Dataset,
        *,
        metric: Literal["rmse", "mse", "mae"] = "rmse",
        linear_scaling: bool = True,
    ) -> None: ...
    def evaluate(self, tree: Tree) -> float | None:
        """The error on the training sample, None if a value isn't finite."""
    def __call__(self, tree: Tree) -> float | None: ...
    def error(self, tree: Tree, sample: Sample) -> float | None:
        """The error on ``sample``, e.g. the test sample, after the scaling fitted on the
        training sample; None if a prediction isn't finite."""
    def values(self, tree: Tree, sample: Sample) -> np.ndarray:
        """The tree's values at the points of ``sample``, before scaling."""
    def predict(self, tree: Tree, sample: Sample) -> np.ndarray | None:
        """The tree's predictions at the points of ``sample``: its values after the scaling
        fitted on the training sample; None if a value on the training sample isn't finite."""
    def scaling(self, tree: Tree) -> tuple[float, float] | None:
        """The linear scaling ``(intercept, slope)`` fitted on the training sample, ``(0.0,
        1.0)`` without linear scaling, None if a value on the training sample isn't finite."""
    def display(self, tree: Tree) -> str:
        """The tree as text, with its scaling if linear scaling is on: ``intercept + slope *
        (expression)``."""
    @property
    def primitives(self) -> PrimitiveSet:
        """The primitive set."""
    @property
    def dataset(self) -> Dataset:
        """The dataset."""
    @property
    def metric(self) -> str:
        """The error measure: "rmse", "mse" or "mae"."""
    @property
    def linear_scaling(self) -> bool:
        """Whether the error is the one after linear scaling."""

class RegressionProblem:
    """A test problem of symbolic regression, evaluated in Rust: the fitness function
    ``regression()``, the RMSE after linear scaling, on the paper's data with its primitives."""

    def __new__(cls, name: str) -> Self: ...
    @property
    def name(self) -> str:
        """The problem's name, e.g. "Koza-1"."""
    @property
    def formula(self) -> str:
        """The target as a formula, e.g. "x^4 + x^3 + x^2 + x"."""
    @property
    def reference(self) -> str:
        """The paper that defines the problem."""
    @property
    def reference_url(self) -> str | None:
        """Its DOI or URL, or None."""
    def target(self, point: Sequence[float]) -> float:
        """The target at ``point``, a value per variable."""
    def primitives(self) -> PrimitiveSet:
        """The paper's primitive set: build the ``gx.gp.Gp`` from it."""
    def dataset(self) -> Dataset:
        """The training and test samples."""
    def regression(
        self, *, metric: Literal["rmse", "mse", "mae"] = "rmse", linear_scaling: bool = True
    ) -> Regression:
        """The fitness function, with these settings: the RMSE after linear scaling by default,
        what the problem itself evaluates."""
    def __call__(self, tree: Tree) -> float | None:
        """The RMSE after linear scaling, None if a value isn't finite."""

class BooleanProblem:
    """A Boolean problem of ``gx.gp.boolean``, evaluated in Rust: the cases of its truth table a
    tree gets wrong, minimized."""

    def __new__(cls, kind: str, size: int) -> Self: ...
    def primitives(self) -> PrimitiveSet:
        """The paper's primitive set: build the ``gx.gp.Gp`` from it."""
    @property
    def inputs(self) -> int:
        """The number of inputs."""
    @property
    def cases(self) -> int:
        """The number of cases of the truth table, 2^inputs."""
    @property
    def address_bits(self) -> int | None:
        """The multiplexer's address bits; None for even parity."""
    @property
    def reference(self) -> str:
        """The source of the problem."""
    def targets(self) -> list[int]:
        """The right outputs, 64 cases per word: case ``c`` at bit ``c % 64`` of word
        ``c // 64``."""
    def outputs(self, tree: Tree) -> list[int]:
        """The tree's outputs, as ``targets()`` lays them out."""
    def errors(self, tree: Tree) -> int:
        """The number of cases the tree gets wrong: the fitness, 0 at the optimum."""
    def __call__(self, tree: Tree) -> float:
        """The number of cases the tree gets wrong."""

class WithSize:
    """A fitness of trees with their size as a second objective, for ``gx.Nsga2`` with
    ``objectives=["minimize", "minimize"]``: ``(fitness, number of nodes)``, both minimized,
    evaluated in Rust. ``fitness`` is a ``gx.gp.regression.Regression``, a regression problem
    or a Boolean problem: the trade-off between accuracy and size, as a Pareto front."""

    def __init__(self, fitness: Regression | RegressionProblem | BooleanProblem) -> None: ...
    @property
    def fitness(self) -> Any:
        """The fitness of the first objective."""
    def __call__(self, tree: Tree) -> tuple[float, float] | None:
        """``(fitness, size)``, or None for an invalid tree."""
