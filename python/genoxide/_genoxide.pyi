import os
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np

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
) -> dict[str, Any]: ...

class Running:
    """The running algorithm, for a control: valid during the control's call only."""

    def get(self, name: str) -> str: ...
    def set(self, name: str, value: str) -> None: ...
    def reevaluate(self) -> None: ...

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
