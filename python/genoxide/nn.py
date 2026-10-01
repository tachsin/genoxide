"""Neural networks whose weights a genome holds, computed in Rust: for neuroevolution.

A network of fixed structure, :class:`Mlp` (a multilayer perceptron) or :class:`Elman` (with a
recurrent hidden layer), has ``parameters`` weights; its ``representation(bounds)`` is the
:class:`genoxide.Real` genome of them, and ``forward(weights, inputs)`` computes its outputs, in
Rust without the GIL. ``policy(weights)`` drives the pole-balancing tasks of
:mod:`genoxide.problems.control`, whose :class:`~genoxide.problems.control.Balance` scores the
weights in Rust, with no Python call::

    import numpy as np
    import genoxide as gx

    # XOR by a 2-2-1 network, its 9 weights found by CMA-ES
    mlp = gx.nn.Mlp([2, 2, 1], "sigmoid", output_activation="sigmoid")
    inputs = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    targets = np.array([0.0, 1.0, 1.0, 0.0])

    def squared_error(weights):
        return float(np.sum((mlp.forward(weights, inputs)[:, 0] - targets) ** 2))

    cmaes = gx.Cmaes(mlp.representation((-10, 10)), restarts="bipop", objective="minimize", seed=1)
    result = cmaes.run(squared_error, target=0.01, evaluations=100_000)
    print(mlp.forward(result.best_genome, inputs)[:, 0].round(2))

The weights come unit by unit: each unit's weights for its inputs in order, then its bias. The
activations are "identity", "tanh", "sigmoid" (the logistic function), "relu" and "steep_sigmoid"
(NEAT's ``1 / (1 + exp(-4.9 x))``), computed with genoxide's portable math (:mod:`genoxide.math`):
a network's outputs are the same bits on every platform, and as a Rust program's.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import KW_ONLY, dataclass
from functools import cached_property
from typing import Any, Literal

import numpy as np

from . import Real, _genoxide, _whole

__all__ = ["Mlp", "Elman", "Policy", "Activation"]

Activation = Literal["identity", "tanh", "sigmoid", "relu", "steep_sigmoid"]
_ACTIVATIONS = ("identity", "tanh", "sigmoid", "relu", "steep_sigmoid")

Policy = _genoxide.Policy
"""A network with its weights, from ``policy(weights)``: a policy of the control tasks of
:mod:`genoxide.problems.control` that runs in Rust."""


def _activation(name: str, value: Any) -> str:
    if value not in _ACTIVATIONS:
        names = ", ".join(f'"{activation}"' for activation in _ACTIVATIONS)
        raise ValueError(f"{name} is one of {names}, not {value!r}")
    return str(value)


class _Network:
    """What every network has: its description, from which Rust builds it."""

    def _describe(self) -> dict[str, Any]:
        raise NotImplementedError

    @cached_property
    def _native(self) -> _genoxide.Network:
        return _genoxide.Network(json.dumps(self._describe()))

    @property
    def parameters(self) -> int:
        """The number of weights (and biases): the length of a genome."""
        return int(self._native.parameters)

    def representation(self, bounds: tuple[float, float]) -> Real:
        """The genome of the weights: a :class:`genoxide.Real` of ``parameters`` genes, each
        within ``bounds``, ``(low, high)``."""
        return Real(bounds, length=self.parameters)

    def forward(self, weights: Any, inputs: Any) -> np.ndarray:
        """The outputs for ``inputs``, with ``weights``: for a 1-D array of ``inputs`` values, a
        1-D array of ``outputs`` values; for a 2-D array, an input per row, a row of outputs
        each. Computed in Rust, without the GIL: other threads run meanwhile, e.g. a parallel
        run's calls.

        Raises a ``ValueError`` for weights other than ``parameters`` of them, or inputs of
        another length.
        """
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        if weights.ndim != 1:
            raise ValueError(f"weights is a 1-D array, not of shape {weights.shape}")
        rows = np.ascontiguousarray(inputs, dtype=np.float64)
        if rows.ndim == 1:
            outputs: np.ndarray = self._native.forward(weights, rows.reshape(1, -1))[0]
            return outputs
        if rows.ndim != 2:
            raise ValueError(f"inputs is a 1-D or a 2-D array, not of shape {rows.shape}")
        return np.asarray(self._native.forward(weights, rows))

    def policy(self, weights: Any) -> Policy:
        """The network with ``weights``, as a policy of the control tasks of
        :mod:`genoxide.problems.control`: they run it in Rust, without the GIL. Raises a
        ``ValueError`` for weights other than ``parameters`` of them."""
        weights = np.ascontiguousarray(weights, dtype=np.float64)
        if weights.ndim != 1:
            raise ValueError(f"weights is a 1-D array, not of shape {weights.shape}")
        return self._native.policy(weights)


@dataclass(frozen=True, eq=True)
class Mlp(_Network):
    """A multilayer perceptron: layers of units, each fed by every unit of the layer before.

    Unit ``j`` of a layer outputs ``f(sum_i w_ji x_i + b_j)`` of the previous layer's outputs
    ``x``, with ``f`` the ``activation`` in the hidden layers and ``output_activation`` in the
    output layer. A network of layers ``n_0, ..., n_L`` has ``sum n_l (n_(l-1) + 1)`` weights, or
    ``sum n_l n_(l-1)`` without biases.

    Parameters
    ----------
    layers : sequence of int
        The layer sizes from the inputs to the outputs, e.g. ``[4, 16, 1]``: at least two, each
        1 to 2^24 units, and at most 2^24 weights.
    activation : str, default "tanh"
        The hidden layers' activation: "identity", "tanh", "sigmoid", "relu" or
        "steep_sigmoid".
    output_activation : str, optional
        The output layer's activation; "identity" (linear outputs) by default. "tanh" gives
        outputs in (-1, 1), e.g. a force for :mod:`genoxide.problems.control`.
    bias : bool, default True
        Whether the units have biases. Without, a network of odd activations (tanh, the
        identity) is an odd function of its inputs, which suits a symmetric task such as
        balancing poles (Igel 2003).
    """

    layers: Sequence[int]
    activation: Activation = "tanh"
    _: KW_ONLY
    output_activation: Activation | None = None
    bias: bool = True

    def __post_init__(self) -> None:
        # hashable and fixed
        object.__setattr__(self, "layers", tuple(self.layers))

    @property
    def inputs(self) -> int:
        """The number of inputs: the first layer's size."""
        return int(self.layers[0])

    @property
    def outputs(self) -> int:
        """The number of outputs: the last layer's size."""
        return int(self.layers[-1])

    def _describe(self) -> dict[str, Any]:
        if not isinstance(self.bias, (bool, np.bool_)):
            raise ValueError(f"bias is True or False, not {self.bias!r}")
        return {
            "type": "mlp",
            "layers": [_whole("layers", units, plural=True) for units in self.layers],
            "activation": _activation("activation", self.activation),
            "output_activation": _activation(
                "output_activation", self.output_activation or "identity"
            ),
            "bias": bool(self.bias),
        }


@dataclass(frozen=True, eq=True)
class Elman(_Network):
    """An Elman network (Elman 1990): a hidden layer that receives the inputs and its own outputs
    of the previous step (the context), and an output layer fed by the hidden layer. Its memory
    lets a policy compute what it doesn't observe, such as the velocities of
    ``DoublePole(velocities=False)``.

    Hidden unit ``j`` outputs ``f(sum_i w_ji x_i + sum_k u_jk h_k(t - 1) + b_j)``, output unit
    ``o`` outputs ``g(sum_j v_oj h_j(t) + c_o)``. The weights come unit by unit: each hidden
    unit's for the inputs, then for the context, then its bias; then each output unit's for the
    hidden units, then its bias: ``h (n + h + 1) + m (h + 1)`` weights for ``n`` inputs, ``h``
    hidden and ``m`` output units, or ``h (n + h) + m h`` without biases.

    The context starts at 0. ``forward`` takes the rows of its inputs as the steps of one
    sequence, from a context of 0; a :class:`Policy` starts each episode from a context of 0.

    Parameters
    ----------
    inputs, hidden, outputs : int
        The numbers of inputs, hidden (and context) units and outputs, each 1 to 2^24, and at
        most 2^24 weights.
    activation : str, default "tanh"
        The hidden units' activation.
    output_activation : str, optional
        The output units' activation; "identity" by default.
    bias : bool, default True
        Whether the units have biases.
    """

    inputs: int
    hidden: int
    outputs: int
    activation: Activation = "tanh"
    _: KW_ONLY
    output_activation: Activation | None = None
    bias: bool = True

    def _describe(self) -> dict[str, Any]:
        if not isinstance(self.bias, (bool, np.bool_)):
            raise ValueError(f"bias is True or False, not {self.bias!r}")
        return {
            "type": "elman",
            "inputs": _whole("inputs", self.inputs),
            "hidden": _whole("hidden", self.hidden),
            "outputs": _whole("outputs", self.outputs),
            "activation": _activation("activation", self.activation),
            "output_activation": _activation(
                "output_activation", self.output_activation or "identity"
            ),
            "bias": bool(self.bias),
        }
