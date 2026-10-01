"""Control tasks: poles to balance on a cart, driven by a policy such as a network of
:mod:`genoxide.nn`, for neuroevolution. Simulated in Rust.

At each step of 0.02 s, the policy observes the system's state, scaled to about [-1, 1], and
chooses an action in [-1, 1], a force of up to 10 N on the cart; the episode ends when a pole has
fallen or the cart has left the track. The equations are Florian's (2007) corrected ones, with the
settings of Gomez, Schmidhuber and Miikkulainen (2008):

- :class:`CartPole`: one pole, failing beyond 12°; the policy observes ``x``, ``x'``, ``θ`` and
  ``θ'``.
- :class:`DoublePole`: a long and a short pole side by side, failing beyond 36°; the policy
  observes the six variables, or with ``velocities=False`` only ``x``, ``θ₁`` and ``θ₂``, which
  needs a recurrent network such as :class:`genoxide.nn.Elman`.

A task is solved by balancing for :data:`SUCCESS_STEPS` steps, and without velocities also by
passing the generalization test. A policy is a network's ``policy(weights)``, which the task runs
in Rust without the GIL, or any Python callable ``policy(observation, action)`` that writes its
action into ``action[0]``, a numpy array, which is slow: a Python call per step.

:class:`Balance` is the fitness of a network's weights on a task, evaluated in Rust: ``run``
takes it like the test problems of :mod:`genoxide.problems`, with no Python call::

    import genoxide as gx
    from genoxide.problems.control import SUCCESS_STEPS, Balance, CartPole

    # a 4-2-1 network without biases balances the pole for 100,000 steps
    mlp = gx.nn.Mlp([4, 2, 1], "tanh", output_activation="tanh", bias=False)
    task = CartPole()
    cmaes = gx.Cmaes(mlp.representation((-1, 1)), seed=1)
    result = cmaes.run(Balance(task, mlp), target=SUCCESS_STEPS, evaluations=10_000)
    print(task.run(mlp.policy(result.best_genome), SUCCESS_STEPS))

The same run in Rust gives the same result, to the bit, on every platform.

References: Barto, Sutton and Anderson (1983), the cart-pole system; Wieland (1991), poles side
by side; Gruau, Whitley and Pyeatt (1996), the double pole without velocities, its damping
fitness and generalization test; Igel (2003); Florian, R. V. (2007), Correct equations for the
dynamics of the cart-pole system; Gomez, F., Schmidhuber, J. and Miikkulainen, R. (2008),
Accelerated neural evolution through cooperatively coevolved synapses, Journal of Machine
Learning Research 9: 937-965.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from functools import cached_property
from typing import Any, ClassVar, Literal, Union

import numpy as np

from .. import _genoxide, _whole
from ..nn import Elman, Mlp, Policy

__all__ = [
    "SUCCESS_STEPS",
    "DAMPING_STEPS",
    "GENERALIZATION_THRESHOLD",
    "CartPole",
    "DoublePole",
    "Balance",
]

SUCCESS_STEPS: int = _genoxide.SUCCESS_STEPS
"""Steps balanced to solve a task: 100,000, of 0.02 s each."""

DAMPING_STEPS: int = _genoxide.DAMPING_STEPS
"""The steps of an episode of :meth:`DoublePole.damping_fitness`, and of each start of
:meth:`DoublePole.generalization`: 1000."""

GENERALIZATION_THRESHOLD: int = _genoxide.GENERALIZATION_THRESHOLD
"""The starts of :meth:`DoublePole.generalization` that a policy must balance for
:data:`DAMPING_STEPS` steps: 200 of the 625."""

PolicyLike = Union[Policy, Callable[[np.ndarray, np.ndarray], Any]]
"""A network's policy, or a Python callable ``policy(observation, action)``."""


def _steps(name: str, steps: Any) -> int:
    return _whole(name, steps, maximum=2**32 - 1)


class _Task:
    """What every task has: its description, from which Rust builds it."""

    _type: ClassVar[str]

    def _describe(self) -> dict[str, Any]:
        return {"type": self._type}

    @cached_property
    def _native(self) -> _genoxide.Task:
        return _genoxide.Task(json.dumps(self._describe()))

    @property
    def observations(self) -> int:
        """The number of variables a policy observes: a network's inputs."""
        return int(self._native.observations)

    def run(self, policy: PolicyLike, steps: int) -> int:
        """Runs an episode of at most ``steps`` steps from the initial state, after resetting the
        policy (a recurrent network's context): the steps before a pole fell or the cart left the
        track.

        ``policy`` is a network's ``policy(weights)`` with an input per observation and one
        output, run in Rust without the GIL, or a Python callable ``policy(observation,
        action)``, called each step with the observation, a numpy array, to write its action into
        ``action[0]``. An exception in the callable ends the episode, and ``run`` raises it.
        """
        return int(self._native.run(policy, _steps("steps", steps)))

    def solved(self, policy: PolicyLike) -> bool:
        """Whether ``policy`` solves the task: balances it for :data:`SUCCESS_STEPS` steps (and,
        for ``DoublePole(velocities=False)``, passes the generalization test)."""
        return bool(self._native.solved(policy))


@dataclass(frozen=True)
class CartPole(_Task):
    """The cart-pole system of Barto, Sutton and Anderson (1983): one pole of 0.1 kg and a
    half-length of 0.5 m on a cart of 1 kg, on a track from -2.4 m to 2.4 m, balanced by pushing
    the cart. It starts with the pole at 4°, and fails beyond 12° or off the track. The policy
    observes ``x / 2.4``, ``x' / 2``, ``θ / 12°`` and ``θ' / 2``, and outputs one action."""

    _type: ClassVar[str] = "cart_pole"


@dataclass(frozen=True)
class DoublePole(_Task):
    """Two poles side by side on a cart (Wieland 1991), of 0.1 kg and 0.01 kg and half-lengths of
    0.5 m and 0.05 m, balanced together by pushing the cart. It starts with the long pole at 4°,
    and fails beyond 36° or off the track.

    With ``velocities`` (the default), the policy observes the six variables, scaled: ``x``,
    ``x'``, ``θ₁``, ``θ₁'``, ``θ₂``, ``θ₂'``; without, only ``x``, ``θ₁`` and ``θ₂``, and it must
    compute the velocities: a recurrent network such as :class:`genoxide.nn.Elman`. Without
    velocities, :meth:`damping_fitness` is the usual fitness, and a solution must also pass
    :meth:`generalization` (Gruau, Whitley and Pyeatt 1996).
    """

    _type: ClassVar[str] = "double_pole"
    velocities: bool = True

    def _describe(self) -> dict[str, Any]:
        if not isinstance(self.velocities, (bool, np.bool_)):
            raise ValueError(f"velocities is True or False, not {self.velocities!r}")
        return {"type": self._type, "velocities": bool(self.velocities)}

    def damping_fitness(self, policy: PolicyLike) -> float:
        """Gruau, Whitley and Pyeatt's (1996) fitness, to maximize, of an episode of
        :data:`DAMPING_STEPS` steps: ``0.1 f1 + 0.9 f2``, where ``f1 = t / 1000`` for the ``t``
        steps balanced, and ``f2 = 0.75 / sum(|x| + |x'| + |θ₁| + |θ₁'|)`` over the states after
        the last 100 of them, or 0 for ``t < 100``. It rewards a policy that brings the cart and
        the long pole to rest, over one that keeps them up by jiggling the cart."""
        return float(self._native.damping_fitness(policy))

    def generalization(self, policy: PolicyLike) -> int:
        """The generalization test of Gruau, Whitley and Pyeatt (1996): of 625 starts spread over
        the state space, the number from which ``policy`` balances the poles for
        :data:`DAMPING_STEPS` steps. A policy passes with at least
        :data:`GENERALIZATION_THRESHOLD`, 200."""
        return int(self._native.generalization(policy))


@dataclass(frozen=True)
class Balance:
    """The fitness of a network's weights on a task, to maximize, evaluated in Rust.

    ``run`` takes it as its fitness, like a test problem of :mod:`genoxide.problems`, with no
    Python call per genome (``parallel=True`` uses every core); the genome must be a
    :class:`genoxide.Real` of ``network.parameters`` genes, e.g. ``network.representation((-1,
    1))``, and the objective "maximize". Calling ``balance(weights)`` or
    ``balance.evaluate(genomes)`` runs the same Rust code.

    Parameters
    ----------
    task : CartPole or DoublePole
        The task.
    network : genoxide.nn.Mlp or genoxide.nn.Elman
        The network, with an input per observation of the task and one output.
    fitness : {"steps", "damping"}, default "steps"
        "steps": the steps balanced in an episode of at most ``steps`` steps. "damping":
        :meth:`DoublePole.damping_fitness`, for the double pole only, whose episodes are
        :data:`DAMPING_STEPS` long.
    steps : int, default SUCCESS_STEPS
        The longest episode of the "steps" fitness, 0 to 2^32 - 1.
    """

    task: CartPole | DoublePole
    network: Mlp | Elman
    _: KW_ONLY
    fitness: Literal["steps", "damping"] = "steps"
    steps: int = SUCCESS_STEPS

    objective: ClassVar[str] = "maximize"
    """The objective of a run: "maximize"."""

    def _describe(self) -> dict[str, Any]:
        if not isinstance(self.task, (CartPole, DoublePole)):
            raise ValueError(f"task is a CartPole() or a DoublePole(), not {self.task!r}")
        if not isinstance(self.network, (Mlp, Elman)):
            raise ValueError(f"network is a gx.nn.Mlp or a gx.nn.Elman, not {self.network!r}")
        if self.fitness not in ("steps", "damping"):
            raise ValueError(f'fitness is "steps" or "damping", not {self.fitness!r}')
        return {
            "type": "balance",
            "task": self.task._describe(),
            "network": self.network._describe(),
            "fitness": self.fitness,
            "steps": _steps("steps", self.steps),
        }

    def _json(self) -> str:
        return json.dumps(self._describe())

    @property
    def parameters(self) -> int:
        """The number of genes: the network's weights."""
        return self.network.parameters

    def evaluate(self, genomes: Any) -> np.ndarray:
        """The fitness of each row of ``genomes``, a 2-D array of the network's weights, in Rust
        without the GIL."""
        rows = np.ascontiguousarray(genomes, dtype=np.float64)
        if rows.ndim != 2:
            raise ValueError(f"genomes is a 2-D array, a genome per row, not of shape {rows.shape}")
        return np.asarray(_genoxide.balance_evaluate(self._json(), rows))

    def __call__(self, weights: Any) -> float:
        """The fitness of ``weights``, a genome."""
        row = np.ascontiguousarray(weights, dtype=np.float64)
        if row.ndim != 1:
            raise ValueError(f"weights is a 1-D array, not of shape {row.shape}")
        return float(self.evaluate(row.reshape(1, -1))[0])
