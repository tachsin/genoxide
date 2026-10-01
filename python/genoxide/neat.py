"""NEAT's networks: the genomes of a :class:`genoxide.Neat` run, in Rust.

A :class:`Network` is node genes (the inputs, a bias input fixed at 1, the outputs and the hidden
nodes, by id) and connection genes (by innovation number, each enabled or not). A NEAT run's
fitness function gets one, and its result's ``best_genome`` is one. It's read-only:
``inputs``, ``outputs``, ``hidden()``, ``enabled()``, ``nodes()``, ``connections()`` and
``len(network)``, the connection genes; networks compare equal by their genes, hash, and pickle.

``feed_forward()`` compiles it into a :class:`FeedForward` evaluator (a ``ValueError`` if its
enabled connections form a cycle, possible only with ``Neat(feed_forward=False)``), and
``recurrent()`` into a :class:`Recurrent` one, a step of time per ``activate``. Their
``activate(input)`` gives the outputs as a numpy array, computed in Rust, the same bits as a Rust
program's::

    import genoxide as gx

    network = gx.neat.Network.fully_connected(2, 1, [0.5, -0.5, 0.0])
    evaluator = network.feed_forward()
    evaluator.activate([1.0, 1.0])  # array([0.5]): the steepened sigmoid of 0

Both evaluators are policies of the control tasks of :mod:`genoxide.problems.control`, which run
them in Rust without the GIL, from a reset network each episode. Their outputs are the actions;
NEAT's sigmoid outputs, in (0, 1), only push one way, so ``policy(scale=2.0, offset=-1.0)`` gives
a policy whose actions are ``2 output - 1``, in (-1, 1)::

    from genoxide.problems.control import SUCCESS_STEPS, CartPole

    task = CartPole()

    def steps(network):
        policy = network.feed_forward().policy(scale=2.0, offset=-1.0)
        return task.run(policy, SUCCESS_STEPS)

    result = gx.Neat(4, 1, seed=1).run(steps, target=SUCCESS_STEPS, evaluations=100_000)
"""

from __future__ import annotations

from dataclasses import dataclass

from . import _genoxide

__all__ = ["Network", "NodeGene", "ConnectionGene", "FeedForward", "Recurrent", "Species"]

Network = _genoxide.NeatNetwork
NodeGene = _genoxide.NodeGene
ConnectionGene = _genoxide.ConnectionGene
FeedForward = _genoxide.FeedForward
Recurrent = _genoxide.Recurrent


@dataclass(frozen=True)
class Species:
    """A species of a running :class:`genoxide.Neat`, from :attr:`genoxide.RunningNeat.species`."""

    id: int
    """Its id: species are numbered in the order they appear."""
    members: tuple[int, ...]
    """Its members: their positions in the population, as in ``progress.population``."""
    best_fitness: float | None
    """The best score its members ever had, or None."""
    improved: int
    """The generation of its last improvement."""
    created: int
    """The generation it appeared in."""
    representative: Network
    """The network new networks are compared with: a member of the previous generation."""
