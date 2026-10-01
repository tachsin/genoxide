"""Double pole balancing without velocities: evolve the weights of a recurrent neural network that
balances two poles on a cart seeing only their angles and the cart's position, by CMA-ES.

Wieland's (1991) double pole, with Florian's (2007) corrected equations and Gomez, Schmidhuber and
Miikkulainen's (2008) settings, made non-Markovian by Gruau, Whitley and Pyeatt (1996): the
network doesn't see the velocities, and has to infer them from what it saw before. It is an Elman
network: 3 inputs, 3 hidden tanh units that also receive their own previous outputs, and a tanh
output, without biases, 21 weights. The fitness is Gruau et al.'s damping fitness over 1000 steps,
maximized by CMA-ES with its step size bounded below, as Igel (2003) did. The task is solved, by
Gruau et al.'s criteria, when the best network of a generation balances the poles for 100,000
steps and, from 625 other starts, for 1000 steps from at least 200. Then the same task by NEAT.

The task, the networks and the fitness run in Rust (``gx.problems.control``, ``gx.nn``,
``gx.neat``), so the runs are the Rust example's, to the bit, on every platform.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/double_pole_no_velocities/main.py
"""

import numpy as np

import genoxide as gx
from genoxide.problems.control import (
    GENERALIZATION_THRESHOLD,
    SUCCESS_STEPS,
    Balance,
    DoublePole,
)

from trace import Trace, degrees

# 3 inputs, 3 hidden units with a context, 1 output, without biases
elman = gx.nn.Elman(3, 3, 1, "tanh", output_activation="tanh", bias=False)
task = DoublePole(velocities=False)


def extent(weights, steps):
    """The largest |x|, |θ₁| and |θ₂| (in degrees) over an episode of ``steps`` steps."""
    states = np.abs(task.episode(elman.policy(weights), steps))
    return (
        float(np.max(states[:, 0])),
        float(np.max(degrees(states[:, 2]))),
        float(np.max(degrees(states[:, 4]))),
    )


def generation_best(progress):
    """The position of the generation's best network, the first of equals, if it balanced the
    1000 steps of the damping fitness (then its fitness is at least 0.1), or None."""
    scores = progress.scores
    if np.all(np.isnan(scores)):
        return None
    best = int(np.nanargmax(scores))
    return best if scores[best] >= 0.1 else None


def tests(policy):
    """Gruau et al.'s tests: the starts of the generalization test that ``policy`` passed, if it
    balances the poles for 100,000 steps from the start and passes the test, or None."""
    if task.run(policy, SUCCESS_STEPS) < SUCCESS_STEPS:
        return None
    generalization = task.generalization(policy)
    return generalization if generalization >= GENERALIZATION_THRESHOLD else None


# Gruau et al.'s damping fitness over 1000 steps, evaluated in Rust
cmaes = gx.Cmaes(
    elman.representation((-1.0, 1.0)),
    min_step=0.05,
    restarts="ipop",
    objective="maximize",
    seed=1,
)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(elman, task)
solution = None


def on_generation(progress):
    """Records the generation, and runs the tests of the generation's best network: the run
    stops once it passes them."""
    global solution
    trace.record(progress)
    best = generation_best(progress)
    if best is None:
        return True
    weights = progress.population[best]
    generalization = tests(elman.policy(weights))
    if generalization is None:
        return True
    solution = (weights, generalization, progress.evaluations, progress.generation)
    return False


result = cmaes.run(
    Balance(task, elman, fitness="damping"), evaluations=100_000, on_generation=on_generation
)

if solution is None:
    print(f"not solved after {result.evaluations} evaluations")
else:
    weights, generalization, evaluations, generation = solution
    print(f"solved after {evaluations} evaluations in {generation} generations")
    print(
        f"balanced for {SUCCESS_STEPS} steps from the start, and for 1000 steps from "
        f"{generalization} of the 625 generalization starts (at least {GENERALIZATION_THRESHOLD} "
        "needed)"
    )
    print(f"damping fitness {task.damping_fitness(elman.policy(weights)):.6f}")
    position, long, short = extent(weights, SUCCESS_STEPS)
    print(
        f"over the 100000 steps: the cart within {position:.4f} m, the poles within {long:.4f}° "
        f"and {short:.4f}°"
    )
    print("weights: [" + ", ".join(f"{w:.3f}" for w in weights) + "]")
    trace.write(weights)

    # the same task by NEAT, with the paper's settings for it: 1000 recurrent networks that grow
    # from the three inputs and a bias connected to the output, c3 = 3 and a threshold of 4, and
    # new connections with probability 0.3; the damping fitness, and the same success criteria. A
    # recurrent network as a controller: its output, in (0, 1), is the force as 2 × output − 1,
    # and its state is cleared at the start of each episode
    def controller(network):
        return network.recurrent().policy(scale=2.0, offset=-1.0)

    def damping(network):
        return task.damping_fitness(controller(network))

    neat = gx.Neat(
        3,
        1,
        population_size=1000,
        compatibility=(1.0, 1.0, 3.0, 4.0),
        structural_mutation=(0.03, 0.3),
        feed_forward=False,
        seed=1,
    )
    neat_solution = None

    def neat_generation(progress):
        """Runs the tests of the generation's best network: the run stops once it passes them."""
        global neat_solution
        best = generation_best(progress)
        if best is None:
            return True
        network = progress.population[best]
        generalization = tests(controller(network))
        if generalization is None:
            return True
        neat_solution = (network, generalization, progress.evaluations, progress.generation)
        return False

    result = neat.run(damping, evaluations=400_000, on_generation=neat_generation)
    if neat_solution is None:
        print(f"\nNEAT: not solved after {result.evaluations} evaluations")
    else:
        network, generalization, evaluations, generation = neat_solution
        print(f"\nNEAT: solved after {evaluations} evaluations in {generation} generations")
        print(
            f"balanced for {SUCCESS_STEPS} steps from the start, and for 1000 steps from "
            f"{generalization} of the 625 generalization starts"
        )
        print(
            f"the network: {network.hidden()} hidden nodes, {network.enabled()} enabled "
            "connections"
        )
