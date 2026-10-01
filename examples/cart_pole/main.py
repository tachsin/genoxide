"""Cart-pole: evolve the weights of a neural network that balances a pole on a cart for 100,000
steps, by CMA-ES.

The classic control task (Barto, Sutton and Anderson 1983), with Florian's (2007) corrected
equations and Gomez, Schmidhuber and Miikkulainen's (2008) settings: a 1 kg cart on a 4.8 m track,
a pole of 1 m and 0.1 kg starting at 4° from vertical, and a force of up to 10 N every 0.02 s. The
network sees the cart's position and velocity and the pole's angle and angular velocity, and
outputs the force: 4 inputs, 8 hidden tanh units and a tanh output, without biases, 40 weights.
The fitness is the number of steps before the pole passes 12° or the cart leaves the track,
maximized by CMA-ES until a network balances it for 100,000 steps, over 33 minutes of simulated
time. Then the same task by NEAT.

The task, the networks and the fitness run in Rust (``gx.problems.control``, ``gx.nn``,
``gx.neat``), so the runs are the Rust example's, to the bit, on every platform.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/cart_pole/main.py
"""

import numpy as np

import genoxide as gx
from genoxide.problems.control import SUCCESS_STEPS, Balance, CartPole

from trace import Trace, degrees

# 4 inputs, 8 hidden units, 1 output, without biases: Igel's (2003) network for this task
mlp = gx.nn.Mlp([4, 8, 1], "tanh", output_activation="tanh", bias=False)
task = CartPole()


def extent(weights, steps):
    """The largest |x| and |θ| (in degrees) over an episode of ``steps`` steps."""
    states = task.episode(mlp.policy(weights), steps)
    return float(np.max(np.abs(states[:, 0]))), float(np.max(degrees(np.abs(states[:, 2]))))


# the steps balanced, up to 100,000, evaluated in Rust
cmaes = gx.Cmaes(mlp.representation((-1.0, 1.0)), restarts="ipop", objective="maximize", seed=1)
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(mlp, task)
result = cmaes.run(
    Balance(task, mlp), target=SUCCESS_STEPS, evaluations=100_000, on_generation=trace.record
)

print(
    f"balanced for {result.best_fitness:.0f} steps (the goal: {SUCCESS_STEPS}) after "
    f"{result.evaluations} evaluations in {result.generations} generations"
)
weights = result.best_genome
position, angle = extent(weights, SUCCESS_STEPS)
print(f"over the 100000 steps: the cart within {position:.4f} m, the pole within {angle:.4f}°")
print("weights: [" + ", ".join(f"{w:.3f}" for w in weights) + "]")
trace.write(weights)


# the same task by NEAT, with the paper's settings: networks that grow from the inputs and a bias
# connected to the output, whose output, in (0, 1), is the force as 2 × output − 1
def steps(network):
    policy = network.feed_forward().policy(scale=2.0, offset=-1.0)
    return task.run(policy, SUCCESS_STEPS)


neat = gx.Neat(4, 1, seed=1)
result = neat.run(steps, target=SUCCESS_STEPS, evaluations=100_000)
network = result.best_genome
print(
    f"\nNEAT: balanced for {result.best_fitness:.0f} steps after {result.evaluations} evaluations "
    f"in {result.generations} generations"
)
print(f"the network: {network.hidden()} hidden nodes, {network.enabled()} enabled connections")
