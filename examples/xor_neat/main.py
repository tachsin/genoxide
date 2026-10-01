"""XOR by NEAT: evolve a network's structure and weights until it computes XOR, from networks
without hidden nodes.

The NEAT paper's first experiment (Stanley and Miikkulainen 2002): XOR isn't linearly separable,
so a network needs at least one hidden node, which NEAT has to discover. The initial networks
connect the two inputs and the bias straight to the output; mutations add nodes and connections,
speciation protects the new structure while its weights are tuned. The fitness is the paper's,
(4 - sum |error|)^2, maximized, with its fitness sharing, and the run stops at the paper's success
criterion: every output on the right side of 0.5.

The networks run in Rust (``gx.neat.Network.feed_forward``), so the run is the Rust example's, to
the bit, on every platform.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/xor_neat/main.py
"""

import genoxide as gx

from trace import Trace

# the inputs and the expected output
CASES = [
    ((0.0, 0.0), 0.0),
    ((0.0, 1.0), 1.0),
    ((1.0, 0.0), 1.0),
    ((1.0, 1.0), 0.0),
]


def outputs(network):
    """The network's outputs for the four cases."""
    evaluator = network.feed_forward()
    return [float(evaluator.activate(inputs)[0]) for inputs, _ in CASES]


def fitness(network):
    """The paper's fitness: (4 - sum |error|)^2, 16 for a perfect network."""
    error = sum(abs(output - target) for output, (_, target) in zip(outputs(network), CASES))
    return (4.0 - error) * (4.0 - error)


def solves(network):
    """The paper's success criterion: every output on the right side of 0.5."""
    return all(
        (output >= 0.5) == (target == 1.0) for output, (_, target) in zip(outputs(network), CASES)
    )


# the paper's settings, with its fitness sharing: the fitness is maximized and non-negative
neat = gx.Neat(2, 1, population_size=150, sharing="raw", seed=1)
print("XOR by NEAT: 150 networks, from 2 inputs and a bias connected to the output")
# with GENOXIDE_TRACE=<file>, a trace of the run for the plot on the example's page
trace = Trace(CASES)
solution = None


def on_generation(progress):
    """Records the generation, and stops the run at the first generation with a network that
    solves XOR."""
    global solution
    trace.record(progress)
    if solution is None:
        solution = next((network for network in progress.population if solves(network)), None)
    return solution is None


result = neat.run(fitness, generations=1000, on_generation=on_generation)

if solution is None:
    print(f"not solved after {result.generations} generations")
else:
    network = solution
    print(f"solved in generation {result.generations} after {result.evaluations} evaluations\n")
    print(
        f"the network: {network.hidden()} hidden nodes, {network.enabled()} enabled connections "
        f"of {len(network.connections())}"
    )
    kinds = {node.id: node.kind for node in network.nodes()}
    names = {"input": "in{}", "bias": "bias", "output": "out"}

    def name(id):
        return names.get(kinds.get(id, "hidden"), "h{}").format(id)

    for connection in network.connections():
        if connection.enabled:
            print(
                f"  {name(connection.from_):>4} -> {name(connection.to):<4} "
                f"{connection.weight:8.3f}"
            )
    print()
    for output, ((a, b), expected) in zip(outputs(network), CASES):
        print(f"{a:.0f} xor {b:.0f} = {expected:.0f}: {output:.3f}")
trace.write()
