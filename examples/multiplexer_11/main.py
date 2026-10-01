"""Koza's 11-multiplexer: find the Boolean function that uses 3 address bits to select one of 8
data bits, from all 2048 cases of its truth table, by genetic programming.

Trees of Koza's functions (and, or, not, if) and the 11 inputs, evolved by a genetic algorithm
with subtree crossover and a mix of mutations, and double tournaments against bloat. The fitness
is the number of the 2048 cases a tree gets wrong; the run stops when it gets all of them right.

The problem is evaluated in Rust (``gx.gp.boolean``), 64 cases at once, so the run is the Rust
example's, to the bit, on every platform.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its run for the plot on the example's
page, with trace.py.

    python examples/multiplexer_11/main.py
"""

import genoxide as gx

from trace import Trace

POPULATION = 4000

problem = gx.gp.boolean.Multiplexer(3)
print("Koza's 11-multiplexer: 3 address bits select one of 8 data bits, 2048 cases")
print(f"{POPULATION} trees, until every case is right")
print()

# Koza's limits and initialization: depth 17, ramped half-and-half of depths 2 to 6
gp = gx.gp.Gp(problem.primitives())
ga = gx.Ga(
    gp,
    population_size=POPULATION,
    initial_genomes=gp.ramped_half_and_half(POPULATION, 1),
    select=gx.DoubleTournament(7, 1.4),
    crossover=gx.gp.SubtreeCrossover(),
    mutation=gx.gp.Mutations(
        [
            (0.5, gx.gp.SubtreeMutation()),
            (0.3, gx.gp.PointMutation(count=1)),
            (0.1, gx.gp.HoistMutation()),
            (0.1, gx.gp.ShrinkMutation()),
        ]
    ),
    crossover_rate=0.9,
    mutation_rate=0.1,
    objective="minimize",
    seed=1,
)
trace = Trace(problem)
# the problem itself is the fitness: the cases a tree gets wrong, counted in Rust
result = ga.run(problem, target=0.0, generations=50, on_generation=trace.on_generation)

best = result.best_genome
print(
    f"{result.stop_reason.capitalize()} after {result.generations} generations and "
    f"{result.evaluations} evaluations"
)
print(f"cases right: {problem.cases - problem.errors(best)} of {problem.cases}")
print()
print(f"the function, {len(best)} nodes of depth {best.depth}:")
print(best)
trace.write()
