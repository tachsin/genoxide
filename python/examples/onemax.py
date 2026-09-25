"""OneMax, as in DEAP's first tutorial: the 100-bit string with the most ones."""

import genoxide as gx

ga = gx.Ga(
    gx.Binary(100),
    population_size=300,
    select=gx.Tournament(3),
    crossover=gx.PointCrossover(2),
    crossover_rate=0.5,
    mutation=gx.BitFlip(rate=0.05),
    mutation_rate=0.2,
    seed=1,
)
result = ga.run(lambda bits: bits.sum(), target=100, generations=1_000)
print(f"{result.best_fitness:.0f} ones after {result.generations} generations ({result.stop_reason})")
