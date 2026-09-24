"""ZDT1 with NSGA-II, as in pymoo's getting-started guide: 30 variables, two objectives to
minimize, and the final non-dominated front."""

import numpy as np

import genoxide as gx


def zdt1(x):
    """x has a genome per row; the result, a row of objective values per genome."""
    f1 = x[:, 0]
    g = 1 + 9 * x[:, 1:].mean(axis=1)
    return np.column_stack([f1, g * (1 - np.sqrt(f1 / g))])


nsga2 = gx.Nsga2(
    gx.Real((0.0, 1.0), length=30),
    objectives=["minimize", "minimize"],
    population_size=100,
    crossover=gx.SimulatedBinaryCrossover(15),
    crossover_rate=0.9,
    mutation=gx.PolynomialMutation(20, rate=1 / 30),
    seed=1,
)
result = nsga2.run(zdt1, batch=True, evaluations=25_000)
front = result.front_objectives[np.argsort(result.front_objectives[:, 0])]

# the hypervolume of the front, with the reference point (1.1, 1.1)
widths = np.diff(np.append(front[:, 0], 1.1))
hypervolume = np.sum(widths * (1.1 - front[:, 1]))
print(f"{len(front)} solutions on the front, hypervolume {hypervolume:.4f} (the whole front: 0.8767)")
