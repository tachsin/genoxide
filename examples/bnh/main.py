"""BNH: minimize two objectives subject to two constraints, with NSGA-II and Deb's rules.

Binh and Korn's problem, from genoxide's problems.Bnh, whose fitness is the two objectives and the
constraint violation; run evaluates it in Rust. Prints how many solutions of the final front are
feasible, their IGD+ to 500 points of the optimal front, and the front's hypervolume.

    python examples/bnh/main.py
"""

import genoxide as gx

problem = gx.problems.Bnh()
nsga2 = gx.Nsga2(
    problem.genome,
    objectives=problem.objectives,
    population_size=100,
    crossover=gx.SimulatedBinaryCrossover(20),
    mutation=gx.PolynomialMutation(20, rate=0.5),
    seed=1,
)
result = nsga2.run(problem, generations=250)

front = result.front_objectives
feasible = int((result.front_violations == 0).sum())
print(f"{len(front)} solutions on the front, {feasible} feasible")

# IGD+ to the optimal front, x1 = x2 from 0 to 5
optimal = problem.optimal_front(500)
distance = gx.indicators.igd_plus(front, optimal)
print(f"IGD+ to the optimal front: {distance:.4f}")

# the hypervolume with the reference point (210, 55); the whole front's is 210 × 55 − 5000/3, the
# area above f2 = 2 (√(f1/8) − 5)²
volume = gx.indicators.hypervolume(front, [210.0, 55.0])
print(f"hypervolume {volume:.2f} (the whole front: 9883.33)")
