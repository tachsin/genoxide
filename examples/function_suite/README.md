---
title: Function suite
category: continuous
summary: CMA-ES, SHADE and PSO on twelve classic test functions in 10 dimensions, with the error to each known minimum.
reference: "Yao, X., Liu, Y. and Lin, G. (1999). Evolutionary programming made faster. IEEE Transactions on Evolutionary Computation 3(2): 82-102."
reference_url: "https://doi.org/10.1109/4235.771163"
optimum: "an error of 0 on each function"
languages: [rust, python]
order: 62
---

# Function suite

The example runs three algorithms on the scalable functions of genoxide's `problems`, in 10
dimensions: Sphere, the axis-parallel ellipsoid, Schwefel's problem 1.2, Zakharov, Rosenbrock,
Rastrigin, Ackley, Griewank, Schwefel's problem 2.26, Levy, Styblinski-Tang and Michalewicz. The
algorithms are CMA-ES with IPOP restarts, SHADE (differential evolution with adapted parameters)
and PSO with 40 particles. Each run has a budget of 10,000 evaluations per dimension and stops
early within 1e-8 of the known minimum. The table gives the error of each run: the best value it
found minus the minimum.

Each problem brings its bounds, its minimum and its citation; the docs of `genoxide::problems`
give each function and where its definition comes from. Most definitions and bounds here are
those of Yao, Liu and Lin (1999). In Python, `run` evaluates the problems in Rust, so both
versions print the same table.
