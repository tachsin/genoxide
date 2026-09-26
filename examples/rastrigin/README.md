---
title: Rastrigin function
category: continuous
summary: Minimize a 30-dimensional function with a local minimum at every integer point.
reference: "Mühlenbein, H., Schomisch, M. and Born, J. (1991). The parallel genetic algorithm as function optimizer. Parallel Computing 17(6-7): 619-632."
reference_url: "https://doi.org/10.1016/S0167-8191(05)80052-3"
optimum: "0 (at the origin)"
languages: [rust, python]
order: 60
---

# Rastrigin function

The Rastrigin function, 10n + Σ (xᵢ² − 10 cos 2πxᵢ) over [−5.12, 5.12]ⁿ, adds a cosine to a
sphere, which puts a local minimum at every integer point around the global minimum of 0 at the
origin. The example minimizes it in 30 dimensions with two algorithms, each with a budget of
1,000,000 evaluations and a target of 1e-8: CMA-ES with IPOP restarts, whose population doubles
at each restart, and L-SHADE, a differential evolution whose population shrinks over the budget.
It prints the best value each one found and the evaluations it took. The function is genoxide's
`problems::Rastrigin`, which the Python version's `run` evaluates in Rust, so both versions print
the same output.
