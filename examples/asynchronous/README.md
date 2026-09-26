---
title: Asynchronous evaluation
category: engine
summary: Keep every worker busy when evaluations take different times, as simulations often do.
reference: null
reference_url: null
optimum: null
languages: [rust]
order: 110
---

# Asynchronous evaluation

When evaluations take different times, a generational algorithm that evaluates in parallel waits
for the slowest evaluation of each generation. This example evaluates a Rastrigin function that
takes 1 to 8 ms per call, 2,000 times: once with a generational genetic algorithm in parallel, and
once with a steady-state genetic algorithm and an `AsyncEngine`, which gives each worker a new
genome as soon as it finishes. It prints the time, the evaluations per second and the best value
of both. It is Rust only: the Python package has no asynchronous evaluation.
