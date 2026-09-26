# Asynchronous evaluation

When evaluations take different times, as simulations often do, a generational algorithm that
evaluates in parallel waits for the slowest evaluation of each generation. This example evaluates
a Rastrigin function that takes 1 to 8 ms per call, 2,000 times: once with a generational GA in
parallel, and once with a steady-state GA and an `AsyncEngine`, which gives each worker a new
genome as soon as it finishes. It prints the time and evaluations per second of both.

This example is Rust only: the Python package doesn't have asynchronous evaluation.

=== "Rust"

    ```rust
    --8<-- "examples/asynchronous.rs"
    ```
