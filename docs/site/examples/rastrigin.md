# Rastrigin

The Rastrigin function has many local minima around its global minimum of 0 at the origin. The
Rust example minimizes it in 10 dimensions with a genetic algorithm on a real genome, polynomial
mutation, parallel evaluation, progress printed every 200 generations and a time limit. The Python
one minimizes it in 30 dimensions with CMA-ES with IPOP restarts and with L-SHADE, evaluating a
whole generation per call of a vectorized numpy function.

=== "Rust"

    ```rust
    --8<-- "examples/rastrigin.rs"
    ```

=== "Python"

    ```python
    --8<-- "python/examples/rastrigin.py"
    ```
