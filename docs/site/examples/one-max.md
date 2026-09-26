# OneMax

OneMax looks for the bit string with the most ones, the usual first problem for a genetic
algorithm. Both examples use a binary genome with tournament selection, crossover and bit-flip
mutation, and stop at the optimum. The Rust one uses 500 bits with uniform crossover and records
statistics per generation; the Python one uses 100 bits with two-point crossover and crossover and
mutation rates below 1.

=== "Rust"

    ```rust
    --8<-- "examples/one_max.rs"
    ```

=== "Python"

    ```python
    --8<-- "python/examples/onemax.py"
    ```
