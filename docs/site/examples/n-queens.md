# N-Queens

N-Queens places N queens on an N×N board so that no two attack each other, here with N = 64. A
permutation genome puts one queen in each row and each column, so only diagonals can conflict, and
the fitness is the number of conflicts to minimize. The Rust example uses a (μ+λ) genetic
algorithm with swap mutation and no crossover; the Python one uses local search with swap
neighbors and a tabu list.

=== "Rust"

    ```rust
    --8<-- "examples/n_queens.rs"
    ```

=== "Python"

    ```python
    --8<-- "python/examples/n_queens.py"
    ```
