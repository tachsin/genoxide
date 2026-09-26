# Knapsack

The 0/1 knapsack problem chooses the items with the highest total value whose total weight fits a
capacity. The fitness function returns the value and how far the weight exceeds the capacity, so
feasible selections rank above infeasible ones, and infeasible ones closer to the capacity rank
higher. The Rust example has 20 items, keeps a hall of fame of the best three selections and
checks the result against the optimum found by brute force; the Python one has 50 random items.

=== "Rust"

    ```rust
    --8<-- "examples/knapsack.rs"
    ```

=== "Python"

    ```python
    --8<-- "python/examples/knapsack.py"
    ```
