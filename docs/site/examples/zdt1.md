# ZDT1

ZDT1 is a standard multi-objective test problem: two conflicting objectives to minimize over 30
variables in [0, 1], with a convex Pareto front. Both examples run NSGA-II with simulated binary
crossover and polynomial mutation for 25,000 evaluations, then print the size of the final
non-dominated front and its hypervolume with the reference point (1.1, 1.1). The Rust example uses
genoxide's built-in ZDT1 problem and hypervolume indicator; the Python one evaluates a whole
generation per call with numpy.

=== "Rust"

    ```rust
    --8<-- "examples/zdt1.rs"
    ```

=== "Python"

    ```python
    --8<-- "python/examples/zdt1.py"
    ```
