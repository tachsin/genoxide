---
title: SHADE, then L-BFGS-B
category: local
summary: Find the basin of Rastrigin's global minimum in 10 dimensions with SHADE, then polish SHADE's best to the minimum itself with L-BFGS-B, in five evaluations.
reference: "Tanabe, R. and Fukunaga, A. (2013). Success-history based parameter adaptation for differential evolution. IEEE Congress on Evolutionary Computation: 71-78."
reference_url: "https://doi.org/10.1109/CEC.2013.6557555"
optimum: "0 (at the origin)"
languages: [rust, python]
order: 284
---

# SHADE, then L-BFGS-B

## The problem

Rastrigin's function in 10 dimensions:

```text
f(x) = 10 n + Σᵢ (xᵢ² − 10 cos 2πxᵢ),   xᵢ ∈ [−5.12, 5.12]
```

Its minimum is 0, at the origin. The cosine puts a local minimum near every point of the integer
grid: about 10¹⁰ of them in the box.

## What makes it hard

A local method ends in the minimum of the basin it starts in, and from a random start that's almost
never the global one. A global method, here SHADE, finds the global minimum's basin, every gene
within 0.5 of 0, but then closes in slowly: its steps come from differences between individuals,
and each digit it gains takes thousands of evaluations. The two together divide the work: the
global method finds the basin, and a local method that uses the gradient finishes inside it.

## Representation

A `Real` genome of 10 genes in [−5.12, 5.12]: genoxide's `problems::Rastrigin`, which also gives
the analytic gradient, 2xᵢ + 20π sin 2πxᵢ.

## Algorithm

1. **SHADE** (`De` with its defaults: Tanabe and Fukunaga's settings, 100 individuals, seed 1) for
   40,000 evaluations.
2. **L-BFGS-B** (`Lbfgsb`, with its defaults) from SHADE's best, `initial_genome(best)`, with the
   analytic gradient. Inside the basin the function is smooth and nearly quadratic, so L-BFGS-B
   converges in a few iterations. It stops once the largest component of the projected gradient
   is below 1e-5.

For contrast, SHADE alone from the same start, run on until f ≤ 1e-12.

## Output

SHADE's best value every 5,000 evaluations, and how far its best is from the origin. Then L-BFGS-B
round by round, and how far its end is from the origin. Last, SHADE alone. In Python, `run`
evaluates the problem in Rust, so both versions print the same.

[The project page](https://tachsin.gr/projects/genoxide/examples/polish) plays the runs back:
the best value over the evaluations, SHADE then L-BFGS-B against SHADE alone.

## Good results

The minimum is 0 at the origin. After 40,000 evaluations SHADE's best is at f = 1.5e-2, every gene
within 5.0e-3 of 0: in the global minimum's basin. L-BFGS-B takes it to f = 0, exactly, in 5 more
evaluations, every gene within 5.8e-13 of 0: 40,005 evaluations in all.

SHADE alone needs 68,300 evaluations to reach f = 6.0e-13, 28,000 more than the two together,
for a value the polish beats. The hand-over works when the global method has found the right basin:
started after 20,000 evaluations, where SHADE's best is at f = 7.5 in another basin, L-BFGS-B
ends at that basin's minimum, f = 3.0, in 7 evaluations.
