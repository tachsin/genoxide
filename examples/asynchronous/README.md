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

## The problem

This example is about the engine, not about a problem from the literature. Its fitness function
stands in for a simulation whose run time varies from one input to the next. It computes Rastrigin's
function in 6 dimensions, and then sleeps for 1 to 8 ms, depending on the genome. Two algorithms
each get 2,000 evaluations, and the example measures how long they take.

Rastrigin's function, 10n + Σ (xᵢ² − 10 cos 2πxᵢ) over [−5.12, 5.12]ⁿ, has its minimum, 0, at the
origin; the Rastrigin example explains it.

There's no Python version: the Python package has no asynchronous engine.

## What makes it hard

The waiting. A generational algorithm breeds a whole generation, evaluates it in parallel, and waits
for all of it before it breeds the next. A generation takes as long as its slowest evaluations
allow, and workers that finish early sit idle. With evaluations of 1 to 8 ms, a worker can finish a
short one and then wait several times as long for a long one elsewhere.

Evaluated one after another, 2,000 evaluations of 1 to 8 ms would take 2 to 16 seconds.

## Representation

A `Real` genome of 6 genes, each in [−5.12, 5.12]: the point x. The fitness is the function's value,
to minimize.

## Algorithm

Both runs use the same genetic algorithm settings: a population of 40, tournaments of size 3,
simulated binary crossover with η = 15, and polynomial mutation with η = 20 at a rate of 1/6 per
gene.

- The first is generational, with the default elitism of 1. Its `Engine` evaluates each generation
  in parallel, on rayon's threads.
- The second is its steady-state version, built with `build_steady`, and run by an `AsyncEngine`
  with as many workers as rayon has threads. It proposes one child at a time. Each worker gets a new
  child as soon as it finishes an evaluation, and each result replaces the worst individual if it
  isn't worse. No worker waits for another.

genoxide's guide recommends asynchronous evaluation for slow fitness functions whose time varies.
With more than one worker, the order of the results depends on the timing, so the asynchronous run
differs from one run to the next.

## Output

The first line gives the number of evaluations and how many run at a time: the number of workers,
one per CPU thread. Then there is a line per run: its wall-clock time, its evaluations per second,
and the best value it found.

The times depend on the machine. The asynchronous run's best value depends on the order of the
results, so it changes from run to run too.

The project page plays back another run, recorded the same way: its times and values differ from
the ones above.

## Good results

What matters is the time. The asynchronous run finishes the same 2,000 evaluations sooner, since its
workers never wait: in the run of `output.txt`, 0.50 s against 0.70 s, with 20 workers. The times
vary from run to run and machine to machine; the gap stays. Neither run gets near
the minimum, 0, with 2,000 evaluations. Their best values come from different algorithms, so they
don't rank the engines.
