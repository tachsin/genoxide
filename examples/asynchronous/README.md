---
title: Asynchronous evaluation
category: engine
summary: Keep every worker busy when evaluations take different times, as simulations often do.
reference: null
reference_url: null
optimum: "0 (at the origin)"
languages: [rust]
order: 260
trace_note: "Recorded from another run like the one below: its times and values differ."
---

# Asynchronous evaluation

## The problem

This example is about the engine, not about a problem from the literature. Its fitness function
stands in for a simulation whose run time varies from one input to the next. It computes Rastrigin's
function in 4 dimensions, and then sleeps for 0.25 to 2 ms, depending on the genome.

Rastrigin's function, 10n + Σ (xᵢ² − 10 cos 2πxᵢ) over [−5.12, 5.12]ⁿ, has its minimum, 0, at the
origin; the Rastrigin example explains it.

Two runs of the same genetic algorithm evaluate in parallel. The first, asynchronous, goes on until
it finds the minimum; the second, generational, then gets as many evaluations. The example measures
how long each takes.

There's no Python version: the Python package has no asynchronous engine.

## What makes it hard

The waiting. A generational algorithm breeds a whole generation, evaluates it in parallel, and waits
for all of it before it breeds the next. A generation takes as long as its slowest evaluations
allow, and workers that finish early sit idle. With evaluations of 0.25 to 2 ms, a worker can finish
a short one and then wait several times as long for a long one elsewhere.

Evaluated one after another, 6,000 evaluations of 0.25 to 2 ms would take 1.5 to 12 seconds.

For the search, Rastrigin's function has a local minimum near every point of the integer grid, and
the example asks for the minimum to within 10⁻⁶, which takes a genetic algorithm thousands of
evaluations.

## Representation

A `Real` genome of 4 genes, each in [−5.12, 5.12]: the point x. The fitness is the function's value,
to minimize.

## Algorithm

Both runs use the same genetic algorithm settings: a population of 80, tournaments of size 3,
simulated binary crossover with η = 1, and polynomial mutation with η = 50 at a rate of 1/4 per gene.
The small η of the crossover spreads the children widely around their parents, and the large η of
the mutation makes its steps small. I chose these settings for the steady-state run: with a
population of 40 and η of 15 and 20, the usual values, 6 of seeds 1 to 10 (with one worker)
didn't get within 10⁻⁶ of the minimum in 100,000 evaluations.

- The first run is the steady-state version, built with `build_steady`, and run by an `AsyncEngine`
  with as many workers as rayon has threads. It proposes one child at a time. Each worker gets a new
  child as soon as it finishes an evaluation, and each result replaces the worst individual if it
  isn't worse. No worker waits for another. It stops within 10⁻⁶ of the minimum, or after 50,000
  evaluations.
- The second is generational, with the default elitism of 1. Its `Engine` evaluates each generation
  in parallel, on rayon's threads, and stops after as many evaluations as the first run made (it
  finishes its last generation, so it may make a few more).

genoxide's guide recommends asynchronous evaluation for slow fitness functions whose time varies.
With more than one worker, the order of the results depends on the timing, so the asynchronous run
differs from one run to the next.

## Output

The first line gives the function, how long an evaluation takes, and how many run at a time: the
number of workers, one per CPU thread. Then there is a line per run: its wall-clock time, its
evaluations, its evaluations per second, and the best value it found.

The times depend on the machine. The asynchronous run's evaluations and best value depend on the
order of the results, so they change from run to run too.

[The project page](https://tachsin.gr/projects/genoxide/examples/asynchronous) plays back another run, recorded the same way: its times and values differ from
the ones above.

## Good results

The asynchronous run finds the minimum: in the run of `output.txt`, it got within 10⁻⁶ of 0, at
0.00000053, after 5,945 evaluations. Over seeds 1 to 30 with 20 workers, every run got there, after
4,800 to 19,200 evaluations (half of them within 7,300), in 0.3 to 1.3 s; with one worker, which
makes the run reproducible, seeds 1 to 40 all got there, within 22,600 evaluations.

The time is what the example compares. With the same evaluations, the asynchronous run finishes
sooner, since its workers never wait: in the run of `output.txt`, 0.39 s against 0.48 s, with 20
workers, about 15,400 evaluations per second against 12,400. The times vary from run to run and
machine to machine; the gap stays. It's smaller than it would be with a population nearer the
number of workers: with 80 individuals on 20 workers, a generation's idle time is only at its end.

The generational run is a contrast, not a rival: in the runs I tried, it never got within 10⁻⁶ in
as many evaluations. In `output.txt`, it ended at 0.99863, next to the local minimum of about 0.995
one step from the origin along an axis. Its best values come from a different algorithm, so they
don't rank the engines.
