---
title: Asynchronous Bayesian optimization
category: bayesian
summary: Keep 4 workers busy on evaluations of uneven duration with an AsyncEngine, each new point chosen with the ones still being evaluated fantasized, to the global minimum of Hartmann's 3-D function.
reference: "Ginsbourger, D., Le Riche, R. and Carraro, L. (2010). Kriging is well-suited to parallelize optimization. In Computational Intelligence in Expensive Optimization Problems, Springer: 131-162."
reference_url: "https://doi.org/10.1007/978-3-642-10701-6_6"
optimum: "−3.86278 at (0.11461, 0.55565, 0.85255) (best known)"
languages: [rust]
order: 292
trace_note: "Recorded from another run like the one below: its times and values differ."
---

# Asynchronous Bayesian optimization

## The problem

Hartmann's function in 3 dimensions, whose global minimum is −3.86278; the [Hartmann
3-D](../hartmann3/) example gives its formula and constants. It stands for an expensive function
whose evaluations take different times, as simulations often do: each evaluation here sleeps for 10
to 50 ms, a time drawn from a hash of the point's bits and a seed, so a point always takes as long.

Four workers evaluate at a time. The example measures how long the search takes to come within
1e-4 of the minimum, and in how many evaluations.

There's no Python version: the Python package has no asynchronous engine.

## What makes it hard

The waiting. In [batches](../bo_hartmann6/), a round of 4 points is evaluated together, and the
next round can't be chosen before the slowest of the 4 is done: workers that finish early sit idle,
here for up to 40 ms of every round. Choosing a point as soon as a worker is free means choosing it
while 3 others are still being evaluated, with their values unknown.

## Representation

A `Real` genome of 3 genes in [0, 1]: genoxide's `problems::Hartmann3`, evaluated in Rust, after
the sleep.

## Algorithm

`Bo` with its defaults, run by an `AsyncEngine` with 4 workers. `Bo` implements `Incremental`: it
proposes the initial design of 2(n + 1) = 8 points first, one at a time, then each proposal is a
point the model chooses. The model is fitted to every result so far, then the points still being
evaluated are added to it with a fantasized value, as the points of a batch are (Ginsbourger, Le
Riche and Carraro, 2010): the Kriging believer, the model's own mean there (`Fantasy`), which
removes the model's uncertainty at those points, so that the log expected improvement that
chooses the next point looks elsewhere. A result that arrives replaces its fantasy.

With one worker, a seed gives the same run every time. With more, the order of the results depends
on the timing, so runs differ, as with any `AsyncEngine`.

As a contrast, the same search in batches of 4 points (`.batch(4)`), each round evaluated at once
by `Engine::parallel(true)` and waiting for its slowest evaluation. Both stop within 1e-4 of the
minimum, or after 120 evaluations.

## Output

The first line gives the function, how long an evaluation takes, how many run at a time and the
global minimum. Then a line per run: its wall-clock time, its evaluations, its evaluations per
second, its best value and how far that is above the minimum.

The times depend on the machine, and the asynchronous run's evaluations and best value on the order
of the results, so they change from run to run.

[The project page](https://tachsin.gr/projects/genoxide/examples/bo-asynchronous) plays back
another run, recorded the same way: when each worker evaluated, and the best value as it fell.

## Good results

The asynchronous run reaches the minimum: in the run of `output.txt`, 7.1e-7 above it after 46
evaluations, in 0.38 s. Over 25 runs, every one came within 1e-4, after 38 to 61 evaluations,
half of them within 44.

The time is what the example compares. The batches came within 1e-4 too, after 60 evaluations
(every run, as they don't depend on the timing), in 0.78 s: twice as long, since every round waits
for its slowest evaluation, and more evaluations, since a batch's points are chosen 4 at a time,
and an asynchronous proposal knows every result that has arrived. The times vary with the machine;
the gap stays.

The fantasy matters more here than in batches: every proposal has 3 points fantasized, never 0.
With the constant liar's lowest value instead of the believer, in 5 runs without the sleeps, only
2 came within 1e-4 in 120 evaluations: telling the model that the points being evaluated are as
good as the best keeps their region at the best value, with no improvement left there, and pushes
every proposal away from where the search is converging. With the believer, all 5 did, in 39 to
76 evaluations.
