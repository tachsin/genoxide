---
title: Bayesian optimization in batches on Hartmann 6-D
category: bayesian
summary: Reach the global minimum of Hartmann's 6-D function with 4 points a round, chosen by the Kriging believer and evaluated in parallel, in 15 rounds where one point a round takes 36.
reference: "Ginsbourger, D., Le Riche, R. and Carraro, L. (2010). Kriging is well-suited to parallelize optimization. In Computational Intelligence in Expensive Optimization Problems, Springer: 131-162."
reference_url: "https://doi.org/10.1007/978-3-642-10701-6_6"
optimum: "−3.32237 at (0.20169, 0.15001, 0.47687, 0.27533, 0.31165, 0.65730) (best known)"
languages: [rust, python]
order: 291
---

# Bayesian optimization in batches on Hartmann 6-D

## The problem

Hartmann's function in 6 dimensions, a sum of four Gaussian wells on [0, 1]⁶, whose global minimum
is −3.32237; the [Hartmann 6-D](../hartmann6/) example gives its formula and constants. Here it
stands for an expensive function that can be evaluated several times at once: a simulation that
runs for an hour, on 4 machines. What counts is how many rounds of evaluations the search takes,
each as long as one evaluation when its points run side by side, and how many evaluations in all.

## What makes it hard

Its second deepest minimum, −3.2032, is nearly as deep as the global one and far from it, with a
basin as wide: a search that finds it first can stay there. The [Bayesian
optimization](../bayesian_optimization/) of one point at a time already has to choose between
exploring and refining; a batch has to choose 4 points before it knows the value of any.

## Representation

A `Real` genome of 6 genes in [0, 1]: genoxide's `problems::Hartmann6`, evaluated in Rust in both
languages.

## Algorithm

`Bo` with batches of 4 (`.batch(4)`), its other settings the defaults: a Latin hypercube of
2(n + 1) = 14 points, then a Gaussian process with Matérn's 5/2 kernel fitted to every evaluation,
and the log expected improvement maximized for each point.

The 4 points of a round are chosen one after the other, as Ginsbourger, Le Riche and Carraro
(2010) propose: after each, the model is told the point with a fantasized value, without fitting
its hyperparameters again, and the next point maximizes the acquisition of that model. The
fantasy here is the Kriging believer (`bo::Fantasy::KrigingBeliever`, the default): the model's
own mean at the point. It leaves the model's mean where it was and removes its uncertainty at the
point, so the expected improvement there vanishes and the next point goes elsewhere. The constant
liar (`bo::Fantasy::ConstantLiar`) tells the model a fixed value instead, the lowest, mean or
highest value so far: the higher the lie, the farther the next points go.

`Engine::parallel(true)` evaluates the 4 points of a round at once; a seed gives the same points
on any number of threads. The run stops within 1e-4 of the minimum, or after 200 evaluations.

As a contrast, the same search one point a round, with the same seed.

## Output

The first lines give the problem and the method. Then a row per round: its number (0 is the
initial design), the evaluations so far, the best value and its distance above the global
minimum. The last lines give the evaluations and rounds each search took to come within 1e-4 of
the minimum, and the distance of the batches' best point from the minimum's. Both versions print
the same rows: the problem is evaluated in Rust.

[The project page](https://tachsin.gr/projects/genoxide/examples/bo-hartmann6) plots the best value's
distance above the minimum after each round, for both searches.

## Good results

A good result is within 1e-4 of −3.32237. The batches get there in round 15, after 74 evaluations,
3.9e-5 above it; one point a round needs 50 evaluations, but 36 rounds: with an evaluation of an
hour and 4 machines, 15 hours against 36. A batch spends more evaluations, since each of its points
is chosen knowing less than a point chosen after the others' results.

Not every seed finds the global minimum. Over seeds 1 to 20, within 200 evaluations: 13 batch
searches came within 1e-4, after 62 to 90 evaluations (a median of 70, 14 rounds); the other 7
stayed at the second minimum, −3.2032. One point a round: 12 of 20 (a median of 50 evaluations
and 36 rounds), the others at −3.2032 too. The seed of this example, 3, is the first to reach the
global minimum both ways. With the constant liar's lowest value and a larger initial design (30 or
60 points), or the log transform of the values, the share stayed at 10 to 13 of 20 (to 1e-3);
another run from another design is the way out, as the [Hartmann 6-D](../hartmann6/) example
shows for CMA-ES with restarts.

The constant liars do about as well on this problem, to 1e-3 within 200 evaluations: 13 of 20
with the lowest value, 12 with the mean, 11 with the highest, which explores most and needed a
median of 86 evaluations, against the believer's 13. On Branin and Hartmann 3, every seed of 20
reached 1e-3 within 80 evaluations with each fantasy: the believer and the lowest lie in a median
of 34 evaluations on Branin and 30 on Hartmann 3, the higher lies in 42 to 54 and 40.
