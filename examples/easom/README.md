---
title: Easom
category: continuous
summary: Minimize Easom's function, one narrow well in a plane that is flat to within 1e-10 almost everywhere, with CMA-ES from 30 seeds, without and with restarts.
reference: "Easom, E. E. (1990). A Survey of Global Optimization Techniques. M.Eng. thesis, University of Louisville."
reference_url: ""
optimum: "−1 at (π, π)"
languages: [rust, python]
order: 87
---

# Easom

## The problem

Easom's function is a function of two variables to minimize:

```text
f(x₁, x₂) = −cos x₁ cos x₂ exp(−((x₁ − π)² + (x₂ − π)²)),   x₁, x₂ in [−100, 100]
```

It comes from Easom's thesis (1990), a survey of global optimization methods, and probably first
appeared in a journal in Stuckman and Easom (1992, IEEE Transactions on Systems, Man, and
Cybernetics 22(5): 1024-1032). Neither could be read: genoxide takes the definition and the bounds
from Jamil and Yang's (2013) restatement, and they are still to be checked against the original.

The minimum is −1 at (π, π), and it's the global minimum: both cosines are at most 1 in absolute
value, and the exponential is 1 only at (π, π). genoxide's `problems::Easom` gives it as proven.

## What makes it hard

The exponential makes the function a single narrow well around (π, π), and nearly nothing
elsewhere: farther than 4.8 from (π, π), |f| is below 1e-10, and that's all of the box but 0.2%.
Farther than 27.3, the exponential is below the smallest number an f64 can hold, and f is exactly 0.
The part of the well where f is below −0.5 is a disk of area 1.4, in a box of 40,000: a random
point falls in it with a probability of 1 in 28,000, and below −0.01 of 1 in 5,200.

The plane isn't quite flat within 27.3 of (π, π): the cosines make ripples in it, too small to see
but not too small to compare, since an algorithm that only compares values ranks −1e-100 below
−1e-200. The ripples have local minima, such as −8.1e-5 at (4.978, 4.978).

So a search either lands in or near the well by chance, and then goes down it, or it has nothing to
follow: all its points tie at 0, or the ripples lead it to a local minimum a hair below 0.

## Representation

A `Real` genome of 2 genes, each in [−100, 100]: the point (x₁, x₂) itself. The fitness is f, to
minimize. The function, its bounds and its minimum are genoxide's `problems::Easom`.

## Algorithm

CMA-ES (Hansen and Ostermeier, 2001, Evolutionary Computation 9(2): 159-195), which samples a
population from a normal distribution and adapts its mean, step size and covariance matrix, with
genoxide's defaults: a population of 4 + ⌊3 ln 2⌋ = 6, and a step size of 0.3 of each gene's range,
60, from a random start. Its first samples cover much of the box. It runs from seeds 1 to 30,
twice:

- without restarts: a run that doesn't find the well converges somewhere on the plane, and stays
  there until its budget is spent;
- with IPOP restarts (Auger and Hansen, 2005, IEEE CEC 2005: 1769-1776): a run that has converged
  starts again from a random point with twice the population, and so more samples of the box.

Each run stops once its value is within 1e-8 of −1, or after 10,000 evaluations.

## Output

The first line gives the minimum, the seeds and the budget. Then a row per algorithm: how many of
the 30 runs reach the minimum and how many don't, and the median and largest number of evaluations
of the runs that reach it. In Python, `run` evaluates the function in Rust, so both versions print
the same table.

The page's plot shows the 30 runs without restarts, each at its best point so far, over the
function's contour: the well is the light dot at (π, π), inside the marked minimum. A curve gives
the best and the median run's distance above the minimum, on a logarithmic axis. A run's best point
is the best sample it has seen, which isn't always where it converged.

[The project page](https://tachsin.gr/projects/genoxide/examples/easom) plays this run back.

## Good results

A good result reaches −1 in every run. Without restarts, 19 of the 30 runs do, after a median of 642
evaluations. Of the other 11, 5 converge where f is exactly 0, 53 to 108 from (π, π), where all
samples tie, and 6 into ripples: seeds 3, 14 and 23 into the local minimum at (4.978, 4.978), and
seed 10 into its mirror image at (4.978, 1.305), each after samples in the well's side, down to
−0.848, and seeds 22 and 26 at (−4.776, 4.978) and (11.06, 4.978), where f is −3.4e-31. With IPOP
restarts, all 30 runs reach −1, after a median of 1,314 evaluations and at most 2,292: the restarts
make the difference, since a converged run can't move any more.

Over seeds 1 to 1,000, 58% of the runs without restarts reach the minimum.
