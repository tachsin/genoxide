---
title: Himmelblau's function
category: continuous
summary: Find the four global minima of a two-dimensional function by restarting a local search from random points.
reference: "Himmelblau, D. M. (1972). Applied Nonlinear Programming. McGraw-Hill."
reference_url: ""
optimum: "0 (at four points)"
languages: [rust, python]
order: 54
---

# Himmelblau's function

## The problem

Himmelblau's function (1972) is a sum of two squares in two variables:

```text
f(x₁, x₂) = (x₁² + x₂ − 11)² + (x₁ + x₂² − 7)²
```

It is 0 exactly where both squares are 0: where x₁² + x₂ = 11 and x₁ + x₂² = 7. At (3, 2), both
hold. At the origin, f = 11² + 7² = 170.

## What makes it hard

The equations have four solutions, so the function has four global minima, all of value 0: (3, 2),
(−2.805118, 3.131313), (−3.779310, −3.283186) and (3.584428, −1.848127). They lie in four separate
basins. A search that converges to one point finds one of them, and which one depends on where it
starts. Finding all four takes several searches, or a method that keeps several apart.

## Representation

A `Real` genome of 2 genes, each in [−5, 5], the box around the four minima: the point itself. The
fitness is f, to minimize. The function is genoxide's `problems::Himmelblau`, which also gives the
four minima, computed to full precision.

## Algorithm

Twenty independent local searches, each from a random point, with seeds 1 to 20. Each is a hill
climber:

- a step makes 10 neighbors of the current point, each with Gaussian noise on both genes, of
  standard deviation 0.005 (0.0005 of the range of 10);
- the search moves to the best neighbor only if it is strictly better;
- it stops after 3,000 steps, 30,001 evaluations with the starting point.

With steps that small, a search follows its basin down to the minimum at the bottom. Restarting from
random points is the simplest way to find several minima: each start lands in some basin. Each
search's end point is then assigned to the nearest of the four known minima.

## Output

The first line gives the number of searches, the bounds and the length of each search. Then a
table has a row per known minimum: its coordinates, how many searches ended nearest to it, and the
range of values they reached, from the best to the worst. In Python, `run` evaluates the function
in Rust, so both versions print the same table.

The values are small but not 0. The step size is fixed, so near a minimum few neighbors are better
than the current point, and progress slows down to a stop. The smaller the steps, the closer to 0 a
search gets, and the longer it takes to reach a minimum from its start: with steps of 0.01 and
1,000 steps, a quarter of the searches stopped between 1e-6 and 6e-6.

[The project page](https://tachsin.gr/projects/genoxide/examples/himmelblau) plays this run back.

## Good results

Each minimum is worth 0. A good result finds all four, each within 1e-6 of 0. The run finds all
four, with 3 to 6 searches each, and every search ends between 3.4e-10 and 2.1e-7.

With seeds 1 to 600, in 30 groups of 20 searches, every search ends within 5.3e-7 of 0, and 29 of
the 30 groups find all four minima; the other finds three.
