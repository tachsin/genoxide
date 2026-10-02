---
title: Shekel 10
category: continuous
summary: Minimize Shekel's function with 10 narrow wells in 4 dimensions with particle swarms from 30 seeds, where a swarm that gathers early often settles in the wrong well.
reference: "Shekel, J. (1971). Test functions for multimodal search techniques. Proceedings of the 5th Annual Princeton Conference on Information Sciences and Systems, Princeton University. Constants as tabulated in Dixon, L. C. W. and Szegö, G. P. (1978). The global optimisation problem: an introduction. In Towards Global Optimisation 2, North-Holland: 1-15."
reference_url: ""
optimum: "−10.53641 near (4, 4, 4, 4) (best known)"
languages: [rust, python]
order: 83
family: Shekel
tab: Shekel 10
---

# Shekel 10

## The problem

Shekel's function puts a well at each of m points aᵢ, to minimize:

```text
f(x) = −Σᵢ₌₁ᵐ 1 / ((x − aᵢ)ᵀ(x − aᵢ) + cᵢ),   each xⱼ in [0, 10]
```

in 4 dimensions. Each term is −1/cᵢ at its center and falls off with the squared distance from it,
so well i is 1/cᵢ deep and about √cᵢ wide. Here m = 10, with the first ten rows of Dixon and Szegö's
table (Shekel 5 takes the first five, Shekel 7 the first seven):

| i | aᵢ | cᵢ | depth 1/cᵢ |
|---|---|---|---|
| 1 | 4, 4, 4, 4 | 0.1 | 10 |
| 2 | 1, 1, 1, 1 | 0.2 | 5 |
| 3 | 8, 8, 8, 8 | 0.2 | 5 |
| 4 | 6, 6, 6, 6 | 0.4 | 2.5 |
| 5 | 3, 7, 3, 7 | 0.4 | 2.5 |
| 6 | 2, 9, 2, 9 | 0.6 | 1.67 |
| 7 | 5, 5, 3, 3 | 0.3 | 3.33 |
| 8 | 8, 1, 8, 1 | 0.7 | 1.43 |
| 9 | 6, 2, 6, 2 | 0.5 | 2 |
| 10 | 7, 3.6, 7, 3.6 | 0.5 | 2 |

The function is Shekel's (1971), with the constants that Dixon and Szegö (1978) tabulate; they call
it SQRIN10. Neither is online: genoxide takes the constants from Yao, Liu and Lin's (1999,
table XIV) reprint.

The minimum is near a₁ = (4, 4, 4, 4), but not at it: the other wells pull it aside a little, and
f(4, 4, 4, 4) = −10.536284, 1.3e-4 higher. genoxide's `problems::Shekel10` gives the minimum as
−10.536409816692043 at (4.000746531592046, 4.000592934138532, 3.9996633980403224,
3.9995098005868077):
the point where the gradient is 0, computed to 40 digits by Newton's method. Later papers quote
−10.5364 from Dixon and Szegö. It's the best known minimum, not proven global. Jamil and Yang
(2013) put the minimum at (4, 4, 4, 4), with a value that is neither the minimum nor the value
there.

## What makes it hard

Each well is a local minimum, at nearly the value of its own term: −10.5364 at a₁, −5.1756 at a₃,
−5.1285 at a₂, −3.8354 at a₇, −2.8711 at a₄, −2.8066 at a₅, −2.4273 at a₁₀, −2.4217 at a₉,
−1.8595 at a₆ and −1.6766 at a₈. The wells are narrow, and the rest of the box is a
low plateau: at the center, (5, 5, 5, 5), f is −0.86, and in 0.7% of the box only is f below −1.
The deepest well is also the narrowest (√0.1 = 0.32 against 0.45 to 0.84). Local searches from
2,000 random points end in a₁'s well 38% of the time, in a₄'s 20%, in a₁₀'s 11%, in a₅'s 10%, and
in the other six 21%.

A method that learns where good points are from the points it has seen is misled by the first well
it finds: a well of depth 2.5 or 5 is far better than the plateau, and pulls the search in before
the deepest well has been sampled.

## Representation

A `Real` genome of 4 genes, each in [0, 10]: the point x itself. The fitness is f(x), to minimize.
The function, its bounds and its best known minimum are genoxide's `problems::Shekel10`.

## Algorithm

Particle swarm optimization (Kennedy and Eberhart, 1995, Proceedings of ICNN'95: 1942-1948), with
80 particles and Clerc and Kennedy's constriction coefficients (2002, IEEE Transactions on
Evolutionary Computation 6(1): 58-73), genoxide's defaults. Each particle moves towards its own
best point and a best point of the swarm. Two topologies, each from seeds 1 to 30:

- global: every particle follows the best point of the whole swarm. As soon as one particle finds a
  well, the whole swarm heads there;
- ring: each particle follows the best of itself and its two neighbors on a ring. A good point
  spreads one neighbor per step, so parts of the swarm keep exploring other wells for longer.

Each run stops once its value is within 1e-6 of the best known minimum, or after 25,000
evaluations. A run is counted in the well whose center is nearest its best point.

The swarm is larger than the usual 20 to 50 particles, and the budget larger than most runs
need: with 40 particles and 10,000 evaluations, the ring didn't reach the minimum in 2% to 3% of
the runs (seeds 1 to 1,000), and with 80 particles and 25,000 evaluations it reaches it in all of
them.

## Output

The first line gives the best known minimum, the seeds, the budget and the swarm's size. Then a row
per well that some run ended in, with how many runs of each topology ended there; then how many runs
came within 1e-6 of the best known minimum, and the median number of evaluations they needed. In
Python, `run` evaluates the function in Rust, so both versions print the same table.

The page's plot shows the 30 runs of the swarm with the global topology, each at its best point so
far, and a curve of the best and the median run's distance above the best known minimum, on a
logarithmic axis. The points are drawn at their (x₁, x₂), over the function on the plane x₃ = x₁,
x₄ = x₂, which holds every center aᵢ but a₇ = (5, 5, 3, 3): a run in well i is drawn at its
center, and one in a₇'s at (5, 5), where the plane has no well.

[The project page](https://tachsin.gr/projects/genoxide/examples/shekel10) plays this run back.

## Good results

A good result reaches −10.5364 in every run. The ring topology does: all 30 runs come within 1e-6
of it, after a median of 11,160 evaluations. With the global topology, for contrast, 20 runs end in
the deepest well and 19 of them come within 1e-6 of its minimum, after a median of 7,200
evaluations; the other 10 end in six of the other nine wells.

Over seeds 1 to 1,000, the ring topology reaches the minimum in every run, after at most 16,160
evaluations, and the global topology in 68% of the runs, against 47% on [Shekel 5](../shekel5/)
and 58% on [Shekel 7](../shekel7/). With 40 particles and 10,000 evaluations, they reach it in 98%
and 58%.
