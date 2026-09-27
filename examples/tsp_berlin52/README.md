---
title: Travelling salesman (berlin52)
category: permutation
summary: The shortest round trip through 52 locations in Berlin, a TSPLIB instance.
reference: "Reinelt, G. (1991). TSPLIB: a traveling salesman problem library. ORSA Journal on Computing 3(4): 376-384."
reference_url: https://doi.org/10.1287/ijoc.3.4.376
optimum: "7542 (tour length)"
languages: [rust, python]
order: 40
---

# Travelling salesman (berlin52)

## The problem

The travelling salesman problem asks for the shortest round trip that visits each of a set of
locations once and returns to the start. TSPLIB (Reinelt, 1991) is a library of instances with known
optimal tours. berlin52 is one of them: 52 locations in Berlin, given as points in the plane. Its
optimal tour has length 7542.

The distance between two locations is TSPLIB's EUC_2D: the Euclidean distance, rounded to the
nearest integer. Locations 1 and 2, at (565, 575) and (25, 185), are 666 apart.

## What makes it hard

The problem is NP-hard (Garey and Johnson, 1979, Computers and Intractability). With symmetric
distances, 52 locations give 51!/2 ≈ 7.8 × 10⁶⁵ different tours. A search that only accepts shorter
tours stops at the first tour that no single move improves: a local optimum, not necessarily the
optimal tour.

## Representation

A `Permutation` of the 52 locations is the order of the visits; the tour returns from the last one
to the first. Every permutation is a valid tour, so no tour needs repairing. The fitness is the
tour's length, to minimize.

## Algorithm

A local search that keeps one tour and tries one neighbor per step.

The neighbor is an inversion: a random segment of the tour is reversed. This is the 2-opt move: it
removes two edges and reconnects the tour the other way. Flood (1956, Operations Research 4(1):
61-75) suggested the move, and Croes (1958, Operations Research 6(6): 791-812) made it a method.
With symmetric distances, only those two edges change length. genoxide's guide recommends inversion
for tours.

The acceptance is simulated annealing (Kirkpatrick, Gelatt and Vecchi, 1983, Science 220(4598): 671-680).
A shorter or equal tour is always accepted. A tour longer by Δ is accepted with probability exp(−Δ /
T), for a temperature T. T starts at 100 and is multiplied by 0.99996 after every step. At first, a
tour 100 longer is accepted with probability 37%; the optimal tour's edges are 145 long on average.
After 50,000 steps T is about 13, and after 200,000 steps about 0.03, when the search only goes
downhill. Accepting worse tours early lets it leave local optima.

The run stops at the optimum, or after 200,000 evaluations.

## Output

The first line gives the length of the best tour found and the evaluations it took. The second gives
that tour, from location 1, with the locations numbered from 1 as in TSPLIB. The tour returns from
the last location to location 1.

[The project page](https://tachsin.gr/projects/genoxide/examples/tsp-berlin52) plays this run back, on a map of Berlin's districts. TSPLIB gives the locations
no names and no real positions ("52 locations in Berlin (Groetschel)"), so their placement on the
map is for orientation: every location inside Berlin, the dense cluster in the centre.

## Good results

The optimum is 7542. The run reaches it after about 51,000 evaluations, a quarter of its budget.
