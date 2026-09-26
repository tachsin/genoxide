---
title: N-Queens
category: permutation
summary: Place 64 queens on a 64×64 chessboard so that no two attack each other.
reference: "Bezzel, M. (1848). Schachfreund. Berliner Schachzeitung 3, p. 363."
reference_url: null
optimum: "0 (conflicts)"
languages: [rust, python]
order: 30
---

# N-Queens

N-Queens places N queens on an N×N board, here N = 64, so that no two share a row, a column or a
diagonal. A permutation genome gives the column of the queen in each row, so rows and columns never
conflict, and the fitness is the number of pairs of queens on the same diagonal, to minimize.
Permutations have no position-wise crossover, so the example uses a (μ+λ) genetic algorithm with
20 parents, 20 children and swap mutation only. It stops at 0 conflicts and prints the generations
and evaluations it took and the column of each queen.
