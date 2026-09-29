---
title: N-Queens 16×16
category: permutation
summary: Place 16 queens on a 16×16 chessboard so that no two attack each other.
reference: "Bezzel, M. (1848). Zwei Schachfragen. Schachzeitung der Berliner Schachgesellschaft 3: 363 (signed 'Schachfreund')."
reference_url: null
optimum: "0 (conflicts)"
languages: [rust, python]
order: 31
family: N-Queens
tab: 16×16
---

# N-Queens 16×16

## The problem

N-Queens places N queens on an N×N chessboard so that no two attack each other: no two share a row,
a column or a diagonal. Bezzel (1848) posed it for 8 queens on the usual board. Every board from 4×4
up has at least one solution (Bell and Stevens, 2009, Discrete Mathematics 309(1): 1-31). Here
N = 16: the board has 14,772,512 solutions.

On a 4×4 board, queens in rows 1 to 4 at columns 2, 4, 1 and 3 are a solution: no two share a
column, and no two are on a common diagonal.

The same search solves the [8×8](../n_queens_8/), [32×32](../n_queens_32/), [64×64](../n_queens/)
and [128×128](../n_queens_128/) boards.

## What makes it hard

Even with one queen per row and column (below), there are 16! ≈ 2.1 × 10¹³ placements, and 1 in 1.4
million is a solution: a random search would try about as many. A random placement has 10.3
attacking pairs on average.

The fitness is a count of conflicts, a small integer, and many placements share it. A swap moves two
queens, which can fix one diagonal and break another. The search crosses plateaus of equal fitness
on its way to 0.

The difficulty grows with N. The placements grow as N!, and the solutions nearly as fast: for large
N, there are about (0.143 N)ᴺ (Simkin, 2021, arXiv:2107.13460). Nobody has counted them beyond
27×27. With the settings below:

| Board | Placements (N!) | Solutions | Attacking pairs of a random placement | Median evaluations to a solution |
|---|---|---|---|---|
| 8×8 | 40,320 | 92 | 5 | 100 |
| 16×16 | 2.1 × 10¹³ | 14,772,512 | 10.3 | 729 |
| 32×32 | 2.6 × 10³⁵ | not counted | 21 | 2,450 |
| 64×64 | 1.3 × 10⁸⁹ | not counted | 42.3 | 7,110 |
| 128×128 | 3.9 × 10²¹⁵ | not counted | 85 | 16,590 |

A random placement has (2N − 1)/3 attacking pairs on average. The evaluations are the median over
seeds 1 to 100, and each takes time in proportion to N.

## Representation

A `Permutation` of 0 to 15: entry r is the column of the queen in row r. Each row then has one
queen, and each column has one, since a permutation repeats no value. Row and column conflicts are
impossible, and only the diagonals are left.

The fitness counts the pairs of queens on a common diagonal, to minimize. A diagonal with k queens
adds k(k − 1)/2 pairs. There are 31 diagonals in each direction. The Python version counts them
with numpy, a generation at a time.

## Algorithm

A (μ+λ) genetic algorithm without crossover:

- 20 parents, chosen by tournaments of size 2, make 20 children;
- each child is its parent with two entries swapped (swap mutation): two queens trade columns, and
  the genome stays a permutation;
- parents and children compete, and the best 20 survive.

A crossover can't take each row's column from either parent freely: the child would repeat columns.
genoxide's guide suggests (μ+λ) for mutation-only searches and plateaus. The best placement always
survives. On ties, genoxide keeps the children rather than the parents, so the population drifts
across plateaus. The run stops at 0 conflicts, or after 50,000 generations.

## Output

The first line gives the conflicts of the best placement, and the generations and evaluations it
took. The second gives the columns: for each row from the first, the column of its queen, with rows
and columns numbered from 0. Every number from 0 to 15 appears once.

[The project page](https://tachsin.gr/projects/genoxide/examples/n-queens-16) plays this run back.

## Good results

The optimum is 0 conflicts. The run reaches it after 52 generations and 1,060 evaluations. Over
seeds 1 to 100, every run reached it, after 729 evaluations at the median and at most 2,478; over
seeds 1 to 1,000, every run too, after at most 3,916.
