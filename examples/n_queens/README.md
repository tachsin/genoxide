---
title: N-Queens
category: permutation
summary: Place 64 queens on a 64×64 chessboard so that no two attack each other.
reference: "Bezzel, M. (1848). Zwei Schachfragen. Schachzeitung der Berliner Schachgesellschaft 3: 363 (signed 'Schachfreund')."
reference_url: null
optimum: "0 (conflicts)"
languages: [rust, python]
order: 30
---

# N-Queens

## The problem

N-Queens places N queens on an N×N chessboard so that no two attack each other: no two share a row,
a column or a diagonal. Bezzel (1848) posed it for 8 queens on the usual board. The 8×8 board has 92
solutions, and every board from 4×4 up has at least one (Bell and Stevens, 2009, Discrete
Mathematics 309(1): 1-31). Here N = 64.

On a 4×4 board, queens in rows 1 to 4 at columns 2, 4, 1 and 3 are a solution: no two share a
column, and no two are on a common diagonal.

## What makes it hard

The number of placements is huge: 64 queens on 4,096 squares. Even with one queen per row and
column, there are 64! ≈ 1.3 × 10⁸⁹ placements.

The fitness is a count of conflicts, a small integer, and many placements share it. A swap moves two
queens, which can fix one diagonal and break another. The search crosses plateaus of equal fitness
on its way to 0.

## Representation

A `Permutation` of 0 to 63: entry r is the column of the queen in row r. Each row then has one
queen, and each column has one, since a permutation repeats no value. Row and column conflicts are
impossible, and only the diagonals are left.

The fitness counts the pairs of queens on a common diagonal, to minimize. A diagonal with k queens
adds k(k − 1)/2 pairs. There are 127 diagonals in each direction. The Python version counts them
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
and columns numbered from 0. Every number from 0 to 63 appears once.

[The project page](https://tachsin.gr/projects/genoxide/examples/n-queens) plays this run back.

## Good results

The optimum is 0 conflicts. The run reaches it after about 300 generations and 6,000 evaluations.
