---
title: OneMax
category: binary
summary: Find the bit string with the most ones, the "hello world" of genetic algorithms.
reference: "Ackley, D. H. (1987). A Connectionist Machine for Genetic Hillclimbing. Kluwer Academic Publishers."
reference_url: https://doi.org/10.1007/978-1-4613-1997-9
optimum: "500 (all ones)"
languages: [rust, python]
order: 10
---

# OneMax

OneMax counts the ones in a bit string, here of 500 bits, so the optimum is the string of all ones.
The example uses a binary genome and a genetic algorithm with a population of 100, tournament
selection of size 3, uniform crossover and bit-flip mutation at a rate of 1/500. It stops at the
optimum and prints the best count every 50 generations, then the generations and evaluations it
took.
