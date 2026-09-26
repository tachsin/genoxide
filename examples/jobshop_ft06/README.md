---
title: Job shop scheduling (ft06)
category: permutation
summary: Schedule 6 jobs of 6 operations on 6 machines in the shortest time, Fisher and Thompson's 6×6 instance.
reference: "Fisher, H. and Thompson, G. L. (1963). Probabilistic learning combinations of local job-shop scheduling rules. In Muth, J. F. and Thompson, G. L. (eds.), Industrial Scheduling, pp. 225-251. Prentice-Hall."
reference_url: http://people.brunel.ac.uk/~mastjjb/jeb/orlib/jobshopinfo.html
optimum: "55 (makespan)"
languages: [rust, python]
order: 50
---

# Job shop scheduling (ft06)

In the job shop problem, each job goes through the machines in its own order, a machine does one
operation at a time, and the goal is the shortest makespan: the time when the last operation ends.
ft06 has 6 jobs and 6 machines, and its optimal makespan is 55. The example uses a permutation of
the 36 operations as a sequence of jobs (operation k counts for job k / 6), and decodes it into a
semi-active schedule: the n-th time a job appears, its n-th operation starts as early as its job
and its machine allow. A genetic algorithm with a population of 100, order crossover and swap
mutation searches the sequences until it reaches 55; the example prints the best makespan and each machine's jobs in
order.
