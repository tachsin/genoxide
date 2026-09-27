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

## The problem

In the job shop problem, each job is a sequence of operations, and each operation needs one machine
for a fixed time. A job visits the machines in its own order, one operation at a time. A machine
does one operation at a time, without interruption. The goal is the shortest makespan: the time when
the last operation ends.

ft06 is the 6×6 instance of Fisher and Thompson (1963): 6 jobs, each with one operation on each of 6
machines, of 1 to 10 time units. Job 0, for example, takes 1 unit on machine 2, then 3 on machine 0,
6 on machine 1, 7 on machine 3, 3 on machine 5 and 6 on machine 4. The optimal makespan is 55.

## What makes it hard

The job shop problem is NP-hard (Garey, Johnson and Sethi, 1976, Mathematics of Operations Research
1(2): 117-129). Each machine's order of its 6 operations can be chosen, but some combinations of
machine orders deadlock: two machines each wait for the other.

The obvious bounds are far from the optimum. The longest job takes 47 units in all, and the busiest
machine is busy for 43. The optimum, 55, has idle time that no schedule avoids.

## Representation

A `Permutation` of the 36 operations, read as a sequence of jobs: entry k counts for job k / 6,
rounded down, so each job appears 6 times. This permutation with repetition is Bierwirth's encoding
(1995, OR Spektrum 17: 87-92). The n-th time a job appears, its n-th operation is scheduled.

The decoder starts each operation as early as its job and its machine allow, in the order of the
sequence: a semi-active schedule. Every sequence gives a valid schedule, since the order within each
job is kept and nothing can deadlock. Listing an optimal schedule's operations by start time gives a
sequence that decodes to a schedule at least as short, so the optimum is reachable.

The six operations of a job are interchangeable in the sequence. The 36! permutations give 36! /
(6!)⁶ ≈ 2.7 × 10²⁴ different sequences, and many sequences give the same schedule. The fitness is
the makespan, to minimize.

## Algorithm

A genetic algorithm with the operators that genoxide's guide lists for sequences:

- a population of 100;
- tournament selection of size 3;
- order crossover: the child keeps a segment of one parent, and takes the other entries in the
  order they have in the other parent. It goes back to the modified crossover of Davis (1985,
  Proceedings of IJCAI-85: 162-164), which keeps the first part of a parent instead of a segment;
- swap mutation, which exchanges two entries;
- the default generational scheme, which keeps the best individual.

The run stops at the optimum, 55, or after 1,000 generations.

## Output

The first line gives the best makespan and the generations it took. The next six give each machine's
jobs in the order the machine does them, with the jobs and machines numbered from 0. Together, those
orders define the schedule.

The project page plays this run back.

## Good results

The optimum is 55. The run reaches it after about 120 generations.
