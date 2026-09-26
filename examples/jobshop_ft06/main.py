"""Job shop scheduling: Fisher and Thompson's 6×6 instance (ft06), whose shortest makespan is 55.

Six jobs each go through the six machines in their own order, and a machine does one operation at
a time. A permutation of the 36 operations, where operation ``k`` counts for job ``k // 6``, is
read as a sequence of jobs (a permutation with repetition): the n-th time a job appears, its n-th
operation starts as early as its job and its machine allow (a semi-active schedule). A genetic
algorithm with order crossover and swap mutation searches the sequences.

    python examples/jobshop_ft06/main.py
"""

import genoxide as gx

JOBS = 6
MACHINES = 6
# each job's operations in order: (machine, duration)
INSTANCE = [
    [(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)],
    [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
    [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
    [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
    [(2, 9), (1, 3), (4, 5), (5, 4), (0, 3), (3, 1)],
    [(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)],
]
OPTIMUM = 55


def schedule(order):
    """The start time of each job's operations in the semi-active schedule of ``order``."""
    starts = [[0] * MACHINES for _ in range(JOBS)]
    next_step = [0] * JOBS
    job_free = [0] * JOBS
    machine_free = [0] * MACHINES
    for operation in order.tolist():
        job = operation // MACHINES
        machine, duration = INSTANCE[job][next_step[job]]
        start = max(job_free[job], machine_free[machine])
        starts[job][next_step[job]] = start
        next_step[job] += 1
        job_free[job] = start + duration
        machine_free[machine] = start + duration
    return starts


def makespan(order):
    """When the last operation ends."""
    starts = schedule(order)
    return float(max(starts[job][-1] + INSTANCE[job][-1][1] for job in range(JOBS)))


ga = gx.Ga(
    gx.Permutation(JOBS * MACHINES),
    population_size=100,
    select=gx.Tournament(3),
    crossover=gx.OrderCrossover(),
    mutation=gx.SwapMutation(),
    objective="minimize",
    seed=1,
)
result = ga.run(makespan, target=OPTIMUM, generations=1_000)

print(
    f"makespan {result.best_fitness:.0f} after {result.generations} generations "
    f"(the optimum: {OPTIMUM})"
)
# each machine's jobs, in the order it does them
starts = schedule(result.best_genome)
for machine in range(MACHINES):
    jobs = sorted(
        (starts[job][step], job)
        for job, operations in enumerate(INSTANCE)
        for step, (on, _) in enumerate(operations)
        if on == machine
    )
    print(f"machine {machine}: jobs {[job for _, job in jobs]}")
