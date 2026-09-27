"""Job shop scheduling: Fisher and Thompson's 6×6 instance (ft06), whose shortest makespan is 55.

Six jobs each go through the six machines in their own order, and a machine does one operation at
a time. A permutation of the 36 operations, where operation ``k`` counts for job ``k // 6``, is
read as a sequence of jobs (a permutation with repetition): the n-th time a job appears, its n-th
operation starts as early as its job and its machine allow (a semi-active schedule). A genetic
algorithm with order crossover and swap mutation searches the sequences.

With ``GENOXIDE_TRACE=<file>``, it also writes the run's trace for the plot on the example's page:
the best schedule so far, in at most 200 generations.

    python examples/jobshop_ft06/main.py
"""

import json
import math
import os

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


# ---- the trace of the run, for the plot on the example's page -----------------------------------


class Trace:
    """A frame per recorded generation, at most ``most``: every ``every``-th generation, with
    ``every`` doubling whenever there are ``most``, and the last generation."""

    def __init__(self, most):
        self.most, self.every, self.frames, self.last = most, 1, [], None

    def record(self, progress, state):
        """The generation's progress, the median score of its population and the plot's
        ``state``."""
        frame = {
            "generation": progress.generation,
            "evaluations": progress.evaluations,
            "best": progress.best_fitness,
            "median": median(progress.scores),
            "state": state,
        }
        self.push(frame)

    def push(self, frame):
        """Keeps ``frame`` if it's of the ``every``-th generation, or as the last one."""
        if frame["generation"] % self.every:
            self.last = frame
            return
        self.frames.append(frame)
        self.last = None
        if len(self.frames) == self.most:
            self.every *= 2
            self.frames = [frame for frame in self.frames if frame["generation"] % self.every == 0]

    def write(self, path, settings):
        """Writes the settings and the frames to ``path``, a frame per line."""
        frames = ",\n".join(to_json(frame) for frame in self.frames + [self.last] if frame)
        with open(path, "w", encoding="utf-8", newline="\n") as file:
            file.write(f'{to_json(settings)[:-1]},"frames":[\n{frames}\n]}}\n')


def median(scores):
    """The median of the valid scores, None without any."""
    scores = sorted(float(score) for score in scores if not math.isnan(score))
    middle = len(scores) // 2
    if not scores:
        return None
    return scores[middle] if len(scores) % 2 else (scores[middle - 1] + scores[middle]) / 2


def to_json(value):
    """Compact JSON with sorted keys, and numbers rounded to 6 significant digits, as the Rust
    example writes it."""
    return json.dumps(rounded(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def rounded(value):
    if isinstance(value, dict):
        return {key: rounded(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(item) for item in value]
    if isinstance(value, float):
        return float(f"{value:.5e}") if math.isfinite(value) else None
    return value


# -------------------------------------------------------------------------------------------------

ga = gx.Ga(
    gx.Permutation(JOBS * MACHINES),
    population_size=100,
    select=gx.Tournament(3),
    crossover=gx.OrderCrossover(),
    mutation=gx.SwapMutation(),
    objective="minimize",
    seed=1,
)


def operations(order):
    """The operations of the schedule of ``order``: [job, operation, machine, start, end]."""
    starts = schedule(order)
    return [
        [job, step, machine, starts[job][step], starts[job][step] + duration]
        for job, steps in enumerate(INSTANCE)
        for step, (machine, duration) in enumerate(steps)
    ]


trace = Trace(200) if "GENOXIDE_TRACE" in os.environ else None


def record(progress):
    if trace:
        trace.record(progress, {"best": operations(progress.best_genome)})


result = ga.run(makespan, target=OPTIMUM, generations=1_000, on_generation=record)

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
if trace:
    trace.write(
        os.environ["GENOXIDE_TRACE"],
        {
            "format": 1,
            "example": "jobshop_ft06",
            "objective": "minimize",
            "x_label": "generations",
            "y_label": "makespan",
            "log_y": False,
            "optimum": float(OPTIMUM),
            "plot": "gantt",
            "problem": {"machines": MACHINES, "jobs": INSTANCE},
        },
    )
