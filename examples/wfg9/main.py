"""WFG9: minimize two objectives whose concave front lies behind multimodal, deceptive and
non-separable parameters, with NSGA-II and SMS-EMOA.

The Walking Fish Group's ninth problem, from genoxide's problems.Wfg9, with 2 objectives, 4
position and 20 distance parameters; run evaluates it in Rust. Runs NSGA-II with simulated binary
crossover for 1,000 generations, and SMS-EMOA with blend crossover for 5,000, and prints their
fronts after 1,000 and 5,000: the size, the IGD+ to 500 points of the optimal front, the
hypervolume, and how far the solutions are from the front.

With ``GENOXIDE_TRACE=<file>``, it also writes a trace of its runs for the plot on the example's
page, with trace.py.

    python examples/wfg9/main.py
"""

import math

import genoxide as gx

from trace import Trace

# the reference point of the hypervolume: 1.1 times the front's nadir point (2, 4)
REFERENCE = [2.2, 4.4]

# the generation of NSGA-II's report, and of SMS-EMOA's first
FIRST = 1_000

# the generation of SMS-EMOA's last report
LAST = 5_000

# 2 objectives, k = 4 position and l = 20 distance parameters
problem = gx.problems.Wfg9(2)
optimal = problem.optimal_front(500)
# with GENOXIDE_TRACE=<file>, a trace of the runs for the plot on the example's page
trace = Trace(REFERENCE)


def distance(f1, f2):
    """How far a point is from the front: the d with (f₁ − d, f₂ − d) on it, where
    ((f₁ − d) / 2)² + ((f₂ − d) / 4)² = 1, the smaller root of that quadratic. WFG adds the
    distance parameters' value, x_M, to both objectives: d is that value."""
    a = 0.25 + 0.0625
    b = f1 / 2 + f2 / 8
    c = f1 * f1 / 4 + f2 * f2 / 16 - 1
    return max((b - math.sqrt(b * b - 4 * a * c)) / (2 * a), 0.0)


def report(name, generations, front):
    """Prints the size of a front, its IGD+ to the optimal front, also with the objectives scaled
    to [0, 1] over the front's ranges, 2 and 4, its hypervolume, and the distances of its points
    from the front."""
    igd = gx.indicators.igd_plus(front, optimal)
    scaled = gx.indicators.igd_plus(front / [2.0, 4.0], optimal / [2.0, 4.0])
    volume = gx.indicators.hypervolume(front, REFERENCE)
    distances = [distance(f1, f2) for f1, f2 in front.tolist()]
    total = 0.0
    for d in distances:
        total += d
    print(
        f"{name:<8} after {generations} generations: {len(front)} solutions, IGD+ {igd:.4f} "
        f"(scaled {scaled:.4f}), hypervolume {volume:.4f}"
    )
    print(
        f"         distance from the front {min(distances):.4f} to {max(distances):.4f}, "
        f"mean {total / len(distances):.4f}"
    )


def run(name, algorithm, reports):
    """Runs ``algorithm`` up to the last of ``reports``, and reports its front after each of
    them."""
    record = trace.fronts(name)
    fronts = []

    def on_generation(progress):
        if progress.generation in reports:
            fronts.append(progress.front_objectives)
        if record:
            record(progress)

    algorithm.run(problem, generations=reports[-1], on_generation=on_generation)
    for generations, front in zip(reports, fronts):
        report(name, generations, front)


# polynomial mutation at a rate of 1/24, one gene per child on average
settings = dict(
    objectives=problem.objectives,
    population_size=100,
    mutation=gx.PolynomialMutation(20, rate=1 / 24),
    seed=1,
)
nsga2 = gx.Nsga2(problem.genome, crossover=gx.SimulatedBinaryCrossover(15), **settings)
run("NSGA-II", nsga2, [FIRST])
# blend crossover: each gene of a child drawn from the parents' interval, widened by 0.3 of its
# length on each side
sms_emoa = gx.SmsEmoa(problem.genome, crossover=gx.BlendCrossover(0.3), **settings)
run("SMS-EMOA", sms_emoa, [FIRST, LAST])

# the box up to the reference point, less the quarter ellipse under the front, of area 2π
whole = REFERENCE[0] * REFERENCE[1] - 2 * math.pi
print(f"the whole front: hypervolume {whole:.4f}")
trace.write()
