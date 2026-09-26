"""Pressure vessel design (Sandgren, 1990): the cheapest cylindrical vessel with hemispherical heads
that holds 1,296,000 cubic inches, a constrained mixed discrete-continuous problem. The best known
cost is 6059.714335.

The variables are the thickness of the shell and of the heads, multiples of 0.0625 inch, and the
inner radius and the length of the shell. Genes 0 and 1 are real numbers from 1 to 99, rounded to
the nearest whole number of 0.0625-inch plates. The fitness function returns the cost and the
violation of the four constraints, which Deb's feasibility rules compare. SHADE, a differential
evolution, searches the genes.

    python examples/pressure_vessel/main.py
"""

import math

import genoxide as gx

BEST_KNOWN = 6059.714335


def design(x):
    """The thicknesses of the shell and the heads, the radius and the length."""
    return math.floor(x[0] + 0.5) * 0.0625, math.floor(x[1] + 0.5) * 0.0625, x[2], x[3]


def at_least(value, limit):
    """The violation of ``value >= limit``: how far ``value`` is below ``limit``, or 0."""
    return max(0.0, limit - value)


def cost(x):
    """The cost of the material, forming and welding, and the violation of the constraints."""
    shell, head, radius, length = design(x.tolist())
    cost = (
        0.6224 * shell * radius * length
        + 1.7781 * head * radius * radius
        + 3.1661 * shell * shell * length
        + 19.84 * shell * shell * radius
    )
    volume = math.pi * radius * radius * length + 4.0 / 3.0 * math.pi * radius * radius * radius
    # the volume constraint relative to the volume, of the order of the others
    violation = (
        at_least(shell, 0.0193 * radius)
        + at_least(head, 0.00954 * radius)
        + at_least(volume, 1_296_000.0) / 1_296_000.0
        + at_least(240.0, length)
    )
    return cost, violation


de = gx.De(gx.Real([(1, 99), (1, 99), (10, 200), (10, 200)]), objective="minimize", seed=1)
result = de.run(cost, evaluations=50_000)

shell, head, radius, length = design(result.best_genome.tolist())
print(
    f"cost {result.best_fitness:.6f} after {result.evaluations} evaluations "
    f"(the best known: {BEST_KNOWN})"
)
print(f"violation {result.violation:.6f}")
print(f"shell {shell:.4f}, heads {head:.4f}, radius {radius:.6f}, length {length:.6f}")
