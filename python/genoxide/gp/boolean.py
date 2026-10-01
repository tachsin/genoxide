"""Boolean problems: Koza's multiplexer and even-parity functions, learned from their whole truth
tables, evaluated in Rust.

Each is a fitness function for trees of its paper's primitives, from its ``primitives()``: the
number of cases of the truth table that a tree gets wrong, minimized, 0 at the optimum (Koza's
standardized fitness), 64 cases at once::

    import genoxide as gx

    # the 6-multiplexer: 2 address bits select one of 4 data bits
    problem = gx.gp.boolean.Multiplexer(2)
    ga = gx.Ga(
        gx.gp.Gp(problem.primitives()),
        population_size=500,
        select=gx.Tournament(7),
        crossover=gx.gp.SubtreeCrossover(),
        mutation=gx.gp.SubtreeMutation(),
        mutation_rate=0.1,
        objective="minimize",
        seed=1,
    )
    result = ga.run(problem, target=0, generations=50)
    print(result.best_fitness, result.best_genome)  # 0.0: all 64 cases right

They are among the problems McDermott et al. (2012) found overused; they stay genetic
programming's classic tests with an exact optimum. ``outputs(tree)`` and ``targets()`` give the
truth tables, 64 cases per word: case ``c`` at bit ``c % 64`` of word ``c // 64``, input ``i`` (in
the order of the set's terminals) being bit ``i`` of ``c``. A tree's ``evaluate(x)`` gives its
outputs at any inputs, a row each.
"""

from __future__ import annotations

from .. import _genoxide, _whole

__all__ = ["BooleanProblem", "Multiplexer", "EvenParity"]

BooleanProblem = _genoxide.BooleanProblem


class Multiplexer(BooleanProblem):
    """Koza's (1992) Boolean multiplexer: ``address_bits`` address bits, 1 to 4, select one of
    ``2^address_bits`` data bits, which is the output. 3 is Koza's 11-multiplexer, 2048 cases;
    2 the 6-multiplexer, 64 cases.

    Its primitives are Koza's: the functions ``and``, ``or``, ``not`` and ``if`` (of three
    arguments: the second if the first is true, else the third), and the inputs ``a0`` to
    ``a{k-1}`` (the address, ``a0`` its least significant bit), then ``d0`` to ``d{2^k-1}`` (the
    data). One type: the untyped genetic programming of the paper.
    """

    def __new__(cls, address_bits: int = 3) -> Multiplexer:
        bits = _whole("Multiplexer.address_bits", address_bits)
        return super().__new__(cls, "multiplexer", bits)

    @property
    def address_bits(self) -> int:
        """The number of address bits."""
        return int(super().address_bits or 0)


class EvenParity(BooleanProblem):
    """Koza's (1992) even-parity function: true when an even number of the ``inputs`` inputs, 2
    to 20, are true. Even-3 to even-5 are the usual sizes; each input more doubles the cases.

    Its primitives are Koza's: the functions ``and``, ``or``, ``nand`` and ``nor``, and the
    inputs ``d0`` to ``d{n-1}``. One type.
    """

    def __new__(cls, inputs: int) -> EvenParity:
        return super().__new__(cls, "even_parity", _whole("EvenParity.inputs", inputs))
