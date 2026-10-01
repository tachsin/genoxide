"""MMA and GCMMA: the closed-form optimum of a constrained problem, the KKT conditions with their
multipliers, the test problems' gradients evaluated in Rust, checkpoints, the errors of a wrong
fitness result, and the same run as in Rust."""

import struct

import numpy as np
import pytest

import genoxide as gx


def total(values):
    """The sum in order, as Rust adds."""
    return float(np.cumsum(values)[-1])


def volume_problem(n, budget):
    """The sum of c / x, c from 1 to 9, subject to sum(x) <= budget."""
    c = 1.0 + (np.arange(n) % 9).astype(np.float64)
    ones = np.ones((1, n))

    def volume(x):
        return total(c / x), -c / (x * x), np.array([total(x) - budget]), ones

    return c, volume


def projection(x):
    """|x - (2, 1, -1)|^2 subject to x0 + x1 + x2 <= 1, 0.8 - x1 <= 0 and x0 - x1 - 3 <= 0: the
    minimum 0.36 at (1.6, 0.8, -1.4), the first two constraints active with multipliers 0.8 and
    0.4, the third inactive."""
    p = np.array([2.0, 1.0, -1.0])
    g = np.array([x.sum() - 1.0, 0.8 - x[1], x[0] - x[1] - 3.0])
    jacobian = np.array([[1.0, 1.0, 1.0], [0.0, -1.0, 0.0], [1.0, -1.0, 0.0]])
    return float(((x - p) ** 2).sum()), 2.0 * (x - p), g, jacobian


@pytest.mark.parametrize("method", ["mma", "gcmma"])
def test_the_closed_form_optimum(method):
    n = 1_000
    c, volume = volume_problem(n, float(n))
    mma = gx.Mma(
        gx.Real((0.01, 10), length=n),
        method=method,
        initial_genome=np.full(n, 0.5),
        objective="minimize",
    )
    result = mma.run(volume, gradient=True, constraints=1, evaluations=500)
    assert result.stop_reason == "converged"
    exact = n * np.sqrt(c) / np.sqrt(c).sum()
    assert np.max(np.abs(result.best_genome - exact) / exact) < 1e-7
    assert result.violation == 0.0


@pytest.mark.parametrize("method", ["mma", "gcmma"])
def test_kkt_conditions_and_multipliers(method):
    seen = {}

    def control(running, progress):
        seen.update(
            multipliers=running.multipliers,
            converged=running.converged,
            iterations=running.iterations,
            inner=running.inner_iterations,
            kkt=running.kkt_residual,
        )

    mma = gx.Mma(
        gx.Real((-5, 5), length=3), method=method, initial_genome=[-2, 3, 4], objective="minimize"
    )
    result = mma.run(projection, gradient=True, constraints=3, evaluations=1_000, control=control)
    assert result.stop_reason == "converged"
    assert result.best_fitness == pytest.approx(0.36, abs=1e-12)
    assert np.allclose(result.best_genome, [1.6, 0.8, -1.4], atol=1e-7)
    assert seen["multipliers"] == pytest.approx([0.8, 0.4, 0.0], abs=1e-6)
    assert all(multiplier >= 0 for multiplier in seen["multipliers"])
    assert seen["converged"] in ("kkt", "step")
    assert seen["kkt"] < 1e-8
    assert seen["iterations"] > 0
    assert (seen["inner"] > 0) == (method == "gcmma")


def test_maximizing_is_minimizing_the_negated_score():
    def negated(x):
        value, gradient, g, jacobian = projection(x)
        return -value, -gradient, g, jacobian

    genome = gx.Real((-5, 5), length=3)
    minimized = gx.Mma(genome, initial_genome=[-2, 3, 4], objective="minimize").run(
        projection, gradient=True, constraints=3, evaluations=1_000
    )
    maximized = gx.Mma(genome, initial_genome=[-2, 3, 4]).run(
        negated, gradient=True, constraints=3, evaluations=1_000
    )
    assert np.array_equal(minimized.best_genome, maximized.best_genome)
    assert maximized.best_fitness == -minimized.best_fitness


def test_without_constraints_and_with_a_problem_evaluated_in_rust():
    sphere = gx.problems.Sphere(10)
    in_rust = gx.Mma(sphere.genome, method="gcmma", objective="minimize", seed=3).run(
        sphere, evaluations=500
    )
    assert in_rust.stop_reason == "converged"
    assert in_rust.best_fitness < 1e-15

    # the same function in numpy, with its gradient: the same run
    def numpy_sphere(x):
        return total(x * x), 2.0 * x

    in_python = gx.Mma(sphere.genome, method="gcmma", objective="minimize", seed=3).run(
        numpy_sphere, gradient=True, evaluations=500
    )
    assert in_python.evaluations == in_rust.evaluations
    assert np.array_equal(in_python.best_genome, in_rust.best_genome)

    # and with the gradient from a function of its own
    separate = gx.Mma(sphere.genome, method="gcmma", objective="minimize", seed=3).run(
        lambda x: total(x * x), gradient=lambda x: 2.0 * x, evaluations=500
    )
    assert np.array_equal(separate.best_genome, in_rust.best_genome)


def test_a_problem_without_a_gradient_is_an_error():
    g06 = gx.problems.cec2006.G06()
    with pytest.raises(ValueError, match="gradient"):
        gx.Mma(g06.genome, objective="minimize").run(g06, evaluations=10)


def test_wrong_results_are_errors():
    mma = gx.Mma(gx.Real((-5, 5), length=3), initial_genome=[0, 0, 0], objective="minimize")
    # MMA needs the gradient: no finite differences
    with pytest.raises(ValueError, match="gradient"):
        mma.run(lambda x: 1.0, evaluations=10)
    with pytest.raises(TypeError, match="tuple"):
        mma.run(lambda x: 1.0, gradient=True, evaluations=10)
    with pytest.raises(TypeError, match="tuple"):
        mma.run(lambda x: (1.0, x), gradient=True, constraints=3, evaluations=10)
    with pytest.raises(ValueError, match="gradient"):
        mma.run(lambda x: (1.0, np.zeros(2)), gradient=True, evaluations=10)
    with pytest.raises(ValueError, match="constraint values"):
        mma.run(
            lambda x: (1.0, x, np.zeros(2), np.zeros((2, 3))),
            gradient=True,
            constraints=3,
            evaluations=10,
        )
    with pytest.raises(ValueError, match="jacobian"):
        mma.run(
            lambda x: (1.0, x, np.zeros(3), np.zeros((3, 2))),
            gradient=True,
            constraints=3,
            evaluations=10,
        )
    # constraints come with the score and its gradient, one point at a time
    with pytest.raises(ValueError, match="gradient=True"):
        mma.run(projection, gradient=lambda x: x, constraints=3, evaluations=10)
    with pytest.raises(ValueError, match="batch"):
        mma.run(projection, gradient=True, constraints=3, batch=True, evaluations=10)
    with pytest.raises(ValueError, match="method"):
        gx.Mma(gx.Real((0, 1), length=2), method="sqp").run(
            projection, gradient=True, constraints=3, evaluations=10
        )
    with pytest.raises(ValueError, match="move_limit"):
        gx.Mma(gx.Real((0, 1), length=2), move_limit=0).run(lambda x: (0.0, x), gradient=True, evaluations=10)
    with pytest.raises(ValueError, match="initial_genome"):
        gx.Mma(gx.Real((0, 1), length=2), initial_genome=[2, 0]).run(
            lambda x: (0.0, x), evaluations=10
        )
    with pytest.raises(ValueError, match="Real genome"):
        gx.Mma(gx.Binary(4)).run(lambda x: (0.0, x), gradient=True, evaluations=10)
    # an invalid initial point
    with pytest.raises(ValueError, match="initial_genome"):
        mma.run(lambda x: (float("nan"), x), gradient=True, evaluations=10)

    def raises(x):
        raise RuntimeError("from the fitness function")

    with pytest.raises(RuntimeError, match="from the fitness function"):
        mma.run(raises, gradient=True, evaluations=10)


def test_seeded_runs_repeat():
    mma = gx.Mma(gx.Real((-5, 5), length=3), objective="minimize", seed=4)
    first = mma.run(projection, gradient=True, constraints=3, generations=3)
    again = mma.run(projection, gradient=True, constraints=3, generations=3)
    assert np.array_equal(first.best_genome, again.best_genome)
    other = gx.Mma(gx.Real((-5, 5), length=3), objective="minimize", seed=5)
    assert not np.array_equal(
        other.run(projection, gradient=True, constraints=3, generations=0).best_genome,
        mma.run(projection, gradient=True, constraints=3, generations=0).best_genome,
    )


def test_a_resumed_run_converges_as_an_uninterrupted_one(tmp_path):
    path = tmp_path / "mma.ckpt"
    mma = gx.Mma(gx.Real((-5, 5), length=3), method="gcmma", objective="minimize", seed=2)
    whole = mma.run(projection, gradient=True, constraints=3, evaluations=1_000)
    assert whole.stop_reason == "converged"
    first = mma.run(projection, gradient=True, constraints=3, generations=7, checkpoint=path, checkpoint_every=3)
    assert first.stop_reason == "generations"
    resumed = mma.run(projection, gradient=True, constraints=3, evaluations=1_000, resume=path)
    assert resumed.best_fitness == whole.best_fitness
    assert np.array_equal(resumed.best_genome, whole.best_genome)
    assert (resumed.evaluations, resumed.generations) == (whole.evaluations, whole.generations)


def test_reevaluation_after_the_function_changes():
    scale = {"factor": 1.0}

    def scaled(x):
        value, gradient, g, jacobian = projection(x)
        return scale["factor"] * value, scale["factor"] * gradient, g, jacobian

    def control(running, progress):
        if progress.generation == 3 and scale["factor"] == 1.0:
            scale["factor"] = 10.0
            running.reevaluate()

    mma = gx.Mma(gx.Real((-5, 5), length=3), initial_genome=[-2, 3, 4], objective="minimize")
    result = mma.run(scaled, gradient=True, constraints=3, evaluations=1_000, control=control)
    assert result.stop_reason == "converged"
    assert result.best_fitness == pytest.approx(3.6, abs=1e-10)


def test_a_run_as_in_rust():
    # tests/mma.rs's a_portable_result: GCMMA on 50 genes for 12 generations, the same bits
    n = 50
    _, volume = volume_problem(n, float(n))
    result = gx.Mma(
        gx.Real((0.01, 10), length=n),
        method="gcmma",
        initial_genome=np.full(n, 0.5),
        objective="minimize",
    ).run(volume, gradient=True, constraints=1, generations=12)
    current = {}

    def last(progress):
        current["x"] = progress.population[0]

    gx.Mma(
        gx.Real((0.01, 10), length=n),
        method="gcmma",
        initial_genome=np.full(n, 0.5),
        objective="minimize",
    ).run(volume, gradient=True, constraints=1, generations=12, on_generation=last)
    x = current["x"]
    bits = [struct.unpack("<Q", struct.pack("<d", value))[0] for value in (x[0], x[17], x[49])]
    assert bits == [0x3FDF1AE5703113A6, 0x3FF67451FE1277ED, 0x3FF12760DAD20B0D]
    assert result.best_fitness == struct.unpack("<d", struct.pack("<Q", 0x406B84D0583E68B4))[0]
