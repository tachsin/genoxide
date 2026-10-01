"""L-BFGS-B: convergence, the sources of the gradient and their cost, bounds, its settings and
control, and the same run as in Rust."""

import math

import numpy as np
import pytest

import genoxide as gx


def rosenbrock(x):
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


def rosenbrock_gradient(x):
    gradient = np.zeros_like(x)
    valley = x[1:] - x[:-1] ** 2
    gradient[:-1] += -400.0 * x[:-1] * valley - 2.0 * (1.0 - x[:-1])
    gradient[1:] += 200.0 * valley
    return gradient


def classic_start(n):
    return [-1.2, 1.0] * (n // 2)


def lbfgsb(n=10, **settings):
    settings.setdefault("initial_genome", classic_start(n))
    return gx.Lbfgsb(gx.Real((-5, 5), length=n), objective="minimize", **settings)


def test_a_problem_converges_with_its_gradient_in_rust():
    problem = gx.problems.Rosenbrock(100)
    result = gx.Lbfgsb(
        problem.genome,
        initial_genome=classic_start(100),
        gradient_tolerance=1e-8,
        function_tolerance=0.0,
        objective="minimize",
    ).run(problem, evaluations=10_000)
    assert result.stop_reason == "converged"
    assert result.best_fitness < 1e-16
    assert np.allclose(result.best_genome, 1.0, atol=1e-7)
    # one evaluation per iteration or line-search trial: far below forward differences' cost
    assert result.evaluations < 1_000


def test_every_way_to_give_the_gradient_takes_the_same_path():
    def both(x):
        return rosenbrock(x), rosenbrock_gradient(x)

    def batch(genomes):
        return np.array([rosenbrock(x) for x in genomes])

    def batch_gradient(genomes):
        return np.array([rosenbrock_gradient(x) for x in genomes])

    def batch_both(genomes):
        return batch(genomes), batch_gradient(genomes)

    runs = [
        lbfgsb().run(rosenbrock, gradient=rosenbrock_gradient, evaluations=5_000),
        lbfgsb().run(both, gradient=True, evaluations=5_000),
        lbfgsb().run(batch, gradient=batch_gradient, batch=True, evaluations=5_000),
        lbfgsb().run(batch_both, gradient=True, batch=True, evaluations=5_000),
        # a list is a gradient too
        lbfgsb().run(rosenbrock, gradient=lambda x: list(rosenbrock_gradient(x)), evaluations=5_000),
    ]
    first = runs[0]
    assert first.stop_reason == "converged"
    assert first.best_fitness < 1e-9
    for run in runs[1:]:
        assert run.best_fitness == first.best_fitness
        assert np.array_equal(run.best_genome, first.best_genome)
        assert run.evaluations == first.evaluations


def test_the_gradients_cost_is_visible():
    seen = {}

    def control(algorithm, progress):
        seen.update(
            gradients=algorithm.gradients,
            gradient_evaluations=algorithm.gradient_evaluations,
            stencil_evaluations=algorithm.stencil_evaluations,
            iterations=algorithm.iterations,
            converged=algorithm.converged,
            projected_gradient=algorithm.projected_gradient,
            pairs=algorithm.pairs,
        )

    supplied = lbfgsb().run(
        rosenbrock, gradient=rosenbrock_gradient, evaluations=50_000, control=control
    )
    assert seen["gradients"] == "supplied"
    assert seen["stencil_evaluations"] == 0
    assert seen["gradient_evaluations"] == supplied.evaluations
    assert seen["converged"] in ("projected_gradient", "relative_decrease")
    assert 0 < seen["pairs"] <= 10
    # without a gradient: forward differences, n points more per gradient, in one generation
    forward = lbfgsb().run(rosenbrock, evaluations=50_000, control=control)
    assert seen["gradients"] == "forward"
    assert seen["stencil_evaluations"] == 10 * seen["gradient_evaluations"]
    assert forward.evaluations == 11 * seen["gradient_evaluations"]
    assert forward.generations + 1 == seen["gradient_evaluations"]
    assert forward.best_fitness < 1e-8
    assert forward.evaluations > 5 * supplied.evaluations
    central = lbfgsb(gradients="central").run(
        rosenbrock, gradient=rosenbrock_gradient, evaluations=50_000, control=control
    )
    assert seen["gradients"] == "central"
    assert central.evaluations == 21 * seen["gradient_evaluations"]


def test_a_supplied_gradient_must_be_given():
    with pytest.raises(ValueError, match="gradients"):
        lbfgsb(gradients="supplied").run(rosenbrock, evaluations=100)


def test_a_run_as_in_rust():
    # tests/lbfgsb.rs has the same run, with the same results
    problem = gx.problems.Rosenbrock(4)
    result = gx.Lbfgsb(
        problem.genome, memory=5, restarts=2, objective="minimize", seed=5
    ).run(problem, evaluations=20_000)
    assert result.stop_reason == "converged"
    assert (result.evaluations, result.generations) == (271, 270)
    assert result.best_fitness == 2.059739817072154e-12


def test_maximizing_takes_the_same_path_as_minimizing_the_negation():
    minimized = lbfgsb(seed=3, initial_genome=None).run(
        rosenbrock, gradient=rosenbrock_gradient, evaluations=5_000
    )
    maximized = gx.Lbfgsb(gx.Real((-5, 5), length=10), objective="maximize", seed=3).run(
        lambda x: -rosenbrock(x), gradient=lambda x: -rosenbrock_gradient(x), evaluations=5_000
    )
    assert np.array_equal(minimized.best_genome, maximized.best_genome)
    assert minimized.best_fitness == -maximized.best_fitness


def test_a_minimum_on_the_bounds_is_landed_on():
    # the sphere centered at 3 in [-2, 2]: the minimum at the corner (2, 2, 2), every point in
    # the box
    outside = []

    def shifted(x):
        if np.any(x < -2) or np.any(x > 2):
            outside.append(x.copy())
        return float(np.sum((x - 3.0) ** 2)), 2.0 * (x - 3.0)

    result = gx.Lbfgsb(gx.Real((-2, 2), length=3), objective="minimize", seed=1).run(
        shifted, gradient=True, evaluations=1_000
    )
    assert result.stop_reason == "converged"
    assert list(result.best_genome) == [2.0, 2.0, 2.0]
    assert result.best_fitness == 3.0
    assert not outside


def test_the_memory_can_change_and_a_reevaluation_drops_the_pairs():
    memories = []

    def control(algorithm, progress):
        if progress.generation == 5:
            algorithm.memory = 3
        if progress.generation == 8:
            algorithm.reevaluate()
        memories.append((algorithm.memory, algorithm.pairs))

    result = lbfgsb(n=20, memory=8).run(
        rosenbrock, gradient=rosenbrock_gradient, evaluations=5_000, control=control
    )
    assert result.stop_reason == "converged"
    assert memories[4][0] == 8 and memories[5][0] == 3
    assert all(pairs <= 3 for _, pairs in memories[5:])
    with pytest.raises(ValueError, match="memory"):

        def zero(algorithm, progress):
            algorithm.memory = 0

        lbfgsb().run(rosenbrock, gradient=rosenbrock_gradient, evaluations=100, control=zero)


def test_restarts_and_the_criterion_of_a_run():
    criteria = []

    def control(algorithm, progress):
        criteria.append((algorithm.restart_count, algorithm.converged))

    problem = gx.problems.Rastrigin(2)
    result = gx.Lbfgsb(problem.genome, restarts=40, objective="minimize", seed=2).run(
        problem, evaluations=100_000, control=control
    )
    assert result.stop_reason == "converged"
    assert criteria[-1][0] == 40
    assert criteria[-1][1] is not None
    assert result.best_fitness < 1.0


def test_a_resumed_run_converges_as_an_uninterrupted_one(tmp_path):
    path = tmp_path / "lbfgsb.ckpt"
    whole = lbfgsb(seed=4).run(rosenbrock, gradient=rosenbrock_gradient, evaluations=5_000)
    first = lbfgsb(seed=4).run(
        rosenbrock,
        gradient=rosenbrock_gradient,
        generations=17,
        checkpoint=path,
        checkpoint_every=17,
    )
    assert first.generations == 17
    resumed = lbfgsb(seed=4).run(
        rosenbrock, gradient=rosenbrock_gradient, evaluations=5_000, resume=path
    )
    assert resumed.best_fitness == whole.best_fitness
    assert np.array_equal(resumed.best_genome, whole.best_genome)
    assert resumed.evaluations == whole.evaluations


def test_wrong_gradients_are_errors():
    with pytest.raises(ValueError, match="own gradient"):
        lbfgsb().run(gx.problems.Rosenbrock(10), gradient=rosenbrock_gradient, evaluations=10)
    with pytest.raises(TypeError, match="gradient"):
        lbfgsb().run(rosenbrock, gradient=3, evaluations=10)
    with pytest.raises(TypeError, match="tuple"):
        lbfgsb().run(rosenbrock, gradient=True, evaluations=10)
    with pytest.raises(ValueError, match="9 values, for 10 genes"):
        lbfgsb().run(rosenbrock, gradient=lambda x: np.zeros(9), evaluations=10)
    with pytest.raises(ValueError, match="shape"):
        lbfgsb().run(
            lambda genomes: np.zeros(len(genomes)),
            gradient=lambda genomes: np.zeros((len(genomes), 3, 2)),
            batch=True,
            evaluations=10,
        )

    def failing(x):
        raise RuntimeError("no gradient here")

    with pytest.raises(RuntimeError, match="no gradient here"):
        lbfgsb().run(rosenbrock, gradient=failing, evaluations=10)


def test_an_invalid_value_is_stepped_back_from():
    # undefined (None) beyond x0 = 1.5: the run ends below it
    def bounded(x):
        if x[0] > 1.5:
            return None, np.zeros_like(x)
        return float(np.sum((x - 2.0) ** 2)), 2.0 * (x - 2.0)

    result = gx.Lbfgsb(
        gx.Real((-5, 5), length=2), initial_genome=[0.0, 0.0], objective="minimize"
    ).run(bounded, gradient=True, evaluations=1_000)
    assert result.stop_reason == "converged"
    assert 1.0 < result.best_genome[0] <= 1.5
    assert math.isfinite(result.best_fitness)


@pytest.mark.parametrize(
    ("settings", "message"),
    [
        ({"memory": 0}, "memory"),
        ({"gradients": "exact"}, "gradients"),
        ({"difference_step": 1e-6}, "difference_step"),
        ({"gradients": "forward", "difference_step": 0.0}, "gradients"),
        ({"gradient_tolerance": -1.0}, "gradient_tolerance"),
        ({"function_tolerance": math.inf}, "function_tolerance"),
        ({"max_line_search": 0}, "max_line_search"),
        ({"restarts": 0}, "restarts"),
        ({"initial_genome": [9.0] * 10}, "initial_genome"),
    ],
)
def test_invalid_settings_are_errors(settings, message):
    with pytest.raises(ValueError, match=message):
        lbfgsb(**settings).run(rosenbrock, evaluations=10)


def test_genomes_other_than_real_are_errors():
    with pytest.raises(ValueError, match="Real"):
        gx.Lbfgsb(gx.Binary(5)).run(lambda bits: 0.0, evaluations=10)
