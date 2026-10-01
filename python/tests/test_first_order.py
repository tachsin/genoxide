"""First-order methods: gradients from Python and from the problems, the step rules, schedules
by control, bounds, settings, and the same runs as in Rust."""

import numpy as np
import pytest

import genoxide as gx

T = np.linspace(0.0, 1.0, 50)
Y = 2.0 * T - 1.0


def loss(p):
    return float(np.sum((p[0] * T + p[1] - Y) ** 2))


def gradient(p):
    r = p[0] * T + p[1] - Y
    return np.array([2.0 * np.sum(r * T), 2.0 * np.sum(r)])


def fit(**settings):
    defaults = {"step": "adam", "learning_rate": 0.05, "objective": "minimize", "seed": 1}
    return gx.FirstOrder(gx.Real((-5, 5), length=2), **{**defaults, **settings})


def same(a, b):
    return (
        a.best_fitness == b.best_fitness
        and np.array_equal(a.best_genome, b.best_genome)
        and (a.generations, a.evaluations, a.stop_reason)
        == (b.generations, b.evaluations, b.stop_reason)
    )


def test_a_supplied_gradient_fits_a_line():
    result = fit().run(loss, gradient=gradient, generations=20_000)
    assert result.stop_reason == "converged"
    assert np.allclose(result.best_genome, [2.0, -1.0], atol=1e-6)
    # one evaluation per generation
    assert result.evaluations == result.generations + 1
    # the same run from a function that returns (fitness, gradient)
    both = fit().run(lambda p: (loss(p), gradient(p)), gradient=True, generations=20_000)
    assert same(result, both)


def test_without_a_gradient_finite_differences_take_its_place():
    result = fit().run(loss, generations=20_000)
    assert result.stop_reason == "converged"
    assert np.allclose(result.best_genome, [2.0, -1.0], atol=1e-5)
    # the point and one more per gene each generation
    assert result.evaluations == 3 * (result.generations + 1)
    central = fit(gradients="central").run(loss, generations=20_000)
    assert central.evaluations == 5 * (central.generations + 1)
    # finite differences even with a gradient given
    forward = fit(gradients="forward").run(loss, gradient=gradient, generations=20_000)
    assert same(forward, result)


@pytest.mark.parametrize(
    "settings",
    [
        {"step": "gradient", "learning_rate": 0.01},
        {"step": "momentum", "learning_rate": 0.005, "momentum": 0.9},
        {"step": "nesterov", "learning_rate": 0.005, "momentum": 0.9},
        {"step": "adam", "learning_rate": 0.05},
        {"step": "adamw", "learning_rate": 0.05, "weight_decay": 0.0},
    ],
)
def test_every_rule_converges(settings):
    result = fit(**settings).run(loss, gradient=gradient, generations=50_000)
    assert result.stop_reason == "converged"
    assert np.allclose(result.best_genome, [2.0, -1.0], atol=1e-5)


def test_the_problems_give_their_gradients_in_rust():
    problem = gx.problems.Sphere(2)
    in_rust = fit().run(problem, generations=10_000)
    assert in_rust.evaluations == in_rust.generations + 1
    # the same function and gradient in Python, to the bit (2 genes: numpy's sum is x0² + x1²)
    in_python = fit().run(
        lambda x: float(np.sum(x * x)), gradient=lambda x: 2.0 * x, generations=10_000
    )
    assert same(in_rust, in_python)


def test_a_run_as_in_rust():
    # tests/first_order.rs has the same run, with the same results
    problem = gx.problems.Rosenbrock(4)
    result = gx.FirstOrder(
        problem.genome,
        step="adam",
        learning_rate=0.02,
        restarts=2,
        objective="minimize",
        seed=5,
    ).run(problem, evaluations=200_000)
    assert result.stop_reason == "converged"
    assert result.evaluations == 85_572
    assert result.best_fitness == 4.017523816634646e-13


def test_a_learning_rate_schedule_by_control():
    rates = []

    def halve(running, progress):
        assert isinstance(running, gx.RunningFirstOrder)
        rates.append(running.learning_rate)
        running.learning_rate = 0.05 * 0.5 ** (progress.generation // 100)

    result = fit().run(loss, gradient=gradient, generations=20_000, control=halve)
    assert result.stop_reason == "converged"
    assert rates[0] == 0.05
    assert rates[-1] < 0.05

    # the multiplier scales every step: eta 0.5 with alpha 0.1 is alpha 0.05 for gradient descent
    def half(running, progress):
        running.multiplier = 0.5

    plain = fit(step="gradient", learning_rate=0.005).run(
        loss, gradient=gradient, generations=200
    )
    scaled = fit(step="gradient", learning_rate=0.01).run(
        loss, gradient=gradient, generations=200, control=half
    )
    assert scaled.best_fitness == pytest.approx(plain.best_fitness, rel=1e-6)


def test_the_running_method_reads_its_state():
    seen = []

    def read(running, progress):
        seen.append(
            (
                running.converged,
                running.iterations,
                running.steps,
                running.gradient_norm,
                running.gradient,
                running.gradients,
            )
        )

    result = fit().run(loss, gradient=gradient, generations=20_000, control=read)
    converged, iterations, steps, norm, last, gradients = seen[-1]
    assert converged == "gradient"
    assert iterations == steps == result.generations
    assert norm <= 1e-6
    assert np.max(np.abs(last)) == norm
    assert gradients == "supplied"
    assert seen[0][0] is None and seen[0][1] == 0


def test_reevaluation_keeps_the_memory():
    shift = {"value": 0.0}
    steps = []

    def moving(p):
        return loss(p - shift["value"])

    def moving_gradient(p):
        return gradient(p - shift["value"])

    def control(running, progress):
        steps.append(running.steps)
        if progress.generation == 50:
            shift["value"] = 0.5
            running.reevaluate()

    result = fit().run(moving, gradient=moving_gradient, generations=20_000, control=control)
    # no generation and no step for the re-evaluation
    assert steps[:52] == list(range(52))
    assert result.stop_reason == "converged"
    assert np.allclose(result.best_genome, [2.5, -0.5], atol=1e-5)


def test_maximizing_climbs_the_score():
    result = fit(objective="maximize").run(
        lambda p: -loss(p), gradient=lambda p: -gradient(p), generations=20_000
    )
    assert result.stop_reason == "converged"
    assert np.allclose(result.best_genome, [2.0, -1.0], atol=1e-6)


def test_points_stay_in_the_bounds():
    # the minimum of (x - 3)² outside [-1, 1]: the method lands on the bound
    def outside(x):
        return float(np.sum((x - 3.0) ** 2))

    def outside_gradient(x):
        return 2.0 * (x - 3.0)

    points = []
    result = gx.FirstOrder(
        gx.Real((-1, 1), length=3), step="adam", learning_rate=0.5, objective="minimize", seed=2
    ).run(
        outside,
        gradient=outside_gradient,
        generations=1_000,
        on_generation=lambda progress: points.append(progress.population[0]),
    )
    assert result.stop_reason == "converged"
    assert np.array_equal(result.best_genome, [1.0, 1.0, 1.0])
    assert all(np.all(np.abs(point) <= 1.0) for point in points)


def test_a_resumed_run_equals_an_uninterrupted_one(tmp_path):
    path = tmp_path / "first_order.ckpt"
    whole = fit(step="nesterov", learning_rate=0.005, momentum=0.9).run(
        loss, gradient=gradient, generations=300
    )
    fit(step="nesterov", learning_rate=0.005, momentum=0.9).run(
        loss, gradient=gradient, generations=100, checkpoint=path, checkpoint_every=50
    )
    resumed = fit(step="nesterov", learning_rate=0.005, momentum=0.9).run(
        loss, gradient=gradient, generations=300, resume=path
    )
    assert same(whole, resumed)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"step": "sgd"}, "step is"),
        ({"step": "gradient", "learning_rate": None}, "needs learning_rate"),
        ({"step": "momentum", "learning_rate": 0.1}, "needs momentum"),
        ({"step": "adam", "momentum": 0.9}, "momentum doesn't go"),
        ({"step": "adamw"}, "needs weight_decay"),
        ({"step": "gradient", "learning_rate": 0.1, "beta1": 0.9}, "beta1 doesn't go"),
        ({"step": "momentum", "learning_rate": 0.1, "momentum": 1.0}, "momentum"),
        ({"learning_rate": -1.0}, "learning rate"),
        ({"beta2": 1.0}, "beta2"),
        ({"epsilon": 0.0}, "epsilon"),
        ({"gradients": "exact"}, "gradients is"),
        ({"difference_step": 1e-6}, "difference_step"),
        ({"gradients": "central", "difference_step": 0.0}, "step"),
        ({"gradient_tolerance": -1.0}, "gradient_tolerance"),
        ({"step_tolerance": float("nan")}, "step_tolerance"),
        ({"restarts": 0}, "restarts"),
        ({"initial_genome": [9.0, 0.0]}, "initial_genome"),
    ],
)
def test_invalid_settings_are_errors(settings, message):
    with pytest.raises(ValueError, match=message):
        fit(**settings).run(loss, generations=1)


def test_a_batch_takes_its_gradients_as_rows():
    # a generation is one genome: the same run, a batch of one at a time
    def losses(genomes):
        return np.array([loss(genome) for genome in genomes])

    def gradients(genomes):
        return np.array([gradient(genome) for genome in genomes])

    alone = fit().run(loss, gradient=gradient, generations=20_000)
    assert same(alone, fit().run(losses, gradient=gradients, batch=True, generations=20_000))
    both = fit().run(
        lambda genomes: (losses(genomes), gradients(genomes)),
        gradient=True,
        batch=True,
        generations=20_000,
    )
    assert same(alone, both)


def test_gradients_are_checked():
    with pytest.raises(ValueError, match="2 genes"):
        fit().run(loss, gradient=lambda p: np.zeros(3), generations=1)
    with pytest.raises(TypeError, match=r"\(value, gradient\)"):
        fit().run(loss, gradient=True, generations=1)
    with pytest.raises(TypeError, match="gradient"):
        fit().run(loss, gradient="yes", generations=1)
    with pytest.raises(ValueError, match="its own"):
        fit().run(gx.problems.Sphere(2), gradient=gradient, generations=1)
    # "supplied" needs one
    with pytest.raises(ValueError, match="Supplied"):
        fit(gradients="supplied").run(loss, generations=1)
    # an exception in the gradient is raised
    def broken(p):
        raise RuntimeError("no gradient here")

    with pytest.raises(RuntimeError, match="no gradient here"):
        fit().run(loss, gradient=broken, generations=1)


def test_genomes_other_than_real_are_errors():
    with pytest.raises(ValueError, match="Real genome"):
        gx.FirstOrder(gx.Integer((0, 5), length=2)).run(lambda x: 0.0, generations=1)
