"""Bayesian optimization and its Gaussian process: the minima of the classic problems, the same runs
as in Rust, the running algorithm's model, checkpoints and the settings."""

import math

import numpy as np
import pytest

import genoxide as gx

BRANIN_MINIMUM = 5 / (4 * math.pi)


def test_branin_in_tens_of_evaluations():
    problem = gx.problems.Branin()
    result = gx.Bo(problem.genome, objective="minimize", seed=1).run(problem, evaluations=40)
    assert result.best_fitness - BRANIN_MINIMUM < 1e-3
    assert result.evaluations == 40
    # the design is generation 0, then a point per generation
    assert result.generations == 40 - 6


def test_a_run_as_in_rust():
    # tests/bo.rs has the same run in Rust, with the same best value, to the bit
    problem = gx.problems.Branin()
    bo = gx.Bo(problem.genome, objective="minimize", seed=1)
    for function, parallel in [(problem, False), (lambda x: problem(x), False), (problem, True)]:
        result = bo.run(function, evaluations=30, parallel=parallel)
        assert result.best_fitness == 0.39798370755715595
        assert result.best_genome.tolist() == [9.424508858501198, 2.484571084777484]


def test_the_log_transform_on_goldstein_price():
    problem = gx.problems.GoldsteinPrice()
    bo = gx.Bo(problem.genome, output="log", objective="minimize", seed=1)
    result = bo.run(problem, target=3.001, evaluations=70)
    assert result.stop_reason == "target"


def test_maximizing_mirrors_minimizing():
    problem = gx.problems.Branin()
    minimized = gx.Bo(problem.genome, objective="minimize", seed=4).run(problem, evaluations=15)
    maximized = gx.Bo(problem.genome, objective="maximize", seed=4).run(
        lambda x: -problem(x), evaluations=15
    )
    assert np.array_equal(maximized.best_genome, minimized.best_genome)
    assert maximized.best_fitness == -minimized.best_fitness


def test_every_acquisition_and_the_other_kernel():
    problem = gx.problems.Branin()
    for acquisition, kernel in [
        ("ei", "matern52"),
        (gx.ProbabilityOfImprovement(0.01), "matern52"),
        (gx.UpperConfidenceBound(4.0), "matern52"),
        ("log-ei", "squared_exponential"),
    ]:
        bo = gx.Bo(
            problem.genome, acquisition=acquisition, kernel=kernel, objective="minimize", seed=1
        )
        result = bo.run(problem, evaluations=40)
        assert result.best_fitness < BRANIN_MINIMUM + 0.05, (acquisition, kernel)


def test_initial_genomes_come_first_and_points_are_never_asked_twice():
    genome = gx.Real((0, 1), length=3)
    given = np.array([[0.5, 0.5, 0.5], [0.1, 0.9, 0.2]])
    seen = []

    def sphere(x):
        seen.append(x.copy())
        return float(np.sum((x - 0.3) ** 2))

    gx.Bo(genome, initial_genomes=given, objective="minimize", seed=1).run(
        sphere, evaluations=20
    )
    assert len(seen) == 20
    assert np.array_equal(seen[0], given[0]) and np.array_equal(seen[1], given[1])
    assert len({tuple(x) for x in seen}) == 20


def test_invalid_points_enter_the_model_at_the_worst_value():
    problem = gx.problems.Branin()

    def partial(x):
        return None if x[0] + x[1] > 14 else problem(x)

    result = gx.Bo(problem.genome, objective="minimize", seed=1).run(partial, evaluations=45)
    assert result.best_fitness < BRANIN_MINIMUM + 1e-2


def test_the_running_algorithm_gives_its_model_and_acquisition():
    problem = gx.problems.Branin()
    seen = {}

    def control(algorithm, progress):
        if progress.generation == 0:
            assert algorithm.model is None
            assert algorithm.initial_points == 6
            with pytest.raises(ValueError, match="no model"):
                algorithm.acquisition_at([0.0, 5.0])
            return
        model = algorithm.model
        assert len(model) == 5 + progress.generation
        newest = progress.population[-1]
        values = algorithm.acquisition_at(np.array([newest, [0.0, 5.0]]))
        assert values.shape == (2,) and np.all(np.isfinite(values))
        # the newest point maximized the acquisition
        grid = np.array([[x1, x2] for x1 in np.linspace(-5, 10, 16) for x2 in np.linspace(0, 15, 16)])
        assert values[0] >= algorithm.acquisition_at(grid).max() - 1e-9
        seen["acquisition"] = algorithm.acquisition
        algorithm.acquisition = gx.UpperConfidenceBound(9.0 / (1 + progress.generation))
        with pytest.raises(ValueError, match="beta"):
            algorithm.acquisition = gx.UpperConfidenceBound(-1.0)

    result = gx.Bo(problem.genome, objective="minimize", seed=1).run(
        problem, evaluations=20, control=control
    )
    assert seen["acquisition"] == gx.UpperConfidenceBound(9.0 / 14)
    assert result.best_fitness < 1.0


def test_reevaluation_asks_every_point_again():
    problem = gx.problems.Branin()
    calls = []
    shift = {"value": 0.0}

    def shifted(x):
        calls.append(x.copy())
        return problem(x) + shift["value"]

    def control(algorithm, progress):
        if progress.generation == 3 and shift["value"] == 0.0:
            shift["value"] = 10.0
            algorithm.reevaluate()

    result = gx.Bo(problem.genome, objective="minimize", seed=2).run(
        shifted, generations=5, control=control
    )
    # 6 points, 3 more, all 9 again, then 2 more
    assert len(calls) == 6 + 3 + 9 + 2
    assert result.best_fitness > 10


def test_a_checkpoint_resumes_the_run(tmp_path):
    problem = gx.problems.Branin()
    bo = gx.Bo(problem.genome, objective="minimize", seed=3)
    whole = bo.run(problem, evaluations=20)
    path = tmp_path / "bo.ckpt"
    bo.run(problem, evaluations=12, checkpoint=path, checkpoint_every=2)
    resumed = bo.run(problem, evaluations=20, resume=path)
    assert resumed.best_fitness == whole.best_fitness
    assert np.array_equal(resumed.best_genome, whole.best_genome)
    assert resumed.evaluations == whole.evaluations


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"acquisition": "ucb"}, "acquisition"),
        ({"acquisition": gx.UpperConfidenceBound(-1.0)}, "beta"),
        ({"acquisition": gx.ProbabilityOfImprovement(math.nan)}, "xi"),
        ({"kernel": "rbf"}, "kernel"),
        ({"output": "rank"}, "output"),
        ({"noise": -1.0}, "noise"),
        ({"noise": gx.model.gp.Learned(0.0)}, "noise"),
        ({"initial_points": 0}, "initial_points"),
        ({"initial_genomes": [0.5, 0.5]}, "initial_genomes"),
        ({"initial_genomes": [[0.5, 0.5], [0.5, 0.5]]}, "initial_genomes"),
        ({"initial_genomes": [[2.0, 0.5]]}, "initial_genomes"),
        ({"raw_samples": 0}, "raw_samples"),
        ({"acquisition_starts": 2000}, "acquisition_starts"),
        ({"hyperparameter_starts": 0}, "hyperparameter_starts"),
    ],
)
def test_wrong_settings_are_value_errors(settings, message):
    bo = gx.Bo(gx.Real((0, 1), length=2), **settings)
    with pytest.raises(ValueError, match=message):
        bo.run(lambda x: float(np.sum(x)), evaluations=10)


def test_a_bo_needs_a_real_genome():
    with pytest.raises(ValueError, match="Real"):
        gx.Bo(gx.Binary(4)).run(lambda x: float(x.sum()), evaluations=10)


# ---- the Gaussian process ------------------------------------------------------------------------


def test_two_points_are_the_formulas_worked_out_by_hand():
    # Rasmussen and Williams (2006): the mean m + k*' K^-1 (y - m) (eq. 2.38), the variance
    # s_f^2 - k*' K^-1 k* (eq. 2.26) and the log marginal likelihood (eq. 2.30), the squared
    # exponential kernel in a gene of [-2, 6]
    x, y = np.array([0.5, 2.0]), np.array([1.0, 4.0])
    length, signal, noise, mean = 2.0, 3.0, 0.1, 1.5
    model = gx.model.gp.GaussianProcess.fit(
        gx.Real((-2, 6), length=1),
        x.reshape(-1, 1),
        y,
        kernel="squared_exponential",
        hyperparameters=gx.model.gp.Hyperparameters(mean, (length,), signal, noise),
    )

    def k(a, b):
        return signal * math.exp(-0.5 * ((a - b) / length) ** 2)

    covariance = np.array([[k(a, b) for b in x] for a in x]) + noise * np.eye(2)
    alpha = np.linalg.solve(covariance, y - mean)
    for at in [-2.0, 1.0, 1.7, 6.0]:
        kstar = np.array([k(at, b) for b in x])
        expected_mean = mean + kstar @ alpha
        expected_variance = signal - kstar @ np.linalg.solve(covariance, kstar)
        means, variances = model.predict([[at]])
        assert means[0] == pytest.approx(expected_mean, rel=1e-12)
        assert variances[0] == pytest.approx(expected_variance, rel=1e-11)
    log_likelihood = (
        -0.5 * (y - mean) @ alpha
        - 0.5 * math.log(np.linalg.det(covariance))
        - math.log(2 * math.pi)
    )
    assert model.log_marginal_likelihood == pytest.approx(log_likelihood, rel=1e-12)
    h = model.hyperparameters
    assert (h.mean, h.length_scales[0], h.signal_variance, h.noise_variance) == pytest.approx(
        (mean, length, signal, noise), rel=1e-15
    )
    assert model.kernel == "squared_exponential" and len(model) == 2 and model.genes == 1


def test_a_fitted_model_interpolates_and_its_gradients_match_differences():
    genome = gx.Real([(-1, 2), (0, 3)])
    rng = np.random.default_rng(1)
    points = rng.uniform([-1, 0], [2, 3], size=(15, 2))
    values = np.sin(2 * points[:, 0]) + 0.5 * points[:, 1] ** 2

    model = gx.model.gp.GaussianProcess.fit(genome, points, values)
    means, variances = model.predict(points)
    assert np.allclose(means, values, atol=1e-6)
    assert np.all(variances < 1e-8 * model.hyperparameters.signal_variance)
    at = np.array([0.3, 1.1])
    mean, variance, dmean, dvariance = model.predict_with_gradient(at)
    assert (mean, variance) == (model.predict(at)[0][0], model.predict(at)[1][0])
    for i in range(2):
        h = np.zeros(2)
        h[i] = 1e-6
        plus, minus = model.predict(at + h), model.predict(at - h)
        assert dmean[i] == pytest.approx((plus[0][0] - minus[0][0]) / 2e-6, rel=1e-6, abs=1e-6)
        assert dvariance[i] == pytest.approx((plus[1][0] - minus[1][0]) / 2e-6, abs=1e-6)
    # a seed repeats the fit; learned noise is another model
    again = gx.model.gp.GaussianProcess.fit(genome, points, values)
    assert again.hyperparameters == model.hyperparameters
    noisy = gx.model.gp.GaussianProcess.fit(genome, points, values, noise=gx.model.gp.Learned())
    assert noisy.hyperparameters.noise_variance > 0


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"points": np.zeros((0, 2)), "values": np.zeros(0)}, "point"),
        ({"points": np.zeros((3, 3)), "values": np.zeros(3)}, "points"),
        ({"points": np.zeros((3, 2)), "values": np.zeros(2)}, "value"),
        ({"points": np.zeros((2, 2)), "values": [0.0, math.inf]}, "value"),
        ({"kernel": "rbf"}, "kernel"),
        ({"noise": -1.0}, "noise"),
        ({"starts": 0}, "starts"),
    ],
)
def test_wrong_model_settings_are_value_errors(arguments, message):
    arguments = {"points": [[0.1, 0.2], [0.5, 0.5]], "values": [1.0, 2.0], **arguments}
    with pytest.raises(ValueError, match=message):
        gx.model.gp.GaussianProcess.fit(gx.Real((0, 1), length=2), **arguments)
    model = gx.model.gp.GaussianProcess.fit(gx.Real((0, 1), length=2), [[0.1, 0.2]], [1.0])
    with pytest.raises(ValueError, match="genes"):
        model.predict([[0.5]])


# ---- batches, constraints and integer genomes ----------------------------------------------------


def test_a_batch_runs_as_in_rust_and_the_same_in_parallel():
    # the bo_hartmann6 example: 4 points a round, 15 rounds to within 1e-4
    problem = gx.problems.Hartmann6()
    bo = gx.Bo(problem.genome, batch=4, objective="minimize", seed=3)
    target = problem.optimum.value + 1e-4
    result = bo.run(problem, target=target, evaluations=200, parallel=True)
    assert (result.evaluations, result.generations) == (74, 15)
    assert result.best_fitness == -3.322329435896455
    again = bo.run(problem, target=target, evaluations=200)
    assert again.best_fitness == result.best_fitness
    assert np.array_equal(again.best_genome, result.best_genome)


def test_the_fantasies_and_the_batch_change_during_a_run():
    problem = gx.problems.Branin()
    seen = []

    def control(running, progress):
        seen.append((running.batch, running.fantasy, running.constraints))
        running.batch = 2
        running.fantasy = "liar-max"

    bo = gx.Bo(problem.genome, batch=3, fantasy="liar-mean", objective="minimize", seed=1)
    result = bo.run(problem, generations=3, control=control)
    assert seen[0] == (3, "liar-mean", 0)
    assert seen[1] == (2, "liar-max", 0)
    # the design of 6, then 2 a round: the control runs after generation 0 already
    assert result.evaluations == 6 + 2 + 2 + 2
    for fantasy in ("believer", "liar-min", "liar-mean", "liar-max"):
        bo = gx.Bo(problem.genome, batch=4, fantasy=fantasy, objective="minimize", seed=2)
        assert bo.run(problem, generations=10).best_fitness < 0.397887 + 0.05


def toy(x):
    """Gramacy et al.'s (2016) toy problem: x1 + x2 and its two constraints' values."""
    wave = gx.math.sin(2.0 * math.pi * (x[0] * x[0] - 2.0 * x[1]))
    return x[0] + x[1], np.array(
        [1.5 - x[0] - 2.0 * x[1] - 0.5 * wave, x[0] * x[0] + x[1] * x[1] - 1.5]
    )


def test_constrained_bayesian_optimization_reaches_the_feasible_minimum():
    # the bo_constrained example: 21 evaluations to within 1e-5 of 0.5997880520100676
    minimum = 0.5997880520100676
    probabilities = []

    def control(running, progress):
        if running.model is not None:
            points = progress.population
            probabilities.append(running.probability_of_feasibility_at(points))
            assert running.constraints == 2

    bo = gx.Bo(gx.Real((0.0, 1.0), length=2), objective="minimize", seed=1)
    result = bo.run(toy, constraints=2, target=minimum + 1e-5, evaluations=60, control=control)
    assert result.evaluations == 21
    assert result.best_fitness - minimum <= 1e-5
    assert np.all(toy(result.best_genome)[1] <= 0.0)
    # at the points evaluated, about 0 or 1: the models interpolate the values
    assert np.all((probabilities[-1] < 0.5) | (probabilities[-1] > 0.5))
    # in parallel, the same run
    parallel = bo.run(toy, constraints=2, target=minimum + 1e-5, evaluations=60, parallel=True)
    assert np.array_equal(parallel.best_genome, result.best_genome)
    # a test problem gives its constraints' values in Rust
    g24 = gx.problems.cec2006.G24()
    bo = gx.Bo(g24.genome, objective="minimize", seed=1)
    result = bo.run(g24, target=g24.optimum.value + 1e-4, evaluations=60)
    assert result.stop_reason == "target"


def test_wrong_constraints_are_errors():
    bo = gx.Bo(gx.Real((0.0, 1.0), length=2), objective="minimize", seed=1)
    with pytest.raises(ValueError, match="2 constraint values, for 3"):
        bo.run(toy, constraints=3, evaluations=10)
    with pytest.raises(TypeError, match="value, constraint values"):
        bo.run(lambda x: float(x[0]), constraints=1, evaluations=10)
    with pytest.raises(ValueError, match="batch=False"):
        bo.run(toy, constraints=2, batch=True, evaluations=10)
    with pytest.raises(ValueError, match="own constraints"):
        bo.run(gx.problems.cec2006.G24(), constraints=2, evaluations=10)
    # and so does a shifted or rotated one: the wrappers pass the values on
    g24 = gx.problems.cec2006.G24()
    for wrapped in [gx.problems.Shifted(g24, seed=1), gx.problems.Rotated(g24, seed=1)]:
        with pytest.raises(ValueError, match="own constraints"):
            bo.run(wrapped, constraints=2, evaluations=10)
    ucb = gx.Bo(gx.Real((0.0, 1.0), length=2), acquisition=gx.UpperConfidenceBound(2.0))
    with pytest.raises(ValueError, match="upper confidence bound"):
        ucb.run(toy, constraints=2, evaluations=10)


def test_integer_genes_are_searched_on_their_lattice():
    def quadratic(x):
        return float((x[0] - 2.6) ** 2 + 2 * (x[1] + 1.3) ** 2 + (x[2] - 0.4) ** 2)

    integer = gx.Integer((-10, 10), length=3)
    bo = gx.Bo(integer, initial_genomes=[[0, 0, 0]], objective="minimize", seed=1)
    calls = []

    def recorded(x):
        calls.append(tuple(int(gene) for gene in x))
        return quadratic(x)

    # the integer minimum, (3, -1, 0): 0.16 + 0.18 + 0.16
    result = bo.run(recorded, target=0.5, evaluations=60)
    assert result.stop_reason == "target"
    assert list(result.best_genome) == [3, -1, 0]
    assert calls[0] == (0, 0, 0)
    assert len(set(calls)) == len(calls)
    with pytest.raises(ValueError, match="whole numbers"):
        gx.Bo(integer, initial_genomes=[[0.5, 0, 0]]).run(quadratic, evaluations=10)


def test_a_checkpoint_resumes_a_constrained_batch(tmp_path):
    space = gx.Real((0.0, 1.0), length=2)
    bo = gx.Bo(space, batch=2, objective="minimize", seed=4)
    whole = bo.run(toy, constraints=2, evaluations=20)
    path = tmp_path / "bo.ckpt"
    bo.run(toy, constraints=2, evaluations=12, checkpoint=path, checkpoint_every=1)
    resumed = bo.run(toy, constraints=2, evaluations=20, resume=path)
    assert resumed.best_fitness == whole.best_fitness
    assert np.array_equal(resumed.best_genome, whole.best_genome)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"batch": 0}, "batch"),
        ({"fantasy": "liar"}, "fantasy"),
    ],
)
def test_wrong_batch_settings_are_value_errors(settings, message):
    bo = gx.Bo(gx.Real((0, 1), length=2), **settings)
    with pytest.raises(ValueError, match=message):
        bo.run(lambda x: float(np.sum(x)), evaluations=10)
