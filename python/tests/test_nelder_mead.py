"""The Nelder-Mead simplex method: convergence, restarts, speculative asks, its settings, and the
same runs as in Rust."""

import math

import numpy as np
import pytest

import genoxide as gx


def rosenbrock(x):
    return 100 * (x[1] - x[0] * x[0]) ** 2 + (1 - x[0]) ** 2


def sphere(x):
    return float(np.sum(x * x))


@pytest.mark.parametrize("coefficients", ["adaptive", "standard"])
def test_it_converges_on_rosenbrock_from_the_classic_start(coefficients):
    nelder_mead = gx.NelderMead(
        gx.Real((-5, 5), length=2),
        coefficients=coefficients,
        initial_genome=np.array([-1.2, 1.0]),
        objective="minimize",
    )
    result = nelder_mead.run(rosenbrock, evaluations=10_000)
    assert result.stop_reason == "converged"
    assert result.best_fitness < 1e-15
    assert np.allclose(result.best_genome, 1.0, atol=1e-7)
    assert result.evaluations < 10_000


def test_a_run_as_in_rust():
    # tests/nelder_mead.rs has the same runs in Rust, evaluated in Rust, with the same results
    problem = gx.problems.Rosenbrock(4)
    for speculative, evaluations, generations in [(False, 1985, 1969), (True, 4815, 1202)]:
        result = gx.NelderMead(
            problem.genome,
            coefficients=(1.0, 2.5, 0.4, 0.6),
            initial_step=0.2,
            tolerance=5e-8,
            restarts=2,
            speculative=speculative,
            objective="minimize",
            seed=5,
        ).run(problem, evaluations=20_000)
        assert result.stop_reason == "converged"
        assert (result.evaluations, result.generations) == (evaluations, generations)
        assert result.best_fitness == 6.607094554363915e-14


def test_python_batch_and_parallel_evaluation_give_the_same_run():
    problem = gx.problems.Rosenbrock(3)
    nelder_mead = gx.NelderMead(problem.genome, restarts=1, objective="minimize", seed=3)
    in_rust = nelder_mead.run(problem, evaluations=20_000)
    for batch, parallel in [(False, False), (True, False), (False, True)]:
        function = problem.evaluate if batch else (lambda x: problem(x))
        again = nelder_mead.run(function, evaluations=20_000, batch=batch, parallel=parallel)
        assert again.best_fitness == in_rust.best_fitness
        assert np.array_equal(again.best_genome, in_rust.best_genome)
        assert (again.evaluations, again.generations, again.stop_reason) == (
            in_rust.evaluations,
            in_rust.generations,
            in_rust.stop_reason,
        )


@pytest.mark.parametrize("seed", [5, 7])
def test_speculative_asks_take_the_same_path_in_fewer_generations(seed):
    genome = gx.Real((-5, 5), length=4)

    def run(speculative):
        simplexes = []

        def keep(progress):
            if not simplexes or not np.array_equal(simplexes[-1], progress.population):
                simplexes.append(progress.population)

        nelder_mead = gx.NelderMead(
            genome, speculative=speculative, objective="minimize", seed=seed
        )
        result = nelder_mead.run(
            rosenbrock_n, evaluations=50_000, parallel=speculative, on_generation=keep
        )
        return result, simplexes

    plain, plain_simplexes = run(False)
    speculative, speculative_simplexes = run(True)
    assert plain.stop_reason == speculative.stop_reason == "converged"
    # the same simplexes, one after another
    assert len(speculative_simplexes) == len(plain_simplexes)
    for a, b in zip(plain_simplexes, speculative_simplexes):
        assert np.array_equal(a, b)
    assert speculative.generations < plain.generations
    assert speculative.evaluations > plain.evaluations
    # the best of the evaluated points: the speculative ones include every plain one, and a
    # rejected trial point can be better than the simplex
    assert speculative.best_fitness <= plain.best_fitness
    if seed == 5:
        assert speculative.best_fitness == plain.best_fitness
        assert np.array_equal(speculative.best_genome, plain.best_genome)


def rosenbrock_n(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


def test_restarts_reach_the_minima_of_every_basin():
    # Himmelblau's function has four minima, all with the value 0
    problem = gx.problems.Himmelblau()
    ends = []
    restarts = []

    def control(nelder_mead, progress):
        assert isinstance(nelder_mead, gx.RunningNelderMead)
        if nelder_mead.converged:
            assert nelder_mead.size <= 1e-9
            # the simplex, best first
            ends.append((progress.population[0], progress.scores[0]))
        restarts.append(nelder_mead.restart_count)

    nelder_mead = gx.NelderMead(problem.genome, restarts=19, objective="minimize", seed=8)
    result = nelder_mead.run(problem, evaluations=100_000, control=control)
    assert result.stop_reason == "converged"
    assert restarts[-1] == 19
    assert len(ends) == 20
    minima = np.array(problem.optimum.solutions)
    found = set()
    for genome, score in ends:
        assert score < 1e-15
        distances = np.max(np.abs(minima - genome), axis=1)
        assert distances.min() < 1e-7
        found.add(int(distances.argmin()))
    assert found == {0, 1, 2, 3}


def test_the_running_algorithm_reads_the_simplex():
    seen = []

    def control(nelder_mead, progress):
        seen.append(
            (
                nelder_mead.converged,
                nelder_mead.size,
                nelder_mead.iterations,
                nelder_mead.restart_count,
            )
        )
        with pytest.raises(AttributeError):
            nelder_mead.size = 0.5

    result = gx.NelderMead(gx.Real((-5, 5), length=3), objective="minimize", seed=1).run(
        sphere, evaluations=100_000, control=control
    )
    assert result.stop_reason == "converged"
    # the first simplex: a step of 0.1 of each range
    assert seen[0] == (False, pytest.approx(1.0), 0, 0)
    converged, size, iterations, restart_count = seen[-1]
    assert converged and size <= 1e-9 and restart_count == 0
    assert 0 < iterations <= result.generations
    assert [state[0] for state in seen].count(True) == 1


def test_maximizing_takes_the_same_path_as_minimizing_the_negation():
    genome = gx.Real((-5, 5), length=3)
    minimized = gx.NelderMead(genome, objective="minimize", seed=4).run(
        sphere, evaluations=10_000
    )
    maximized = gx.NelderMead(genome, seed=4).run(lambda x: -sphere(x), evaluations=10_000)
    assert np.array_equal(minimized.best_genome, maximized.best_genome)
    assert minimized.evaluations == maximized.evaluations
    assert maximized.best_fitness == -minimized.best_fitness


def test_a_stop_condition_is_still_needed():
    with pytest.raises(ValueError, match="needs a stop condition"):
        gx.NelderMead(gx.Real((-5, 5), length=2), objective="minimize").run(sphere)
    # a stop condition met in the generation that converges is the stop reason
    nelder_mead = gx.NelderMead(gx.Real((-5, 5), length=2), objective="minimize", seed=9)
    converged = nelder_mead.run(sphere, evaluations=10_000)
    assert converged.stop_reason == "converged"
    at_the_end = nelder_mead.run(sphere, generations=converged.generations)
    assert at_the_end.stop_reason == "generations"
    assert at_the_end.best_fitness == converged.best_fitness


def test_fixed_genes_stay_and_points_stay_in_the_bounds():
    # the minimum of the sphere is outside [1, 3] x [2, 2]: at (1, 2), on the bound, which the
    # simplex approaches (mirrored at the bound) rather than lands on
    nelder_mead = gx.NelderMead(gx.Real([(1.0, 3.0), (2.0, 2.0)]), objective="minimize", seed=6)
    genomes = []
    result = nelder_mead.run(
        sphere, evaluations=10_000, on_generation=lambda p: genomes.append(p.population)
    )
    assert result.stop_reason == "converged"
    assert np.allclose(result.best_genome, [1.0, 2.0], rtol=0, atol=1e-8)
    assert result.best_fitness - 5.0 < 1e-7
    for population in genomes:
        assert np.all(population[:, 0] >= 1.0) and np.all(population[:, 0] <= 3.0)
        assert np.all(population[:, 1] == 2.0)


@pytest.mark.parametrize(
    "settings, message",
    [
        ({"coefficients": "fancy"}, "coefficients is"),
        ({"coefficients": (1.0, 2.0, 0.5)}, "coefficients is"),
        ({"coefficients": 3}, "coefficients is"),
        ({"coefficients": (1.0, 0.5, 0.5, 0.5)}, "invalid setting `coefficients`"),
        ({"coefficients": (1.0, 2.0, 1.5, 0.5)}, "invalid setting `coefficients`"),
        ({"coefficients": (1.0, 2.0, 0.5, "half")}, "coefficients.shrink"),
        ({"coefficients": (1.0, math.inf, 0.5, 0.5)}, "coefficients.expansion"),
        ({"initial_step": 0.0}, "invalid setting `initial_step`"),
        ({"initial_step": 1.5}, "invalid setting `initial_step`"),
        ({"initial_step": math.nan}, "initial_step is a finite number"),
        ({"tolerance": 0.0}, "invalid setting `tolerance`"),
        ({"tolerance": 1.0}, "invalid setting `tolerance`"),
        ({"restarts": 0}, "restarts is at least 1"),
        ({"restarts": -2}, "restarts is at least 1"),
        ({"restarts": 2.0}, "restarts is a whole number"),
        ({"restarts": True}, "restarts is a whole number"),
        ({"speculative": "yes"}, "speculative is True or False"),
        ({"initial_genome": [1.0, 2.0, 3.0]}, "invalid setting `initial_genome`: expected 2"),
        ({"initial_genome": [9.0, 0.0]}, "invalid setting `initial_genome`: gene 0 is 9"),
        ({"initial_genome": [[1.0, 2.0]]}, "initial_genome is a sequence"),
        ({"initial_genome": [1.0, math.nan]}, "initial_genome are finite numbers"),
        ({"objective": "up"}, "objective is"),
        ({"seed": -1}, "seed"),
    ],
)
def test_invalid_settings_are_errors(settings, message):
    nelder_mead = gx.NelderMead(gx.Real((-5, 5), length=2), **settings)
    with pytest.raises(ValueError, match=message):
        nelder_mead.run(sphere, generations=1)


def test_genomes_other_than_real_are_errors():
    for genome in [gx.Binary(4), gx.Integer((0, 3), length=2), gx.Permutation(4)]:
        with pytest.raises(ValueError, match="NelderMead needs a Real genome"):
            gx.NelderMead(genome).run(lambda x: 0.0, generations=1)
    adaptive = gx.AdaptiveReal(gx.Real((0, 1), length=2), 0.3)
    with pytest.raises(ValueError, match="NelderMead needs a Real genome"):
        gx.NelderMead(adaptive).run(lambda x: 0.0, generations=1)
    with pytest.raises(ValueError, match="Real.bounds"):
        gx.NelderMead(gx.Real([(1.0, 1.0), (2.0, 2.0)])).run(lambda x: 0.0, generations=1)
    with pytest.raises(ValueError, match="genome is"):
        gx.NelderMead("real").run(lambda x: 0.0, generations=1)
    with pytest.raises(ValueError, match="multi-objective algorithm"):
        zdt1 = gx.problems.Zdt1(30)
        gx.NelderMead(zdt1.genome).run(zdt1, generations=1)


def test_a_resumed_run_converges_as_an_uninterrupted_one(tmp_path):
    path = tmp_path / "nelder-mead.ckpt"
    nelder_mead = gx.NelderMead(
        gx.Real((-5, 5), length=4), restarts=2, objective="minimize", seed=2
    )
    whole = nelder_mead.run(rosenbrock_n, evaluations=50_000)
    assert whole.stop_reason == "converged"
    first = nelder_mead.run(rosenbrock_n, generations=300, checkpoint=path, checkpoint_every=50)
    assert first.stop_reason == "generations"
    resumed = nelder_mead.run(rosenbrock_n, evaluations=50_000, resume=path)
    assert resumed.stop_reason == "converged"
    assert resumed.best_fitness == whole.best_fitness
    assert np.array_equal(resumed.best_genome, whole.best_genome)
    assert (resumed.evaluations, resumed.generations) == (whole.evaluations, whole.generations)
    # other settings can't resume it
    other = gx.NelderMead(gx.Real((-5, 5), length=4), restarts=3, objective="minimize", seed=2)
    with pytest.raises(ValueError, match="other settings"):
        other.run(rosenbrock_n, evaluations=50_000, resume=path)


def test_an_absolute_step_works_in_any_box():
    # with a fraction of the range, a box of ±1e10 would start with a step of 2e9
    def run(width):
        return gx.NelderMead(
            gx.Real((-width, width), length=2),
            initial_genome=[-1.2, 1.0],
            initial_step_absolute=0.5,
            objective="minimize",
        ).run(gx.problems.Rosenbrock(2), evaluations=10_000)

    narrow = run(10.0)
    assert narrow.stop_reason == "converged"
    assert narrow.best_fitness < 1e-15
    for width in (1e3, 1e6, 1e10):
        wide = run(width)
        assert wide.best_fitness == narrow.best_fitness
        assert wide.evaluations == narrow.evaluations


def test_one_initial_step_at_a_time():
    genome = gx.Real((-1, 1), length=2)
    with pytest.raises(ValueError, match="not both"):
        gx.NelderMead(genome, initial_step=0.1, initial_step_absolute=0.5).run(
            lambda x: 0.0, evaluations=10
        )
    for bad in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError, match="initial_step_absolute"):
            gx.NelderMead(genome, initial_step_absolute=bad).run(lambda x: 0.0, evaluations=10)
