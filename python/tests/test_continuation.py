"""Continuation: stages of one problem with the callbacks, the state kept or reset, budgets,
exceptions, checkpoints, the gradient methods it wraps, and the settings."""

import numpy as np
import pytest

import genoxide as gx

CENTER = np.array([0.3, -1.2, 2.0])
EPSILON = [1.0, 0.1, 0.01]


class Smoothed:
    """Σ √((xᵢ − cᵢ)² + ε²), a smoothed Σ |xᵢ − cᵢ|, with its gradient: ε set by stage."""

    def __init__(self):
        self.epsilon = EPSILON[0]
        self.stages = []

    def on_stage(self, index):
        self.stages.append(index)
        self.epsilon = EPSILON[index]

    def __call__(self, x):
        root = np.sqrt((x - CENTER) ** 2 + self.epsilon**2)
        return float(np.sum(root)), (x - CENTER) / root


def adam(**settings):
    defaults = {
        "step": "adam",
        "learning_rate": 0.05,
        "gradient_tolerance": 1e-9,
        "objective": "minimize",
        "seed": 1,
    }
    return gx.FirstOrder(gx.Real((-5, 5), length=3), **{**defaults, **settings})


def staged(algorithm, problem, **settings):
    return gx.Continuation(algorithm, stages=len(EPSILON), on_stage=problem.on_stage, **settings)


def test_the_stages_run_to_convergence():
    problem = Smoothed()
    lbfgsb = gx.Lbfgsb(
        gx.Real((-5, 5), length=3), gradient_tolerance=1e-10, objective="minimize", seed=1
    )
    finished = []
    continuation = staged(
        lbfgsb, problem, on_stage_finished=lambda stage, point: finished.append((stage, point))
    )
    result = continuation.run(problem, gradient=True, evaluations=10_000)
    assert result.stop_reason == "converged"
    assert problem.stages == [0, 1, 2]
    assert len(result.stages) == 3
    assert [stage for stage, _ in finished] == list(result.stages)
    for index, stage in enumerate(result.stages):
        assert isinstance(stage, gx.Stage)
        assert stage.index == index
        assert stage.end == "finished"
        assert stage.best_fitness == pytest.approx(3 * EPSILON[index], abs=1e-9)
        assert isinstance(finished[index][1], np.ndarray)
    assert sum(stage.evaluations for stage in result.stages) == result.evaluations
    assert sum(stage.generations for stage in result.stages) == result.generations
    assert np.allclose(result.best_genome, CENTER, atol=1e-8)
    # other algorithms have no stages
    assert (
        gx.Lbfgsb(gx.Real((-5, 5), length=3), objective="minimize", seed=1)
        .run(problem, gradient=True, evaluations=1_000)
        .stages
        == ()
    )


def test_a_budget_ends_each_stage():
    problem = Smoothed()
    result = staged(adam(), problem, generations=7).run(problem, gradient=True, generations=1_000)
    assert result.stop_reason == "converged"
    assert result.generations == 21
    assert [stage.generations for stage in result.stages] == [7, 7, 7]
    assert [stage.end for stage in result.stages] == ["generations"] * 3
    assert [stage.evaluations for stage in result.stages] == [8, 8, 8]


def steps(keep):
    """Adam's step count t at each generation of a run of 3 stages of 5 generations."""
    problem = Smoothed()
    counts = []
    staged(adam(), problem, generations=5, keep=keep).run(
        problem,
        gradient=True,
        generations=1_000,
        control=lambda running, progress: counts.append(running.steps),
    )
    return counts


def test_adam_keeps_its_state_or_only_the_point():
    # t counts on across the stages, or starts again with each
    assert steps("state") == list(range(16))
    # reset as each stage begins, at the end of the last
    assert steps("point") == [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5]


def test_runs_repeat_and_resume(tmp_path):
    def run(**settings):
        problem = Smoothed()
        return staged(adam(), problem, generations=10).run(problem, gradient=True, **settings)

    whole = run(generations=1_000)
    again = run(generations=1_000)
    assert np.array_equal(whole.best_genome, again.best_genome)
    assert whole.stages == again.stages
    # saved within the second stage, and at its end
    for split in (14, 20):
        path = tmp_path / f"continuation-{split}.ckpt"
        run(generations=split, checkpoint=path, checkpoint_every=split)
        resumed = run(generations=1_000, resume=path)
        assert resumed.best_fitness == whole.best_fitness
        assert np.array_equal(resumed.best_genome, whole.best_genome)
        assert (resumed.generations, resumed.evaluations) == (whole.generations, whole.evaluations)
        assert resumed.stages == whole.stages


def test_mma_with_constraints_in_stages():
    # |x − (2, 1, −1)|² + ε Σ x², subject to x₀ + x₁ + x₂ ≤ 1: ε falls to 0
    weights = [1.0, 0.1, 0.0]
    state = {"stage": 0}

    def penalized(x):
        p = np.array([2.0, 1.0, -1.0])
        e = weights[state["stage"]]
        value = float(((x - p) ** 2).sum() + e * (x**2).sum())
        gradient = 2.0 * (x - p) + 2.0 * e * x
        return value, gradient, np.array([x.sum() - 1.0]), np.array([[1.0, 1.0, 1.0]])

    mma = gx.Mma(
        gx.Real((-5, 5), length=3),
        method="gcmma",
        initial_genome=[0.0, 0.0, 0.0],
        objective="minimize",
    )
    continuation = gx.Continuation(
        mma, stages=3, on_stage=lambda index: state.__setitem__("stage", index)
    )
    result = continuation.run(penalized, gradient=True, constraints=1, evaluations=2_000)
    assert result.stop_reason == "converged"
    assert len(result.stages) == 3
    # the projection of (2, 1, −1) onto the plane: (5/3, 2/3, −4/3)
    assert np.allclose(result.best_genome, [5 / 3, 2 / 3, -4 / 3], atol=1e-6)
    assert result.stages[2].best_fitness == pytest.approx(1 / 3)


def test_lbfgsb_can_keep_its_pairs():
    problem = Smoothed()
    pairs = []
    stage = {"start": True}

    def on_stage(index):
        problem.on_stage(index)
        stage["start"] = True

    def control(running, progress):
        if stage["start"]:
            pairs.append(running.pairs)
            stage["start"] = False

    for keep_pairs in (False, True):
        pairs.clear()
        lbfgsb = gx.Lbfgsb(
            gx.Real((-5, 5), length=3), keep_pairs=keep_pairs, objective="minimize", seed=1
        )
        gx.Continuation(lbfgsb, stages=2, on_stage=on_stage, generations=4).run(
            problem, gradient=True, generations=100, control=control
        )
        # at the first generation of each stage: none dropped only when they're kept
        assert (pairs[1] > 0) == keep_pairs, pairs


def test_exceptions_of_the_callbacks_stop_the_run():
    problem = Smoothed()

    def failing(index):
        if index == 1:
            raise KeyError("no second stage")

    with pytest.raises(KeyError, match="no second stage"):
        gx.Continuation(adam(), stages=2, on_stage=failing, generations=3).run(
            problem, gradient=True, generations=100
        )

    def finished(stage, point):
        raise ZeroDivisionError("stage finished")

    calls = []
    with pytest.raises(ZeroDivisionError, match="stage finished"):
        gx.Continuation(
            adam(),
            stages=3,
            on_stage=calls.append,
            generations=3,
            on_stage_finished=finished,
        ).run(problem, gradient=True, generations=100)
    # the run stopped before the second stage's parameters
    assert calls == [0]


@pytest.mark.parametrize(
    "settings, error, message",
    [
        ({"stages": 0}, ValueError, "stages is at least 1"),
        ({"stages": 2, "generations": 0}, ValueError, "generations is at least 1"),
        ({"stages": 2, "keep": "both"}, ValueError, 'keep is "state" or "point"'),
        ({"stages": 2, "on_stage": 3}, TypeError, "on_stage isn't callable"),
        ({"stages": 2, "on_stage_finished": 3}, TypeError, "on_stage_finished isn't callable"),
    ],
)
def test_settings_are_checked(settings, error, message):
    problem = Smoothed()
    continuation = gx.Continuation(adam(), **{"on_stage": problem.on_stage, **settings})
    with pytest.raises(error, match=message):
        continuation.run(problem, gradient=True, generations=10)


def test_only_gradient_methods_are_wrapped():
    problem = Smoothed()
    nelder_mead = gx.NelderMead(gx.Real((-5, 5), length=3), objective="minimize")
    with pytest.raises(ValueError, match="gx.FirstOrder, gx.Lbfgsb or gx.Mma"):
        gx.Continuation(nelder_mead, stages=2, on_stage=problem.on_stage).run(  # type: ignore[arg-type]
            problem, gradient=True, generations=10
        )
    with pytest.raises(ValueError, match="constraints need a gx.Mma"):
        gx.Continuation(adam(), stages=2, on_stage=problem.on_stage).run(
            problem, gradient=True, constraints=1, generations=10
        )
