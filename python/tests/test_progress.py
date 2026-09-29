"""Progress objects: their scalars are set at once, and the population's arrays are made when first
read, from the run's copy of the population, then kept."""

import copy
import dataclasses
import pickle

import numpy as np
import pytest

import genoxide as gx

# a genome of each type, with operators for it; 70 and 128 bits end inside and at a 64-bit word
GENOMES = {
    "binary-70": (gx.Binary(70), gx.UniformCrossover(), gx.BitFlip(rate=0.05), bool),
    "binary-128": (gx.Binary(128), gx.PointCrossover(2), gx.BitFlip(rate=0.02), bool),
    "integer": (
        gx.Integer((-3, 9), length=6),
        gx.UniformCrossover(),
        gx.UniformMutation(rate=0.3),
        np.int64,
    ),
    "real": (
        gx.Real((-5.0, 5.0), length=4),
        gx.SimulatedBinaryCrossover(15.0),
        gx.PolynomialMutation(rate=0.5),
        np.float64,
    ),
    "permutation": (gx.Permutation(7), gx.OrderCrossover(), gx.SwapMutation(), np.int64),
}


class Recorder:
    """A fitness function that records what it returns for each genome: a score and a violation,
    or None (an invalid solution) for some genomes."""

    def __init__(self):
        self.values = {}

    def __call__(self, genome):
        genes = genome.astype(np.float64)
        weights = np.arange(1, len(genes) + 1)
        total = float(np.sum(genes * weights))
        if int(abs(total) * 7) % 5 == 0:
            value = None
        else:
            value = (total, max(0.0, float(genes[0]) - 0.5))
        self.values[genome.tobytes()] = value
        return value


def ga(name, **settings):
    genome, crossover, mutation, _ = GENOMES[name]
    return gx.Ga(
        genome,
        population_size=12,
        select=gx.Tournament(2),
        crossover=crossover,
        mutation=mutation,
        seed=3,
        **settings,
    )


def assert_matches(progress, recorder, dtype):
    """The progress's population, scores and violations are what the fitness function got and
    returned: the eager arrays of earlier versions."""
    population, scores, violations = progress.population, progress.scores, progress.violations
    assert population.dtype == dtype and population.ndim == 2
    assert len(population) == len(scores) == len(violations) == 12
    assert scores.dtype == violations.dtype == np.float64
    for genome, score, violation in zip(population, scores, violations):
        value = recorder.values[genome.tobytes()]
        if value is None:
            assert np.isnan(score) and np.isnan(violation)
        else:
            assert (score, violation) == value


@pytest.mark.parametrize("name", GENOMES)
def test_the_population_is_the_one_evaluated(name):
    recorder = Recorder()
    progress = []
    ga(name).run(recorder, generations=8, on_generation=progress.append)
    assert len(progress) == 9
    dtype = GENOMES[name][3]
    # read after the run: each progress object kept its generation's population
    for state in progress:
        assert_matches(state, recorder, dtype)
    if name == "permutation":
        assert all(sorted(row) == list(range(7)) for row in progress[-1].population)


@pytest.mark.parametrize("name", GENOMES)
def test_read_in_the_callback_or_after_it_the_same(name):
    during = []

    def read(progress):
        during.append((progress.population, progress.scores, progress.violations))

    kept = []
    ga(name).run(Recorder(), generations=6, on_generation=read)
    ga(name).run(Recorder(), generations=6, on_generation=kept.append)
    assert len(during) == len(kept) == 7
    for (population, scores, violations), state in zip(during, kept):
        assert np.array_equal(population, state.population)
        assert np.array_equal(scores, state.scores, equal_nan=True)
        assert np.array_equal(violations, state.violations, equal_nan=True)


def test_the_arrays_are_made_when_first_read_and_then_kept():
    progress = []
    ga("binary-70").run(Recorder(), generations=3, on_generation=progress.append)
    state = progress[-1]
    lazy = ("population", "scores", "violations")
    # nothing made for the population before it's read: the run's copy only
    assert not any(name in vars(state) for name in lazy)
    assert type(state._population).__name__ == "Snapshot"
    assert state.population is state.population
    assert "population" in vars(state)
    assert "scores" not in vars(state) and "violations" not in vars(state)
    assert state.scores is state.scores and state.violations is state.violations


def test_a_progress_object_is_read_only():
    progress = []
    ga("real").run(Recorder(), generations=2, on_generation=progress.append)
    state = progress[-1]
    for name in ("generation", "best_fitness", "population", "scores"):
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(state, name, None)
        with pytest.raises(AttributeError):
            delattr(state, name)
    assert state.generation == 2
    assert not dataclasses.is_dataclass(state)


def test_repr_and_pattern_matching():
    progress = []
    ga("real").run(Recorder(), generations=2, on_generation=progress.append)
    state = progress[-1]
    assert repr(state) == (
        f"Progress(generation=2, evaluations={state.evaluations}, seconds={state.seconds!r}, "
        f"best_fitness={state.best_fitness!r})"
    )
    match state:
        case gx.Progress(generation, evaluations, _, best, _, population, scores, _):
            assert (generation, evaluations, best) == (2, state.evaluations, state.best_fitness)
            assert population is state.population and scores is state.scores


@pytest.mark.parametrize("name", GENOMES)
def test_copies_and_pickles_have_the_arrays(name):
    progress = []
    ga(name).run(Recorder(), generations=2, on_generation=progress.append)
    state = progress[-1]
    for other in (pickle.loads(pickle.dumps(state)), copy.deepcopy(state), copy.copy(state)):
        assert type(other) is gx.Progress
        assert repr(other) == repr(state)
        assert np.array_equal(other.best_genome, state.best_genome)
        assert np.array_equal(other.population, state.population)
        assert other.population.dtype == state.population.dtype
        assert np.array_equal(other.scores, state.scores, equal_nan=True)
        assert np.array_equal(other.violations, state.violations, equal_nan=True)


def test_control_and_on_generation_get_the_same_population():
    seen = {}

    def on_generation(progress):
        seen[progress.generation] = progress

    def control(running, progress):
        other = seen[progress.generation]
        assert other is not progress
        assert other.evaluations == progress.evaluations
        assert other.best_fitness == progress.best_fitness
        assert np.array_equal(other.population, progress.population)
        assert np.array_equal(other.scores, progress.scores, equal_nan=True)
        assert np.array_equal(other.violations, progress.violations, equal_nan=True)
        assert np.array_equal(other.best_genome, progress.best_genome)
        seen[progress.generation] = None

    ga("integer").run(Recorder(), generations=5, on_generation=on_generation, control=control)
    assert list(seen) == list(range(6)) and all(state is None for state in seen.values())


MULTI_ARRAYS = ("population", "objectives", "violations", "front_objectives", "front_violations")


def objectives(x):
    """Two objectives and a violation, or None for some genomes."""
    if x[1] > 0.9:
        return None
    return [x[0], (1 - x[0]) * (1 + x[1:].sum())], max(0.0, 0.3 - x[1])


def test_multi_progress_arrays():
    recorded = {}

    def fitness(x):
        value = objectives(x)
        recorded[x.tobytes()] = value
        return value

    progress = []
    gx.Nsga2(
        gx.Real((0.0, 1.0), length=3),
        objectives=["minimize", "minimize"],
        population_size=16,
        crossover=gx.SimulatedBinaryCrossover(15.0),
        mutation=gx.PolynomialMutation(rate=0.5),
        seed=5,
    ).run(fitness, generations=6, on_generation=progress.append)
    assert len(progress) == 7
    for state in progress:
        assert not any(name in vars(state) for name in MULTI_ARRAYS)
        assert state.population.shape == (16, 3) and state.objectives.shape == (16, 2)
        for genome, values, violation in zip(state.population, state.objectives, state.violations):
            value = recorded[genome.tobytes()]
            if value is None:
                assert np.isnan(values).all() and np.isnan(violation)
            else:
                assert list(values) == value[0] and violation == value[1]
        assert state.front_objectives.shape == (state.front_size, 2)
        assert state.front_violations.shape == (state.front_size,)
        # NaN for an invalid solution: -1 to compare
        rows = np.nan_to_num(np.column_stack([state.objectives, state.violations]), nan=-1.0)
        front = np.column_stack([state.front_objectives, state.front_violations])
        front = np.nan_to_num(front, nan=-1.0)
        assert all((rows == row).all(axis=1).any() for row in front)
    state = progress[-1]
    assert repr(state) == (
        f"MultiProgress(generation=6, evaluations={state.evaluations}, "
        f"seconds={state.seconds!r}, front_size={state.front_size})"
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        state.front_size = 0
    for other in (pickle.loads(pickle.dumps(state)), copy.deepcopy(state)):
        assert type(other) is gx.MultiProgress and repr(other) == repr(state)
        for name in MULTI_ARRAYS:
            assert np.array_equal(getattr(other, name), getattr(state, name), equal_nan=True)
