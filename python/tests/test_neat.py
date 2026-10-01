"""gx.Neat, NEAT's networks (gx.neat), their evaluators and their policies on the control tasks."""

import pickle

import numpy as np
import pytest

import genoxide as gx
from genoxide.problems.control import SUCCESS_STEPS, CartPole, DoublePole

CASES = [((0.0, 0.0), 0.0), ((0.0, 1.0), 1.0), ((1.0, 0.0), 1.0), ((1.0, 1.0), 0.0)]
INPUTS = np.array([inputs for inputs, _ in CASES])
TARGETS = np.array([target for _, target in CASES])


def xor(network):
    """The paper's fitness, (4 - sum |error|)^2, 16 for a perfect network."""
    outputs = network.feed_forward().activate(INPUTS)[:, 0]
    error = float(np.cumsum(np.abs(outputs - TARGETS))[-1])
    return (4.0 - error) * (4.0 - error)


def same(a, b):
    return (
        a.best_fitness == b.best_fitness
        and a.best_genome == b.best_genome
        and (a.generations, a.evaluations, a.stop_reason)
        == (b.generations, b.evaluations, b.stop_reason)
    )


def steep_sigmoid(x):
    return 1.0 / (1.0 + gx.math.exp(-4.9 * x))


# --- runs --------------------------------------------------------------------------------------


def test_neat_solves_xor():
    # the Rust doc example's run, with the paper's fitness sharing
    result = gx.Neat(2, 1, sharing="raw", seed=1).run(xor, target=15.0, generations=500)
    assert result.stop_reason == "target"
    assert result.best_fitness >= 15.0
    network = result.best_genome
    assert isinstance(network, gx.neat.Network)
    assert network.hidden() >= 1  # XOR needs a hidden node
    assert xor(network) == result.best_fitness
    outputs = network.feed_forward().activate(INPUTS)[:, 0]
    assert np.array_equal(outputs >= 0.5, TARGETS == 1.0)
    # the same seed repeats the run
    assert same(result, gx.Neat(2, 1, sharing="raw", seed=1).run(xor, target=15.0, generations=500))


def test_the_settings_reach_the_run():
    settings = {
        "population_size": 40,
        "compatibility": (1.0, 1.0, 3.0, 4.0),
        "weight_mutation": (0.9, 0.2),
        "weight_deviations": (0.5, 2.0),
        "structural_mutation": (0.2, 0.3),
        "reproduction": (0.5, 0.01, 0.5),
        "selection": (3, 0.5),
        "stagnation": 5,
        "activation": "tanh",
        "feed_forward": False,
        "initial": "unconnected",
        "sharing": "normalized",
        "objective": "minimize",
        "seed": 7,
    }
    seen = []

    def error(network):
        outputs = network.recurrent().activate(INPUTS)[:, 0]
        return float(np.sum((outputs - TARGETS) ** 2))

    result = gx.Neat(3 - 1, 1, **settings).run(
        error, generations=10, on_generation=lambda progress: seen.append(progress) and None
    )
    assert result.generations == 10
    assert result.evaluations <= 40 * 11
    # unconnected networks to start from
    first = seen[0].population
    assert len(first) == 40 and all(network.enabled() == 0 for network in first)
    # the activation of the outputs and the hidden nodes
    for network in seen[-1].population:
        for node in network.nodes():
            assert node.activation == ("tanh" if node.kind in ("output", "hidden") else "identity")
    # minimized: the best score is the smallest seen
    assert result.best_fitness == min(np.nanmin(progress.scores) for progress in seen)
    # each setting makes a difference
    for name, value in [
        ("compatibility", (1.0, 1.0, 0.4, 3.0)),
        ("weight_mutation", (0.8, 0.1)),
        ("structural_mutation", (0.03, 0.05)),
        ("initial", "fully_connected"),
    ]:
        other = gx.Neat(2, 1, **{**settings, name: value}).run(error, generations=10)
        assert not same(result, other), name


def test_defaults_are_the_papers():
    # the defaults given explicitly change nothing
    explicit = gx.Neat(
        2,
        1,
        population_size=150,
        compatibility=(1.0, 1.0, 0.4, 3.0),
        weight_mutation=(0.8, 0.1),
        weight_deviations=(1.0, 1.0),
        structural_mutation=(0.03, 0.05),
        reproduction=(0.25, 0.001, 0.75),
        selection=(5, 0.2),
        stagnation=15,
        activation="steep_sigmoid",
        feed_forward=True,
        initial="fully_connected",
        sharing="normalized",
        objective="maximize",
        seed=3,
    )
    implicit = gx.Neat(2, 1, seed=3)
    assert same(explicit.run(xor, generations=5), implicit.run(xor, generations=5))


@pytest.mark.parametrize(
    "settings, setting",
    [
        ({"inputs": 0}, "inputs"),
        ({"outputs": 2**24 + 1}, "outputs"),
        ({"inputs": 1.0}, "inputs"),
        ({"population_size": 0}, "population_size"),
        ({"compatibility": (1.0, 1.0, -0.4, 3.0)}, "compatibility"),
        ({"compatibility": (1.0, 1.0, 0.4, 0.0)}, "compatibility"),
        ({"compatibility": (1.0, 1.0, 0.4)}, "compatibility"),
        ({"compatibility": "1, 1, 0.4, 3"}, "compatibility"),
        ({"weight_mutation": (1.5, 0.1)}, "weight_mutation"),
        ({"weight_mutation": (0.8, float("nan"))}, "weight_mutation"),
        ({"weight_deviations": (0.0, 1.0)}, "weight_deviations"),
        ({"structural_mutation": (0.03, -0.05)}, "structural_mutation"),
        ({"reproduction": (0.25, 0.001, 1.75)}, "reproduction"),
        ({"selection": (5, 0.0)}, "selection"),
        ({"selection": (5.0, 0.2)}, "selection"),
        ({"stagnation": 0}, "stagnation"),
        ({"activation": "softmax"}, "activation"),
        ({"feed_forward": 1}, "feed_forward"),
        ({"initial": "empty"}, "initial"),
        ({"sharing": "shared"}, "sharing"),
        ({"sharing": "raw", "objective": "minimize"}, "sharing"),
        ({"objective": "maximise"}, "objective"),
        ({"seed": -1}, "seed"),
    ],
)
def test_wrong_settings_name_the_setting(settings, setting):
    arguments = {"inputs": 2, "outputs": 1, **settings}
    neat = gx.Neat(arguments.pop("inputs"), arguments.pop("outputs"), **arguments)
    with pytest.raises(ValueError, match=setting):
        neat.run(xor, generations=1)


def test_raw_sharing_needs_valid_non_negative_scores():
    with pytest.raises(ValueError, match="fitness"):
        gx.Neat(2, 1, sharing="raw", seed=1).run(lambda network: -1.0, generations=2)


def test_neat_takes_networks_not_problems():
    with pytest.raises(ValueError, match="gx.neat.Network"):
        gx.Neat(2, 1, seed=1).run(gx.problems.Sphere(3), generations=1)


def test_batch_and_parallel_runs_equal_the_plain_one():
    neat = gx.Neat(2, 1, population_size=50, seed=4)
    plain = neat.run(xor, generations=15)
    batches = []

    def batch(networks):
        batches.append(len(networks))
        assert isinstance(networks, tuple)
        return np.array([xor(network) for network in networks])

    assert same(plain, neat.run(batch, batch=True, generations=15))
    assert batches[0] == 50 and sum(batches) == plain.evaluations
    assert same(plain, neat.run(xor, parallel=True, generations=15))


def test_progress_has_the_networks():
    seen = []
    result = gx.Neat(2, 1, population_size=30, seed=2).run(
        xor, generations=8, on_generation=seen.append
    )
    assert [progress.generation for progress in seen] == list(range(9))
    last = seen[-1]
    assert isinstance(last, gx.NeatProgress)
    assert isinstance(last.population, tuple) and len(last.population) == 30
    assert all(isinstance(network, gx.neat.Network) for network in last.population)
    assert last.scores.shape == (30,) and last.violations.shape == (30,)
    assert last.best_genome == result.best_genome and last.best_fitness == result.best_fitness
    # the scores are the networks'
    assert all(score == xor(network) for network, score in zip(last.population, last.scores))
    # a progress object pickles, its networks with it
    again = pickle.loads(pickle.dumps(last))
    assert again.population == last.population and again.best_genome == last.best_genome
    assert np.array_equal(again.scores, last.scores)


def test_running_neat_has_the_species():
    seen = {}

    def control(running, progress):
        assert isinstance(running, gx.RunningNeat)
        assert isinstance(progress, gx.NeatProgress)
        species = running.species
        seen[progress.generation] = species
        assert running.seed == 5
        assert running.innovations >= 3  # the initial networks' connections
        # each network in one species
        members = sorted(member for one in species for member in one.members)
        assert members == list(range(len(progress.population)))
        for one in species:
            assert isinstance(one, gx.neat.Species)
            assert isinstance(one.representative, gx.neat.Network)
            assert one.created <= one.improved <= progress.generation
            assert one.best_fitness is None or one.best_fitness <= progress.best_fitness

    gx.Neat(2, 1, compatibility=(1.0, 1.0, 0.4, 1.0), seed=5).run(
        xor, generations=10, control=control
    )
    assert len(seen) == 11
    assert len(seen[10]) > 1
    ids = [one.id for one in seen[10]]
    assert ids == sorted(ids)
    with pytest.raises(ValueError, match="no setting"):
        gx.Neat(2, 1, seed=5).run(
            xor, generations=1, control=lambda running, _: running._set("species", [])
        )


def test_a_resumed_run_equals_an_uninterrupted_one(tmp_path):
    neat = gx.Neat(2, 1, sharing="raw", seed=1)
    whole = neat.run(xor, generations=30)
    path = tmp_path / "neat.ckpt"
    neat.run(xor, generations=12, checkpoint=path, checkpoint_every=5)
    assert same(whole, neat.run(xor, generations=30, resume=path))
    with pytest.raises(ValueError, match="other settings"):
        gx.Neat(2, 1, seed=1).run(xor, generations=30, resume=path)


# --- networks and evaluators -------------------------------------------------------------------


def test_a_fully_connected_network():
    network = gx.neat.Network.fully_connected(2, 1, [0.5, -0.5, 0.25])
    assert (network.inputs, network.outputs, network.hidden(), network.enabled()) == (2, 1, 0, 3)
    assert len(network) == 3
    kinds = [(node.id, node.kind, node.activation) for node in network.nodes()]
    assert kinds == [
        (0, "input", "identity"),
        (1, "input", "identity"),
        (2, "bias", "identity"),
        (3, "output", "steep_sigmoid"),
    ]
    genes = [(c.innovation, c.from_, c.to, c.weight, c.enabled) for c in network.connections()]
    assert genes == [(0, 0, 3, 0.5, True), (1, 1, 3, -0.5, True), (2, 2, 3, 0.25, True)]
    # by hand: the steepened sigmoid of the weighted sum, the bias's input 1
    evaluator = network.feed_forward()
    for inputs in ([0.0, 0.0], [1.0, 1.0], [2.0, -1.0]):
        expected = steep_sigmoid(0.5 * inputs[0] - 0.5 * inputs[1] + 0.25)
        assert evaluator.activate(inputs)[0] == expected
    assert "hidden=0" in repr(network)
    with pytest.raises(ValueError, match="weights"):
        gx.neat.Network.fully_connected(2, 1, [0.5])
    with pytest.raises(ValueError, match="inputs"):
        gx.neat.Network.fully_connected(0, 1, [0.5])


def test_networks_compare_hash_and_pickle():
    a = gx.neat.Network.fully_connected(2, 1, [0.5, -0.5, 0.25])
    b = gx.neat.Network.fully_connected(2, 1, [0.5, -0.5, 0.25])
    c = gx.neat.Network.fully_connected(2, 1, [0.5, -0.5, 0.5])
    assert a == b and a != c and hash(a) == hash(b)
    assert len({a, b, c}) == 2
    assert pickle.loads(pickle.dumps(a)) == a
    with pytest.raises(ValueError, match="NEAT network"):
        gx._genoxide.neat_network("{}")


def test_activate_takes_rows_lists_and_an_output_array():
    network = gx.Neat(2, 1, sharing="raw", seed=1).run(xor, target=15.0, generations=500)
    evaluator = network.best_genome.feed_forward()
    rows = evaluator.activate(INPUTS)
    assert rows.shape == (4, 1) and rows.dtype == np.float64
    for row, inputs in zip(rows, INPUTS):
        assert evaluator.activate(inputs)[0] == row[0]
        assert evaluator.activate(list(inputs))[0] == row[0]
        out = np.zeros(1)
        assert evaluator.activate(inputs, out) is out and out[0] == row[0]
    # integers and lists of rows, through numpy
    assert np.array_equal(evaluator.activate(INPUTS.astype(np.int64)), rows)
    assert np.array_equal(evaluator.activate(INPUTS.tolist()), rows)
    with pytest.raises(ValueError, match="2 inputs"):
        evaluator.activate([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="2 inputs"):
        evaluator.activate(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="outputs"):
        evaluator.activate([1.0, 2.0], np.zeros(2))
    with pytest.raises(ValueError, match="1-D or a 2-D"):
        evaluator.activate(["a", "b"])


def recurrent_xor(network):
    """XOR's fitness of a recurrent network: the four cases as a sequence."""
    outputs = network.recurrent().activate(INPUTS)[:, 0]
    error = float(np.cumsum(np.abs(outputs - TARGETS))[-1])
    return (4.0 - error) * (4.0 - error)


def recurrent_networks():
    """The networks of a short run with recurrent connections."""
    seen = []
    gx.Neat(
        2, 1, structural_mutation=(0.3, 0.5), feed_forward=False, population_size=50, seed=1
    ).run(recurrent_xor, generations=10, on_generation=seen.append)
    return [network for progress in seen for network in progress.population]


def test_the_recurrent_evaluator_steps_through_time():
    networks = recurrent_networks()
    # some networks have cycles: they have no feed-forward evaluator
    cyclic = 0
    for network in networks:
        try:
            network.feed_forward()
        except ValueError as error:
            assert "cycle" in str(error)
            cyclic += 1
    assert cyclic > 0
    network = max(networks, key=lambda network: (network.hidden(), network.enabled()))
    assert network.hidden() > 0
    sequence = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, -0.5]])
    evaluator = network.recurrent()
    steps = [evaluator.activate(inputs)[0] for inputs in sequence]
    # the rows of a 2-D input are the steps of a sequence, from the current state
    evaluator.reset()
    assert evaluator.activate(sequence)[:, 0].tolist() == steps
    # reset forgets the past; a fresh evaluator starts from 0 too
    evaluator.reset()
    assert [evaluator.activate(inputs)[0] for inputs in sequence] == steps
    assert network.recurrent().activate(sequence)[:, 0].tolist() == steps


# --- policies ----------------------------------------------------------------------------------


def python_policy(evaluator, scale=1.0, offset=0.0):
    """The evaluator as a Python callable policy, slow but the same arithmetic."""

    def act(observation, action):
        action[0] = scale * evaluator.activate(observation)[0] + offset

    return act


def cart_pole_networks():
    seen = []
    gx.Neat(4, 1, population_size=20, seed=3).run(
        lambda network: network.enabled(), generations=3, on_generation=seen.append
    )
    return [network for progress in seen for network in progress.population][::7]


def test_evaluators_and_their_policies_balance_in_rust():
    task = CartPole()
    for network in cart_pole_networks():
        mapped = task.run(network.feed_forward().policy(scale=2.0, offset=-1.0), 2_000)
        python = task.run(python_policy(network.feed_forward(), 2.0, -1.0), 2_000)
        assert mapped == python
        # the evaluator itself: its outputs are the actions
        plain = task.run(network.feed_forward(), 2_000)
        assert plain == task.run(network.feed_forward().policy(), 2_000)
        assert plain == task.run(python_policy(network.feed_forward()), 2_000)
        # the same with the recurrent evaluator, a step late through hidden nodes
        assert task.run(network.recurrent(), 2_000) == task.run(
            python_policy(network.recurrent()), 2_000
        )
        # an episode's states, up to the step that failed
        states = task.episode(network.feed_forward().policy(scale=2.0, offset=-1.0), 2_000)
        assert states.shape == (min(mapped + 1, 2_000), 4)
        assert np.all(np.abs(states[:-1, 0]) <= 2.4)


def test_a_recurrent_policy_starts_each_episode_afresh():
    task = DoublePole(velocities=False)
    seen = []
    gx.Neat(
        3, 1, structural_mutation=(0.3, 0.5), feed_forward=False, population_size=30, seed=2
    ).run(lambda network: network.enabled(), generations=10, on_generation=seen.append)
    network = max(seen[-1].population, key=lambda network: (network.hidden(), len(network)))
    assert network.hidden() > 0
    evaluator = network.recurrent()
    evaluator.activate([5.0, 5.0, 5.0])  # a state the task's copy doesn't keep
    policy = evaluator.policy(scale=2.0, offset=-1.0)
    first = task.damping_fitness(policy)
    assert first == task.damping_fitness(policy)
    fresh = network.recurrent()
    assert first == task.damping_fitness(python_policy(fresh, 2.0, -1.0))


def test_policies_are_checked():
    network = gx.neat.Network.fully_connected(2, 1, [0.5, -0.5, 0.25])
    with pytest.raises(ValueError, match="4 inputs"):
        CartPole().run(network.feed_forward(), 10)
    with pytest.raises(ValueError, match="scale"):
        network.feed_forward().policy(scale=float("inf"))
    with pytest.raises(TypeError):
        network.feed_forward().policy(2.0, -1.0)
    assert "NEAT" in repr(network.recurrent().policy(scale=2.0, offset=-1.0))


def test_neat_balances_the_pole():
    # the cart_pole example's NEAT run: the initial population has a solution
    task = CartPole()

    def steps(network):
        return task.run(network.feed_forward().policy(scale=2.0, offset=-1.0), SUCCESS_STEPS)

    result = gx.Neat(4, 1, seed=1).run(steps, target=SUCCESS_STEPS, evaluations=100_000)
    assert (result.best_fitness, result.evaluations, result.generations) == (SUCCESS_STEPS, 150, 0)
    assert task.solved(result.best_genome.feed_forward().policy(scale=2.0, offset=-1.0))
