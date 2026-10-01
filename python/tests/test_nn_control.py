"""gx.nn's networks, gx.problems.control's tasks and Balance, and gx.math."""

import math

import numpy as np
import pytest

import genoxide as gx
from genoxide.problems.control import (
    DAMPING_STEPS,
    GENERALIZATION_THRESHOLD,
    SUCCESS_STEPS,
    Balance,
    CartPole,
    DoublePole,
)
from test_docs import docstring_examples

ACTIVATIONS = {
    "identity": lambda x: x,
    "tanh": np.tanh,
    "sigmoid": lambda x: 1 / (1 + np.exp(-x)),
    "relu": lambda x: np.maximum(x, 0.0),
    "steep_sigmoid": lambda x: 1 / (1 + np.exp(-4.9 * x)),
}


def numpy_forward(layers, activation, output_activation, bias, weights, inputs):
    """An MLP's forward pass in numpy: each unit's weights for its inputs, then its bias."""
    values = np.atleast_2d(inputs)
    start = 0
    for layer in range(1, len(layers)):
        n, units = layers[layer - 1], layers[layer]
        per_unit = n + int(bias)
        matrix = weights[start : start + units * per_unit].reshape(units, per_unit)
        start += units * per_unit
        sums = values @ matrix[:, :n].T
        if bias:
            sums = sums + matrix[:, n]
        last = layer == len(layers) - 1
        values = ACTIVATIONS[output_activation if last else activation](sums)
    return values


@pytest.mark.parametrize("activation", list(ACTIVATIONS))
@pytest.mark.parametrize("bias", [True, False])
def test_mlp_forward_equals_a_numpy_forward_pass(activation, bias):
    layers = [3, 5, 4, 2]
    mlp = gx.nn.Mlp(layers, activation, output_activation="tanh", bias=bias)
    expected_parameters = sum(
        layers[i] * (layers[i - 1] + int(bias)) for i in range(1, len(layers))
    )
    assert mlp.parameters == expected_parameters
    assert (mlp.inputs, mlp.outputs) == (3, 2)
    rng = np.random.default_rng(1)
    weights = rng.uniform(-1, 1, mlp.parameters)
    inputs = rng.uniform(-2, 2, (10, 3))
    outputs = mlp.forward(weights, inputs)
    assert outputs.shape == (10, 2) and outputs.dtype == np.float64
    expected = numpy_forward(layers, activation, "tanh", bias, weights, inputs)
    assert np.allclose(outputs, expected, rtol=1e-12, atol=1e-12)
    # one input as a 1-D array gives a 1-D array, the same row
    assert np.array_equal(mlp.forward(weights, inputs[3]), outputs[3])
    # lists do
    assert np.array_equal(mlp.forward(list(weights), inputs.tolist()), outputs)


def test_the_networks_use_portable_math():
    mlp = gx.nn.Mlp([1, 1], "identity", output_activation="tanh", bias=False)
    x = np.linspace(-3, 3, 101)
    assert np.array_equal(mlp.forward([1.0], x[:, None])[:, 0], gx.math.tanh(x))


def test_elman_forward_is_a_sequence():
    # one input, one hidden unit that adds the input to its previous output: a running sum
    elman = gx.nn.Elman(1, 1, 1, "identity", bias=False)
    assert elman.parameters == 3
    outputs = elman.forward([1.0, 1.0, 1.0], [[1.0], [2.0], [3.0]])
    assert outputs[:, 0].tolist() == [1.0, 3.0, 6.0]
    # each call starts from a context of 0
    assert elman.forward([1.0, 1.0, 1.0], [5.0]).tolist() == [5.0]
    with_bias = gx.nn.Elman(3, 4, 2, "tanh")
    assert with_bias.parameters == 4 * (3 + 4 + 1) + 2 * (4 + 1)


def test_network_settings_are_checked():
    with pytest.raises(ValueError, match="layers"):
        gx.nn.Mlp([3]).parameters
    with pytest.raises(ValueError, match="layers"):
        gx.nn.Mlp([3, 0, 1]).parameters
    with pytest.raises(ValueError, match="layers"):
        gx.nn.Mlp([3, 1.5, 1]).parameters
    with pytest.raises(ValueError, match="activation"):
        gx.nn.Mlp([3, 1], "softmax").parameters
    with pytest.raises(ValueError, match="output_activation"):
        gx.nn.Mlp([3, 1], output_activation="linear").parameters
    with pytest.raises(ValueError, match="bias"):
        gx.nn.Mlp([3, 1], bias=1).parameters
    with pytest.raises(ValueError, match="hidden"):
        gx.nn.Elman(3, 0, 1).parameters
    mlp = gx.nn.Mlp([3, 2, 1])
    with pytest.raises(ValueError, match="11 weights, not 10"):
        mlp.forward(np.zeros(10), np.zeros(3))
    with pytest.raises(ValueError, match="3 inputs"):
        mlp.forward(np.zeros(11), np.zeros((4, 2)))
    with pytest.raises(ValueError, match="11 weights, not 12"):
        mlp.policy(np.zeros(12))


def test_representation():
    mlp = gx.nn.Mlp([4, 3, 1])
    assert mlp.representation((-1.0, 1.0)) == gx.Real((-1.0, 1.0), length=19)


def test_a_policy_in_task_run_equals_its_network():
    mlp = gx.nn.Mlp([4, 2, 1], "tanh", output_activation="tanh", bias=False)
    weights = np.array([0.5, 0.2, 1.0, 0.3, -0.1, 0.4, 0.8, 0.2, 1.0, 0.5])
    task = CartPole()
    steps = task.run(mlp.policy(weights), 1000)

    # the same network, as a Python policy calling its forward pass
    def policy(observation, action):
        action[0] = mlp.forward(weights, observation)[0]

    assert task.run(policy, 1000) == steps
    assert task.observations == 4
    assert DoublePole().observations == 6
    assert DoublePole(velocities=False).observations == 3


def test_a_python_callable_policy():
    # push the cart towards where the pole leans, harder the faster it falls
    def push(observation, action):
        x, velocity, angle, angular_velocity = observation
        action[0] = 2.0 * angle + angular_velocity + 0.1 * x + 0.2 * velocity

    task = CartPole()
    assert task.run(push, 1000) == 1000
    # doing nothing lets the pole fall
    assert task.run(lambda observation, action: None, 1000) < 100

    def broken(observation, action):
        raise RuntimeError("no policy")

    with pytest.raises(RuntimeError, match="no policy"):
        task.run(broken, 1000)
    with pytest.raises(TypeError, match="policy"):
        task.run(42, 10)


def test_a_policy_of_the_wrong_size_is_an_error():
    policy = gx.nn.Mlp([6, 1]).policy(np.zeros(7))
    with pytest.raises(ValueError, match="CartPole needs a network of 4 inputs and 1 output"):
        CartPole().run(policy, 10)
    with pytest.raises(ValueError, match="DoublePole needs a network of 3 inputs"):
        DoublePole(velocities=False).run(policy, 10)


def test_double_pole_damping_and_generalization():
    elman = gx.nn.Elman(3, 5, 1, "tanh", output_activation="tanh", bias=False)
    weights = elman.representation((-1, 1)).random_genome(3)
    task = DoublePole(velocities=False)
    policy = elman.policy(weights)
    steps = task.run(policy, DAMPING_STEPS)
    fitness = task.damping_fitness(policy)
    assert fitness >= 0.1 * steps / DAMPING_STEPS
    assert 0 <= task.generalization(policy) <= 625
    assert not task.solved(policy)
    with pytest.raises(ValueError, match="damping_fitness is DoublePole's"):
        CartPole()._native.damping_fitness(policy)
    assert (SUCCESS_STEPS, DAMPING_STEPS, GENERALIZATION_THRESHOLD) == (100_000, 1000, 200)


def test_balance_in_run_equals_a_python_loop_over_task_run():
    mlp = gx.nn.Mlp([4, 2, 1], "tanh", output_activation="tanh", bias=False)
    task = CartPole()

    def cmaes():
        return gx.Cmaes(mlp.representation((-1, 1)), seed=1)

    balance = Balance(task, mlp)
    native = cmaes().run(balance, target=SUCCESS_STEPS, evaluations=10_000)
    looped = cmaes().run(
        lambda weights: task.run(mlp.policy(weights), SUCCESS_STEPS),
        target=SUCCESS_STEPS,
        evaluations=10_000,
    )
    assert native.stop_reason == "target"
    assert native.best_fitness == looped.best_fitness == SUCCESS_STEPS
    assert np.array_equal(native.best_genome, looped.best_genome)
    assert native.evaluations == looped.evaluations
    assert task.solved(mlp.policy(native.best_genome))
    # parallel: the same run
    parallel = cmaes().run(balance, target=SUCCESS_STEPS, evaluations=10_000, parallel=True)
    assert np.array_equal(native.best_genome, parallel.best_genome)
    # calling it runs the same Rust code
    assert balance(native.best_genome) == SUCCESS_STEPS
    genomes = np.stack([native.best_genome, np.zeros(10)])
    assert balance.evaluate(genomes).tolist() == [
        task.run(mlp.policy(genome), SUCCESS_STEPS) for genome in genomes
    ]


def test_balance_with_the_damping_fitness():
    elman = gx.nn.Elman(3, 2, 1, "tanh", output_activation="tanh", bias=False)
    task = DoublePole(velocities=False)
    balance = Balance(task, elman, fitness="damping")
    weights = elman.representation((-1, 1)).random_genome(5)
    assert balance(weights) == task.damping_fitness(elman.policy(weights))
    short = Balance(CartPole(), gx.nn.Mlp([4, 1]), steps=50)
    assert short(np.ones(5)) <= 50


def test_balance_checks_the_run():
    mlp = gx.nn.Mlp([4, 2, 1], bias=False)
    balance = Balance(CartPole(), mlp)
    with pytest.raises(ValueError, match="10 weights, but the genome has 9 genes"):
        gx.Cmaes(gx.Real((-1, 1), length=9), seed=1).run(balance, generations=1)
    with pytest.raises(ValueError, match="maximizes"):
        gx.Cmaes(mlp.representation((-1, 1)), objective="minimize").run(balance, generations=1)
    with pytest.raises(ValueError, match="Balance needs a Real genome"):
        gx.Ga(
            gx.Binary(10),
            population_size=10,
            select=gx.Tournament(2),
            crossover=gx.UniformCrossover(),
            mutation=gx.BitFlip(rate=0.1),
        ).run(balance, generations=1)
    with pytest.raises(ValueError, match="single-objective"):
        gx.Nsga2(
            mlp.representation((-1, 1)),
            objectives=["maximize", "maximize"],
            population_size=10,
            crossover=gx.UniformCrossover(),
            mutation=gx.PolynomialMutation(20, rate=0.1),
        ).run(balance, generations=1)
    with pytest.raises(ValueError, match="CartPole needs a network of 4 inputs"):
        Balance(CartPole(), gx.nn.Mlp([3, 1]))(np.zeros(4))
    with pytest.raises(ValueError, match="damping fitness is DoublePole's"):
        Balance(CartPole(), mlp, fitness="damping")(np.zeros(10))
    with pytest.raises(ValueError, match="fitness"):
        Balance(CartPole(), mlp, fitness="time")(np.zeros(10))
    with pytest.raises(ValueError, match="steps"):
        Balance(CartPole(), mlp, steps=-1)(np.zeros(10))


def test_math_is_portable_math():
    x = np.linspace(-10, 10, 201)
    for name in ("sin", "cos", "tan", "atan", "sinh", "cosh", "tanh", "exp", "cbrt"):
        assert np.allclose(getattr(gx.math, name)(x), getattr(np, name)(x), rtol=1e-14), name
    assert gx.math.sin(0.5) == pytest.approx(math.sin(0.5), rel=1e-15)
    assert isinstance(gx.math.cos(0.0), float) and gx.math.cos(0.0) == 1.0
    assert gx.math.log(1.0) == 0.0 and gx.math.log2(8.0) == 3.0 and gx.math.log10(100.0) == 2.0
    assert gx.math.atan2(1.0, 1.0) == pytest.approx(math.pi / 4)
    assert gx.math.pow(2.0, 10.0) == 1024.0 and gx.math.hypot(3.0, 4.0) == 5.0
    assert gx.math.pow(np.array([2.0, 3.0]), 2.0).tolist() == [4.0, 9.0]
    assert gx.math.exp(np.zeros((2, 3))).shape == (2, 3)


def test_the_docstring_examples_run():
    for module in (gx.nn, gx.math, gx.problems.control):
        examples = docstring_examples(module)
        assert examples, module.__name__
        # later examples use what earlier ones define
        namespace = {}
        for example in examples:
            assert "gx." in example
            exec(compile(example, f"{module.__name__}.__doc__", "exec"), namespace)
