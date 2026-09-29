"""gx.problems and gx.indicators: the test problems evaluated in Rust, and the indicators of
fronts."""

import dataclasses
import math

import numpy as np
import pytest

import genoxide as gx

PROBLEMS = [
    gx.problems.Sphere,
    gx.problems.AxisParallelEllipsoid,
    gx.problems.Schwefel1_2,
    gx.problems.Rastrigin,
    gx.problems.Rosenbrock,
    gx.problems.Ackley,
    gx.problems.Griewank,
    gx.problems.Schwefel2_26,
    gx.problems.Levy,
    gx.problems.Zakharov,
    gx.problems.StyblinskiTang,
    gx.problems.Michalewicz,
    gx.problems.Himmelblau,
    gx.problems.Branin,
    gx.problems.GoldsteinPrice,
    gx.problems.SixHumpCamel,
    gx.problems.Hartmann3,
    gx.problems.Hartmann6,
    gx.problems.Shekel5,
    gx.problems.Shekel7,
    gx.problems.Shekel10,
    gx.problems.Easom,
    gx.problems.Eggholder,
    gx.problems.SchafferF6,
]

# the problems whose minimum is known numerically, not proven
NUMERICAL = {"Hartmann3", "Hartmann6", "Shekel5", "Shekel7", "Shekel10", "Eggholder"}


CONSTRAINED = [
    gx.problems.cec2006.G01,
    gx.problems.cec2006.G02,
    gx.problems.cec2006.G03,
    gx.problems.cec2006.G04,
    gx.problems.cec2006.G05,
    gx.problems.cec2006.G06,
    gx.problems.cec2006.G07,
    gx.problems.cec2006.G08,
    gx.problems.cec2006.G09,
    gx.problems.cec2006.G10,
    gx.problems.cec2006.G11,
    gx.problems.cec2006.G12,
    gx.problems.cec2006.G13,
    gx.problems.cec2006.G14,
    gx.problems.cec2006.G15,
    gx.problems.cec2006.G16,
    gx.problems.cec2006.G17,
    gx.problems.cec2006.G18,
    gx.problems.cec2006.G19,
    gx.problems.cec2006.G20,
    gx.problems.cec2006.G21,
    gx.problems.cec2006.G22,
    gx.problems.cec2006.G23,
    gx.problems.cec2006.G24,
    gx.problems.engineering.WeldedBeam,
    gx.problems.engineering.WeldedBeamRagsdell,
    gx.problems.engineering.PressureVessel,
    gx.problems.engineering.TensionCompressionSpring,
    gx.problems.engineering.SpeedReducer,
    gx.problems.engineering.ThreeBarTruss,
    gx.problems.engineering.CantileverBeam,
    gx.problems.engineering.CarSideImpact,
]


MULTI_PROBLEMS = [
    gx.problems.Zdt1,
    gx.problems.Zdt2,
    gx.problems.Zdt3,
    gx.problems.Zdt4,
    gx.problems.Zdt6,
    gx.problems.Schaffer1,
    gx.problems.Schaffer2,
    gx.problems.FonsecaFleming,
    gx.problems.Kursawe,
    gx.problems.Poloni,
    gx.problems.Viennet1,
    gx.problems.Viennet2,
    gx.problems.Viennet3,
    gx.problems.Bnh,
    gx.problems.Srn,
    gx.problems.Tnk,
    gx.problems.Osy,
    gx.problems.Constr,
    gx.problems.Ctp1,
    gx.problems.Ctp2,
    gx.problems.Ctp3,
    gx.problems.Ctp4,
    gx.problems.Ctp5,
    gx.problems.Ctp6,
    gx.problems.Ctp7,
    gx.problems.Ctp8,
    gx.problems.Mw1,
    gx.problems.Mw2,
    gx.problems.Mw3,
    gx.problems.Mw5,
    gx.problems.Mw6,
    gx.problems.Mw7,
    gx.problems.Mw9,
    gx.problems.Mw10,
    gx.problems.Mw11,
    gx.problems.Mw12,
    gx.problems.Mw13,
    gx.problems.Dtlz1,
    gx.problems.Dtlz2,
    gx.problems.Dtlz3,
    gx.problems.Dtlz4,
    gx.problems.Dtlz5,
    gx.problems.Dtlz6,
    gx.problems.Dtlz7,
    gx.problems.ConvexDtlz2,
    gx.problems.ScaledDtlz1,
    gx.problems.ScaledDtlz2,
    gx.problems.InvertedDtlz1,
    gx.problems.Wfg1,
    gx.problems.Wfg2,
    gx.problems.Wfg3,
    gx.problems.Wfg4,
    gx.problems.Wfg5,
    gx.problems.Wfg6,
    gx.problems.Wfg7,
    gx.problems.Wfg8,
    gx.problems.Wfg9,
    gx.problems.Mw4,
    gx.problems.Mw8,
    gx.problems.Mw14,
    gx.problems.C1Dtlz1,
    gx.problems.C1Dtlz3,
    gx.problems.C2Dtlz2,
    gx.problems.ConvexC2Dtlz2,
    gx.problems.C3Dtlz1,
    gx.problems.C3Dtlz4,
]

# on bit strings, and so not in the registry of real problems
BINARY_PROBLEMS = [gx.problems.Zdt5]


def test_the_classes_are_the_rust_registry():
    assert [cls().name for cls in PROBLEMS + CONSTRAINED] == gx._genoxide.problem_names()
    two = [cls().name for cls in MULTI_PROBLEMS if len(cls().objectives) == 2]
    three = [cls().name for cls in MULTI_PROBLEMS if len(cls().objectives) == 3]
    # the problems with any number of objectives have 3 by default, and close both lists (the
    # constrained DTLZ problems with a radius for 3 objectives only in the three-objective one)
    names = gx._genoxide.multi_problem_names(2)
    assert two == names[: len(two)]
    assert set(names[len(two) :]) <= set(three)
    assert three == gx._genoxide.multi_problem_names(3)
    classes = PROBLEMS + MULTI_PROBLEMS + BINARY_PROBLEMS
    assert sorted(cls.__name__ for cls in classes) == sorted(
        name for name in gx.problems.__all__ if name not in ("Problem", "MultiProblem", "Optimum")
    )
    submodules = gx.problems.cec2006.__all__ + gx.problems.engineering.__all__
    names = [cls.__name__ for cls in CONSTRAINED] + ["GearTrain", "EQUALITY_TOLERANCE"]
    assert sorted(names) == sorted(submodules)


@pytest.mark.parametrize("cls", PROBLEMS)
def test_every_problem_describes_itself(cls):
    problem = cls()
    assert problem.objective == "minimize"
    assert problem.reference
    assert problem.reference_url is None or problem.reference_url.startswith("https://")
    assert cls.__doc__
    genome = problem.genome
    assert isinstance(genome, gx.Real)
    bounds = np.array(genome._describe()["bounds"])
    assert bounds.shape == (problem.dimensions, 2)
    optimum = problem.optimum
    assert optimum.proven == (problem.name not in NUMERICAL)
    assert optimum.solutions.shape[1] == problem.dimensions
    for solution in optimum.solutions:
        assert np.all(bounds[:, 0] <= solution) and np.all(solution <= bounds[:, 1])
        assert problem(solution) == pytest.approx(optimum.value, rel=1e-12, abs=1e-12)


@pytest.mark.parametrize("cls", PROBLEMS)
def test_calls_and_batches_agree(cls):
    problem = cls()
    genomes = np.random.default_rng(1).uniform(-2, 2, size=(20, problem.dimensions))
    scores = problem.evaluate(genomes)
    assert scores.shape == (20,)
    assert scores.dtype == np.float64
    assert [problem(genome) for genome in genomes] == list(scores)


def test_values_at_chosen_points():
    # 1 − 10 cos 2π + 10 in the first dimension, 0 in the others
    assert gx.problems.Rastrigin(3)([1.0, 0.0, 0.0]) == pytest.approx(1.0)
    # Rosenbrock's starting point: 100 (1 − 1.44)² + (−2.2)²
    assert gx.problems.Rosenbrock(2)([-1.2, 1.0]) == pytest.approx(24.2)
    # 1 + 4 + 9
    assert gx.problems.Sphere(3)([1, -2, 3]) == 14
    # the local minima that Goldstein and Price list
    assert gx.problems.GoldsteinPrice()([1.8, 0.2]) == pytest.approx(84)
    assert gx.problems.Himmelblau()([0, 0]) == 170
    assert gx.problems.Branin().optimum.value == pytest.approx(5 / (4 * math.pi))
    assert len(gx.problems.Himmelblau().optimum.solutions) == 4
    assert gx.problems.Michalewicz(10).optimum.value == pytest.approx(-9.6601517, abs=1e-7)


def test_values_of_the_functions_with_tables():
    # the minima that Dixon and Szegö report, to their digits
    assert round(gx.problems.Hartmann3().optimum.value, 5) == -3.86278
    assert round(gx.problems.Hartmann6().optimum.value, 5) == -3.32237
    assert round(gx.problems.Shekel5().optimum.value, 4) == -10.1532
    assert round(gx.problems.Shekel7().optimum.value, 4) == -10.4029
    assert round(gx.problems.Shekel10().optimum.value, 4) == -10.5364
    # at (4, 4, 4, 4), the squared distances to a₁ … a₅ are 0, 36, 64, 16 and 20
    expected = -(1 / 0.1 + 1 / 36.2 + 1 / 64.2 + 1 / 16.4 + 1 / 20.4)
    assert gx.problems.Shekel5()([4, 4, 4, 4]) == pytest.approx(expected, rel=1e-12)
    assert gx.problems.Easom()([math.pi, math.pi]) == -1
    assert gx.problems.Easom()([-100, 100]) == 0
    # −47 sin √47 at the origin
    expected = -47 * math.sin(math.sqrt(47))
    assert gx.problems.Eggholder()([0, 0]) == pytest.approx(expected, rel=1e-12)
    assert gx.problems.Eggholder().optimum.solutions[0][0] == 512
    assert gx.problems.SchafferF6()([0, 0]) == 0
    assert gx.problems.SchafferF6()([3, 4]) == gx.problems.SchafferF6()([0, -5])
    assert gx.problems.Hartmann6().genome == gx.Real((0.0, 1.0), length=6)
    assert gx.problems.Shekel10().genome == gx.Real((0.0, 10.0), length=4)


def test_sizes():
    assert gx.problems.Rastrigin().dimensions == 30
    assert gx.problems.Michalewicz().dimensions == 10
    assert gx.problems.Rastrigin(dimensions=5).genome == gx.Real((-5.12, 5.12), length=5)
    assert gx.problems.Branin().genome == gx.Real([(-5.0, 10.0), (0.0, 15.0)])
    assert gx.problems.Rastrigin(4) == gx.problems.Rastrigin(4)
    assert gx.problems.Rastrigin(4) != gx.problems.Rastrigin(5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        gx.problems.Rastrigin(4).dimensions = 5


@pytest.mark.parametrize(
    "problem, message",
    [
        (gx.problems.Rosenbrock(1), "Rosenbrock.dimensions is at least 2, not 1"),
        (gx.problems.Sphere(0), "Sphere.dimensions is at least 1, not 0"),
        (gx.problems.Sphere(2.0), "Sphere.dimensions is a whole number"),
    ],
)
def test_wrong_sizes_are_errors(problem, message):
    with pytest.raises(ValueError, match=message):
        problem.genome
    with pytest.raises(ValueError, match=message):
        gx.Cmaes(gx.Real((0, 1), length=2), seed=1).run(problem, generations=1)


def test_wrong_genomes_are_errors():
    problem = gx.problems.Sphere(3)
    with pytest.raises(ValueError, match="Sphere takes genomes of 3 genes, not 2"):
        problem([1.0, 2.0])
    with pytest.raises(ValueError, match="a genome is a 1-D array"):
        problem([[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError, match="evaluate takes a 2-D array"):
        problem.evaluate([1.0, 2.0, 3.0])


def test_a_native_run_reaches_the_optimum_and_repeats():
    problem = gx.problems.Rastrigin(10)
    de = gx.De(problem.genome, objective=problem.objective, seed=1)
    target = problem.optimum.value + 1e-8
    result = de.run(problem, target=target, evaluations=200_000)
    assert result.stop_reason == "target"
    assert result.best_fitness <= target
    again = de.run(problem, target=target, evaluations=200_000)
    assert again.evaluations == result.evaluations
    assert np.array_equal(again.best_genome, result.best_genome)


@pytest.mark.parametrize("parallel", [False, True])
def test_a_native_run_equals_a_run_with_python_calls(parallel):
    problem = gx.problems.Ackley(5)
    cmaes = gx.Cmaes(problem.genome, objective="minimize", seed=3)
    native = cmaes.run(problem, generations=50, parallel=parallel)
    python = cmaes.run(lambda x: problem(x), generations=50)
    batch = cmaes.run(problem.evaluate, generations=50, batch=True)
    for other in (python, batch):
        assert other.best_fitness == native.best_fitness
        assert np.array_equal(other.best_genome, native.best_genome)
        assert other.evaluations == native.evaluations


def test_a_native_run_needs_a_matching_genome_and_one_objective():
    problem = gx.problems.Sphere(3)
    with pytest.raises(ValueError, match="Sphere has 3 dimensions, but the genome has 2 genes"):
        gx.Cmaes(gx.Real((0, 1), length=2), objective=problem.objective, seed=1).run(
            problem, generations=1
        )
    with pytest.raises(ValueError, match="Sphere needs a Real genome"):
        ga = gx.Ga(
            gx.Integer((0, 5), length=3),
            objective=problem.objective,
            population_size=10,
            select=gx.Tournament(2),
            crossover=gx.UniformCrossover(),
            mutation=gx.UniformMutation(rate=0.5),
        )
        ga.run(problem, generations=1)
    nsga2 = gx.Nsga2(
        problem.genome,
        objectives=["minimize", "minimize"],
        population_size=10,
        crossover=gx.SimulatedBinaryCrossover(15),
        mutation=gx.PolynomialMutation(20, rate=0.5),
    )
    with pytest.raises(ValueError, match="Sphere has one objective"):
        nsga2.run(problem, generations=1)


def test_a_native_run_calls_on_generation():
    problem = gx.problems.Sphere(4)
    seen = []

    def stop_at_three(progress):
        seen.append(progress.generation)
        return progress.generation < 3

    pso = gx.Pso(problem.genome, population_size=10, objective="minimize", seed=1)
    result = pso.run(problem, generations=100, on_generation=stop_at_three)
    assert result.stop_reason == "aborted"
    assert seen == [0, 1, 2, 3]


# ---- constrained and engineering problems -------------------------------------------------------


@pytest.mark.parametrize("cls", CONSTRAINED)
def test_every_constrained_problem_describes_itself(cls):
    problem = cls()
    assert isinstance(problem, gx.problems.Problem)
    assert problem.objective == "minimize"
    assert problem.reference and cls.__doc__
    assert problem.reference_url is None or problem.reference_url.startswith("https://")
    genome = problem.genome
    assert isinstance(genome, gx.Real)
    bounds = np.array(genome._describe()["bounds"])
    assert bounds.shape == (problem.dimensions, 2)
    assert problem.constraint_count > 0
    assert problem.constraints(bounds.mean(axis=1)).shape == (problem.constraint_count,)
    optimum = problem.optimum
    assert optimum is not None
    for solution in optimum.solutions:
        assert np.all(bounds[:, 0] <= solution) and np.all(solution <= bounds[:, 1])
        value, violation = problem(solution)
        # best known values and their solutions are printed to a few digits
        tolerance = 1e-12 if optimum.proven else 2e-4
        assert value == pytest.approx(optimum.value, rel=tolerance)
        assert violation >= 0


@pytest.mark.parametrize("cls", CONSTRAINED)
def test_constrained_calls_and_batches_agree(cls):
    problem = cls()
    bounds = np.array(problem.genome._describe()["bounds"])
    genomes = np.random.default_rng(1).uniform(bounds[:, 0], bounds[:, 1], size=(20, len(bounds)))
    scores, violations = problem.evaluate(genomes)
    assert scores.shape == violations.shape == (20,)
    assert np.all(violations >= 0)
    assert [problem(genome) for genome in genomes] == list(zip(scores, violations))


def test_constrained_values_at_chosen_points():
    # g01 at x₁₀ = 100, the rest 0: f = −100, and g₁, g₂ exceed 10 by 90, g₄, g₇ exceed 0 by 100
    x = np.zeros(13)
    x[9] = 100
    assert gx.problems.cec2006.G01()(x) == (-100.0, 380.0)
    # g06 at (13, 0): 27 − 8000, and g₁ = −64 − 25 + 100
    assert gx.problems.cec2006.G06()([13.0, 0.0]) == (-7973.0, 11.0)
    # g03's optimum is −(1 + δ)⁵, and −1 with no tolerance
    assert gx.problems.cec2006.G03().optimum.value == pytest.approx(-(1.0001**5))
    assert gx.problems.cec2006.G03(tolerance=0).optimum.value == -1.0
    assert gx.problems.cec2006.G05(tolerance=1e-6).optimum is None
    # the equality of g03 at the origin: h = −1
    assert gx.problems.cec2006.G03().constraints(np.zeros(10)).tolist() == [-1.0]
    # the three-bar truss at (1, 1): 100 (2√2 + 1), feasible
    value, violation = gx.problems.engineering.ThreeBarTruss()([1.0, 1.0])
    assert value == pytest.approx(100 * (2 * math.sqrt(2) + 1)) and violation == 0
    # Coello's welded beam solution, 1.74830941
    value, violation = gx.problems.engineering.WeldedBeam()([0.2088, 3.4205, 8.9975, 0.2100])
    assert value == pytest.approx(1.74830941, rel=1e-8) and violation == 0


def test_cec2006_g07_to_g18_at_chosen_points():
    cec2006 = gx.problems.cec2006
    # g07 at the origin: 100 + 4·25 + 9 + 2 + 7·121 + 2·100 + 49 + 45, and g₆, g₇, g₈ exceed 0
    # by 8, 34 and 768
    assert cec2006.G07()(np.zeros(10)) == (1352.0, 810.0)
    # g09 at the origin: 100 + 5·144 + 3·121, feasible
    assert cec2006.G09()(np.zeros(7)) == (1183.0, 0.0)
    # g10 at the lower corner: 100 + 1000 + 1000, and g₆ = −10000 + 1250000 + 10000 − 25000
    assert cec2006.G10()([100, 1000, 1000, 10, 10, 10, 10, 10]) == (2100.0, 1_225_000.0)
    # g11's optimum is 3/4 − δ at (±√(1/2 − δ), 1/2), and 3/4 with no tolerance
    optimum = cec2006.G11().optimum
    assert optimum.value == pytest.approx(0.7499) and optimum.solutions.shape == (2, 2)
    assert cec2006.G11(tolerance=0).optimum.value == 0.75
    # g12 halfway between sphere centers: 3·0.5² − 0.0625 from the nearest
    value, violation = cec2006.G12()([1.5, 1.5, 1.5])
    assert value == pytest.approx(-0.6325) and violation == 0.6875
    assert cec2006.G12().constraint_count == 1
    assert cec2006.G16().constraint_count == 38
    # g17: the report's x* is feasible, at 8853.53401643571, above the best known
    problem = cec2006.G17()
    x = [
        201.784467214523659,
        99.9999999999999005,
        383.071034852773266,
        420,
        -10.9076584514292652,
        0.0731482312084287128,
    ]
    value, violation = problem(x)
    assert value == pytest.approx(8853.53401643571, rel=1e-13) and violation == 0
    assert problem.optimum.value == pytest.approx(8853.5338748065, rel=1e-15)
    assert not problem.optimum.proven
    for cls in (cec2006.G13, cec2006.G14, cec2006.G15, cec2006.G17):
        assert cls(tolerance=1e-6).optimum is None
    # g18's best known is −√3/2 to its digits
    assert cec2006.G18().optimum.value == pytest.approx(-math.sqrt(3) / 2, rel=1e-15)


def test_cec2006_g19_to_g24_at_chosen_points():
    cec2006 = gx.problems.cec2006
    # g19 at the origin: f = 0, and gⱼ = −eⱼ, violated by 15 + 27 + 36 + 18 + 12
    assert cec2006.G19()(np.zeros(15)) == (0.0, 108.0)
    # x₁₁…x₁₅ = 1 alone: f = Σ cᵢⱼ + 2 Σ dⱼ = 50 + 60, and only g₃ = 44 − 30 + 36 is violated
    x = np.zeros(15)
    x[10:] = 1
    assert cec2006.G19()(x) == (110.0, 50.0)
    # g20: the report's x* is infeasible, g₁ = (x₁ + x₁₃)/(Σ x + 0.1) = 0.1438
    problem = cec2006.G20()
    value, violation = problem(problem.optimum.solutions[0])
    assert value == pytest.approx(0.2049794002, rel=1e-9)
    assert violation == pytest.approx(0.14375, abs=1e-4)
    assert not problem.optimum.proven and problem.constraint_count == 20
    # g21 at x₄ = 100, x₅ = ln 800, x₆ = ln 400, x₇ = ln 500: only h₁ = 5000 ln 2 is violated
    value, violation = cec2006.G21()([50, 0, 0, 100, math.log(800), math.log(400), math.log(500)])
    assert value == 50 and violation == pytest.approx(5000 * math.log(2) - 1e-4, rel=1e-12)
    assert cec2006.G22().constraint_count == 20
    assert cec2006.G22().optimum.value == pytest.approx(236.430975504001, rel=1e-15)
    # g23: x₂ = x₄ = x₇ = 100, x₈ = 200, x₉ = 0.01 meets every constraint exactly, at −400
    assert cec2006.G23()([0, 100, 0, 100, 0, 0, 100, 200, 0.01]) == (-400.0, 0.0)
    assert cec2006.G23().optimum.solutions[0][7] == 200
    # g24: the two parts of the feasible region meet at (1, 0)
    assert cec2006.G24()([1.0, 0.0]) == (-1.0, 0.0)
    assert cec2006.G24()([1.0, 0.5]) == (-1.5, 0.5)
    assert cec2006.G24().optimum.proven
    for cls in (cec2006.G20, cec2006.G21, cec2006.G22, cec2006.G23):
        assert cls(tolerance=1e-6).optimum is None


def test_invalid_tolerances_are_errors():
    with pytest.raises(ValueError, match="equality tolerance is a finite number"):
        gx.problems.cec2006.G03(tolerance=-1.0).genome
    with pytest.raises(ValueError, match="equality tolerance is a finite number"):
        gx.problems.cec2006.G17(tolerance=-0.5).genome


def test_mixed_problems_round_their_discrete_genes():
    vessel = gx.problems.engineering.PressureVessel()
    assert vessel.design([0.83, 0.43, 50.0, 100.0]).tolist() == [0.8125, 0.4375, 50.0, 100.0]
    reducer = gx.problems.engineering.SpeedReducer()
    assert reducer.design([3.0, 0.75, 20.4, 8.0, 8.0, 3.0, 5.0])[2] == 20.0
    with pytest.raises(ValueError, match="PressureVessel takes genomes of 4 genes, not 3"):
        vessel.design([1.0, 1.0, 1.0])
    # the same design, and so the same fitness, for genes that round alike
    assert vessel([0.83, 0.43, 50.0, 100.0]) == vessel([0.8125, 0.4375, 50.0, 100.0])
    assert not hasattr(gx.problems.engineering.WeldedBeam(), "design")


def test_the_gear_train_is_an_integer_problem():
    problem = gx.problems.engineering.GearTrain()
    assert problem.genome == gx.Integer((12, 60), length=4)
    assert problem.constraint_count == 0
    optimum = problem.optimum
    assert optimum.proven and optimum.solutions.dtype == np.int64
    assert len(optimum.solutions) == 4
    for solution in optimum.solutions:
        assert problem(solution) == optimum.value
    assert optimum.value == pytest.approx(2.7008571e-12, rel=1e-7)
    assert problem.evaluate(np.array([[17, 14, 33, 50]])) == pytest.approx([1.362e-9], rel=1e-3)
    with pytest.raises(ValueError, match="GearTrain takes whole numbers as genes, not 16.5"):
        problem([16.5, 19, 43, 49])
    ga = gx.Ga(
        problem.genome,
        objective=problem.objective,
        population_size=50,
        select=gx.Tournament(2),
        crossover=gx.UniformCrossover(),
        mutation=gx.UniformMutation(count=1),
        seed=2,
    )
    native = ga.run(problem, generations=30)
    python = ga.run(lambda x: problem(x), generations=30)
    assert native.best_fitness == python.best_fitness
    assert np.array_equal(native.best_genome, python.best_genome)
    with pytest.raises(ValueError, match="GearTrain needs an Integer genome"):
        gx.De(gx.Real((12, 60), length=4), objective="minimize", seed=1).run(
            problem, generations=1
        )


def test_a_constrained_native_run_equals_a_run_with_python_calls():
    problem = gx.problems.engineering.PressureVessel()
    de = gx.De(problem.genome, objective=problem.objective, seed=4)
    native = de.run(problem, evaluations=3_000)
    python = de.run(lambda x: problem(x), evaluations=3_000)
    assert native.best_fitness == python.best_fitness
    assert native.violation == python.violation
    assert np.array_equal(native.best_genome, python.best_genome)


# ---- multi-objective problems ------------------------------------------------------------------


@pytest.mark.parametrize("cls", MULTI_PROBLEMS)
def test_every_multi_objective_problem_describes_itself(cls):
    problem = cls()
    assert isinstance(problem, gx.problems.MultiProblem)
    assert not isinstance(problem, gx.problems.Problem)
    assert set(problem.objectives) == {"minimize"}
    assert problem.reference
    assert problem.reference_url is None or problem.reference_url.startswith("https://")
    assert cls.__doc__
    bounds = np.array(problem.genome._describe()["bounds"])
    assert bounds.shape == (problem.dimensions, 2)
    genome = bounds.mean(axis=1)
    assert problem.constraints(genome).shape == (problem.constraint_count,)
    front = problem.optimal_front(20)
    if front is None:
        # KUR, POL, VNT2 and VNT3 have ideal and nadir points without a known front
        assert (problem.ideal_point is None) == (problem.nadir_point is None)
        if problem.ideal_point is not None:
            assert np.all(problem.ideal_point < problem.nadir_point)
    else:
        count = len(problem.objectives)
        assert front.shape[1] == count and len(front) >= 20
        assert np.all(problem.ideal_point <= front.min(axis=0) + 1e-9)
        assert np.all(front.max(axis=0) <= problem.nadir_point + 1e-9)


@pytest.mark.parametrize("cls", MULTI_PROBLEMS)
def test_multi_objective_calls_and_batches_agree(cls):
    problem = cls()
    bounds = np.array(problem.genome._describe()["bounds"])
    genomes = np.random.default_rng(1).uniform(bounds[:, 0], bounds[:, 1], size=(20, len(bounds)))
    values = problem.evaluate(genomes)
    if problem.constraint_count:
        values, violations = values
        assert violations.shape == (20,) and np.all(violations >= 0)
        calls = [problem(genome) for genome in genomes]
        assert [violation for _, violation in calls] == list(violations)
        assert np.array_equal(np.array([objectives for objectives, _ in calls]), values)
    else:
        assert np.array_equal(np.array([problem(genome) for genome in genomes]), values)
    assert values.shape == (20, len(problem.objectives))


def test_multi_objective_values_at_chosen_points():
    # ZDT1 at (0.25, 1/9, …): g = 2, f₂ = 2 (1 − √0.125)
    x = np.full(30, 1 / 9)
    x[0] = 0.25
    assert gx.problems.Zdt1()(x) == pytest.approx([0.25, 2 - math.sqrt(2) / 2])
    # BNH at (0, 3): outside the first constraint's circle by 25 + 9 − 25
    objectives, violation = gx.problems.Bnh()([0.0, 3.0])
    assert list(objectives) == [36.0, 29.0] and violation == 9.0
    assert list(gx.problems.Bnh().constraints([0.0, 0.0])) == pytest.approx([0.0, -65.3])
    # SRN at (−2.5, 2.5), on the second constraint's boundary
    objectives, violation = gx.problems.Srn()([-2.5, 2.5])
    assert list(objectives) == pytest.approx([24.5, -24.75]) and violation == 0.0
    # VNT1 at the origin
    assert list(gx.problems.Viennet1()([0.0, 0.0])) == [1.0, 2.0, 3.0]
    # DTLZ2 on its front: the squared objectives sum to 1
    front = gx.problems.Dtlz2(objectives=4).optimal_front(35)
    assert front.shape == (35, 4)
    assert np.allclose((front**2).sum(axis=1), 1)
    assert gx.problems.Kursawe().optimal_front(10) is None
    assert gx.problems.Constr().ideal_point == pytest.approx([7 / 18, 1])


def test_dtlz5_to_7_values_at_chosen_points():
    # DTLZ5 with the distance variables at 1: g = 2.5, θ₂ = π (1 + 5x₂) / 14 = π/4 at x₂ = 0.5;
    # at x₁ = 0, 3.5 (cos π/4, sin π/4, 0)
    x = np.ones(12)
    x[:2] = [0.0, 0.5]
    assert gx.problems.Dtlz5()(x) == pytest.approx([3.5 / math.sqrt(2), 3.5 / math.sqrt(2), 0])
    # DTLZ6 there: g = 10, θ₂ = π (1 + 20x₂) / 44 = π/4, a radius of 11
    assert gx.problems.Dtlz6()(x) == pytest.approx([11 / math.sqrt(2), 11 / math.sqrt(2), 0])
    # DTLZ7 with the distance variables at 0: g = 1, and at f₁ = f₂ = 0.5, sin 1.5π = −1, h = 3
    x = np.zeros(22)
    x[:2] = 0.5
    assert list(gx.problems.Dtlz7()(x)) == pytest.approx([0.5, 0.5, 6.0])
    # the fronts: DTLZ5's curve for 3 objectives, unknown for 4; DTLZ7's regions
    front = gx.problems.Dtlz5().optimal_front(20)
    assert front.shape == (20, 3)
    assert np.allclose(front[:, 0], front[:, 1]) and np.allclose((front**2).sum(axis=1), 1)
    four = gx.problems.Dtlz6(objectives=4)
    assert four.optimal_front(20) is None and four.nadir_point is None
    assert list(four.ideal_point) == [0.0] * 4
    assert gx.problems.Dtlz5().nadir_point == pytest.approx([1 / math.sqrt(2)] * 2 + [1])
    front = gx.problems.Dtlz7(objectives=2).optimal_front(30)
    phi = front[:, 0] * (1 + np.sin(3 * math.pi * front[:, 0]))
    assert front.shape == (30, 2) and np.allclose(front[:, 1], 4 - phi)
    assert gx.problems.Dtlz7().nadir_point == pytest.approx([0.8594008566447239] * 2 + [6])
    assert gx.problems.Dtlz7().dimensions == 22
    assert gx.problems.Dtlz5(objectives=5).dimensions == 14


# ---- ZDT5, on bit strings -----------------------------------------------------------------------


def test_zdt5_describes_itself():
    problem = gx.problems.Zdt5()
    assert isinstance(problem, gx.problems.MultiProblem)
    assert problem.name == "ZDT5"
    assert problem.genome == gx.Binary(80)
    assert problem.dimensions == 80
    assert problem.objectives == ["minimize", "minimize"]
    assert problem.constraint_count == 0
    assert problem.reference and problem.reference_url.startswith("https://")
    assert gx.problems.Zdt5.__doc__
    assert list(problem.ideal_point) == [1.0, 10 / 31]
    assert list(problem.nadir_point) == [31.0, 10.0]
    front = problem.optimal_front(31)
    assert front.tolist() == [[f1, 10 / f1] for f1 in range(1, 32)]
    assert problem.optimal_front(3).tolist() == [[1.0, 10.0], [16.0, 10 / 16], [31.0, 10 / 31]]
    small = gx.problems.Zdt5(first_bits=3, substrings=2)
    assert small.genome == gx.Binary(13)
    assert list(small.nadir_point) == [4.0, 2.0]


def test_zdt5_values_at_chosen_points():
    problem = gx.problems.Zdt5()
    # all zeros: f₁ = 1, and every substring at its deceptive attractor, v = 2: g = 20
    assert list(problem(np.zeros(80, dtype=bool))) == [1.0, 20.0]
    # all ones: f₁ = 31, g = 10
    assert list(problem(np.ones(80))) == [31.0, 10 / 31]
    # 7 ones in x₁, and substrings with 0 to 4 ones then five full: g = 2 + 3 + 4 + 5 + 6 + 5
    x = np.zeros(80, dtype=bool)
    x[:7] = True
    for i, ones in enumerate([0, 1, 2, 3, 4, 5, 5, 5, 5, 5]):
        x[30 + 5 * i : 30 + 5 * i + ones] = True
    assert list(problem(x)) == [8.0, 25 / 8]
    genomes = np.random.default_rng(1).integers(0, 2, size=(20, 80)).astype(bool)
    values = problem.evaluate(genomes)
    assert values.shape == (20, 2)
    assert np.array_equal(np.array([problem(genome) for genome in genomes]), values)
    # f₁ f₂ = g, at least 10
    assert np.all(values[:, 0] * values[:, 1] >= 10)
    assert problem.constraints(np.ones(80)).shape == (0,)


def test_zdt5_rejects_what_isnt_its_genome():
    problem = gx.problems.Zdt5()
    with pytest.raises(ValueError, match="ZDT5 takes bits, 0 or 1, as genes, not 0.5"):
        problem(np.full(80, 0.5))
    with pytest.raises(ValueError, match="ZDT5 takes genomes of 80 bits, not 79"):
        problem(np.zeros(79))
    with pytest.raises(ValueError, match="ZDT5 takes genomes of 80 bits, not 3"):
        problem.constraints([0, 1, 1])
    with pytest.raises(ValueError, match="Zdt5.substrings is at least 1, not 0"):
        gx.problems.Zdt5(substrings=0).genome
    nsga2 = gx.Nsga2(
        gx.Real((0, 1), length=80),
        objectives=problem.objectives,
        population_size=10,
        crossover=gx.SimulatedBinaryCrossover(15),
        mutation=gx.PolynomialMutation(20, rate=0.5),
    )
    with pytest.raises(ValueError, match="ZDT5 needs a Binary genome"):
        nsga2.run(problem, generations=1)
    with pytest.raises(ValueError, match="ZDT5 has 80 bits, but the genome has 79"):
        _zdt5_nsga2(gx.Binary(79)).run(problem, generations=1)
    with pytest.raises(ValueError, match="Bnh needs a Real genome|BNH needs a Real genome"):
        _zdt5_nsga2(gx.Binary(2)).run(gx.problems.Bnh(), generations=1)


def _zdt5_nsga2(genome):
    return gx.Nsga2(
        genome,
        objectives=["minimize", "minimize"],
        population_size=40,
        crossover=gx.UniformCrossover(),
        mutation=gx.BitFlip(rate=1 / 80),
        seed=3,
    )


@pytest.mark.parametrize("parallel", [False, True])
def test_a_native_zdt5_run_equals_a_run_with_python_calls(parallel):
    problem = gx.problems.Zdt5()
    algorithm = _zdt5_nsga2(problem.genome)
    native = algorithm.run(problem, generations=30, parallel=parallel)
    python = algorithm.run(lambda x: problem(x), generations=30)
    batch = algorithm.run(problem.evaluate, generations=30, batch=True)
    for other in (python, batch):
        assert np.array_equal(other.front_objectives, native.front_objectives)
        assert np.array_equal(other.front_genomes, native.front_genomes)
        assert other.evaluations == native.evaluations
    assert native.front_genomes.dtype == bool
    assert np.all(native.front_objectives[:, 0] * native.front_objectives[:, 1] >= 10)


@pytest.mark.parametrize("first_bits, substrings", [(3, 2), (30, 34)])
def test_bits_reach_a_python_function_as_they_are(first_bits, substrings):
    # 13 bits, part of a word, and 200, three words and 8 bits: a run with Python calls is the
    # native run only if every bit of every genome reaches Python unchanged
    problem = gx.problems.Zdt5(first_bits=first_bits, substrings=substrings)
    algorithm = _zdt5_nsga2(problem.genome)
    native = algorithm.run(problem, generations=10)
    python = algorithm.run(lambda x: problem(x), generations=10)
    threads = algorithm.run(lambda x: problem(x), generations=10, parallel=True)
    batch = algorithm.run(problem.evaluate, generations=10, batch=True)
    for other in (python, threads, batch):
        assert np.array_equal(other.front_objectives, native.front_objectives)
        assert np.array_equal(other.front_genomes, native.front_genomes)


def _spread_point(n, a, b):
    """z with zᵢ = 2i ((a i + b) mod 1), as in the Rust tests."""
    i = np.arange(1, n + 1, dtype=np.float64)
    return 2 * i * np.fmod(a * i + b, 1.0)


# the values of the authors' C++ toolkit (version 2006.03.28), compiled and run at the point
WFG_VALUES = [
    (gx.problems.Wfg1, [2.9521691109522714, 0.97445227500261]),
    (gx.problems.Wfg2, [1.5793517175960388, 3.8576901485363813]),
    (gx.problems.Wfg3, [1.9138827838827839, 1.9238827838827837]),
    (gx.problems.Wfg4, [1.4056727977183536, 3.7515601023124354]),
    (gx.problems.Wfg5, [2.0899502646748944, 3.2324642099371106]),
    (gx.problems.Wfg6, [2.470388403034529, 2.5745763998994367]),
    (gx.problems.Wfg7, [1.3020135240884834, 4.037767908965662]),
    (gx.problems.Wfg8, [2.4877866969669267, 2.7674219485001066]),
    (gx.problems.Wfg9, [2.624356071590899, 2.9327440749083484]),
]


@pytest.mark.parametrize("cls, expected", WFG_VALUES)
def test_wfg_values_match_the_toolkit(cls, expected):
    problem = cls(objectives=2, position=2, distance=4)
    assert problem.genome == gx.Real([(0.0, 2.0 * i) for i in range(1, 7)])
    assert list(problem(_spread_point(6, 0.37, 0.11))) == pytest.approx(expected, rel=1e-13)


def test_wfg_sizes_and_fronts():
    # k = 4 for 2 objectives, 2 (M − 1) for more, and l = 20
    assert gx.problems.Wfg1(objectives=2).dimensions == 24
    assert gx.problems.Wfg4().dimensions == 24
    assert gx.problems.Wfg9(objectives=5).dimensions == 28
    assert gx.problems.Wfg2(objectives=4, position=3, distance=6).dimensions == 9
    assert gx.problems.Wfg5(4) == gx.problems.Wfg5(objectives=4)
    # the concave fronts: Σ (fₘ / 2m)² = 1, spanning [0, 2m]
    problem = gx.problems.Wfg4(objectives=3)
    front = problem.optimal_front(91)
    assert front.shape == (91, 3)
    assert np.allclose(((front / [2, 4, 6]) ** 2).sum(axis=1), 1)
    assert list(problem.ideal_point) == [0, 0, 0]
    assert list(problem.nadir_point) == [2, 4, 6]
    # WFG2's disconnected front and WFG1's: exactly the points asked for, with 2 objectives
    assert gx.problems.Wfg2(objectives=2).optimal_front(50).shape == (50, 2)
    assert len(gx.problems.Wfg1(objectives=4).optimal_front(50)) >= 50
    # WFG3: the segment from (0, 4) to (2, 0) with 2 objectives; not known with more
    front = gx.problems.Wfg3(objectives=2).optimal_front(5)
    assert front.tolist() == [[0, 4], [0.5, 3], [1, 2], [1.5, 1], [2, 0]]
    wfg3 = gx.problems.Wfg3()
    assert wfg3.optimal_front(10) is None and wfg3.nadir_point is None
    # WFG9's optimal distance parameters, from the last back: on the front
    k, l = 4, 6
    y = [0.35]
    for count in range(1, l):
        y.append(0.35 ** (1 / (0.02 + 1.96 * np.mean(y))))
    y = np.r_[np.full(k, 0.5), y[::-1]]
    f = gx.problems.Wfg9(objectives=3, position=k, distance=l)(y * 2 * np.arange(1, k + l + 1))
    assert ((f / [2, 4, 6]) ** 2).sum() == pytest.approx(1, rel=1e-12)


@pytest.mark.parametrize(
    "problem, message",
    [
        (gx.problems.Wfg1(objectives=3, position=3), "Wfg1.position is a multiple of 2"),
        (gx.problems.Wfg2(distance=5), "Wfg2.distance is even, not 5"),
        (gx.problems.Wfg4(distance=0), "Wfg4.distance is at least 1, not 0"),
        (gx.problems.Wfg4(objectives=7), "Wfg4.objectives is at most 6, not 7"),
        (gx.problems.Wfg6(position=0), "Wfg6.position is at least 1, not 0"),
    ],
)
def test_wrong_wfg_sizes_are_errors(problem, message):
    with pytest.raises(ValueError, match=message):
        problem.genome


def test_wfg_sizes_are_checked_in_rust_too():
    description = '{"type": "wfg2", "objectives": 3, "position": 4, "distance": 3}'
    with pytest.raises(ValueError, match="WFG2 needs an even number of distance parameters"):
        gx._genoxide.problem_info(description)
    description = '{"type": "wfg1", "objectives": 3, "position": 3, "distance": 4}'
    with pytest.raises(ValueError, match="WFG1 needs a positive multiple of 2"):
        gx._genoxide.problem_info(description)


def test_dtlz_variants_at_chosen_points():
    # DTLZ1 (3 objectives) at 0.5 everywhere is (0.125, 0.125, 0.25) with g = 0: scaled by 1, 10
    # and 100, and inverted to 0.5 − f
    x = np.full(7, 0.5)
    assert list(gx.problems.ScaledDtlz1()(x)) == pytest.approx([0.125, 1.25, 25.0])
    assert list(gx.problems.InvertedDtlz1()(x)) == pytest.approx([0.375, 0.375, 0.25])
    # DTLZ2 there is (1/2, 1/2, 1/√2): convex, the first two to the fourth power, the last squared
    x = np.full(12, 0.5)
    assert list(gx.problems.ConvexDtlz2()(x)) == pytest.approx([0.0625, 0.0625, 0.5])
    assert list(gx.problems.ScaledDtlz2(factor=2)(x)) == pytest.approx(
        [0.5, 1.0, 2 * math.sqrt(2)]
    )
    # the fronts and their corners
    assert gx.problems.ScaledDtlz2(factor=2).nadir_point == pytest.approx([1.0, 2.0, 4.0])
    assert gx.problems.ScaledDtlz1(objectives=5).nadir_point[-1] == pytest.approx(5000.0)
    front = gx.problems.InvertedDtlz1().optimal_front(91)
    assert front.shape == (91, 3) and np.allclose(front.sum(axis=1), 1.0)
    front = gx.problems.ConvexDtlz2().optimal_front(91)
    assert np.allclose(front[:, 2] + np.sqrt(front[:, :2]).sum(axis=1), 1.0)
    assert gx.problems.ScaledDtlz1(objectives=4, variables=10).dimensions == 10


def test_mw_problems_at_chosen_points():
    # MW11 at x₁ = 1 with g₃ = 1: the isolated optimal point (1, 1), feasible
    x = [1.0]
    for _ in range(14):
        x.append(1 - (x[-1] - 0.5) ** 2)
    objectives, violation = gx.problems.Mw11()(np.array(x))
    assert list(objectives) == [1.0, 1.0] and violation == 0.0
    assert gx.problems.Mw11().optimal_front(50).shape == (50, 2)
    # MW7 at x₁ = 0 with g₃ = 1: (0, 1), inside the second constraint's radius 1.15
    x[0] = 0.0
    for i in range(1, 15):
        x[i] = 1 - (x[i - 1] - 0.5) ** 2
    objectives, violation = gx.problems.Mw7()(np.array(x))
    assert list(objectives) == pytest.approx([0.0, 1.0]) and violation == pytest.approx(0.3225)
    # MW4: the simplex; MW8: the unit sphere; MW14's corners
    assert np.allclose(gx.problems.Mw4().optimal_front(91).sum(axis=1), 1.0)
    front = gx.problems.Mw8(objectives=4).optimal_front(50)
    assert len(front) >= 50 and np.allclose((front**2).sum(axis=1), 1.0)
    assert gx.problems.Mw14().nadir_point == pytest.approx([1.5, 1.5, 5.0])
    assert gx.problems.Mw12().nadir_point == pytest.approx([1.3164, 1.0], abs=1e-4)
    assert gx.problems.Mw5().constraint_count == 3
    assert gx.problems.Mw4(objectives=5).dimensions == 17
    assert gx.problems.Mw1(variables=5).dimensions == 5


@pytest.mark.parametrize(
    "problem, message",
    [
        (gx.problems.ScaledDtlz1(factor=0), "ScaledDtlz1.factor is finite and positive, not 0"),
        (gx.problems.ScaledDtlz2(factor="a"), "ScaledDtlz2.factor is a number"),
        (gx.problems.ConvexDtlz2(objectives=7), "ConvexDtlz2.objectives is at most 6, not 7"),
        (gx.problems.Mw1(variables=2), "Mw1.variables is at least 3, not 2"),
        (gx.problems.Mw4(variables=3), "Mw4.variables is at least 4, not 3"),
        (gx.problems.Mw8(objectives=1), "Mw8.objectives is at least 2, not 1"),
    ],
)
def test_wrong_dtlz_variant_and_mw_sizes_are_errors(problem, message):
    with pytest.raises((ValueError, TypeError), match=message):
        problem.genome


def test_mw_and_dtlz_variant_sizes_are_checked_in_rust_too():
    description = '{"type": "mw14", "objectives": 3, "variables": 3}'
    with pytest.raises(ValueError, match="MW with this many objectives needs at least 4"):
        gx._genoxide.problem_info(description)
    description = '{"type": "scaled_dtlz1", "objectives": 3, "variables": null, "factor": -1.0}'
    with pytest.raises(ValueError, match="a scaling factor is finite and positive"):
        gx._genoxide.problem_info(description)


def _non_dominated_2d(values):
    """The non-dominated rows of an (n, 2) array."""
    values = values[np.lexsort((values[:, 1], values[:, 0]))]
    best = np.minimum.accumulate(values[:, 1])
    keep = np.r_[True, values[1:, 1] < best[:-1]]
    return values[keep]


def test_ideal_and_nadir_points_without_a_known_front():
    # POL: the ends of the front, at (−3, −1) and (1, 2)
    pol = gx.problems.Poloni()
    assert list(pol.ideal_point) == [1.0, 0.0]
    assert pol([-3.0, -1.0]) == pytest.approx([pol.nadir_point[0], 0.0], rel=1e-14, abs=1e-14)
    assert pol([1.0, 2.0]) == pytest.approx([1.0, pol.nadir_point[1]], rel=1e-14)
    # VNT2: the objectives' minima, and the other objectives there
    vnt2 = gx.problems.Viennet2()
    assert list(vnt2.ideal_point) == [3.0, -17.0, -13.0]
    assert list(vnt2.nadir_point) == [883 / 208, -2109 / 128, -35858 / 2975]
    assert vnt2([0.5, 0.25])[:2] == pytest.approx(vnt2.nadir_point[:2], rel=1e-15)
    # VNT3: f₁ at t = 14π/3, f₂ at the origin, f₃ at t = 4π/3
    vnt3 = gx.problems.Viennet3()
    assert list(vnt3.ideal_point) == [0.0, 15.0, -0.1]
    t = 14 * math.pi / 3
    u = 4 * math.pi / 3
    expected = [t / 2 + math.sin(t), 460 / 27, 1 / (u + 1) - 1.1 * math.exp(-u)]
    assert list(vnt3.nadir_point) == pytest.approx(expected, rel=1e-14)
    # KUR: at x = 0 and at f₂'s minimum; a random sample's non-dominated points stay between them
    for n in (2, 3, 5):
        kur = gx.problems.Kursawe(n)
        ideal, nadir = kur.ideal_point, kur.nadir_point
        assert list(kur(np.zeros(n))) == [ideal[0], nadir[1]]
        f = kur(np.full(n, -1.1527408475499261))
        assert f == pytest.approx([nadir[0], ideal[1]], rel=1e-14)
    rng = np.random.default_rng(1)
    kur_ends = [[0.0] * 3, [-1.1527408475499261] * 3]
    for problem, ends in [(gx.problems.Kursawe(), kur_ends), (pol, [[-3.0, -1.0], [1.0, 2.0]])]:
        bounds = np.array(problem.genome._describe()["bounds"])
        genomes = rng.uniform(bounds[:, 0], bounds[:, 1], size=(200_000, len(bounds)))
        front = _non_dominated_2d(problem.evaluate(np.vstack([genomes, ends])))
        assert np.all(front >= problem.ideal_point - 1e-12)
        assert np.all(front <= problem.nadir_point + 1e-12)


def test_multi_objective_sizes():
    assert gx.problems.Zdt1().dimensions == 30
    assert gx.problems.Zdt4().dimensions == 10
    assert gx.problems.Zdt4().genome == gx.Real([(0.0, 1.0)] + [(-5.0, 5.0)] * 9)
    assert gx.problems.Dtlz1().dimensions == 7
    assert gx.problems.Dtlz2(objectives=4).dimensions == 13
    assert gx.problems.Dtlz2(objectives=4).objectives == ["minimize"] * 4
    assert gx.problems.Dtlz2(objectives=2, variables=5).dimensions == 5
    assert gx.problems.Dtlz2(3) == gx.problems.Dtlz2(objectives=3)
    assert gx.problems.Kursawe(5).dimensions == 5
    assert gx.problems.Bnh().genome == gx.Real((-15.0, 30.0), length=2)
    assert gx.problems.Osy().constraint_count == 6


@pytest.mark.parametrize(
    "problem, message",
    [
        (gx.problems.Zdt1(1), "Zdt1.variables is at least 2, not 1"),
        (gx.problems.Kursawe(1), "Kursawe.variables is at least 2, not 1"),
        (gx.problems.FonsecaFleming(0), "FonsecaFleming.variables is at least 1, not 0"),
        (gx.problems.Dtlz2(objectives=7), "Dtlz2.objectives is at most 6, not 7"),
        (gx.problems.Dtlz2(objectives=1), "Dtlz2.objectives is at least 2, not 1"),
        (gx.problems.Dtlz1(objectives=4, variables=3), "Dtlz1.variables is at least 4, not 3"),
    ],
)
def test_wrong_multi_objective_sizes_are_errors(problem, message):
    with pytest.raises(ValueError, match=message):
        problem.genome


def test_ctp_values_at_chosen_points():
    # CTP1 at the origin: f = (0, 1), inside both constraints (0.858 − 1, 0.728 − 1)
    objectives, violation = gx.problems.Ctp1()([0.0, 0.0])
    assert list(objectives) == [0.0, 1.0] and violation == 0.0
    assert list(gx.problems.Ctp1().constraints([0.0, 0.0])) == pytest.approx([-0.142, -0.272])
    # CTP1 at (1, 0), the unconstrained front's end: both constraints break
    e = math.exp(-1)
    expected = 0.858 * math.exp(-0.541) - e + 0.728 * math.exp(-0.295) - e
    assert gx.problems.Ctp1()([1.0, 0.0])[1] == pytest.approx(expected)
    # CTP2-CTP5 at the origin, on the line where the constraint's right-hand side is 0
    for cls in (gx.problems.Ctp2, gx.problems.Ctp3, gx.problems.Ctp4, gx.problems.Ctp5):
        objectives, violation = cls()([0.0, 0.0])
        assert list(objectives) == [0.0, 1.0] and violation == 0.0
    # f₂ = g (1 − √(f₁/g)): at (0.25, 3), g = 4 and f₂ = 3
    assert list(gx.problems.Ctp6()([0.25, 3.0])[0]) == [0.25, 3.0]
    # CTP3's and CTP4's fronts are the same 13 points on the line f₂ = 1 − tan(0.2π) f₁
    front = gx.problems.Ctp3().optimal_front(13)
    assert np.array_equal(front, gx.problems.Ctp4().optimal_front(13))
    assert len(np.unique(front, axis=0)) == 13
    assert np.allclose(front[:, 1], 1 - math.tan(0.2 * math.pi) * front[:, 0])
    # CTP7's front, but its point at f₁ = 0, lies on f₂ = 1 − √f₁
    front = gx.problems.Ctp7().optimal_front(200)
    assert np.allclose(front[1:, 1], 1 - np.sqrt(front[1:, 0]))
    assert gx.problems.Ctp7().nadir_point == pytest.approx([1.0, 1.0446206], abs=1e-7)
    assert gx.problems.Ctp8().constraint_count == 2
    assert gx.problems.Ctp6().genome == gx.Real([(0.0, 1.0), (0.0, 10.0)])


def test_constrained_dtlz_values_at_chosen_points():
    # C1-DTLZ1 at (0.5, …): DTLZ1's front point (0.125, 0.125, 0.25), feasible
    objectives, violation = gx.problems.C1Dtlz1()(np.full(7, 0.5))
    assert list(objectives) == [0.125, 0.125, 0.25] and violation == 0.0
    assert list(gx.problems.C1Dtlz1().constraints(np.full(7, 0.5))) == pytest.approx([-1 / 12])
    # C3-DTLZ1 there: DTLZ1's front breaks all three constraints, 1 − (S + fⱼ)
    assert gx.problems.C3Dtlz1()(np.full(7, 0.5))[1] == pytest.approx(1.0)
    # C3-DTLZ4 at 2 × (1, 0, 0): g = 1 with four distance variables at 1, on the boundary
    objectives, violation = gx.problems.C3Dtlz4()([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.5])
    assert list(objectives) == pytest.approx([2.0, 0.0, 0.0]) and violation == 0.0
    # C2-DTLZ2 at the corner (1, 0, 0), inside its sphere, and at (1/√2, 1/√2, 0), outside all
    x = np.full(12, 0.5)
    x[:2] = [0.0, 0.0]
    assert gx.problems.C2Dtlz2()(x)[1] == 0.0
    assert list(gx.problems.C2Dtlz2().constraints(x)) == pytest.approx([-0.16])
    x[1] = 0.5
    assert gx.problems.C2Dtlz2()(x)[1] > 0.2
    # the fronts: on the unit sphere, on the convex front outside the cylinder, and on the
    # constraints' boundaries
    front = gx.problems.C2Dtlz2().optimal_front(91)
    assert len(front) >= 91 and np.allclose((front**2).sum(axis=1), 1)
    front = gx.problems.ConvexC2Dtlz2().optimal_front(91)
    assert np.allclose(front[:, 2] + np.sqrt(front[:, 0]) + np.sqrt(front[:, 1]), 1)
    front = gx.problems.C3Dtlz1().optimal_front(91)
    assert np.allclose(front.sum(axis=1) + front.min(axis=1), 1)
    assert gx.problems.C3Dtlz4().nadir_point == pytest.approx([2, 2, 2])


def test_constrained_dtlz_sizes():
    assert gx.problems.C1Dtlz1().dimensions == 7
    assert gx.problems.C1Dtlz3().dimensions == 12
    assert gx.problems.C3Dtlz4().dimensions == 7
    assert gx.problems.C3Dtlz4(objectives=5).constraint_count == 5
    assert gx.problems.C2Dtlz2(objectives=2).optimal_front(10).shape == (10, 2)
    assert gx.problems.C1Dtlz3(objectives=5).name == "C1-DTLZ3"
    assert gx.problems.C1Dtlz3(objectives=4, radius=10).dimensions == 13
    assert gx.problems.C2Dtlz2(objectives=3, radius=0.3).constraints(np.full(12, 0.5)).shape == (1,)


@pytest.mark.parametrize(
    "problem, message",
    [
        (gx.problems.C1Dtlz3(objectives=4), "C1-DTLZ3 has no radius for 4 objectives"),
        (gx.problems.ConvexC2Dtlz2(objectives=2), "convex C2-DTLZ2 has no radius for 2"),
        (gx.problems.C2Dtlz2(radius=0), "C2-DTLZ2 needs a finite radius above 0, not 0"),
        (gx.problems.C3Dtlz1(objectives=7), "C3Dtlz1.objectives is at most 6, not 7"),
        (gx.problems.C3Dtlz4(variables=2), "C3Dtlz4.variables is at least 3, not 2"),
    ],
)
def test_wrong_constrained_dtlz_settings_are_errors(problem, message):
    with pytest.raises(ValueError, match=message):
        problem.genome


def nsga2(genome, objectives, **settings):
    return gx.Nsga2(
        genome,
        objectives=objectives,
        population_size=40,
        crossover=gx.SimulatedBinaryCrossover(20),
        mutation=gx.PolynomialMutation(20, rate=0.5),
        seed=2,
        **settings,
    )


@pytest.mark.parametrize(
    "cls", [gx.problems.Bnh, gx.problems.Zdt1, gx.problems.Viennet1, gx.problems.Wfg9]
)
@pytest.mark.parametrize("parallel", [False, True])
def test_a_native_multi_objective_run_equals_a_run_with_python_calls(cls, parallel):
    problem = cls()
    algorithm = nsga2(problem.genome, problem.objectives)
    native = algorithm.run(problem, generations=30, parallel=parallel)
    python = algorithm.run(lambda x: problem(x), generations=30)
    batch = algorithm.run(problem.evaluate, generations=30, batch=True)
    for other in (python, batch):
        assert np.array_equal(other.front_objectives, native.front_objectives)
        assert np.array_equal(other.front_violations, native.front_violations)
        assert np.array_equal(other.front_genomes, native.front_genomes)
        assert other.evaluations == native.evaluations


def test_a_native_multi_objective_run_reaches_the_front():
    problem = gx.problems.Bnh()
    result = nsga2(problem.genome, problem.objectives).run(problem, generations=200)
    assert np.all(result.front_violations == 0)
    distance = gx.indicators.igd_plus(result.front_objectives, problem.optimal_front(200))
    assert distance < 1.0
    for algorithm in (
        gx.Spea2(
            problem.genome,
            objectives=problem.objectives,
            population_size=40,
            crossover=gx.SimulatedBinaryCrossover(20),
            mutation=gx.PolynomialMutation(20, rate=0.5),
            seed=1,
        ),
        gx.Moead(
            problem.genome,
            objectives=problem.objectives,
            weights=gx.das_dennis(2, 39),
            crossover=gx.SimulatedBinaryCrossover(20),
            mutation=gx.PolynomialMutation(20, rate=0.5),
            seed=1,
        ),
    ):
        result = algorithm.run(problem, generations=100)
        assert len(result.front_objectives) > 5


def test_a_single_objective_problem_is_minimized():
    # the algorithms maximize by default: a test problem, minimized, rejects that
    rastrigin = gx.problems.Rastrigin(5)
    with pytest.raises(ValueError, match="Rastrigin minimizes its objective"):
        gx.Cmaes(rastrigin.genome, seed=1).run(rastrigin, generations=1)
    gear_train = gx.problems.engineering.GearTrain()
    with pytest.raises(ValueError, match="minimizes its objective"):
        gx.LocalSearch(gear_train.genome, neighbor=gx.UniformMutation(count=1), seed=1).run(
            gear_train, generations=1
        )
    result = gx.Cmaes(rastrigin.genome, objective=rastrigin.objective, seed=1).run(
        rastrigin, evaluations=2_000
    )
    assert result.best_fitness < 50


def test_a_native_multi_objective_run_needs_matching_settings():
    problem = gx.problems.Bnh()
    with pytest.raises(ValueError, match="Bnh has 2 objectives: use a multi-objective algorithm"):
        gx.Cmaes(problem.genome, seed=1).run(problem, generations=1)
    with pytest.raises(ValueError, match="BNH has 2 objectives, but the algorithm has 3"):
        nsga2(problem.genome, ["minimize"] * 3).run(problem, generations=1)
    with pytest.raises(ValueError, match="BNH minimizes its objectives"):
        nsga2(problem.genome, ["minimize", "maximize"]).run(problem, generations=1)
    with pytest.raises(ValueError, match="BNH has 2 dimensions, but the genome has 3 genes"):
        nsga2(gx.Real((0, 1), length=3), problem.objectives).run(problem, generations=1)
    with pytest.raises(ValueError, match="BNH takes genomes of 2 genes, not 3"):
        problem([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="BNH takes genomes of 2 genes, not 1"):
        problem.constraints([1.0])
    with pytest.raises(ValueError, match="points is at least 0"):
        problem.optimal_front(-1)


# ---- indicators ------------------------------------------------------------------------------

FRONT = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])


def test_hypervolume():
    # 3 + 2 + 1 unit squares below the reference point (4, 4)
    assert gx.indicators.hypervolume(FRONT, [4.0, 4.0]) == 6.0
    # maximizing both: the same squares, mirrored
    mirrored = -FRONT
    assert gx.indicators.hypervolume(mirrored, [-4, -4], ["maximize", "maximize"]) == 6.0
    # a point beyond the reference point adds nothing
    assert gx.indicators.hypervolume([[5.0, 0.0]], [4.0, 4.0]) == 0.0
    assert gx.indicators.hypervolume([], [4.0, 4.0]) == 0.0
    # three objectives: the unit cube below (1, 1, 1)
    assert gx.indicators.hypervolume([[0.0, 0.0, 0.0]], [1.0, 1.0, 1.0]) == 1.0


def test_distances():
    reference = np.array([[0.0, 1.0], [1.0, 0.0]])
    assert gx.indicators.igd(reference, reference) == 0.0
    # (0, 1) covers itself, and is √2 from (1, 0)
    assert gx.indicators.igd([[0.0, 1.0]], reference) == pytest.approx(math.sqrt(2) / 2)
    assert gx.indicators.gd([[0.0, 1.0]], reference) == 0.0
    # better than the reference front: IGD sees a distance, IGD+ doesn't
    assert gx.indicators.igd([[0.0, 0.0]], reference) == 1.0
    assert gx.indicators.igd_plus([[0.0, 0.0]], reference) == 0.0
    assert gx.indicators.igd_plus([[0.0, 0.0]], reference, ["maximize", "maximize"]) == 1.0
    assert gx.indicators.igd([], reference) == math.inf
    assert math.isnan(gx.indicators.igd(reference, []))


def test_spread():
    reference = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
    # evenly spaced from extreme to extreme
    assert gx.indicators.spread(reference, reference) == 0.0
    assert gx.indicators.spread(reference[:1], reference) == 1.0


def test_indicators_check_their_arguments():
    with pytest.raises(ValueError, match="the front has 3 objectives and the reference front 2"):
        gx.indicators.igd(np.zeros((2, 3)), np.zeros((2, 2)))
    with pytest.raises(ValueError, match="the indicators take 2 to 6 objectives, not 7"):
        gx.indicators.igd(np.zeros((2, 7)), np.zeros((2, 7)))
    with pytest.raises(ValueError, match="objectives has 1 entries, for 2 objectives"):
        gx.indicators.igd_plus(FRONT, FRONT, ["minimize"])
    with pytest.raises(ValueError, match='"maximize" or "minimize", not "best"'):
        gx.indicators.spread(FRONT, FRONT, ["best", "minimize"])
    with pytest.raises(ValueError, match="reference_point is a 1-D array"):
        gx.indicators.hypervolume(FRONT, [[4.0, 4.0]])
    with pytest.raises(ValueError, match="front is a 2-D array"):
        gx.indicators.gd([1.0, 2.0], FRONT)


def test_indicators_of_a_run():
    problem_front = np.column_stack([np.linspace(0, 1, 101), 1 - np.sqrt(np.linspace(0, 1, 101))])

    def zdt1(x):
        g = 1 + 9 * x[:, 1:].mean(axis=1)
        return np.column_stack([x[:, 0], g * (1 - np.sqrt(x[:, 0] / g))])

    nsga2 = gx.Nsga2(
        gx.Real((0.0, 1.0), length=10),
        objectives=["minimize", "minimize"],
        population_size=40,
        crossover=gx.SimulatedBinaryCrossover(15),
        mutation=gx.PolynomialMutation(20, rate=0.1),
        seed=1,
    )
    result = nsga2.run(zdt1, batch=True, generations=100)
    front = result.front_objectives
    assert gx.indicators.igd_plus(front, problem_front) < gx.indicators.igd_plus(
        front + 1, problem_front
    )
    assert 0 < gx.indicators.hypervolume(front, [1.1, 1.1]) < 0.8767
