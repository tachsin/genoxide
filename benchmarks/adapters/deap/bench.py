"""Benchmark adapter for DEAP.

Usage:
    python bench.py <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
    python bench.py values <problem> <size>    (one JSON solution per line on stdin)

The first prints one JSON line per solver per seed, see ../../README.md for the fields; the second
prints the value (or objectives) of each solution with the fitness functions below.

The methods, their settings, where DEAP's docs and examples show them, what was left out and the
separate test runs are in docs/benchmarks/libraries/deap.md. Every setting below cites the DEAP
example it comes from (DEAP 1.4.4: https://github.com/DEAP/deap/tree/1.4.4/examples).
"""

import os

# one thread (rule 4.3): the CMA-ES's eigendecomposition is numpy's BLAS
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import json  # noqa: E402
import math  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import deque  # noqa: E402
from itertools import chain  # noqa: E402

import numpy  # noqa: E402
from deap import algorithms, base, cma, creator, tools  # noqa: E402

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("IndividualMax", list, fitness=creator.FitnessMax)
creator.create("IndividualMin", list, fitness=creator.FitnessMin)
creator.create("FitnessMin2", base.Fitness, weights=(-1.0, -1.0))
creator.create("FitnessMin3", base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create("Individual2", list, fitness=creator.FitnessMin2)
creator.create("Individual3", list, fitness=creator.FitnessMin3)

REAL_TARGET = 0.01


class Budget:
    """Counts the fitness evaluations (rule 3) and keeps the best solution evaluated. A run stops at
    the target, at max_evaluations or at max_seconds (rule 2.1). `start` is the run's clock."""

    def __init__(self, problem, size, max_evaluations, max_seconds, start):
        self.start = start
        self.evaluations = 0
        self.max_evaluations = max_evaluations
        self.deadline = start + max_seconds
        self.maximize = problem == "onemax"
        self.target = {"onemax": size, "nqueens": 0}.get(problem, REAL_TARGET)
        self.best = -math.inf if self.maximize else math.inf
        self.solution = None
        # the first evaluation that reaches the target: (its number, seconds since the start)
        self.first_hit = None
        # the evaluations when the last generation started (rule 2.3), marked by the solvers below
        self.generation_start = 0
        # the box of a continuous or multi-objective problem, and the evaluated solutions outside it
        # (rule 2.4)
        if problem in REAL_PROBLEMS:
            self.bounds = REAL_PROBLEMS[problem][1:]
        elif problem in FRONT_PROBLEMS:
            self.bounds = (0.0, 1.0)
        else:
            self.bounds = None
        self.outside = 0

    def count(self, solution):
        """Counts one evaluation (rule 3), and whether the solution is outside the box (rule 2.4)."""
        self.evaluations += 1
        if self.bounds is not None:
            low, high = self.bounds
            if any(v < low or v > high for v in solution):
                self.outside += 1

    def wrap(self, function):
        """The fitness function with the counter around it: every call on one solution counts."""

        def counted(individual):
            self.count(individual)
            values = function(individual)
            value = values[0]
            if value > self.best if self.maximize else value < self.best:
                self.best = value
                self.solution = list(individual)
                if self.first_hit is None and self.reached():
                    self.first_hit = (self.evaluations, time.perf_counter() - self.start)
            return values

        return counted

    def new_generation(self):
        """Marks the start of a generation: an initial population, or a loop of a solver below."""
        self.generation_start = self.evaluations

    def last_generation(self):
        """The evaluations since the start of the last generation (rule 2.3)."""
        return self.evaluations - self.generation_start

    def reached(self):
        return self.best >= self.target if self.maximize else self.best <= self.target

    def full(self):
        """Whether the evaluation budget is spent: the solvers check it before each evaluation, so a
        run never goes past it; the target and the time cap are checked between generations."""
        return self.evaluations >= self.max_evaluations

    def exhausted(self):
        return self.evaluations >= self.max_evaluations or time.perf_counter() >= self.deadline

    def done(self):
        return self.reached() or self.exhausted()


# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to benchmarks/problems.py, in plain Python as DEAP's examples write
# them (e.g. examples/ga/onemax.py, examples/ga/nqueens.py, deap/benchmarks). DEAP's fitness is a
# tuple.
# -------------------------------------------------------------------------------------------------


def onemax(individual):
    return (sum(individual),)


def nqueens(individual):
    """Diagonal conflicts of the queens at (i, individual[i]): for each diagonal, its queens minus
    one (O(n), like examples/ga/nqueens.py, which counts the conflicting pairs instead)."""
    size = len(individual)
    left_diagonal = [0] * (2 * size - 1)
    right_diagonal = [0] * (2 * size - 1)
    for i in range(size):
        left_diagonal[i + individual[i]] += 1
        right_diagonal[size - 1 - i + individual[i]] += 1
    conflicts = 0
    for i in range(2 * size - 1):
        if left_diagonal[i] > 1:
            conflicts += left_diagonal[i] - 1
        if right_diagonal[i] > 1:
            conflicts += right_diagonal[i] - 1
    return (conflicts,)


def shift(upper):
    """Rastrigin and Ackley are shifted, so an optimum at the origin can't favour operators that
    drift towards 0: gene i is measured from s_i = 0.8 upper (2 ((37 i + 11) mod 101) / 101 - 1),
    within 80% of the box, computed in this order, as problems.py."""
    return [0.8 * upper * (2 * ((37 * i + 11) % 101) / 101 - 1) for i in range(1000)]


RASTRIGIN_SHIFT = shift(5.12)
ACKLEY_SHIFT = shift(32.768)


def rastrigin(individual):
    return (
        10 * len(individual)
        + sum((x - s) ** 2 - 10 * math.cos(2 * math.pi * (x - s))
              for x, s in zip(individual, RASTRIGIN_SHIFT)),
    )


def rosenbrock(individual):
    return (
        sum(100 * (b - a * a) ** 2 + (1 - a) ** 2 for a, b in zip(individual, individual[1:])),
    )


def ackley(individual):
    n = len(individual)
    squares = sum((x - s) ** 2 for x, s in zip(individual, ACKLEY_SHIFT)) / n
    cosines = sum(math.cos(2 * math.pi * (x - s)) for x, s in zip(individual, ACKLEY_SHIFT)) / n
    return (-20 * math.exp(-0.2 * math.sqrt(squares)) - math.exp(cosines) + 20 + math.e,)


# the real-valued problems: fitness function and bounds
REAL_PROBLEMS = {
    "rastrigin": (rastrigin, -5.12, 5.12),
    "rosenbrock": (rosenbrock, -5.0, 10.0),
    "ackley": (ackley, -32.768, 32.768),
}
MULTIMODAL = {"rastrigin", "ackley"}


def zdt_g(x):
    return 1 + 9 * sum(x[1:]) / (len(x) - 1)


def zdt1(x):
    g = zdt_g(x)
    return x[0], g * (1 - math.sqrt(x[0] / g))


def zdt2(x):
    g = zdt_g(x)
    return x[0], g * (1 - (x[0] / g) ** 2)


def zdt3(x):
    g = zdt_g(x)
    return x[0], g * (1 - math.sqrt(x[0] / g) - x[0] / g * math.sin(10 * math.pi * x[0]))


def dtlz2(x, objectives):
    g = sum((v - 0.5) ** 2 for v in x[objectives - 1:])
    values = []
    for m in range(objectives):
        f = 1 + g
        for v in x[:objectives - 1 - m]:
            f *= math.cos(v * math.pi / 2)
        if m > 0:
            f *= math.sin(x[objectives - 1 - m] * math.pi / 2)
        values.append(f)
    return tuple(values)


def dtlz1(x, objectives):
    tail = x[objectives - 1:]
    g = 100 * (len(tail) + sum((v - 0.5) ** 2 - math.cos(20 * math.pi * (v - 0.5)) for v in tail))
    values = []
    for m in range(objectives):
        f = 0.5 * (1 + g)
        for v in x[:objectives - 1 - m]:
            f *= v
        if m > 0:
            f *= 1 - x[objectives - 1 - m]
        values.append(f)
    return tuple(values)


# (fitness function of the size, number of variables, number of objectives), all minimized, all
# variables in [0, 1]; the size of DTLZ is its number of objectives, with k = 10 (DTLZ2) and 5
# (DTLZ1) distance variables
FRONT_PROBLEMS = {
    "zdt1": (lambda size: zdt1, lambda size: size, lambda size: 2),
    "zdt2": (lambda size: zdt2, lambda size: size, lambda size: 2),
    "zdt3": (lambda size: zdt3, lambda size: size, lambda size: 2),
    "dtlz2": (lambda size: lambda x: dtlz2(x, size), lambda size: size + 9, lambda size: size),
    "dtlz1": (lambda size: lambda x: dtlz1(x, size), lambda size: size + 4, lambda size: size),
}


# -------------------------------------------------------------------------------------------------
# Binary and permutation: DEAP's GA examples with algorithms.eaSimple
# -------------------------------------------------------------------------------------------------


def ea_simple(toolbox, population_size, cxpb, mutpb, budget):
    """algorithms.eaSimple (deap/algorithms.py), with a stop at the target and at the time cap
    between generations, and at the budget: select, varAnd, evaluate the individuals whose fitness
    is invalid (varAnd keeps the fitness of an individual it neither crossed nor mutated), replace
    the population."""
    population = toolbox.population(n=population_size)
    budget.new_generation()
    for individual in population:
        individual.fitness.values = toolbox.evaluate(individual)
    generations = 0
    while not budget.done():
        generations += 1
        budget.new_generation()
        offspring = toolbox.select(population, len(population))
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)
        for individual in offspring:
            if not individual.fitness.valid:
                if budget.full():
                    break
                individual.fitness.values = toolbox.evaluate(individual)
        population[:] = offspring
    return generations


def solve_onemax(size, mode, budget):
    # examples/ga/onemax.py (docs: "One Max Problem"): 300 individuals, two-point crossover,
    # tournament of 3, eaSimple with cxpb 0.5 and mutpb 0.2. The matched scenarios are defined from
    # it: the only difference is the bit-flip probability 1 / n, about 0.2 bits per offspring.
    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.IndividualMax, toolbox.attr_bool, size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", budget.wrap(onemax))
    toolbox.register("mate", tools.cxTwoPoint)
    indpb = 1.0 / size if mode == "matched" else 0.05
    toolbox.register("mutate", tools.mutFlipBit, indpb=indpb)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return ea_simple(toolbox, 300, 0.5, 0.2, budget)


def solve_nqueens(size, budget):
    # examples/ga/nqueens.py: permutations, partially matched crossover, shuffle-indexes mutation
    # with indpb 2 / n, tournament of 3, 300 individuals, eaSimple with cxpb 0.5 and mutpb 0.2
    toolbox = base.Toolbox()
    toolbox.register("permutation", random.sample, range(size), size)
    toolbox.register("individual", tools.initIterate, creator.IndividualMin, toolbox.permutation)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", budget.wrap(nqueens))
    toolbox.register("mate", tools.cxPartialyMatched)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=2.0 / size)
    toolbox.register("select", tools.selTournament, tournsize=3)
    return ea_simple(toolbox, 300, 0.5, 0.2, budget)


# -------------------------------------------------------------------------------------------------
# Real-valued: BIPOP-CMA-ES and DE
# -------------------------------------------------------------------------------------------------


def bounded_evaluate(budget, function, low, high, size):
    """The fitness function within the box bounds, as DEAP's examples/es/cma_mo.py does for a box
    (its CMA-ES and DE are unbounded): tools.ClosestValidPenalty evaluates the closest point
    within the bounds and adds 1e6 times the squared distance to it. So every solution the fitness
    function evaluates is within the bounds (rule 2.4): a sample outside them is repaired by
    clipping, and keeps the penalized value. The counter is around the fitness function itself,
    inside DEAP's repair: it sees the repaired point, so `outside` is 0 by construction."""
    lower, upper = numpy.full(size, low), numpy.full(size, high)

    def valid(individual):
        return not (any(individual < lower) or any(individual > upper))

    def closest_feasible(individual):
        return numpy.minimum(upper, numpy.maximum(lower, numpy.array(individual)))

    def distance(feasible_ind, original_ind):
        return sum((f - o) ** 2 for f, o in zip(feasible_ind, original_ind))

    toolbox = base.Toolbox()
    toolbox.register("evaluate", budget.wrap(function))
    toolbox.decorate("evaluate", tools.ClosestValidPenalty(valid, closest_feasible, 1.0e6, distance))
    return toolbox


def solve_bipop_cmaes(problem, size, budget):
    """BI-population CMA-ES, as examples/es/cma_bipop.py (docs: "Controlling the Stopping Criteria:
    BI-POP CMA-ES"), line for line, with the budget and the target as extra stops.

    The example's domain is [-5, 5]: it starts each run at a uniform random point of [-4, 4] and
    with the large-population regime's sigma 2, "1/5th of the domain". Here that's the inner 80%
    of the problem's bounds and a fifth of their width. Its 9 stop criteria end each CMA-ES run and
    its BIPOP restarts start the next (rule 2.2: the library's restart mechanism); the adapter adds
    one criterion: DEAP's Strategy raises numpy.linalg.LinAlgError when its covariance matrix
    degenerates, and that ends the run too. The example ends after 10 runs (NRESTARTS, the first and
    9 restarts), a limit that's only a budget, so it's lifted (rule 2.2): the restarts go on to the
    budget with the example's rule for choosing the regime, less its clause that makes the tenth run
    a large-population one."""
    function, low, high = REAL_PROBLEMS[problem]
    N = size
    toolbox = bounded_evaluate(budget, function, low, high, N)
    width = high - low
    SIGMA0 = width / 5  # the example: 2.0, 1/5th of the domain [-5 5]

    nsmallpopruns = 0
    smallbudget = list()
    largebudget = list()
    lambda0 = 4 + int(3 * numpy.log(N))
    regime = 1
    i = 0
    generations = 0

    while not budget.done():
        # The first regime is enforced on the first restart. The second regime is run if its
        # allocated budget is smaller than the allocated large population regime budget
        if i > 0 and sum(smallbudget) < sum(largebudget):
            lambda_ = int(lambda0 * (0.5 * (2**(i - nsmallpopruns) * lambda0) / lambda0)**(numpy.random.rand()**2))
            # the example: 2 * 10**(-2 * numpy.random.rand()), SIGMA0 times 10**(-2 U)
            sigma = SIGMA0 * 10**(-2 * numpy.random.rand())
            nsmallpopruns += 1
            regime = 2
            smallbudget += [0]
        else:
            lambda_ = 2**(i - nsmallpopruns) * lambda0
            sigma = SIGMA0
            regime = 1
            largebudget += [0]

        t = 0

        # Set the termination criterion constants
        if regime == 1:
            MAXITER = 100 + 50 * (N + 3)**2 / numpy.sqrt(lambda_)
        elif regime == 2:
            MAXITER = 0.5 * largebudget[-1] / lambda_
        TOLHISTFUN = 10**-12
        TOLHISTFUN_ITER = 10 + int(numpy.ceil(30. * N / lambda_))
        EQUALFUNVALS = 1. / 3.
        EQUALFUNVALS_K = int(numpy.ceil(0.1 + lambda_ / 4.))
        TOLX = 10**-12
        TOLUPSIGMA = 10**20
        STAGNATION_ITER = int(numpy.ceil(0.2 * t + 120 + 30. * N / lambda_))
        NOEFFECTAXIS_INDEX = t % N

        equalfunvalues = list()
        bestvalues = list()
        medianvalues = list()
        mins = deque(maxlen=TOLHISTFUN_ITER)

        # the example: a centroid in [-4, 4]**D
        centroid = numpy.random.uniform(low + 0.1 * width, high - 0.1 * width, N)
        strategy = cma.Strategy(centroid=centroid, sigma=sigma, lambda_=lambda_)

        conditions = {"MaxIter": False, "TolHistFun": False, "EqualFunVals": False,
                      "TolX": False, "TolUpSigma": False, "Stagnation": False,
                      "ConditionCov": False, "NoEffectAxis": False, "NoEffectCoor": False,
                      "Degenerate": False}

        while not any(conditions.values()) and not budget.done():
            generations += 1
            budget.new_generation()
            # Generate a new population
            population = strategy.generate(creator.IndividualMin)

            # Evaluate the individuals
            for ind in population:
                if budget.full():
                    break
                ind.fitness.values = toolbox.evaluate(ind)
            if budget.done():
                break
            minimum = min(ind.fitness.values[0] for ind in population)

            # Update the strategy with the evaluated individuals
            try:
                strategy.update(population)
            except numpy.linalg.LinAlgError:
                conditions["Degenerate"] = True
                break

            # Count the number of times the k'th best solution is equal to the best solution
            # At this point the population is sorted (method update)
            if population[-1].fitness == population[-EQUALFUNVALS_K].fitness:
                equalfunvalues.append(1)

            # Log the best and median value of this population
            bestvalues.append(population[-1].fitness.values)
            medianvalues.append(population[int(round(len(population) / 2.))].fitness.values)

            # First run does not count into the budget
            if regime == 1 and i > 0:
                largebudget[-1] += lambda_
            elif regime == 2:
                smallbudget[-1] += lambda_

            t += 1
            STAGNATION_ITER = int(numpy.ceil(0.2 * t + 120 + 30. * N / lambda_))
            NOEFFECTAXIS_INDEX = t % N

            if t >= MAXITER:
                # The maximum number of iteration per CMA-ES ran
                conditions["MaxIter"] = True

            mins.append(minimum)
            if (len(mins) == mins.maxlen) and max(mins) - min(mins) < TOLHISTFUN:
                # The range of the best values is smaller than the threshold
                conditions["TolHistFun"] = True

            if t > N and sum(equalfunvalues[-N:]) / float(N) > EQUALFUNVALS:
                # In 1/3rd of the last N iterations the best and k'th best solutions are equal
                conditions["EqualFunVals"] = True

            if all(strategy.pc < TOLX) and all(numpy.sqrt(numpy.diag(strategy.C)) < TOLX):
                # All components of pc and sqrt(diag(C)) are smaller than the threshold
                conditions["TolX"] = True

            # Need to transfor strategy.diagD[-1]**2 from pyp/numpy.float64 to python
            # float to avoid OverflowError
            if strategy.sigma / sigma > float(strategy.diagD[-1]**2) * TOLUPSIGMA:
                # The sigma ratio is bigger than a threshold
                conditions["TolUpSigma"] = True

            if len(bestvalues) > STAGNATION_ITER and len(medianvalues) > STAGNATION_ITER and \
               numpy.median(bestvalues[-20:]) >= numpy.median(bestvalues[-STAGNATION_ITER:-STAGNATION_ITER + 20]) and \
               numpy.median(medianvalues[-20:]) >= numpy.median(medianvalues[-STAGNATION_ITER:-STAGNATION_ITER + 20]):
                # Stagnation occurred
                conditions["Stagnation"] = True

            if strategy.cond > 10**14:
                # The condition number is bigger than a threshold
                conditions["ConditionCov"] = True

            if all(strategy.centroid == strategy.centroid + 0.1 * strategy.sigma * strategy.diagD[-NOEFFECTAXIS_INDEX] * strategy.B[-NOEFFECTAXIS_INDEX]):
                # The coordinate axis std is too low
                conditions["NoEffectAxis"] = True

            if any(strategy.centroid == strategy.centroid + 0.2 * strategy.sigma * numpy.diag(strategy.C)):
                # The main axis std has no effect
                conditions["NoEffectCoor"] = True

        i += 1
    return generations


def mut_de(y, a, b, c, f):
    """examples/de/sphere.py, mutDE"""
    for i in range(len(y)):
        y[i] = a[i] + f * (b[i] - c[i])
    return y


def cx_exponential(x, y, cr):
    """examples/de/sphere.py, cxExponential, as the example has it: it stops copying from the
    mutant with probability cr after each gene, where exponential crossover continues with that
    probability (see the library's page)"""
    size = len(x)
    index = random.randrange(size)
    # Loop on the indices index -> end, then on 0 -> index
    for i in chain(range(index, size), range(0, index)):
        x[i] = y[i]
        if random.random() < cr:
            break
    return x


def solve_de(problem, size, budget):
    """DE as DEAP's examples/de: examples/de/sphere.py for the multimodal problems (it minimizes
    Griewank: DE/rand/1 with exponential crossover, F 0.8, CR 0.8, 10 n agents, a generation's
    children replace their agents at its end), examples/de/basic.py for Rosenbrock (it minimizes
    the sphere: DE/rand/1 with binomial crossover, F 1, CR 0.25, 300 agents, each child replaces
    its agent at once). Both start uniformly within the bounds (the examples: [-3, 3]) and run to
    the budget by themselves (the examples run NGEN generations)."""
    function, low, high = REAL_PROBLEMS[problem]
    toolbox = bounded_evaluate(budget, function, low, high, size)
    toolbox.register("attr_float", random.uniform, low, high)
    toolbox.register("individual", tools.initRepeat, creator.IndividualMin, toolbox.attr_float, size)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("select", tools.selRandom, k=3)
    multimodal = problem in MULTIMODAL
    if multimodal:
        toolbox.register("mutate", mut_de, f=0.8)
        toolbox.register("mate", cx_exponential, cr=0.8)
        mu = size * 10
    else:
        CR, F = 0.25, 1
        mu = 300

    pop = toolbox.population(n=mu)
    budget.new_generation()
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)
    generations = 0
    while not budget.done():
        generations += 1
        budget.new_generation()
        if multimodal:
            # examples/de/sphere.py
            children = []
            for agent in pop:
                # We must clone everything to ensure independence
                a, b, c = [toolbox.clone(ind) for ind in toolbox.select(pop)]
                x = toolbox.clone(agent)
                y = toolbox.clone(agent)
                y = toolbox.mutate(y, a, b, c)
                z = toolbox.mate(x, y)
                del z.fitness.values
                children.append(z)
            for i, ind in enumerate(children):
                if budget.full():
                    break
                ind.fitness.values = toolbox.evaluate(ind)
                if ind.fitness > pop[i].fitness:
                    pop[i] = ind
        else:
            # examples/de/basic.py
            for k, agent in enumerate(pop):
                if budget.full():
                    break
                a, b, c = toolbox.select(pop)
                y = toolbox.clone(agent)
                index = random.randrange(size)
                for i, value in enumerate(agent):
                    if i == index or random.random() < CR:
                        y[i] = a[i] + F * (b[i] - c[i])
                y.fitness.values = toolbox.evaluate(y)
                if y.fitness > agent.fitness:
                    pop[k] = y
    return generations


# -------------------------------------------------------------------------------------------------
# Multi-objective: NSGA-II and NSGA-III with the matched settings
# -------------------------------------------------------------------------------------------------


def solve_front(problem, size, solver, budget):
    """NSGA-II as examples/ga/nsga2.py, NSGA-III as examples/ga/nsga3.py, with the matched settings
    of every library: 100 individuals (92 with 3 objectives), SBX with eta 15 at 0.9 and polynomial
    mutation with eta 20 at 1 / n for NSGA-II; Das-Dennis reference points with 99 divisions (12
    with 3 objectives), SBX with eta 30 at 1 and the same mutation for NSGA-III. Returns the final
    population and the generations."""
    function, variables, objectives = (f(size) for f in FRONT_PROBLEMS[problem])
    n = variables
    population_size = 100 if objectives == 2 else 92
    individual = creator.Individual2 if objectives == 2 else creator.Individual3
    toolbox = base.Toolbox()
    toolbox.register("attribute", random.random)
    toolbox.register("individual", tools.initRepeat, individual, toolbox.attribute, n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", counted_front(budget, function))
    eta = 15.0 if solver == "nsga2" else 30.0
    toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0.0, up=1.0, eta=eta)
    toolbox.register("mutate", tools.mutPolynomialBounded, low=0.0, up=1.0, eta=20.0, indpb=1.0 / n)
    if solver == "nsga2":
        toolbox.register("select", tools.selNSGA2)
    else:
        reference = tools.uniform_reference_points(objectives, 99 if objectives == 2 else 12)
        toolbox.register("select", tools.selNSGA3, ref_points=reference)

    population = toolbox.population(n=population_size)
    budget.new_generation()
    for member in population:
        member.fitness.values = toolbox.evaluate(member)
    if solver == "nsga2":
        # nsga2.py: "This is just to assign the crowding distance to the individuals" (nsga3.py
        # has no such step)
        population = toolbox.select(population, len(population))
    generations = 0
    while not budget.exhausted():
        generations += 1
        budget.new_generation()
        if solver == "nsga2":
            # nsga2.py: crowded binary tournament, then each pair crossed with probability 0.9 and
            # both children mutated
            offspring = [toolbox.clone(member) for member in tools.selTournamentDCD(population, len(population))]
            for a, b in zip(offspring[::2], offspring[1::2]):
                if random.random() <= 0.9:
                    toolbox.mate(a, b)
                toolbox.mutate(a)
                toolbox.mutate(b)
                del a.fitness.values, b.fitness.values
        else:
            # nsga3.py: varAnd with cxpb 1 and mutpb 1
            offspring = algorithms.varAnd(population, toolbox, 1.0, 1.0)
        for member in offspring:
            if not member.fitness.valid:
                member.fitness.values = toolbox.evaluate(member)
        population = toolbox.select(population + offspring, population_size)
    return population, generations


def counted_front(budget, function):
    def counted(individual):
        budget.count(individual)
        return function(individual)

    return counted


# -------------------------------------------------------------------------------------------------


def solvers_of(problem, size, mode):
    """(solver name, function of the budget returning the generations) of a problem."""
    if problem == "onemax":
        return [("ga", lambda budget: solve_onemax(size, mode, budget))]
    if problem == "nqueens":
        return [("ga", lambda budget: solve_nqueens(size, budget))]
    if problem in REAL_PROBLEMS:
        return [("cma_es", lambda budget: solve_bipop_cmaes(problem, size, budget)),
                ("de", lambda budget: solve_de(problem, size, budget))]
    return None


def values(problem, size):
    """Prints the value, or the objectives, of each solution read from stdin."""
    if problem in FRONT_PROBLEMS:
        function = FRONT_PROBLEMS[problem][0](size)
    else:
        function = {"onemax": onemax, "nqueens": nqueens}.get(problem) or REAL_PROBLEMS[problem][0]
    for line in sys.stdin:
        if line.strip():
            result = function(json.loads(line))
            print(json.dumps(list(result) if problem in FRONT_PROBLEMS else result[0]), flush=True)


def main():
    if len(sys.argv) == 4 and sys.argv[1] == "values":
        values(sys.argv[2], int(sys.argv[3]))
        return
    if len(sys.argv) != 8:
        print(__doc__, file=sys.stderr)
        sys.exit(2)
    problem, size, mode = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    seed_from, seed_to = int(sys.argv[4]), int(sys.argv[5])
    max_evaluations, max_seconds = int(sys.argv[6]), float(sys.argv[7])

    for seed in range(seed_from, seed_to + 1):
        if problem in FRONT_PROBLEMS:
            for solver in ("nsga2", "nsga3"):
                random.seed(seed)
                numpy.random.seed(seed)
                start = time.perf_counter()
                budget = Budget(problem, size, max_evaluations, max_seconds, start)
                population, generations = solve_front(problem, size, solver, budget)
                elapsed = time.perf_counter() - start
                # rule 7.2, after the clock: the first non-dominated front of the final population
                front = tools.sortNondominated(population, len(population), first_front_only=True)[0]
                print(json.dumps({
                    "library": "deap", "solver": solver, "problem": problem, "size": size, "mode": mode,
                    "seed": seed, "time_s": round(elapsed, 6), "generations": generations,
                    "evaluations": budget.evaluations, "last_generation": budget.last_generation(),
                    "outside": budget.outside,
                    "front": [list(member.fitness.values) for member in front],
                    "solutions": [list(member) for member in front],
                }), flush=True)
            continue

        solvers = solvers_of(problem, size, mode)
        if solvers is None:
            print(f"unknown problem {problem}", file=sys.stderr)
            sys.exit(2)
        for solver, solve in solvers:
            # DEAP draws from Python's random; its CMA-ES from numpy's
            random.seed(seed)
            numpy.random.seed(seed)
            start = time.perf_counter()
            budget = Budget(problem, size, max_evaluations, max_seconds, start)
            generations = solve(budget)
            elapsed = time.perf_counter() - start
            result = {
                "library": "deap", "solver": solver, "problem": problem, "size": size, "mode": mode,
                "seed": seed, "time_s": round(elapsed, 6), "generations": generations,
                "evaluations": budget.evaluations, "last_generation": budget.last_generation(),
            }
            if problem in REAL_PROBLEMS:
                result.update(outside=budget.outside, best=float(budget.best),
                              solution=[float(v) for v in budget.solution])
            else:
                result.update(best=int(budget.best), solution=[int(v) for v in budget.solution])
            result.update(target=budget.target, success=bool(budget.reached()),
                          first_hit=budget.first_hit and {"evaluations": budget.first_hit[0],
                                                          "time_s": round(budget.first_hit[1], 6)})
            print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
