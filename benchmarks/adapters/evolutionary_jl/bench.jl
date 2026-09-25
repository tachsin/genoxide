# Benchmark adapter for Evolutionary.jl (https://github.com/wildart/Evolutionary.jl).
#
# Usage: julia --project=<this folder> --threads=1 bench.jl <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
#        julia --project=<this folder> bench.jl --version
# Prints one JSON line per solver per seed, see ../../README.md for the fields.
#
# Every solver is warmed up (compiled) with a tiny untimed run of the same problem before the
# timed runs, so time_s holds the optimization only.

using Evolutionary
using LinearAlgebra
using Random
using Logging

BLAS.set_num_threads(1)

# CMAES stops when its covariance matrix degenerates (NaN) and logs the whole matrix with @error:
# keep stderr readable (the run still ends there, with its true numbers)
disable_logging(Logging.Error)

# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to the ones in the other adapters. Evolutionary.jl minimizes.
# -------------------------------------------------------------------------------------------------

# maximize the number of ones: minimize its negative
onemax(x::AbstractVector{Bool}) = -count(x)

# Number of diagonal conflicts, O(n) (DEAP examples/ga/nqueens.py). Julia permutations are 1-based:
# queen i is in column p[i], the diagonal indices are shifted by one compared to the Python code.
function nqueens(p::AbstractVector{<:Integer})
    n = length(p)
    left = zeros(Int, 2n - 1)
    right = zeros(Int, 2n - 1)
    @inbounds for i in 1:n
        left[i + p[i] - 1] += 1
        right[n - i + p[i]] += 1
    end
    conflicts = 0
    @inbounds for i in 1:(2n - 1)
        left[i] > 1 && (conflicts += left[i] - 1)
        right[i] > 1 && (conflicts += right[i] - 1)
    end
    return conflicts
end

# The shift of Rastrigin and Ackley, so that the optimum isn't at the origin:
# s_i = 2 ((37 i + 11) mod 101) / 101 - 1 for the 0-based gene index i
shift(index::Integer) = 2 * ((37 * (index - 1) + 11) % 101) / 101 - 1

function rastrigin(x::AbstractVector{<:Real})
    s = 10.0 * length(x)
    for (index, v) in enumerate(x)
        y = v - shift(index)
        s += y * y - 10.0 * cos(2π * y)
    end
    return s
end

function rosenbrock(x::AbstractVector{<:Real})
    s = 0.0
    for i in 1:(length(x) - 1)
        s += 100.0 * (x[i + 1] - x[i]^2)^2 + (1.0 - x[i])^2
    end
    return s
end

function ackley(x::AbstractVector{<:Real})
    n = length(x)
    squares = 0.0
    cosines = 0.0
    for (index, v) in enumerate(x)
        y = v - shift(index)
        squares += y * y
        cosines += cos(2π * y)
    end
    return -20.0 * exp(-0.2 * sqrt(squares / n)) - exp(cosines / n) + 20.0 + ℯ
end

# (function, lower bound, upper bound)
const REAL_PROBLEMS = Dict(
    "rastrigin" => (rastrigin, -5.12, 5.12),
    "rosenbrock" => (rosenbrock, -5.0, 10.0),
    "ackley" => (ackley, -32.768, 32.768),
)
const REAL_TARGET = 0.01

# multi-objective problems, in-place F .= f(x), all variables in [0, 1]
zdt_g(x) = 1.0 + 9.0 * sum(@view x[2:end]) / (length(x) - 1)

function zdt1!(F, x)
    g = zdt_g(x)
    F[1] = x[1]
    F[2] = g * (1.0 - sqrt(x[1] / g))
    return F
end

function zdt2!(F, x)
    g = zdt_g(x)
    F[1] = x[1]
    F[2] = g * (1.0 - (x[1] / g)^2)
    return F
end

function zdt3!(F, x)
    g = zdt_g(x)
    F[1] = x[1]
    F[2] = g * (1.0 - sqrt(x[1] / g) - x[1] / g * sin(10π * x[1]))
    return F
end

# DTLZ2 with M = length(F) objectives and k = length(x) - M + 1 = 10
function dtlz2!(F, x)
    m = length(F)
    g = 0.0
    for v in @view x[m:end]
        g += (v - 0.5)^2
    end
    for j in 1:m
        f = 1.0 + g
        for v in @view x[1:(m - j)]
            f *= cos(v * π / 2)
        end
        j > 1 && (f *= sin(x[m - j + 1] * π / 2))
        F[j] = f
    end
    return F
end

# DTLZ1 with M = length(F) objectives and k = length(x) - M + 1 = 5
function dtlz1!(F, x)
    m = length(F)
    k = length(x) - m + 1
    s = 0.0
    for v in @view x[m:end]
        s += (v - 0.5)^2 - cos(20π * (v - 0.5))
    end
    g = 100.0 * (k + s)
    for j in 1:m
        f = 0.5 * (1.0 + g)
        for v in @view x[1:(m - j)]
            f *= v
        end
        j > 1 && (f *= 1.0 - x[m - j + 1])
        F[j] = f
    end
    return F
end

# (function, variables for the size, objectives for the size, population size)
const FRONT_PROBLEMS = Dict(
    "zdt1" => (zdt1!, size -> size, size -> 2, 100),
    "zdt2" => (zdt2!, size -> size, size -> 2, 100),
    "zdt3" => (zdt3!, size -> size, size -> 2, 100),
    # size: the number of objectives, with k = 10 (DTLZ2) and k = 5 (DTLZ1)
    "dtlz2" => (dtlz2!, size -> size + 9, size -> size, 92),
    "dtlz1" => (dtlz1!, size -> size + 4, size -> size, 92),
)

# -------------------------------------------------------------------------------------------------
# The budget: counts the evaluations and keeps the best value, stops at the target, at
# max_evaluations or at max_seconds (checked by the callback after every generation).
# -------------------------------------------------------------------------------------------------

mutable struct Budget
    evaluations::Int
    best::Float64
    const max_evaluations::Int
    const max_seconds::Float64
    const target::Float64
    start::Float64
end

Budget(max_evaluations, max_seconds, target = -Inf) =
    Budget(0, Inf, max_evaluations, max_seconds, target, time())

function counted(f, budget::Budget)
    return function (x)
        budget.evaluations += 1
        value = f(x)
        value < budget.best && (budget.best = value)
        return value
    end
end

function counted!(f!, budget::Budget)
    return function (F, x)
        budget.evaluations += 1
        return f!(F, x)
    end
end

exhausted(budget::Budget) =
    budget.best <= budget.target ||
    budget.evaluations >= budget.max_evaluations ||
    time() - budget.start >= budget.max_seconds

# Options without the library's own stagnation stops (AbsDiff over successive_f_tol generations),
# so that every library stops at the same criteria: the target, the evaluations or the time.
options(budget::Budget, rng) = Evolutionary.Options(
    iterations = typemax(Int),
    successive_f_tol = typemax(Int),
    time_limit = budget.max_seconds,
    callback = record -> exhausted(budget),
    rng = rng,
)

# -------------------------------------------------------------------------------------------------
# Operators for the matched configuration
# -------------------------------------------------------------------------------------------------

# Bit-flip of every gene with probability p (DEAP's mutFlipBit). Evolutionary's `flip` flips exactly
# one random bit.
function bitflip_per_gene(p::Float64)
    return function (x::AbstractVector{Bool}; rng::AbstractRNG = Random.default_rng())
        @inbounds for i in eachindex(x)
            rand(rng) < p && (x[i] = !x[i])
        end
        return x
    end
end

# Crossover with probability p, else copies of the parents (as DEAP's varAnd, which clones). The GA
# runs with crossoverRate = 1 and this operator: with crossoverRate < 1, Evolutionary passes the
# parents themselves (not copies) to the offspring, the in-place mutation then also changes every
# other offspring that references the same parent, and in NSGA2 the parent itself, whose fitness
# isn't evaluated again.
function crossover_with_probability(crossover, p::Float64)
    return function (a, b; rng::AbstractRNG = Random.default_rng())
        return rand(rng) < p ? crossover(a, b; rng = rng) : (copy(a), copy(b))
    end
end

# -------------------------------------------------------------------------------------------------
# Solvers: (name, run(budget, rng) -> result) per problem
# -------------------------------------------------------------------------------------------------

function onemax_solvers(size, mode)
    if mode == "matched"
        # as DEAP eaSimple: population 300, tournament 3, two-point crossover with probability 0.5,
        # bit-flip with probability 1 / size on 20% of the children, no elitism (ɛ = 0).
        # Differences: Evolutionary's tournament draws its contestants without replacement from a
        # shuffled population (DEAP: with replacement); the crossover and bit-flip are the wrappers
        # above (the library's own `flip` flips exactly one bit).
        method = () -> GA(
            populationSize = 300,
            selection = tournament(3),
            crossover = crossover_with_probability(TPX, 0.5),
            crossoverRate = 1.0,
            mutation = bitflip_per_gene(1.0 / size),
            mutationRate = 0.2,
            ɛ = 0,
        )
    else
        # test/onemax.jl of Evolutionary.jl: tournament(3), flip, TPX, mutation 0.05, crossover 0.85,
        # population 100 (the test's population is the genome size, 100)
        method = () -> GA(
            populationSize = 100,
            selection = tournament(3),
            crossover = TPX,
            crossoverRate = 0.85,
            mutation = flip,
            mutationRate = 0.05,
        )
    end
    run = function (budget, rng)
        individual = () -> bitrand(rng, size)
        return Evolutionary.optimize(counted(onemax, budget), individual, method(), options(budget, rng))
    end
    return [("ga", run)]
end

function nqueens_solvers(size)
    # test/n-queens.jl of Evolutionary.jl: population 100, tournament(5), crossover 0.89, mutation
    # 0.06; the test tries 5 mutations × 5 crossovers, this is its first pair (PMX, inversion)
    ga = function (budget, rng)
        method = GA(
            populationSize = 100,
            selection = tournament(5),
            crossover = PMX,
            crossoverRate = 0.89,
            mutation = inversion,
            mutationRate = 0.06,
        )
        return Evolutionary.optimize(counted(nqueens, budget), () -> randperm(rng, size), method, options(budget, rng))
    end
    # test/n-queens.jl: (20+100)-ES with a GA mutation, ρ = 1 (the first of its mutations, inversion)
    es = function (budget, rng)
        method = ES(mutation = mutationwrapper(inversion), μ = 20, ρ = 1, λ = 100, selection = :plus)
        return Evolutionary.optimize(counted(nqueens, budget), () -> randperm(rng, size), method, options(budget, rng))
    end
    return [("ga", ga), ("es", es)]
end

function real_solvers(problem, size)
    f, lower, upper = REAL_PROBLEMS[problem]
    # the bounds: a random initial population within them, and clipping of the offspring
    constraints = () -> BoxConstraints(lower, upper, size)
    solve(method) = (budget, rng) ->
        Evolutionary.optimize(counted(f, budget), constraints(), method(), options(budget, rng))

    # test/rosenbrock.jl of Evolutionary.jl: GA(populationSize = 100, ɛ = 0.1, selection = rouletteinv,
    # crossover = IC(0.2), mutation = BGA(fill(0.5, N))) with the default rates (crossover 0.8,
    # mutation 0.1); the only single real-valued GA configuration of its tests and docs
    ga = () -> GA(
        populationSize = 100,
        ɛ = 0.1,
        selection = rouletteinv,
        crossover = IC(0.2),
        mutation = BGA(fill(0.5, size)),
    )
    # test/rosenbrock.jl: DE(populationSize = 100), i.e. DE/rand/1/bin with F = 0.9, Cr = 0.5
    de = () -> DE(populationSize = 100)
    # test/rastrigin.jl and test/rosenbrock.jl: CMAES(lambda = 100), i.e. (50,100)-CMA-ES, σ0 = 0.5,
    # from a random point of the domain
    cma_es = () -> CMAES(lambda = 100)
    if problem == "rosenbrock"
        # test/rosenbrock.jl: (15/3+100)-ES, isotropic self-adaptive gaussian mutation
        es = () -> ES(
            initStrategy = IsotropicStrategy(size),
            recombination = average, srecombination = average,
            mutation = gaussian, smutation = gaussian,
            μ = 15, ρ = 3, λ = 100, selection = :plus,
        )
    else
        # test/rastrigin.jl: (15/15,100)-σ-SA-ES, anisotropic self-adaptive gaussian mutation
        es = () -> ES(
            initStrategy = AnisotropicStrategy(size),
            recombination = average, srecombination = average,
            mutation = gaussian, smutation = gaussian,
            μ = 15, λ = 100, selection = :comma,
        )
    end
    return [("ga", solve(ga)), ("de", solve(de)), ("cma_es", solve(cma_es)), ("es", solve(es))]
end

# -------------------------------------------------------------------------------------------------
# Multi-objective: NSGA-II with the matched settings
#
# NSGA2_BROKEN: NSGA2 of Evolutionary.jl 0.12.0 doesn't work: update_state! (src/nsga2.jl)
# reorders the parents (`parents .= state.population[fitidx]`) but not their objective values in
# `state.fitpop[:, 1:N]`, nor their ranks and crowding distances, so from the second generation on
# it sorts and selects on the values of other individuals. With the library's defaults
# (NSGA2(populationSize = 100)) on ZDT1, the recorded objective values of its "fittest" differ from
# their genomes' by up to 3.7 after 250 generations, and the final population has no point inside
# the reference point (1.1, 1.1): the hypervolume is 0.
# The multi-objective scenarios run it anyway, as the library's users get it, and their results
# show the bug. The bug isn't worked around; the front printed is the true objective values of the
# final population. EVOLUTIONARY_JL_NSGA2=0 skips them (prints nothing).
# -------------------------------------------------------------------------------------------------

# the untimed warm-up run of every solver before the timed ones: a few generations, which compile
# everything the timed runs call
const WARM_UP_EVALUATIONS = 1000

non_dominated(points) = [p for p in points if !any(q -> all(q .<= p) && any(q .< p), points)]

# NSGA-II with the matched settings of every library: population 100 (92 for DTLZ), SBX with η 15
# at 0.9, polynomial mutation with η 20 at 1 / n; binary tournament on rank and crowding distance
# (the library's default). Differences: Evolutionary's SBX and PLM are the unbounded variants
# (the offspring are clipped to [0, 1] by the box constraints), its SBX crosses each variable
# with probability 0.5 (as pymoo's default).
function run_front(problem, size, budget, rng)
    f!, variables, objectives, population_size = FRONT_PROBLEMS[problem]
    n = variables(size)
    m = objectives(size)
    method = NSGA2(
        populationSize = population_size,
        crossover = crossover_with_probability(SBX(0.5, 15), 0.9),
        crossoverRate = 1.0,
        mutation = PLM(1.0; η = 20, pm = 1.0 / n),
        mutationRate = 1.0,
    )
    # the initial population, uniform in [0, 1]; NSGA2 replaces its members in place, so it
    # holds the final population afterwards
    population = [rand(rng, n) for _ in 1:population_size]
    result = Evolutionary.optimize(
        counted!(f!, budget), zeros(m), BoxConstraints(0.0, 1.0, n), method, population, options(budget, rng),
    )
    return result, population, f!, m
end

# -------------------------------------------------------------------------------------------------
# Output
# -------------------------------------------------------------------------------------------------

json_number(x::Integer) = string(x)
json_number(x::Real) = isfinite(x) ? repr(Float64(x)) : (isnan(x) ? "NaN" : (x > 0 ? "Infinity" : "-Infinity"))

function print_line(fields)
    parts = String[]
    for (key, value) in fields
        text = value isa AbstractString ? "\"$value\"" :
            value isa Bool ? string(value) :
            value isa AbstractVector ? "[" * join(("[" * join(json_number.(p), ",") * "]" for p in value), ",") * "]" :
            json_number(value)
        push!(parts, "\"$key\":$text")
    end
    println("{" * join(parts, ",") * "}")
    return flush(stdout)
end

function main(args)
    if length(args) == 1 && args[1] == "--version"
        println(pkgversion(Evolutionary))
        return
    end
    if length(args) != 7
        println(stderr, "usage: bench.jl <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>")
        exit(2)
    end
    problem, size, mode = args[1], parse(Int, args[2]), args[3]
    seed_from, seed_to = parse(Int, args[4]), parse(Int, args[5])
    max_evaluations, max_seconds = parse(Int, args[6]), parse(Float64, args[7])

    if haskey(FRONT_PROBLEMS, problem)
        # NSGA2 of Evolutionary.jl 0.12.0 has a bug (see NSGA2_BROKEN): it runs anyway, and its
        # results show the bug, unless EVOLUTIONARY_JL_NSGA2=0
        if get(ENV, "EVOLUTIONARY_JL_NSGA2", "1") == "0"
            println(stderr, "evolutionary_jl: $problem skipped (EVOLUTIONARY_JL_NSGA2=0)")
            return
        end
        # warm-up: compile with a tiny untimed run
        run_front(problem, size, Budget(WARM_UP_EVALUATIONS, 10.0), Xoshiro(0))
        for seed in seed_from:seed_to
            Random.seed!(seed)
            rng = Xoshiro(seed)
            budget = Budget(max_evaluations, max_seconds)
            start = time_ns()
            result, population, f!, m = run_front(problem, size, budget, rng)
            elapsed = (time_ns() - start) / 1.0e9
            # the objective values of the final population (not counted: reporting only)
            front = non_dominated([f!(zeros(m), x) for x in population])
            print_line([
                "library" => "evolutionary_jl", "solver" => "nsga2", "problem" => problem, "size" => size,
                "mode" => mode, "seed" => seed, "time_s" => round(elapsed, digits = 6),
                "generations" => Evolutionary.iterations(result), "evaluations" => budget.evaluations,
                "front" => front,
            ])
        end
        return
    end

    if problem == "onemax"
        solvers = onemax_solvers(size, mode)
        target = -size  # minimized
    elseif problem == "nqueens"
        solvers = nqueens_solvers(size)
        target = 0
    elseif haskey(REAL_PROBLEMS, problem)
        solvers = real_solvers(problem, size)
        target = REAL_TARGET
    else
        # unsupported problem: print nothing
        println(stderr, "evolutionary_jl: unsupported problem $problem")
        return
    end

    # warm-up: compile every solver with a tiny untimed run of the same problem
    for (_, run) in solvers
        run(Budget(WARM_UP_EVALUATIONS, 10.0, target), Xoshiro(0))
    end

    for seed in seed_from:seed_to, (solver, run) in solvers
        Random.seed!(seed)
        rng = Xoshiro(seed)
        budget = Budget(max_evaluations, max_seconds, target)
        start = time_ns()
        result = run(budget, rng)
        elapsed = (time_ns() - start) / 1.0e9
        best = problem == "onemax" ? -Int(budget.best) : problem == "nqueens" ? Int(budget.best) : budget.best
        success = problem == "onemax" ? best >= size : budget.best <= target
        print_line([
            "library" => "evolutionary_jl", "solver" => solver, "problem" => problem, "size" => size,
            "mode" => mode, "seed" => seed, "time_s" => round(elapsed, digits = 6),
            "generations" => Evolutionary.iterations(result), "evaluations" => budget.evaluations,
            "best" => best, "target" => problem == "onemax" ? size : target, "success" => success,
        ])
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
