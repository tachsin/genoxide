# Benchmark adapter for Evolutionary.jl (https://github.com/wildart/Evolutionary.jl, docs:
# https://wildart.github.io/Evolutionary.jl/stable/).
#
# Usage: julia --project=<this folder> --threads=1 bench.jl <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
#        julia --project=<this folder> bench.jl values <problem> <size>   # one JSON solution per line on stdin
#        julia --project=<this folder> bench.jl --version
# Prints one JSON line per solver per seed, see ../../README.md for the fields.
#
# The methods, their settings and where Evolutionary.jl recommends them are explained in
# docs/benchmarks/libraries/evolutionary_jl.md; each one is cited next to its code below.
#
# Every solver is warmed up (compiled) with an untimed run of the same problem, 1,000 evaluations
# and seed 1000, before the timed runs (rule 4.2), so time_s holds the optimization only.

using Evolutionary
using LinearAlgebra
using Random
using Logging

BLAS.set_num_threads(1)

# CMAES stops when its covariance matrix degenerates (NaN) and logs the whole matrix with @error:
# keep stderr readable (the run restarts there, see run_restarting)
disable_logging(Logging.Error)

# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to problems.py. Evolutionary.jl minimizes.
# -------------------------------------------------------------------------------------------------

# maximize the number of ones: minimize its negative
onemax(x::AbstractVector{Bool}) = -count(x)

# Diagonal conflicts: for each diagonal, its queens minus one. Julia permutations are 1-based:
# queen i is in column p[i], the diagonal indices are shifted by one compared to problems.py.
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

# multi-objective problems, in place F .= f(x) (the in-place form of docs/src/tutorial.md), all
# variables in [0, 1]
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
# The budget: counts every evaluation (rule 3) and keeps the best value and solution; stops at the
# target, at max_evaluations or at max_seconds (checked by the callback after every generation).
# -------------------------------------------------------------------------------------------------

mutable struct Budget
    evaluations::Int
    best::Float64
    solution::Any
    # evaluated solutions outside the bounds (rule 2.4), as the library proposed them
    outside::Int
    const max_evaluations::Int
    const max_seconds::Float64
    const target::Float64
    start::Float64
    # multi-objective: the objective values of every solution evaluated, to report the final
    # population's without evaluating it again
    const values::Dict{Vector{Float64}, Vector{Float64}}
end

Budget(max_evaluations, max_seconds, target = -Inf) =
    Budget(0, Inf, nothing, 0, max_evaluations, max_seconds, target, time(), Dict{Vector{Float64}, Vector{Float64}}())

outside(x, bounds) = bounds !== nothing && any(v -> v < bounds[1] || v > bounds[2], x)

# `bounds`: (lower, upper) of every variable, to count the solutions outside them
function counted(f, budget::Budget; bounds = nothing)
    return function (x)
        budget.evaluations += 1
        outside(x, bounds) && (budget.outside += 1)
        value = f(x)
        if value < budget.best
            budget.best = value
            budget.solution = copy(x)
        end
        return value
    end
end

function counted!(f!, budget::Budget)
    return function (F, x)
        budget.evaluations += 1
        outside(x, (0.0, 1.0)) && (budget.outside += 1)
        f!(F, x)
        budget.values[copy(x)] = copy(F)
        return F
    end
end

exhausted(budget::Budget) =
    budget.best <= budget.target ||
    budget.evaluations >= budget.max_evaluations ||
    time() - budget.start >= budget.max_seconds

# Rule 2.2. The iteration limit, only a budget, is lifted (typemax(Int); the library's default is
# 1,000, 1,500 for CMAES). The convergence test is the library's: each method's default metric
# (AbsDiff(1e-12) for GA and CMAES, AbsDiff(1e-10) for DE and ES, GD for NSGA2) below its tolerance
# for more than `successive_f_tol` generations (src/api/optimize.jl); `successive_f_tol` is the
# default 10, or the value of the library's example for the problem type. It ends the attempt, and
# run_restarting starts a new one. The callback of Options (docs/src/tutorial.md, "General
# options") ends the run at the target, the budget or the time cap, after every generation.
options(budget::Budget, rng; successive_f_tol = 10) = Evolutionary.Options(
    iterations = typemax(Int),
    successive_f_tol = successive_f_tol,
    callback = record -> exhausted(budget),
    rng = rng,
)

# An attempt that ends before the target, the budget or the cap has converged (the convergence
# test above), or, for CMAES, its covariance matrix broke down (update_state! returns true when the
# eigendecomposition fails, src/cmaes.jl). Evolutionary.jl has no restart mechanism, so the method
# starts again from a new random start with the seed seed * 1000 + restart (rule 2.2). The best
# solution and every evaluation are kept in `budget`. Returns (generations, restarts).
function run_restarting(start, budget::Budget, seed)
    generations = 0
    restart = 0
    while true
        rng = Xoshiro(restart == 0 ? seed : seed * 1000 + restart)
        result = start(rng)
        generations += Evolutionary.iterations(result)
        exhausted(budget) && return generations, restart
        restart += 1
    end
end

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

# Crossover with probability p, else copies of the parents (as DEAP's varAnd, which clones). The
# matched runs use crossoverRate = 1 and this operator: with crossoverRate < 1, Evolutionary passes
# the parents themselves (not copies) to the offspring, and the in-place mutation then also changes
# every other offspring that references the same parent, and the elites (see the library's page,
# "Bugs found"). The idiomatic runs use the library's own crossoverRate, as its users get it.
function crossover_with_probability(crossover, p::Float64)
    return function (a, b; rng::AbstractRNG = Random.default_rng())
        return rand(rng) < p ? crossover(a, b; rng = rng) : (copy(a), copy(b))
    end
end

# -------------------------------------------------------------------------------------------------
# Solvers: (name, run(budget, seed) -> (generations, restarts)) per problem
# -------------------------------------------------------------------------------------------------

function onemax_solvers(size, mode)
    if mode == "matched"
        # as DEAP eaSimple: population 300, tournament 3, two-point crossover with probability 0.5,
        # bit-flip with probability 1 / size on 20% of the children, no elitism (ɛ = 0).
        # Differences: Evolutionary's tournament draws its contestants without replacement from a
        # shuffled population (DEAP: with replacement); the crossover and bit-flip are the wrappers
        # above (the library's own `flip` flips exactly one bit). The GA's convergence test (AbsDiff
        # of the best value, with the default successive_f_tol) restarts it, as in every scenario.
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
        # test/onemax.jl of Evolutionary.jl, its only binary example: GA with tournament(3), flip,
        # TPX, mutationRate 0.05, crossoverRate 0.85, population = the genome size (100)
        method = () -> GA(
            populationSize = size,
            selection = tournament(3),
            crossover = TPX,
            crossoverRate = 0.85,
            mutation = flip,
            mutationRate = 0.05,
        )
    end
    run = (budget, seed) -> run_restarting(budget, seed) do rng
        Evolutionary.optimize(counted(onemax, budget), () -> bitrand(rng, size), method(), options(budget, rng))
    end
    return [("ga", run)]
end

function nqueens_solvers(size)
    # test/n-queens.jl of Evolutionary.jl, its only permutation example: GA with population 100,
    # tournament(5), crossoverRate 0.89, mutationRate 0.06; the test tries 5 mutations × 5
    # crossovers as equals, this is its first pair (PMX, inversion; the library's defaults, `genop`,
    # aren't among them), with successive_f_tol = 30
    ga = (budget, seed) -> run_restarting(budget, seed) do rng
        method = GA(
            populationSize = 100,
            selection = tournament(5),
            crossover = PMX,
            crossoverRate = 0.89,
            mutation = inversion,
            mutationRate = 0.06,
        )
        Evolutionary.optimize(counted(nqueens, budget), () -> randperm(rng, size), method, options(budget, rng; successive_f_tol = 30))
    end
    # test/n-queens.jl: (20+100)-ES with a GA mutation (mutationwrapper), ρ = 1, default options;
    # the test tries 5 mutations with :plus and :comma: the first mutation (inversion), and the
    # library's default selection, :plus
    es = (budget, seed) -> run_restarting(budget, seed) do rng
        method = ES(mutation = mutationwrapper(inversion), μ = 20, ρ = 1, λ = 100, selection = :plus)
        Evolutionary.optimize(counted(nqueens, budget), () -> randperm(rng, size), method, options(budget, rng))
    end
    return [("ga", ga), ("es", es)]
end

function real_solvers(problem, size)
    f, lower, upper = REAL_PROBLEMS[problem]
    # the bounds (docs/src/constraints.md, "Box Constrained Optimization"): a random initial
    # population within them, and clipping of the offspring
    solve(method; successive_f_tol = 10) = (budget, seed) -> run_restarting(budget, seed) do rng
        Evolutionary.optimize(
            counted(f, budget; bounds = (lower, upper)), BoxConstraints(lower, upper, size), method(),
            options(budget, rng; successive_f_tol),
        )
    end

    # Rule 6.2: the documentation states what two methods are for: CMA-ES (docs/src/cmaes.md: for
    # "difficult (non-convex, ill-conditioned, multi-modal, rugged, noisy) optimization problems in
    # continuous search spaces") and DE (docs/src/de.md: "used for multidimensional real-valued
    # functions"). The third is the library's example for the problem type: the ES of
    # test/rastrigin.jl for the multimodal problems, the GA of the getting-started example on the
    # Sphere function (docs/src/index.md) for the unimodal one. Their settings are those examples'.
    if problem == "rosenbrock"
        # docs/src/tutorial.md minimizes Rosenbrock with CMAES() and the default options: (10,20)-CMA-ES
        # with σ0 = 0.5, from a random point of the domain
        cma_es = solve(() -> CMAES())
        # test/rosenbrock.jl: DE(populationSize = 100) with the default options, i.e. DE/rand/1/bin
        # with F = 0.9, Cr = 0.5
        de = solve(() -> DE(populationSize = 100))
        # docs/src/index.md, "Getting started": GA(populationSize = 100, selection = susinv,
        # crossover = DC, mutation = PLM()) on the Sphere function, with the default options and
        # rates (crossover 0.8, mutation 0.1)
        third = ("ga", solve(() -> GA(populationSize = 100, selection = susinv, crossover = DC, mutation = PLM())))
    else
        # test/rastrigin.jl: CMAES(lambda = 100) with the default options: (50,100)-CMA-ES with
        # σ0 = 0.5, from a random point of the domain
        cma_es = solve(() -> CMAES(lambda = 100))
        # test/rastrigin.jl: DE with population 100 and F = 0.9, and successive_f_tol = 25; the
        # test varies the selection, the recombination and n, which take the library's defaults
        # (random, BINX(0.5), 1): DE/rand/1/bin
        de = solve(() -> DE(populationSize = 100, F = 0.9); successive_f_tol = 25)
        # test/rastrigin.jl, with iterations = 1000 (lifted): (15/15,100)-σ-SA-ES, anisotropic
        # self-adaptive gaussian mutation
        third = ("es", solve(() -> ES(
            initStrategy = AnisotropicStrategy(size),
            recombination = average, srecombination = average,
            mutation = gaussian, smutation = gaussian,
            μ = 15, λ = 100, selection = :comma,
        )))
    end
    return [("cma_es", cma_es), ("de", de), third]
end

# -------------------------------------------------------------------------------------------------
# Multi-objective: NSGA-II with the matched settings
#
# NSGA2 of Evolutionary.jl 0.12.0 has a bug (SciML/Evolutionary.jl#174): update_state!
# (src/nsga2.jl) reorders the parents (`parents .= state.population[fitidx]`) but not their
# objective values in `state.fitpop[:, 1:N]`, nor their ranks and crowding distances, so from the
# second generation on it sorts and selects on the values of other individuals. The multi-objective
# scenarios run it anyway, as the library's users get it (rule 8.4), and their results show the
# bug. The front printed holds the true objective values of the final population, recorded when
# they were evaluated. EVOLUTIONARY_JL_NSGA2=0 skips them (prints nothing).
# -------------------------------------------------------------------------------------------------

# the untimed warm-up run of every solver before the timed ones (rule 4.2)
const WARM_UP_EVALUATIONS = 1000
const WARM_UP_SEED = 1000

non_dominated(points) = [i for (i, p) in enumerate(points) if !any(q -> all(q .<= p) && any(q .< p), points)]

# NSGA-II with the matched settings of every library: population 100 (92 for DTLZ), SBX with η 15
# at 0.9, polynomial mutation with η 20 at 1 / n; binary tournament on rank and crowding distance
# (the library's default). Differences: Evolutionary's SBX and PLM are the unbounded variants
# (the offspring are clipped to [0, 1] by the box constraints), its SBX crosses each variable
# with probability 0.5 (as pymoo's default).
function run_front(problem, size, budget, seed)
    f!, variables, objectives, population_size = FRONT_PROBLEMS[problem]
    n = variables(size)
    m = objectives(size)
    population = Vector{Float64}[]
    generations, restarts = run_restarting(budget, seed) do rng
        method = NSGA2(
            populationSize = population_size,
            crossover = crossover_with_probability(SBX(0.5, 15), 0.9),
            crossoverRate = 1.0,
            mutation = PLM(1.0; η = 20, pm = 1.0 / n),
            mutationRate = 1.0,
        )
        # the initial population, uniform in [0, 1]; NSGA2 replaces its members in place, so it
        # holds the final population afterwards (rule 7.2): after a restart, the last attempt's.
        # The library's convergence test (its GD metrics, test/moea.jl's default options) applies.
        population = [rand(rng, n) for _ in 1:population_size]
        Evolutionary.optimize(
            counted!(f!, budget), zeros(m), BoxConstraints(0.0, 1.0, n), method, population, options(budget, rng),
        )
    end
    return generations, restarts, population
end

# -------------------------------------------------------------------------------------------------
# Output
# -------------------------------------------------------------------------------------------------

json_value(x::Bool) = string(x)
json_value(x::Integer) = string(x)
json_value(x::Real) = isfinite(x) ? repr(Float64(x)) : (isnan(x) ? "NaN" : (x > 0 ? "Infinity" : "-Infinity"))
json_value(x::AbstractString) = "\"$x\""
json_value(x::AbstractVector) = "[" * join((json_value(v) for v in x), ",") * "]"

function print_line(fields)
    println("{" * join(("\"$key\":" * json_value(value) for (key, value) in fields), ",") * "}")
    return flush(stdout)
end

# `values <problem> <size>`: one JSON solution per line on stdin, its value (or objectives) per
# line on stdout, with the fitness functions above (rule 1.2)
function print_values(problem, size)
    for line in eachline(stdin)
        text = strip(line, ['[', ']', ' ', '\t', '\r'])
        isempty(strip(line)) && continue
        x = [parse(Float64, strip(s)) for s in split(text, ',') if !isempty(strip(s))]
        if problem == "onemax"
            println(json_value(-onemax(x .!= 0)))
        elseif problem == "nqueens"
            println(json_value(nqueens(round.(Int, x) .+ 1)))
        elseif haskey(REAL_PROBLEMS, problem)
            println(json_value(REAL_PROBLEMS[problem][1](x)))
        else
            f!, _, objectives, _ = FRONT_PROBLEMS[problem]
            println(json_value(f!(zeros(objectives(size)), x)))
        end
    end
    return
end

function main(args)
    if length(args) == 1 && args[1] == "--version"
        println(pkgversion(Evolutionary))
        return
    end
    if length(args) == 3 && args[1] == "values"
        print_values(args[2], parse(Int, args[3]))
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
        if get(ENV, "EVOLUTIONARY_JL_NSGA2", "1") == "0"
            println(stderr, "evolutionary_jl: $problem skipped (EVOLUTIONARY_JL_NSGA2=0)")
            return
        end
        run_front(problem, size, Budget(WARM_UP_EVALUATIONS, 10.0), WARM_UP_SEED)
        for seed in seed_from:seed_to
            Random.seed!(seed)
            budget = Budget(max_evaluations, max_seconds)
            start = time_ns()
            generations, restarts, population = run_front(problem, size, budget, seed)
            elapsed = (time_ns() - start) / 1.0e9
            points = [budget.values[x] for x in population]
            front = non_dominated(points)
            print_line([
                "library" => "evolutionary_jl", "solver" => "nsga2", "problem" => problem, "size" => size,
                "mode" => mode, "seed" => seed, "time_s" => round(elapsed, digits = 6),
                "generations" => generations, "evaluations" => budget.evaluations, "restarts" => restarts,
                "outside" => budget.outside, "front" => points[front], "solutions" => population[front],
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

    for (_, run) in solvers
        run(Budget(WARM_UP_EVALUATIONS, 10.0, target), WARM_UP_SEED)
    end

    for seed in seed_from:seed_to, (solver, run) in solvers
        Random.seed!(seed)
        budget = Budget(max_evaluations, max_seconds, target)
        start = time_ns()
        generations, restarts = run(budget, seed)
        elapsed = (time_ns() - start) / 1.0e9
        if problem == "onemax"
            best, solution = -Int(budget.best), Int.(budget.solution)
        elseif problem == "nqueens"
            best, solution = Int(budget.best), budget.solution .- 1
        else
            best, solution = budget.best, budget.solution
        end
        success = problem == "onemax" ? best >= size : budget.best <= target
        print_line([
            "library" => "evolutionary_jl", "solver" => solver, "problem" => problem, "size" => size,
            "mode" => mode, "seed" => seed, "time_s" => round(elapsed, digits = 6),
            "generations" => generations, "evaluations" => budget.evaluations, "restarts" => restarts,
            (haskey(REAL_PROBLEMS, problem) ? ["outside" => budget.outside] : [])...,
            "best" => best, "target" => problem == "onemax" ? size : target, "success" => success,
            "solution" => solution,
        ])
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
