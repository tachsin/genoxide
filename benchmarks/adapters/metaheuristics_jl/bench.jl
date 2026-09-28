# Benchmark adapter for Metaheuristics.jl (https://github.com/jmejia8/Metaheuristics.jl, docs:
# https://jmejia8.github.io/Metaheuristics.jl/stable/).
#
# Usage: julia --project=<this folder> --threads=1 bench.jl <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
#        julia --project=<this folder> bench.jl values <problem> <size>   # one JSON solution per line on stdin
#        julia --project=<this folder> bench.jl --version
# Prints one JSON line per solver per seed, see ../../README.md for the fields.
#
# The methods, their settings and where Metaheuristics.jl recommends them are explained in
# docs/benchmarks/libraries/metaheuristics_jl.md; each one is cited next to its code below. "The
# guide" is the "Quick Selection Guide" of docs/src/algorithms/index.md
# (https://jmejia8.github.io/Metaheuristics.jl/stable/algorithms/).
#
# Every solver is warmed up (compiled) with an untimed, unprinted run of the same problem, with
# the seed 999,999, 50,000 evaluations and the scenario's time cap, before the timed runs (rule
# 4.2), so time_s holds the optimization only.

using Metaheuristics
using LinearAlgebra
using Random

BLAS.set_num_threads(1)

# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to problems.py. Metaheuristics.jl minimizes.
# -------------------------------------------------------------------------------------------------

onemax(x::AbstractVector{Bool}) = -count(x)

# Diagonal conflicts: for each diagonal, its queens minus one. 1-based: queen i in column p[i]
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
# s_i = 0.8 upper (2 ((37 i + 11) mod 101) / 101 - 1) for the 0-based gene index i, with `upper`
# the box's upper bound, computed in this order (problems.py). Computed once, before any run, for up
# to MAX_GENES genes.
shift(index::Integer, upper) = 0.8 * upper * (2 * ((37 * (index - 1) + 11) % 101) / 101 - 1)
const MAX_GENES = 1024
const RASTRIGIN_SHIFT = [shift(index, 5.12) for index in 1:MAX_GENES]
const ACKLEY_SHIFT = [shift(index, 32.768) for index in 1:MAX_GENES]

function rastrigin(x::AbstractVector{<:Real})
    s = 10.0 * length(x)
    for (v, offset) in zip(x, RASTRIGIN_SHIFT)
        y = v - offset
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
    for (v, offset) in zip(x, ACKLEY_SHIFT)
        y = v - offset
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

# -------------------------------------------------------------------------------------------------
# The budget: counts every evaluation (rule 3) and keeps the best value and solution. A termination
# criterion stops the run at the target, at max_evaluations or at max_seconds, checked after every
# iteration. The clock starts when the budget is created, just before the run (rule 4.1).
# -------------------------------------------------------------------------------------------------

mutable struct Budget
    evaluations::Int
    best::Float64
    solution::Any
    # evaluated solutions outside the bounds (rule 2.4), as the library proposed them
    outside::Int
    # the first evaluation whose value reaches the target, and the clock then (-1: not yet)
    first_hit_evaluations::Int
    first_hit_seconds::Float64
    # the evaluations at the end of the last generation, and that generation's (rule 2.3)
    generation_end::Int
    last_generation::Int
    const max_evaluations::Int
    const max_seconds::Float64
    const target::Float64
    const start::UInt64
end

Budget(max_evaluations, max_seconds, target) =
    Budget(0, Inf, nothing, 0, -1, 0.0, 0, 0, max_evaluations, max_seconds, target, time_ns())

seconds(budget::Budget) = (time_ns() - budget.start) / 1.0e9

# marks the end of a generation
function end_generation!(budget::Budget)
    budget.last_generation = budget.evaluations - budget.generation_end
    budget.generation_end = budget.evaluations
    return
end

# the evaluations since the start of the last generation (rule 2.3): of one cut short, or of the
# last one that ended
last_generation(budget::Budget) =
    budget.evaluations > budget.generation_end ? budget.evaluations - budget.generation_end : budget.last_generation

# optimize's logger, called after the initial population and after every iteration
# (optimize/before.jl and during.jl): it marks the generation's end
generation_logger(budget::Budget) = status -> end_generation!(budget)

outside(x, bounds) = bounds !== nothing && any(v -> v < bounds[1] || v > bounds[2], x)

exhausted(budget::Budget) =
    budget.best <= budget.target ||
    budget.evaluations >= budget.max_evaluations ||
    seconds(budget) >= budget.max_seconds

# the "first_hit" field of a single-objective run
first_hit(budget::Budget) = budget.first_hit_evaluations < 0 ? nothing :
    (evaluations = budget.first_hit_evaluations, time_s = round(budget.first_hit_seconds, digits = 6))

# a user-defined termination criterion, as the library's own (src/termination/budget.jl)
struct BudgetTermination <: Metaheuristics.AbstractTermination
    budget::Budget
end
Metaheuristics.stop_check(status, criterion::BudgetTermination) = exhausted(criterion.budget)

# `bounds`: (lower, upper) of every variable, to count the solutions outside them
function counted(f, budget::Budget; bounds = nothing)
    return function (x)
        budget.evaluations += 1
        outside(x, bounds) && (budget.outside += 1)
        value = f(x)
        if value < budget.best
            budget.best = value
            budget.solution = copy(x)
            if budget.first_hit_evaluations < 0 && value <= budget.target
                budget.first_hit_evaluations = budget.evaluations
                budget.first_hit_seconds = seconds(budget)
            end
        end
        return value
    end
end

# The options (Options docstring, docs/src/api.md) of one attempt: its seed, and what is left of the
# evaluations and of the time. f_calls_limit is also what ECA uses to switch to exploitation at 95%
# of it (eca_solution in src/algorithms/singleobjective/ECA/ECA.jl). The tolerances are the
# defaults (f_tol 1e-12, f_tol_rel eps(), x_tol 1e-8). The iteration limit, only a budget, is
# lifted (rule 2.2).
options(budget::Budget, seed) = Options(
    f_calls_limit = budget.max_evaluations - budget.evaluations,
    time_limit = budget.max_seconds - seconds(budget),
    iterations = typemax(Int) ÷ 4,
    f_tol = 1e-12,
    seed = seed,
)

# The termination: BudgetTermination, and the library's convergence criteria, which end the attempt
# (rule 2.2): the one optimize checks always (default_stop_check in src/termination/default.jl:
# CheckConvergence, all of AbsoluteFunctionConvergence(f_tol), RelativeFunctionConvergence(f_tol_rel),
# SmallStandardDeviation and RelativeParameterConvergence(x_tol)), and the one it adds when the user
# gives no termination (src/optimize/before.jl): the same CheckConvergence for one objective.
function algorithm_kwargs(budget, seed)
    opts = options(budget, seed)
    convergence = Metaheuristics.CheckConvergence(f_tol_abs = opts.f_tol, f_tol_rel = opts.f_tol_rel, x_tol = opts.x_tol)
    return (options = opts, termination = Metaheuristics.Termination(checkany = [BudgetTermination(budget), convergence]))
end

# An attempt that ends before the target, the budget or the cap has converged. The library has no
# restart after convergence: its Restart (docs/src/algorithms/singleobjective.md, "Restart")
# replaces the population every 100 iterations whatever happens, and keeps the base method's stops.
# So the method starts again from a new random start, with the seed (seed + 1) * 1,000,000 +
# restart (rule 2.2); `budget` keeps the best solution and counts every evaluation.
# Returns (the last attempt's status, iterations, restarts).
function run_restarting(start, budget::Budget, seed)
    iterations = 0
    restart = 0
    while true
        status = start(restart == 0 ? seed : (seed + 1) * 1_000_000 + restart)
        iterations += status.iteration
        exhausted(budget) && return status, iterations, restart
        restart += 1
    end
end

# -------------------------------------------------------------------------------------------------
# Solvers: (name, run(budget, seed) -> status, decode(best solution) -> reported solution)
# -------------------------------------------------------------------------------------------------

bits(x) = Int.(x)
zero_based(p) = p .- 1

function onemax_solvers(size, mode)
    # Matched: not run (rule 6.1). The library has no two-point crossover (its crossovers are
    # UniformCrossover, OrderCrossover, SBX and BinomialCrossover), and the matched scenarios use
    # only the library's own operators.
    mode == "matched" && return nothing
    run = function (budget, seed)
        # the guide: "Binary: Use GA with BitFlipMutation". The binary example of the GA
        # docstring (docs/src/algorithms/singleobjective.md, "GA"): GA() with its defaults,
        # population 100, binary tournament, uniform crossover 0.5, BitFlipMutation(1e-5),
        # elitist replacement
        algorithm = GA(; algorithm_kwargs(budget, seed)...)
        return optimize(counted(onemax, budget), BitArraySpace(size), algorithm; logger = generation_logger(budget))
    end
    return [("ga", run, bits)]
end

function nqueens_solvers(size)
    # the guide: "Permutation-based: Use GA with OrderCrossover or BRKGA".
    # docs/src/tutorials/n-queens.md: optimize(attacks, PermutationSpace(N), GA), i.e. the GA's
    # defaults for permutations (get_parameters in src/algorithms/singleobjective/GA/GA.jl):
    # population 100, binary tournament, OrderCrossover, SlightMutation, elitist replacement
    ga = function (budget, seed)
        N = 100
        algorithm = GA(;
            N = N,
            initializer = Metaheuristics.RandomPermutation(; N),
            selection = TournamentSelection(; N),
            crossover = OrderCrossover(),
            mutation = SlightMutation(),
            environmental_selection = ElitistReplacement(),
            algorithm_kwargs(budget, seed)...,
        )
        return optimize(counted(nqueens, budget), PermutationSpace(size), algorithm; logger = generation_logger(budget))
    end
    # the BRKGA docstring (docs/src/algorithms/combinatorial.md, "BRKGA"): random keys in [0, 1]^n
    # decoded by sortperm, with the defaults (20 elites, 10 mutants, 70 offspring, bias 0.7)
    brkga = function (budget, seed)
        bounds = boxconstraints(lb = zeros(size), ub = ones(size))
        return optimize(
            counted(keys -> nqueens(sortperm(keys)), budget), bounds, BRKGA(; algorithm_kwargs(budget, seed)...);
            logger = generation_logger(budget),
        )
    end
    return [("ga", ga, zero_based), ("brkga", brkga, keys -> sortperm(keys) .- 1)]
end

function real_solvers(problem, size)
    f, lower, upper = REAL_PROBLEMS[problem]
    bounds = boxconstraints(lb = fill(lower, size), ub = fill(upper, size))
    # the bounds: the initial population within them, and each method's own repair (ECA and DE:
    # evo_boundary_repairer!, ECA.jl and DE.jl; PSO: reset_to_violated_bounds!, PSO.jl)
    solve(make) = (budget, seed) -> optimize(
        counted(f, budget; bounds = (lower, upper)), bounds, make(algorithm_kwargs(budget, seed));
        logger = generation_logger(budget),
    )
    # the guide: "Box-constrained (continuous): Use ECA, DE, PSO, or SHADE", the first three, with
    # their defaults (their docstrings, docs/src/algorithms/singleobjective.md). ECA is also the
    # default of optimize and the Quick Start's method on Rastrigin (docs/src/index.md); DE and PSO
    # are "Good for multimodal".
    return [
        # ECA: K = 7, population K·D, η_max = 2, p_exploit 0.95, p_bin 0.02
        ("eca", solve(kwargs -> ECA(; kwargs...)), identity),
        # DE/rand/1/bin: population 10·D, F = 0.7, CR = 0.5 (the code's defaults; its docstring
        # says F = 1.0)
        ("de", solve(kwargs -> DE(; kwargs...)), identity),
        # PSO: population 10·D, C1 = C2 = 2, ω = 0.8
        ("pso", solve(kwargs -> PSO(; kwargs...)), identity),
    ]
end

# the untimed warm-up run of every solver before the timed ones (rule 4.2), with the scenario's
# time cap
const WARM_UP_EVALUATIONS = 50_000
const WARM_UP_SEED = 999_999

# rule 5.3: a solver whose first EARLY_SEEDS runs all hit the time cap (a run that took CAPPED of
# it) without reaching the target runs no more seeds
const EARLY_SEEDS = 3
const CAPPED = 0.98

# -------------------------------------------------------------------------------------------------
# Output
# -------------------------------------------------------------------------------------------------

json_value(::Nothing) = "null"
json_value(x::NamedTuple) = "{" * join(("\"$key\":" * json_value(value) for (key, value) in pairs(x)), ",") * "}"
json_value(x::Bool) = string(x)
json_value(x::Integer) = string(x)
json_value(x::Real) = isfinite(x) ? repr(Float64(x)) : (isnan(x) ? "NaN" : (x > 0 ? "Infinity" : "-Infinity"))
json_value(x::AbstractString) = "\"$x\""
json_value(x::AbstractVector) = "[" * join((json_value(v) for v in x), ",") * "]"

function print_line(fields)
    println("{" * join(("\"$key\":" * json_value(value) for (key, value) in fields), ",") * "}")
    return flush(stdout)
end

# `values <problem> <size>`: one JSON solution per line on stdin, its value per
# line on stdout, with the fitness functions above (rule 1.2)
function print_values(problem, size)
    for line in eachline(stdin)
        isempty(strip(line)) && continue
        text = strip(line, ['[', ']', ' ', '\t', '\r'])
        x = [parse(Float64, strip(s)) for s in split(text, ',') if !isempty(strip(s))]
        if problem == "onemax"
            println(json_value(-onemax(x .!= 0)))
        elseif problem == "nqueens"
            println(json_value(nqueens(round.(Int, x) .+ 1)))
        elseif haskey(REAL_PROBLEMS, problem)
            println(json_value(REAL_PROBLEMS[problem][1](x)))
        end
    end
    return
end

function main(args)
    if length(args) == 1 && args[1] == "--version"
        println(pkgversion(Metaheuristics))
        return
    end
    if length(args) == 3 && args[1] == "values"
        size = parse(Int, args[3])
        size <= MAX_GENES || error("at most $MAX_GENES genes")
        print_values(args[2], size)
        return
    end
    if length(args) != 7
        println(stderr, "usage: bench.jl <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>")
        exit(2)
    end
    problem, size, mode = args[1], parse(Int, args[2]), args[3]
    seed_from, seed_to = parse(Int, args[4]), parse(Int, args[5])
    max_evaluations, max_seconds = parse(Int, args[6]), parse(Float64, args[7])
    size <= MAX_GENES || error("at most $MAX_GENES genes")

    if problem == "onemax"
        solvers = onemax_solvers(size, mode)
        solvers === nothing && return  # not run: print nothing
        target = -size  # minimized
    elseif problem == "nqueens"
        solvers = nqueens_solvers(size)
        target = 0
    elseif haskey(REAL_PROBLEMS, problem)
        solvers = real_solvers(problem, size)
        target = REAL_TARGET
    else
        println(stderr, "metaheuristics_jl: unsupported problem $problem")
        return
    end

    for (_, run, _) in solvers
        budget = Budget(WARM_UP_EVALUATIONS, max_seconds, target)
        run_restarting(s -> run(budget, s), budget, WARM_UP_SEED)
    end

    capped = Dict(solver => 0 for (solver, _, _) in solvers)
    for seed in seed_from:seed_to, (solver, run, decode) in solvers
        index = seed - seed_from
        index >= EARLY_SEEDS && capped[solver] == EARLY_SEEDS && continue
        budget = Budget(max_evaluations, max_seconds, target)  # the clock starts
        _, iterations, restarts = run_restarting(s -> run(budget, s), budget, seed)
        elapsed = seconds(budget)
        best = problem == "onemax" ? -Int(budget.best) : problem == "nqueens" ? Int(budget.best) : budget.best
        success = problem == "onemax" ? best >= size : budget.best <= target
        capped[solver] += index < EARLY_SEEDS && !success && elapsed >= CAPPED * max_seconds
        print_line([
            "library" => "metaheuristics_jl", "solver" => solver, "problem" => problem, "size" => size,
            "mode" => mode, "seed" => seed, "time_s" => round(elapsed, digits = 6),
            "generations" => iterations, "evaluations" => budget.evaluations,
            "last_generation" => last_generation(budget), "restarts" => restarts,
            (haskey(REAL_PROBLEMS, problem) ? ["outside" => budget.outside] : [])...,
            "best" => best, "target" => problem == "onemax" ? size : target, "success" => success,
            "first_hit" => first_hit(budget), "solution" => decode(budget.solution),
        ])
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
