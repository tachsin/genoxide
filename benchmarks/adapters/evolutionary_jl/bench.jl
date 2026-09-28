# Benchmark adapter for Evolutionary.jl (https://github.com/SciML/Evolutionary.jl, docs:
# https://docs.sciml.ai/Evolutionary/stable/).
#
# Usage: julia --project=<this folder> --threads=1 bench.jl <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
#        julia --project=<this folder> bench.jl values <problem> <size>   # one JSON solution per line on stdin
#        julia --project=<this folder> bench.jl --version
# Prints one JSON line per solver per seed, see ../../README.md for the fields.
#
# The matched suite: one method per problem, set to the definition of docs/benchmarks/rules.md with
# the library's own implementation. Evolutionary.jl runs two of the three:
# - onemax 1000 matched: the GA ("ga");
# - rosenbrock 10 matched: CMA-ES ("cma_es").
# Not rastrigin 30: its DE recombines the mutant with the base vector, not the target. Every other
# problem, size or mode prints nothing. The settings, their sources and the differences from the
# definitions are on docs/benchmarks/libraries/evolutionary_jl.md.
#
# Every solver is warmed up (compiled) with an untimed, unprinted run of the same problem, with
# the seed 999,999, 50,000 evaluations and the scenario's time cap, before the timed runs (rule
# 4.2), so time_s holds the optimization only.

using Evolutionary
using LinearAlgebra
using Random
using Logging

BLAS.set_num_threads(1)

# CMAES ends the run when its covariance matrix can't be decomposed, and logs the whole matrix with
# @error: keep stderr readable (the run prints "ended_by", see run_once)
disable_logging(Logging.Error)

# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to problems.py. Evolutionary.jl minimizes.
# -------------------------------------------------------------------------------------------------

# maximize the number of ones: minimize its negative
onemax(x::AbstractVector{Bool}) = -count(x)

function rosenbrock(x::AbstractVector{<:Real})
    s = 0.0
    for i in 1:(length(x) - 1)
        s += 100.0 * (x[i + 1] - x[i]^2)^2 + (1.0 - x[i])^2
    end
    return s
end

const ROSENBROCK_LOWER = -5.0
const ROSENBROCK_UPPER = 10.0
const REAL_TARGET = 0.01

# -------------------------------------------------------------------------------------------------
# The budget: counts every evaluation (rule 3) and keeps the best value and solution; stops at the
# target, at max_evaluations or at max_seconds (checked by the callback after every generation).
# The clock starts when the budget is created, just before the run (rule 4.1).
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

outside(x, bounds) = bounds !== nothing && any(v -> v < bounds[1] || v > bounds[2], x)

# `bounds`: (lower, upper) of every variable, to count the solutions outside them. Counts every
# call, including the one EvolutionaryObjective makes before each attempt to learn the value's
# type (`zero(f(x))`, src/api/objective.jl).
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

exhausted(budget::Budget) =
    budget.best <= budget.target ||
    budget.evaluations >= budget.max_evaluations ||
    seconds(budget) >= budget.max_seconds

# the "first_hit" field of a single-objective run
first_hit(budget::Budget) = budget.first_hit_evaluations < 0 ? nothing :
    (evaluations = budget.first_hit_evaluations, time_s = round(budget.first_hit_seconds, digits = 6))

# The options of a run (docs/src/tutorial.md, "General options"). The matched methods have no
# convergence criterion (rule 2.2): `iterations = typemax(Int)` lifts the iteration limit (the
# library's default is 1,000, 1,500 for CMAES), and `successive_f_tol = typemax(Int)` turns off
# the convergence test, which ends a run once the method's metric (AbsDiff(1e-12) for GA and
# CMAES) has held for more than `successive_f_tol` generations (optimize, src/api/optimize.jl).
# The callback ends the run at the target, the budget or the time cap.
options(budget::Budget, rng) = Evolutionary.Options(
    iterations = typemax(Int),
    successive_f_tol = typemax(Int),
    # called after the initial population and after every generation (optimize, api/optimize.jl),
    # where it marks the generation's end too
    callback = record -> (end_generation!(budget); exhausted(budget)),
    rng = rng,
)

# One run: `start(rng)` with the run's seed in the library's generator (Options' `rng`, used by the
# operators and the initial population). It ends at the target, the budget or the cap, or when the
# library ends it itself: CMAES when its covariance matrix breaks down (update_state! catches the
# failed eigendecomposition and returns true, src/cmaes.jl), on Rosenbrock 10 once σ underflows,
# after about 9,700 generations (its step-size bug, see the page's "Bugs found"). That isn't worked
# around (rule 8.4) and the matched methods have no restarts: the run ends there and says why in
# "ended_by". Both methods evaluate every child, changed or not, so they never stall (rule 2.2).
# Returns (generations, ended_by).
function run_once(start, budget::Budget, seed, ended_by)
    result = start(Xoshiro(seed))
    return Evolutionary.iterations(result), exhausted(budget) ? nothing : ended_by
end

# -------------------------------------------------------------------------------------------------
# OneMax 1000, matched: the GA ("ga")
# -------------------------------------------------------------------------------------------------

# DEAP's eaSimple with the library's own GA and operators: population 300, tournament(3), two-point
# crossover (TPX) with crossoverRate 0.5, `flip` with mutationRate 0.2, generational without elitism
# (ɛ = 0), random initial bits. Differences: `flip` flips exactly one random bit of a mutated child
# (the definition: each bit with probability 1 / size, the same mean); `tournament` draws its
# contestants from a shuffled population, without replacement until it's used up; `TPX` swaps the
# genes from one uniform position to another, both included; every child is evaluated, changed or
# not. A pair that isn't crossed passes the parents themselves to the offspring, not copies (the
# library's bug, see the page), as users get it.
function onemax_ga(size)
    method = () -> GA(
        populationSize = 300,
        selection = tournament(3),
        crossover = TPX,
        crossoverRate = 0.5,
        mutation = flip,
        mutationRate = 0.2,
        ɛ = 0,
    )
    return (budget, seed) -> run_once(budget, seed, "optimize returned") do rng
        Evolutionary.optimize(counted(onemax, budget), () -> bitrand(rng, size), method(), options(budget, rng))
    end
end

# -------------------------------------------------------------------------------------------------
# Rosenbrock 10, matched: CMA-ES ("cma_es")
# -------------------------------------------------------------------------------------------------

# CMAES (src/cmaes.jl, docs/src/cmaes.md) with Hansen's defaults: λ = 4 + ⌊3 ln n⌋ = 10, μ = 5,
# positive weights ∝ ln((λ + 1) / 2) - ln i summing to 1 (μ_eff ≈ 3.17) and zero for the other
# five: the library's default weights are active (negative for the worst five) and can only be
# turned off by passing `weights`. Given weights, the library would take other learning rates
# (c_c = c_σ = 1 / √n, c_μ = μ_eff / n², c_1 = 2 / n²), so c_1, c_c, c_mu and c_sigma are passed with
# the formulas of its own defaults for them (initial_state, src/cmaes.jl, Hansen's 2016 tutorial);
# d_σ is the library's (Hansen's). σ0 = 0.3 (upper - lower) = 4.5, C0 = I. BoxConstraints: the
# mean starts at a uniform random point of the box (the first of the initial population,
# src/api/utilities.jl), and every sample is clipped to the box before it's evaluated (apply!,
# src/api/constraints.jl). The library's bugs in update_state! (the page, "Bugs found": the
# covariance matrix overwritten by eigen!, the step size and the rank-μ update) aren't worked around.
function rosenbrock_cma_es(size)
    lower, upper = ROSENBROCK_LOWER, ROSENBROCK_UPPER
    n = size
    λ = 4 + floor(Int, 3 * log(n))
    μ = λ ÷ 2
    w = [log((λ + 1) / 2) - log(i) for i in 1:μ]
    weights = [w ./ sum(w); zeros(λ - μ)]
    μ_eff = 1 / sum(abs2, weights)
    c_1 = 2 / ((n + 1.3)^2 + μ_eff)
    method = () -> CMAES(
        mu = μ,
        lambda = λ,
        weights = weights,
        c_1 = c_1,
        c_c = (4 + μ_eff / n) / (n + 4 + 2 * μ_eff / n),
        c_mu = min(1 - c_1, 2 * (μ_eff - 2 + 1 / μ_eff) / ((n + 2)^2 + μ_eff)),
        c_sigma = (μ_eff + 2) / (n + μ_eff + 5),
        sigma0 = 0.3 * (upper - lower),
    )
    return (budget, seed) -> run_once(budget, seed, "eigendecomposition failed") do rng
        Evolutionary.optimize(
            counted(rosenbrock, budget; bounds = (lower, upper)), BoxConstraints(lower, upper, n), method(),
            options(budget, rng),
        )
    end
end

# (problem, size) => (solver, run(budget, seed) -> (generations, ended_by)), all matched
const SCENARIOS = Dict(
    ("onemax", 1000) => size -> ("ga", onemax_ga(size)),
    ("rosenbrock", 10) => size -> ("cma_es", rosenbrock_cma_es(size)),
)

# the untimed warm-up run of every solver before the timed ones (rule 4.2), with the scenario's
# time cap
const WARM_UP_EVALUATIONS = 50_000
const WARM_UP_SEED = 999_999

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
        text = strip(line, ['[', ']', ' ', '\t', '\r'])
        isempty(strip(line)) && continue
        x = [parse(Float64, strip(s)) for s in split(text, ',') if !isempty(strip(s))]
        if problem == "onemax"
            println(json_value(-onemax(x .!= 0)))
        elseif problem == "rosenbrock"
            println(json_value(rosenbrock(x)))
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

    if mode != "matched" || !haskey(SCENARIOS, (problem, size))
        # not in the suite: print nothing
        println(stderr, "evolutionary_jl: not run: $problem $size $mode")
        return
    end
    solver, run = SCENARIOS[(problem, size)](size)
    target = problem == "onemax" ? -size : REAL_TARGET  # minimized

    run(Budget(WARM_UP_EVALUATIONS, max_seconds, target), WARM_UP_SEED)

    for seed in seed_from:seed_to
        Random.seed!(seed)
        budget = Budget(max_evaluations, max_seconds, target)  # the clock starts
        outcome = run(budget, seed)
        # the clock stops as the run returns: `run`'s result type isn't known here, so Julia
        # compiles its destructuring on the first timed run (the warm-up doesn't use the result)
        elapsed = seconds(budget)
        generations, ended_by = outcome
        if problem == "onemax"
            best, solution = -Int(budget.best), Int.(budget.solution)
        else
            best, solution = budget.best, budget.solution
        end
        success = problem == "onemax" ? best >= size : budget.best <= target
        print_line([
            "library" => "evolutionary_jl", "solver" => solver, "problem" => problem, "size" => size,
            "mode" => mode, "seed" => seed, "time_s" => round(elapsed, digits = 6),
            "generations" => generations, "evaluations" => budget.evaluations,
            "last_generation" => last_generation(budget),
            (problem == "onemax" ? [] : ["outside" => budget.outside])...,
            "best" => best, "target" => problem == "onemax" ? size : target, "success" => success,
            "first_hit" => first_hit(budget), (ended_by === nothing ? [] : ["ended_by" => ended_by])...,
            "solution" => solution,
        ])
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
