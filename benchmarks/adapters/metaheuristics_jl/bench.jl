# Benchmark adapter for Metaheuristics.jl (https://github.com/jmejia8/Metaheuristics.jl, docs:
# https://jmejia8.github.io/Metaheuristics.jl/stable/).
#
# Usage: julia --project=<this folder> --threads=1 bench.jl <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
#        julia --project=<this folder> bench.jl values <problem> <size>   # one JSON solution per line on stdin
#        julia --project=<this folder> bench.jl --version
# Prints one JSON line per solver per seed, see ../../README.md for the fields.
#
# The matched suite: one method per problem, set to the definition of docs/benchmarks/rules.md with
# the library's own implementation. Metaheuristics.jl runs one of the three:
# - rastrigin 30 matched: DE/rand/1/bin ("de").
# Not onemax 1000 (no two-point crossover) nor rosenbrock 10 (no CMA-ES). Every other problem,
# size or mode prints nothing. The settings, their sources and the differences from the definition
# are on docs/benchmarks/libraries/metaheuristics_jl.md.
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

# The shift of Rastrigin, so that the optimum isn't at the origin:
# s_i = 0.8 upper (2 ((37 i + 11) mod 101) / 101 - 1) for the 0-based gene index i, with `upper`
# the box's upper bound, computed in this order (problems.py). Computed once, before any run, for up
# to MAX_GENES genes.
shift(index::Integer, upper) = 0.8 * upper * (2 * ((37 * (index - 1) + 11) % 101) / 101 - 1)
const MAX_GENES = 1024
const RASTRIGIN_SHIFT = [shift(index, 5.12) for index in 1:MAX_GENES]

function rastrigin(x::AbstractVector{<:Real})
    s = 10.0 * length(x)
    for (v, offset) in zip(x, RASTRIGIN_SHIFT)
        y = v - offset
        s += y * y - 10.0 * cos(2π * y)
    end
    return s
end

const RASTRIGIN_LOWER = -5.12
const RASTRIGIN_UPPER = 5.12
# Rastrigin 30 has no target: a run uses the whole budget (or stops at the time cap), measured by its
# time and its error at the end (the best value; the optimum is 0). The budget's target is -Inf,
# never reached, so no run stops early and first_hit stays null; the run prints "target": null.
const NO_TARGET = -Inf

# -------------------------------------------------------------------------------------------------
# The budget: counts every evaluation (rule 3) and keeps the best value and solution. A termination
# criterion stops the run at max_evaluations or at max_seconds (or at a target, which Rastrigin 30
# doesn't have), checked after every iteration. The clock starts when the budget is created, just
# before the run (rule 4.1).
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

# The options (Options docstring, docs/src/api.md) and termination of a run, with no convergence
# criterion (rule 2.2):
# - its seed (optimize seeds Julia's global generator with it, src/optimize/before.jl, which the
#   operators use), the budget and the time cap; the iteration limit, only a budget, is lifted;
# - optimize always checks CheckConvergence (default_stop_check, src/termination/default.jl), which
#   needs all of AbsoluteFunctionConvergence(f_tol), RelativeFunctionConvergence(f_tol_rel),
#   SmallStandardDeviation and RelativeParameterConvergence(x_tol), each a spread <= its
#   tolerance (src/termination/convergence.jl): negative tolerances turn it off;
# - a user termination replaces the CheckConvergence optimize adds without one
#   (src/optimize/before.jl): BudgetTermination, which ends the run at the budget or the time cap.
# DE evaluates every trial, so it never stalls (rule 2.2), and nothing else ends a run: no restarts.
function algorithm_kwargs(budget, seed)
    options = Options(
        f_calls_limit = budget.max_evaluations,
        time_limit = budget.max_seconds,
        iterations = typemax(Int) ÷ 4,
        f_tol = -1.0,
        f_tol_rel = -1.0,
        x_tol = -1.0,
        seed = seed,
    )
    return (options = options, termination = Metaheuristics.Termination(checkany = [BudgetTermination(budget)]))
end

# -------------------------------------------------------------------------------------------------
# Rastrigin 30, matched: DE/rand/1/bin ("de")
# -------------------------------------------------------------------------------------------------

# DE (src/algorithms/singleobjective/DE/DE.jl, docs/src/algorithms/singleobjective.md): N = 100,
# F = 0.5 and CR = 0.9 fixed (F_min = F_max = F, CR_min = CR_max = CR: no dither), strategy :rand1:
# v = x_r1 + F (x_r2 - x_r3) (DE_mutation, src/operators/mutation/mutation.jl), binomial crossover
# with the target and a forced index j_rand (DE_crossover, src/operators/crossover/uniform.jl),
# generational: all trials are built from the last population, then each replaces its target if
# better (environmental_selection, DE.jl). The initial population is uniform in the box. Bounds:
# a trial gene outside the box is redrawn between the bound it crossed and the best solution's gene
# (evo_boundary_repairer!, src/common/repair.jl). Differences: r1, r2, r3 are distinct but may
# include the target (DE_mutation isn't given its index); a trial replaces its target only if
# strictly better (the definition: also on a tie); the repair.
function rastrigin_de(size)
    lower, upper = RASTRIGIN_LOWER, RASTRIGIN_UPPER
    bounds = boxconstraints(lb = fill(lower, size), ub = fill(upper, size))
    return (budget, seed) -> optimize(
        counted(rastrigin, budget; bounds = (lower, upper)), bounds,
        DE(; N = 100, F = 0.5, CR = 0.9, strategy = :rand1, algorithm_kwargs(budget, seed)...);
        logger = generation_logger(budget),
    )
end

# (problem, size) => (solver, run(budget, seed) -> status), all matched
const SCENARIOS = Dict(
    ("rastrigin", 30) => size -> ("de", rastrigin_de(size)),
)

# the untimed warm-up run of every solver before the timed ones (rule 4.2), with the scenario's
# time cap
const WARM_UP_EVALUATIONS = 50_000
const WARM_UP_SEED = 999_999

# rule 5.3: a solver whose first EARLY_SEEDS runs all hit the time cap (a run that took CAPPED of
# it) runs no more seeds
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
        problem == "rastrigin" && println(json_value(rastrigin(x)))
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

    if mode != "matched" || !haskey(SCENARIOS, (problem, size))
        # not in the suite: print nothing
        println(stderr, "metaheuristics_jl: not run: $problem $size $mode")
        return
    end
    solver, run = SCENARIOS[(problem, size)](size)
    target = NO_TARGET

    run(Budget(WARM_UP_EVALUATIONS, max_seconds, target), WARM_UP_SEED)

    capped = 0
    for seed in seed_from:seed_to
        index = seed - seed_from
        index >= EARLY_SEEDS && capped == EARLY_SEEDS && break
        budget = Budget(max_evaluations, max_seconds, target)  # the clock starts
        status = run(budget, seed)
        # the clock stops as the run returns, before anything is read from its result (whose type
        # isn't known here)
        elapsed = seconds(budget)
        # the iterations, the initial population's included
        iterations = status.iteration
        success = false  # no target
        capped += index < EARLY_SEEDS && elapsed >= CAPPED * max_seconds
        print_line([
            "library" => "metaheuristics_jl", "solver" => solver, "problem" => problem, "size" => size,
            "mode" => mode, "seed" => seed, "time_s" => round(elapsed, digits = 6),
            "generations" => iterations, "evaluations" => budget.evaluations,
            "last_generation" => last_generation(budget), "outside" => budget.outside,
            "best" => budget.best, "target" => nothing, "success" => success,
            "first_hit" => nothing, "solution" => budget.solution,
        ])
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
