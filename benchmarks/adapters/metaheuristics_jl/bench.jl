# Benchmark adapter for Metaheuristics.jl (https://github.com/jmejia8/Metaheuristics.jl).
#
# Usage: julia --project=<this folder> --threads=1 bench.jl <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
#        julia --project=<this folder> bench.jl --version
# Prints one JSON line per solver per seed, see ../../README.md for the fields.
#
# Every solver is warmed up (compiled) with a tiny untimed run of the same problem before the
# timed runs, so time_s holds the optimization only.

using Metaheuristics
using LinearAlgebra
using Random

BLAS.set_num_threads(1)

# -------------------------------------------------------------------------------------------------
# Fitness functions, identical to the ones in the other adapters. Metaheuristics.jl minimizes.
# -------------------------------------------------------------------------------------------------

onemax(x::AbstractVector{Bool}) = -count(x)

# Number of diagonal conflicts, O(n) (DEAP examples/ga/nqueens.py), 1-based: queen i in column p[i]
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

zdt_g(x) = 1.0 + 9.0 * sum(@view x[2:end]) / (length(x) - 1)

function zdt1(x, m)
    g = zdt_g(x)
    return [x[1], g * (1.0 - sqrt(x[1] / g))]
end

function zdt2(x, m)
    g = zdt_g(x)
    return [x[1], g * (1.0 - (x[1] / g)^2)]
end

function zdt3(x, m)
    g = zdt_g(x)
    return [x[1], g * (1.0 - sqrt(x[1] / g) - x[1] / g * sin(10π * x[1]))]
end

# DTLZ2 with m objectives and k = length(x) - m + 1 = 10
function dtlz2(x, m)
    g = 0.0
    for v in @view x[m:end]
        g += (v - 0.5)^2
    end
    F = zeros(m)
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

# DTLZ1 with m objectives and k = length(x) - m + 1 = 5
function dtlz1(x, m)
    k = length(x) - m + 1
    s = 0.0
    for v in @view x[m:end]
        s += (v - 0.5)^2 - cos(20π * (v - 0.5))
    end
    g = 100.0 * (k + s)
    F = zeros(m)
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

# (function, variables for the size, objectives for the size, population size, Das-Dennis divisions)
const FRONT_PROBLEMS = Dict(
    "zdt1" => (zdt1, size -> size, size -> 2, 100, 99),
    "zdt2" => (zdt2, size -> size, size -> 2, 100, 99),
    "zdt3" => (zdt3, size -> size, size -> 2, 100, 99),
    # size: the number of objectives, with k = 10 (DTLZ2) and k = 5 (DTLZ1)
    "dtlz2" => (dtlz2, size -> size + 9, size -> size, 92, 12),
    "dtlz1" => (dtlz1, size -> size + 4, size -> size, 92, 12),
)

# -------------------------------------------------------------------------------------------------
# The budget: counts the evaluations and keeps the best value. A Termination criterion stops the
# run at the target, at max_evaluations or at max_seconds, checked after every iteration.
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

exhausted(budget::Budget) =
    budget.best <= budget.target ||
    budget.evaluations >= budget.max_evaluations ||
    time() - budget.start >= budget.max_seconds

struct BudgetTermination <: Metaheuristics.AbstractTermination
    budget::Budget
end
Metaheuristics.stop_check(status, criterion::BudgetTermination) = exhausted(criterion.budget)

function counted(f, budget::Budget)
    return function (x)
        budget.evaluations += 1
        value = f(x)
        value < budget.best && (budget.best = value)
        return value
    end
end

# multi-objective: (objectives, inequality constraints, equality constraints)
function counted_front(f, m, budget::Budget)
    return function (x)
        budget.evaluations += 1
        return f(x, m), [0.0], [0.0]
    end
end

# The options: the seed, and no library stop other than ours. The evaluation and time limits are
# the library's too (checked after every iteration, MOEA/D after every child); f_tol = -1 turns off
# its convergence stops (a collapsed population for single objectives, RobustConvergence for
# multiple objectives), as in the other adapters.
options(budget::Budget, seed) = Options(
    f_calls_limit = budget.max_evaluations,
    time_limit = budget.max_seconds,
    iterations = typemax(Int) ÷ 4,
    f_tol = -1.0,
    seed = seed,
)

algorithm_kwargs(budget, seed) = (options = options(budget, seed), termination = BudgetTermination(budget))

# -------------------------------------------------------------------------------------------------
# Operators for the matched OneMax: the GA framework of Metaheuristics.jl dispatches on operator
# types, these are the missing ones (two-point crossover with a probability, and bit-flip on a
# fraction of the children).
# -------------------------------------------------------------------------------------------------

struct TwoPointCrossover
    p::Float64
end

# the parents come as population[parent_mask]: pairs (i, i + n) as in UniformCrossover
function Metaheuristics.crossover(population, parameters::TwoPointCrossover)
    n = length(population) ÷ 2
    a = Metaheuristics.positions(population[1:n])
    b = Metaheuristics.positions(population[(n + 1):(2n)])
    d = size(a, 2)
    for i in 1:n
        rand() < parameters.p || continue
        # DEAP's cxTwoPoint
        from = rand(1:d)
        to = rand(1:(d - 1))
        to >= from ? (to += 1) : ((from, to) = (to, from))
        for j in from:(to - 1)
            a[i, j], b[i, j] = b[i, j], a[i, j]
        end
    end
    return [a; b]
end

struct BitFlipSomeChildren
    child_probability::Float64
    gene_probability::Float64
end

function Metaheuristics.mutation!(Q::AbstractMatrix{Bool}, parameters::BitFlipSomeChildren)
    for i in axes(Q, 1)
        rand() < parameters.child_probability || continue
        for j in axes(Q, 2)
            rand() < parameters.gene_probability && (Q[i, j] = !Q[i, j])
        end
    end
    return Q
end

# -------------------------------------------------------------------------------------------------
# Solvers: (name, run(budget, seed) -> status) per problem
# -------------------------------------------------------------------------------------------------

function onemax_solvers(size, mode)
    run = function (budget, seed)
        if mode == "matched"
            # as DEAP eaSimple: population 300, tournament 3 (with replacement, as DEAP), two-point
            # crossover with probability 0.5, bit-flip with probability 1 / size on 20% of the
            # children, generational replacement (no elitism). Metaheuristics' GA recombines
            # every pair (i, i + N/2) of the selected parents instead of neighbours (1, 2).
            algorithm = GA(;
                N = 300,
                selection = TournamentSelection(K = 3, N = 300),
                crossover = TwoPointCrossover(0.5),
                mutation = BitFlipSomeChildren(0.2, 1.0 / size),
                environmental_selection = GenerationalReplacement(),
                algorithm_kwargs(budget, seed)...,
            )
        else
            # the binary example of the GA docstring: GA() with its defaults (population 100,
            # binary tournament, uniform crossover 0.5, bit-flip 1e-5, elitist replacement)
            algorithm = GA(; algorithm_kwargs(budget, seed)...)
        end
        return optimize(counted(onemax, budget), BitArraySpace(size), algorithm)
    end
    return [("ga", run)]
end

function nqueens_solvers(size)
    # docs/src/tutorials/n-queens.md: optimize(attacks, PermutationSpace(N), GA), i.e. the defaults
    # for permutations (population 100, binary tournament, order crossover, slight mutation,
    # elitist replacement)
    run = function (budget, seed)
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
        return optimize(counted(nqueens, budget), PermutationSpace(size), algorithm)
    end
    return [("ga", run)]
end

function real_solvers(problem, size)
    f, lower, upper = REAL_PROBLEMS[problem]
    bounds = boxconstraints(lb = fill(lower, size), ub = fill(upper, size))
    solve(make) = (budget, seed) -> optimize(counted(f, budget), bounds, make(algorithm_kwargs(budget, seed)))
    return [
        # the library's defaults, as in docs/src/examples.md (ECA() on Rastrigin) and the
        # docstrings: ECA (K = 7, N = K·D), DE/rand/1/bin (N = 10·D, F = 0.7, CR = 0.5),
        # PSO (N = 10·D, C1 = C2 = 2, ω = 0.8), GA (population 100, SBX and polynomial mutation)
        ("eca", solve(kwargs -> ECA(; kwargs...))),
        ("de", solve(kwargs -> DE(; kwargs...))),
        ("pso", solve(kwargs -> PSO(; kwargs...))),
        ("ga", solve(kwargs -> GA(;
            N = 100,
            initializer = Metaheuristics.RandomInBounds(; N = 100),
            selection = TournamentSelection(; N = 100),
            crossover = Metaheuristics.SBX(; bounds),
            mutation = PolynomialMutation(; bounds),
            environmental_selection = ElitistReplacement(),
            kwargs...,
        ))),
    ]
end

# The matched settings of every library:
# - NSGA-II, SPEA2, SMS-EMOA: population 100 (92 for DTLZ), SBX η 15 at 0.9, polynomial
#   mutation η 20 at 1 / n.
# - NSGA-III: Das-Dennis directions (12 divisions for 3 objectives, 91; 99 for 2, 100), population
#   92 (100), SBX η 30 at 1, polynomial mutation η 20 at 1 / n.
# - MOEA/D: 100 weights (91 for 3 objectives), 20 neighbours, parents from the neighbourhood at 0.9.
# Differences:
# - the library's p_cr is the probability of crossing each variable, and every pair is crossed
#   (pymoo: pairs at 0.9, variables at 0.5)
# - its NSGA-II and SPEA2 create 2N children per generation, not N
# - its MOEA/D is MOEA/D-DE: DE/rand/1 (F 0.5, CR 1) with polynomial mutation, not SBX, at most
#   2 replacements per child, and Tchebycheff also for DTLZ (no PBI)
function front_solvers(problem, size)
    f, variables, objectives, population, divisions = FRONT_PROBLEMS[problem]
    n = variables(size)
    m = objectives(size)
    bounds = boxconstraints(lb = zeros(n), ub = ones(n))
    solve(make) = (budget, seed) -> optimize(counted_front(f, m, budget), bounds, make(algorithm_kwargs(budget, seed)))
    solvers = [
        ("nsga2", solve(kwargs -> NSGA2(; N = population, η_cr = 15, p_cr = 0.9, η_m = 20, p_m = 1.0 / n, kwargs...))),
        ("nsga3", solve(kwargs -> NSGA3(; N = population, η_cr = 30, p_cr = 1.0, η_m = 20, p_m = 1.0 / n, partitions = divisions, kwargs...))),
        ("spea2", solve(kwargs -> SPEA2(; N = population, η_cr = 15, p_cr = 0.9, η_m = 20, p_m = 1.0 / n, kwargs...))),
        ("moead", solve(kwargs -> MOEAD_DE(gen_ref_dirs(m, divisions); T = 20, δ = 0.9, η = 20, p_m = 1.0 / n, kwargs...))),
        ("sms_emoa", solve(kwargs -> SMS_EMOA(; N = population, η_cr = 15, p_cr = 0.9, η_m = 20, p_m = 1.0 / n, kwargs...))),
    ]
    return solvers
end

# the untimed warm-up run of every solver before the timed ones: a few generations, which compile
# everything the timed runs call
const WARM_UP_EVALUATIONS = 1000

non_dominated(points) = [p for p in points if !any(q -> all(q .<= p) && any(q .< p), points)]

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
        println(pkgversion(Metaheuristics))
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
        solvers = front_solvers(problem, size)
        # warm-up: compile every solver with a tiny untimed run
        for (_, run) in solvers
            run(Budget(WARM_UP_EVALUATIONS, 10.0), 0)
        end
        for seed in seed_from:seed_to, (solver, run) in solvers
            budget = Budget(max_evaluations, max_seconds)
            start = time_ns()
            status = run(budget, seed)
            elapsed = (time_ns() - start) / 1.0e9
            front = non_dominated([Metaheuristics.fval(s) for s in status.population])
            print_line([
                "library" => "metaheuristics_jl", "solver" => solver, "problem" => problem, "size" => size,
                "mode" => mode, "seed" => seed, "time_s" => round(elapsed, digits = 6),
                "generations" => status.iteration, "evaluations" => budget.evaluations, "front" => front,
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
        println(stderr, "metaheuristics_jl: unsupported problem $problem")
        return
    end

    # warm-up: compile every solver with a tiny untimed run of the same problem
    for (_, run) in solvers
        run(Budget(WARM_UP_EVALUATIONS, 10.0, target), 0)
    end

    for seed in seed_from:seed_to, (solver, run) in solvers
        budget = Budget(max_evaluations, max_seconds, target)
        start = time_ns()
        status = run(budget, seed)
        elapsed = (time_ns() - start) / 1.0e9
        best = problem == "onemax" ? -Int(budget.best) : problem == "nqueens" ? Int(budget.best) : budget.best
        success = problem == "onemax" ? best >= size : budget.best <= target
        print_line([
            "library" => "metaheuristics_jl", "solver" => solver, "problem" => problem, "size" => size,
            "mode" => mode, "seed" => seed, "time_s" => round(elapsed, digits = 6),
            "generations" => status.iteration, "evaluations" => budget.evaluations,
            "best" => best, "target" => problem == "onemax" ? size : target, "success" => success,
        ])
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
