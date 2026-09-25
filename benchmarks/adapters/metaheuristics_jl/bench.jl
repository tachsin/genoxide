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
# Every solver is warmed up (compiled) with an untimed run of the same problem, 1,000 evaluations
# and seed 1000, before the timed runs (rule 4.2), so time_s holds the optimization only.

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
# The budget: counts every evaluation (rule 3) and keeps the best value and solution. A termination
# criterion stops the run at the target, at max_evaluations or at max_seconds, checked after every
# iteration.
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
end

Budget(max_evaluations, max_seconds, target = -Inf) =
    Budget(0, Inf, nothing, 0, max_evaluations, max_seconds, target, time())

outside(x, bounds) = bounds !== nothing && any(v -> v < bounds[1] || v > bounds[2], x)

exhausted(budget::Budget) =
    budget.best <= budget.target ||
    budget.evaluations >= budget.max_evaluations ||
    time() - budget.start >= budget.max_seconds

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
        end
        return value
    end
end

# multi-objective: (objectives, inequality constraints, equality constraints), the form the
# multi-objective algorithms take (NSGA2 docstring)
function counted_front(f, m, budget::Budget)
    return function (x)
        budget.evaluations += 1
        outside(x, (0.0, 1.0)) && (budget.outside += 1)
        return f(x, m), [0.0], [0.0]
    end
end

# The options (Options docstring, docs/src/api.md) of one attempt: its seed, and what is left of the
# evaluations and of the time. f_calls_limit is also what ECA uses to switch to exploitation at 95%
# of it (eca_solution in src/algorithms/singleobjective/ECA/ECA.jl). The tolerances are the
# defaults (f_tol 1e-12, f_tol_rel eps(), x_tol 1e-8). The iteration limit, only a budget, is
# lifted (rule 2.2).
options(budget::Budget, seed) = Options(
    f_calls_limit = budget.max_evaluations - budget.evaluations,
    time_limit = budget.max_seconds - (time() - budget.start),
    iterations = typemax(Int) ÷ 4,
    seed = seed,
)

# The termination: BudgetTermination, and the library's convergence criteria, which end the attempt
# (rule 2.2): the one optimize checks always (default_stop_check in src/termination/default.jl:
# CheckConvergence, all of AbsoluteFunctionConvergence(f_tol), RelativeFunctionConvergence(f_tol_rel),
# SmallStandardDeviation and RelativeParameterConvergence(x_tol)), and the one it adds when the user
# gives no termination (src/optimize/before.jl): the same CheckConvergence for one objective,
# RobustConvergence(ftol = f_tol) for several.
function algorithm_kwargs(budget, seed; front = false)
    opts = options(budget, seed)
    convergence = front ? Metaheuristics.RobustConvergence(ftol = opts.f_tol) :
        Metaheuristics.CheckConvergence(f_tol_abs = opts.f_tol, f_tol_rel = opts.f_tol_rel, x_tol = opts.x_tol)
    return (options = opts, termination = Metaheuristics.Termination(checkany = [BudgetTermination(budget), convergence]))
end

# An attempt that ends before the target, the budget or the cap has converged. The library has no
# restart after convergence: its Restart (docs/src/algorithms/singleobjective.md, "Restart")
# replaces the population every 100 iterations whatever happens, and keeps the base method's stops.
# So the method starts again from a new random start, with the seed seed * 1000 + restart (rule
# 2.2); `budget` keeps the best solution and counts every evaluation.
# Returns (the last attempt's status, iterations, restarts).
function run_restarting(start, budget::Budget, seed)
    iterations = 0
    restart = 0
    while true
        status = start(restart == 0 ? seed : seed * 1000 + restart)
        iterations += status.iteration
        exhausted(budget) && return status, iterations, restart
        restart += 1
    end
end

# -------------------------------------------------------------------------------------------------
# Operators for the matched OneMax: the GA framework of Metaheuristics.jl dispatches on operator
# types (docs/src/tutorials/create-metaheuristic.md), these are the missing ones (two-point
# crossover with a probability, and bit-flip on a fraction of the children).
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
# Solvers: (name, run(budget, seed) -> status, decode(best solution) -> reported solution)
# -------------------------------------------------------------------------------------------------

bits(x) = Int.(x)
zero_based(p) = p .- 1

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
            # the guide: "Binary: Use GA with BitFlipMutation". The binary example of the GA
            # docstring (docs/src/algorithms/singleobjective.md, "GA"): GA() with its defaults,
            # population 100, binary tournament, uniform crossover 0.5, BitFlipMutation(1e-5),
            # elitist replacement
            algorithm = GA(; algorithm_kwargs(budget, seed)...)
        end
        return optimize(counted(onemax, budget), BitArraySpace(size), algorithm)
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
        return optimize(counted(nqueens, budget), PermutationSpace(size), algorithm)
    end
    # the BRKGA docstring (docs/src/algorithms/combinatorial.md, "BRKGA"): random keys in [0, 1]^n
    # decoded by sortperm, with the defaults (20 elites, 10 mutants, 70 offspring, bias 0.7)
    brkga = function (budget, seed)
        bounds = boxconstraints(lb = zeros(size), ub = ones(size))
        return optimize(counted(keys -> nqueens(sortperm(keys)), budget), bounds, BRKGA(; algorithm_kwargs(budget, seed)...))
    end
    return [("ga", ga, zero_based), ("brkga", brkga, keys -> sortperm(keys) .- 1)]
end

function real_solvers(problem, size)
    f, lower, upper = REAL_PROBLEMS[problem]
    bounds = boxconstraints(lb = fill(lower, size), ub = fill(upper, size))
    # the bounds: the initial population within them, and each method's own repair (ECA:
    # evo_boundary_repairer!, DE and PSO: reset_to_violated_bounds!)
    solve(make) = (budget, seed) -> optimize(counted(f, budget; bounds = (lower, upper)), bounds, make(algorithm_kwargs(budget, seed)))
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

# The matched settings of every library:
# - NSGA-II, SPEA2, SMS-EMOA: population 100 (92 for DTLZ), SBX η 15 at 0.9, polynomial
#   mutation η 20 at 1 / n.
# - NSGA-III: Das-Dennis directions (12 divisions for 3 objectives, 91; 99 for 2, 100), population
#   92 (100), SBX η 30 at 1, polynomial mutation η 20 at 1 / n.
# - MOEA/D: 100 weights (91 for 3 objectives), 20 neighbours, parents from the neighbourhood at 0.9.
# Differences:
# - the library's p_cr is the probability of crossing each variable, and every pair is crossed
#   (pymoo: pairs at 0.9, variables at 0.5)
# - its NSGA-II and SPEA2 create 2N children per generation, not N (the reproduction of
#   AbstractNSGA in src/algorithms/multiobjective/NSGA2/NSGA2.jl); its SMS-EMOA is steady-state,
#   one child at a time, N per iteration
# - its SMS-EMOA estimates the hypervolume contributions with 3 objectives by Monte Carlo, with its
#   default n_samples = 10,000 samples for every child
# - a converged attempt restarts (rule 2.2), and the front is the last attempt's final population
# Not run (rule 6.1): the library's MOEA/D is MOEAD_DE, whose reproduction is DE/rand/1 with
# polynomial mutation and can't take the matched SBX (MOEAD_DE_reproduction in
# src/algorithms/multiobjective/MOEAD_DE/MOEAD_DE.jl); CCMO is for constrained problems.
function front_solvers(problem, size)
    f, variables, objectives, population, divisions = FRONT_PROBLEMS[problem]
    n = variables(size)
    m = objectives(size)
    # the bounds: the initial population within them, and the library's repair of the offspring
    # (reset_to_violated_bounds! after SBX and polynomial mutation)
    bounds = boxconstraints(lb = zeros(n), ub = ones(n))
    solve(make) = (budget, seed) -> optimize(counted_front(f, m, budget), bounds, make(algorithm_kwargs(budget, seed; front = true)))
    return [
        ("nsga2", solve(kwargs -> NSGA2(; N = population, η_cr = 15, p_cr = 0.9, η_m = 20, p_m = 1.0 / n, kwargs...))),
        ("nsga3", solve(kwargs -> NSGA3(; N = population, η_cr = 30, p_cr = 1.0, η_m = 20, p_m = 1.0 / n, partitions = divisions, kwargs...))),
        ("spea2", solve(kwargs -> SPEA2(; N = population, η_cr = 15, p_cr = 0.9, η_m = 20, p_m = 1.0 / n, kwargs...))),
        ("sms_emoa", solve(kwargs -> SMS_EMOA(; N = population, η_cr = 15, p_cr = 0.9, η_m = 20, p_m = 1.0 / n, kwargs...))),
    ]
end

# the untimed warm-up run of every solver before the timed ones (rule 4.2)
const WARM_UP_EVALUATIONS = 1000
const WARM_UP_SEED = 1000

# rule 5.3: a solver whose first EARLY_SEEDS runs all hit the time cap (a run that took CAPPED of
# it) without reaching the target runs no more seeds
const EARLY_SEEDS = 3
const CAPPED = 0.98

non_dominated(points) = [i for (i, p) in enumerate(points) if !any(q -> all(q .<= p) && any(q .< p), points)]

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
        isempty(strip(line)) && continue
        text = strip(line, ['[', ']', ' ', '\t', '\r'])
        x = [parse(Float64, strip(s)) for s in split(text, ',') if !isempty(strip(s))]
        if problem == "onemax"
            println(json_value(-onemax(x .!= 0)))
        elseif problem == "nqueens"
            println(json_value(nqueens(round.(Int, x) .+ 1)))
        elseif haskey(REAL_PROBLEMS, problem)
            println(json_value(REAL_PROBLEMS[problem][1](x)))
        else
            f, _, objectives, _, _ = FRONT_PROBLEMS[problem]
            println(json_value(f(x, objectives(size))))
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
        solvers = front_solvers(problem, size)
        for (_, run) in solvers
            budget = Budget(WARM_UP_EVALUATIONS, 10.0)
            run_restarting(s -> run(budget, s), budget, WARM_UP_SEED)
        end
        capped = Dict(solver => 0 for (solver, _) in solvers)
        for seed in seed_from:seed_to, (solver, run) in solvers
            index = seed - seed_from
            index >= EARLY_SEEDS && capped[solver] == EARLY_SEEDS && continue
            budget = Budget(max_evaluations, max_seconds)
            start = time_ns()
            status, iterations, restarts = run_restarting(s -> run(budget, s), budget, seed)
            elapsed = (time_ns() - start) / 1.0e9
            capped[solver] += index < EARLY_SEEDS && elapsed >= CAPPED * max_seconds
            # the final population of the last attempt (rule 7.2): the objective values the library
            # evaluated
            points = [Metaheuristics.fval(s) for s in status.population]
            front = non_dominated(points)
            print_line([
                "library" => "metaheuristics_jl", "solver" => solver, "problem" => problem, "size" => size,
                "mode" => mode, "seed" => seed, "time_s" => round(elapsed, digits = 6),
                "generations" => iterations, "evaluations" => budget.evaluations, "restarts" => restarts,
                "outside" => budget.outside, "population" => length(points), "front" => points[front],
                "solutions" => [Metaheuristics.get_position(status.population[i]) for i in front],
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

    for (_, run, _) in solvers
        budget = Budget(WARM_UP_EVALUATIONS, 10.0, target)
        run_restarting(s -> run(budget, s), budget, WARM_UP_SEED)
    end

    capped = Dict(solver => 0 for (solver, _, _) in solvers)
    for seed in seed_from:seed_to, (solver, run, decode) in solvers
        index = seed - seed_from
        index >= EARLY_SEEDS && capped[solver] == EARLY_SEEDS && continue
        budget = Budget(max_evaluations, max_seconds, target)
        start = time_ns()
        _, iterations, restarts = run_restarting(s -> run(budget, s), budget, seed)
        elapsed = (time_ns() - start) / 1.0e9
        best = problem == "onemax" ? -Int(budget.best) : problem == "nqueens" ? Int(budget.best) : budget.best
        success = problem == "onemax" ? best >= size : budget.best <= target
        capped[solver] += index < EARLY_SEEDS && !success && elapsed >= CAPPED * max_seconds
        print_line([
            "library" => "metaheuristics_jl", "solver" => solver, "problem" => problem, "size" => size,
            "mode" => mode, "seed" => seed, "time_s" => round(elapsed, digits = 6),
            "generations" => iterations, "evaluations" => budget.evaluations, "restarts" => restarts,
            (haskey(REAL_PROBLEMS, problem) ? ["outside" => budget.outside] : [])...,
            "best" => best, "target" => problem == "onemax" ? size : target, "success" => success,
            "solution" => decode(budget.solution),
        ])
    end
    return
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
