// Benchmark adapter for openGA (https://github.com/Arash-codedev/openGA), a header-only C++ GA,
// pinned at commit f9b15e7 (build.sh). Its page, with every method, setting and where it comes
// from: docs/benchmarks/libraries/openga.md.
//
// Usage: ga_bench_openga <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
//        ga_bench_openga values <problem> <size>    (one JSON array per line on stdin)
//        ga_bench_openga --version
// Prints one JSON line per solver per seed, see ../../README.md for the fields.
//
// openGA ships no operators: its users write init_genes, eval_solution, crossover (one child per
// call) and mutate, and set the population and rates (openGA.pdf, "User side code", p. 2). So
// "idiomatic" here means the operators and settings of openGA's own examples and of its code
// generator, openGA assist (assist/main.js), which the README recommends for starting a program.
// Header lines below are of src/openGA.hpp at the pinned commit.
//
// The matched scenarios (OneMax matched, the multi-objective ones) aren't run: a matched scenario
// uses only the library's own operators, and openGA has none (no two-point crossover, bit flip,
// SBX or polynomial mutation). The adapter prints nothing for them; `values` still evaluates them.
//
// openGA's loop (EA::Genetic, SOGA mode), which the adapter can't change:
// - every generation keeps the whole population (transfer, L582-599) and adds
//   round(population * crossover_fraction) children (L1669); each child comes from two distinct
//   parents chosen by a rank-based roulette (chance 1 / sqrt(rank + 1); L1132-1147, L1586-1594,
//   L1613-1616), the user's crossover and, with probability mutation_rate, the user's mutation
//   (L1622-1628);
// - SOGA survival (L1021-1061): the elite_count best of parents + children, then a rank-based
//   roulette for the rest. The roulette is normalized by the cumulative chance at index
//   population - 1 (L1145), so it can only pick the previous population (indices < population):
//   a new child survives only through an elite slot. That is openGA's bug
//   https://github.com/Arash-codedev/openGA/issues/30, which the adapter doesn't work around.
// All randomness goes through the library's own std::mt19937_64 (also passed to the operators as
// rnd01), which the adapter seeds per attempt (attempt_seed). The library seeds it from the clock
// (L371-374) and has no setter, so the adapter reaches the private generator with the standard
// explicit-instantiation access idiom. multi_threading is off: the population is created and
// evaluated sequentially (L1557-1561, L1678-1682), on one thread.
//
// The adapter drives the generations itself (solve_init, then solve_next_generation, the two
// halves of solve(), L402-503) and stops after the generation in which the target is reached or
// the evaluation budget or the time limit is used up (run_ga). openGA's stop criteria (L1705-1740),
// rule 2.2: generation_max is only a budget, so it's lifted; the best and average stalls detect
// convergence, so they're kept as each method documents them and end that attempt, and the method
// starts again from a new random start (openGA has no restart mechanism), keeping the best solution
// and counting every evaluation. Attempt 0 uses the run's seed, restart r (seed + 1) * 1000000 + r.
//
// The fitness records the first evaluation whose value reaches the target (`first_hit`). The best
// solution of an attempt is the best of openGA's last generation (the elite keeps it).
//
// Bounds (rule 2.4): the examples' mutation draws the child again while a gene is out of range,
// and their crossover mixes the parents, which stays between them. The adapter counts the
// solutions evaluated outside the bounds (`outside`).

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "openGA.hpp"

#ifndef OPENGA_VERSION
#define OPENGA_VERSION "unknown"
#endif

namespace {

constexpr double PI = 3.14159265358979323846;
constexpr double E = 2.71828182845904523536;
constexpr double REAL_TARGET = 0.01;

// -------------------------------------------------------------------------------------------------
// Genes and costs
// -------------------------------------------------------------------------------------------------

struct Bits {
    std::vector<std::uint8_t> x;  // genes of 0/1
};

struct Permutation {
    std::vector<int> p;  // queen i in row i, column p[i]
};

struct Reals {
    std::vector<double> x;
};

struct Cost {
    double cost;  // minimized
};

using BitsGA = EA::Genetic<Bits, Cost>;
using PermutationGA = EA::Genetic<Permutation, Cost>;
using RealGA = EA::Genetic<Reals, Cost>;

std::vector<double> numbers(const Bits &b) { return {b.x.begin(), b.x.end()}; }
std::vector<double> numbers(const Permutation &q) { return {q.p.begin(), q.p.end()}; }
std::vector<double> numbers(const Reals &r) { return r.x; }

// -------------------------------------------------------------------------------------------------
// Seeding: openGA seeds its private std::mt19937_64 from the clock and has no setter. The explicit
// instantiation of a template may name a private member, which gives a pointer to it.
// -------------------------------------------------------------------------------------------------

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnon-template-friend"  // intended: one function per RngTag<GA>
template <typename GA>
struct RngTag {
    using type = std::mt19937_64 GA::*;
    friend type rng_member(RngTag);
};
#pragma GCC diagnostic pop

template <typename Tag, typename Tag::type Member>
struct RngAccess {
    friend typename Tag::type rng_member(Tag) { return Member; }
};

template struct RngAccess<RngTag<BitsGA>, &BitsGA::rng>;
template struct RngAccess<RngTag<PermutationGA>, &PermutationGA::rng>;
template struct RngAccess<RngTag<RealGA>, &RealGA::rng>;

template <typename GA>
void seed_ga(GA &ga, std::uint64_t seed) {
    (ga.*rng_member(RngTag<GA>())).seed(seed);
}

// the seed of an attempt (rule 2.2): the run's seed first, then (seed + 1) * 1000000 + restart
std::uint64_t attempt_seed(std::uint64_t seed, std::uint64_t attempt) {
    return attempt == 0 ? seed : (seed + 1) * 1000000 + attempt;
}

// -------------------------------------------------------------------------------------------------
// Budget: counts every evaluation in the fitness function and records the first one that reaches
// the target
// -------------------------------------------------------------------------------------------------

using Clock = std::chrono::steady_clock;

struct Budget {
    long long evaluations = 0;
    long long max_evaluations = 0;
    double max_seconds = 0.0;
    double target = 0.0;         // a cost at most this reaches the target (minimized)
    long long outside = 0;       // solutions evaluated outside the bounds (rule 2.4)
    long long hit_evaluations = 0;  // the first evaluation that reaches the target, 0 if none
    double hit_seconds = 0.0;       // the clock at that evaluation
    Clock::time_point start, deadline;

    void record(double cost) {
        evaluations++;
        if (hit_evaluations == 0 && cost <= target) {
            hit_evaluations = evaluations;
            hit_seconds = std::chrono::duration<double>(Clock::now() - start).count();
        }
    }

    bool reached() const { return hit_evaluations > 0; }
    bool exhausted() const { return evaluations >= max_evaluations || Clock::now() >= deadline; }
};

Budget budget;

// -------------------------------------------------------------------------------------------------
// Fitness functions, identical to problems.py
// -------------------------------------------------------------------------------------------------

double onemax(const std::vector<std::uint8_t> &x) {
    int ones = 0;
    for (std::uint8_t v : x) ones += v;
    return ones;
}

// Number of diagonal conflicts, O(n): for each diagonal, its queens minus one
double nqueens(const std::vector<int> &p) {
    const int size = int(p.size());
    std::vector<int> left(2 * size - 1, 0), right(2 * size - 1, 0);
    for (int i = 0; i < size; i++) {
        left[i + p[i]]++;
        right[size - 1 - i + p[i]]++;
    }
    int conflicts = 0;
    for (int i = 0; i < 2 * size - 1; i++) {
        if (left[i] > 1) conflicts += left[i] - 1;
        if (right[i] > 1) conflicts += right[i] - 1;
    }
    return conflicts;
}

// Rastrigin and Ackley are shifted (rule 1.4): gene i is measured from
// s_i = 0.8 * upper * (2 * ((37 * i + 11) % 101) / 101 - 1), upper the box's upper bound, in this
// order. Computed once in main, before any run, for up to MAX_GENES genes.
constexpr size_t MAX_GENES = 1024;
std::vector<double> rastrigin_shift, ackley_shift;

std::vector<double> shifts(double upper) {
    std::vector<double> s(MAX_GENES);
    for (size_t i = 0; i < MAX_GENES; i++)
        s[i] = 0.8 * upper * (2.0 * double((37 * i + 11) % 101) / 101.0 - 1.0);
    return s;
}

double rastrigin(const std::vector<double> &x) {
    double sum = 10.0 * double(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        const double v = x[i] - rastrigin_shift[i];
        sum += v * v - 10.0 * std::cos(2.0 * PI * v);
    }
    return sum;
}

double rosenbrock(const std::vector<double> &x) {
    double sum = 0.0;
    for (size_t i = 0; i + 1 < x.size(); i++) {
        const double a = x[i], b = x[i + 1];
        sum += 100.0 * (b - a * a) * (b - a * a) + (1.0 - a) * (1.0 - a);
    }
    return sum;
}

double ackley(const std::vector<double> &x) {
    const double n = double(x.size());
    double squares = 0.0, cosines = 0.0;
    for (size_t i = 0; i < x.size(); i++) {
        const double v = x[i] - ackley_shift[i];
        squares += v * v;
        cosines += std::cos(2.0 * PI * v);
    }
    return -20.0 * std::exp(-0.2 * std::sqrt(squares / n)) - std::exp(cosines / n) + 20.0 + E;
}

double zdt_g(const std::vector<double> &x) {
    double sum = 0.0;
    for (size_t i = 1; i < x.size(); i++) sum += x[i];
    return 1.0 + 9.0 * sum / double(x.size() - 1);
}

std::vector<double> zdt1(const std::vector<double> &x, int) {
    const double g = zdt_g(x);
    return {x[0], g * (1.0 - std::sqrt(x[0] / g))};
}

std::vector<double> zdt2(const std::vector<double> &x, int) {
    const double g = zdt_g(x);
    return {x[0], g * (1.0 - (x[0] / g) * (x[0] / g))};
}

std::vector<double> zdt3(const std::vector<double> &x, int) {
    const double g = zdt_g(x);
    return {x[0], g * (1.0 - std::sqrt(x[0] / g) - x[0] / g * std::sin(10.0 * PI * x[0]))};
}

std::vector<double> dtlz2(const std::vector<double> &x, int objectives) {
    double g = 0.0;
    for (size_t i = objectives - 1; i < x.size(); i++) g += (x[i] - 0.5) * (x[i] - 0.5);
    std::vector<double> f(objectives);
    for (int m = 0; m < objectives; m++) {
        double v = 1.0 + g;
        for (int i = 0; i < objectives - 1 - m; i++) v *= std::cos(x[i] * PI / 2.0);
        if (m > 0) v *= std::sin(x[objectives - 1 - m] * PI / 2.0);
        f[m] = v;
    }
    return f;
}

std::vector<double> dtlz1(const std::vector<double> &x, int objectives) {
    const double k = double(x.size() - (objectives - 1));
    double sum = 0.0;
    for (size_t i = objectives - 1; i < x.size(); i++)
        sum += (x[i] - 0.5) * (x[i] - 0.5) - std::cos(20.0 * PI * (x[i] - 0.5));
    const double g = 100.0 * (k + sum);
    std::vector<double> f(objectives);
    for (int m = 0; m < objectives; m++) {
        double v = 0.5 * (1.0 + g);
        for (int i = 0; i < objectives - 1 - m; i++) v *= x[i];
        if (m > 0) v *= 1.0 - x[objectives - 1 - m];
        f[m] = v;
    }
    return f;
}

using RealFunction = double (*)(const std::vector<double> &);
using ObjectiveFunction = std::vector<double> (*)(const std::vector<double> &, int);

struct RealProblem {
    RealFunction function;
    double low, high;
};

// the real-valued problems and their bounds (problems.py REAL_BOUNDS)
bool real_problem(const std::string &problem, RealProblem &out) {
    if (problem == "rastrigin")
        out = {rastrigin, -5.12, 5.12};
    else if (problem == "rosenbrock")
        out = {rosenbrock, -5.0, 10.0};
    else if (problem == "ackley")
        out = {ackley, -32.768, 32.768};
    else
        return false;
    return true;
}

// the multi-objective problems, for the `values` command only: variables and objectives for the
// scenario's size (problems.py FRONT_VARIABLES, FRONT_OBJECTIVES)
bool front_problem(const std::string &problem, int size, ObjectiveFunction &function, int &variables,
                   int &objectives) {
    if (problem == "zdt1" || problem == "zdt2" || problem == "zdt3") {
        function = problem == "zdt1" ? zdt1 : problem == "zdt2" ? zdt2 : zdt3;
        variables = size;
        objectives = 2;
    } else if (problem == "dtlz2" || problem == "dtlz1") {
        // size: the number of objectives, with k = 10 (DTLZ2) or 5 (DTLZ1)
        function = problem == "dtlz2" ? dtlz2 : dtlz1;
        objectives = size;
        variables = problem == "dtlz2" ? objectives + 9 : objectives + 4;
    } else {
        return false;
    }
    return true;
}

// -------------------------------------------------------------------------------------------------
// Running a GA and printing the result
// -------------------------------------------------------------------------------------------------

using Rnd01 = std::function<double(void)>;

int random_index(const Rnd01 &rnd01, int n) {
    int i = int(rnd01() * n);
    return i < n ? i : n - 1;
}

template <typename GA>
void configure(GA &ga) {
    ga.problem_mode = EA::GA_MODE::SOGA;
    ga.multi_threading = false;  // single-threaded: sequential evaluation (L1557, L1678)
    ga.verbose = false;
    // generation_max is only a budget, so it's lifted (rule 2.2). The mutation's shrink_scale
    // doesn't depend on it: it's default_shrink_scale(generation) (L531-539), a function of the
    // generation number only.
    ga.generation_max = INT_MAX;
}

// openGA's convergence criteria (L1712-1734): the attempt ends after `best_max` generations whose
// best cost changes by less than `best_tol`, or `average_max` generations whose average cost changes
// by less than `average_tol`; run_ga then starts a new attempt (rule 2.2)
template <typename GA>
void stall(GA &ga, int best_max, double best_tol, int average_max, double average_tol) {
    ga.best_stall_max = best_max;
    ga.tol_stall_best = best_tol;
    ga.average_stall_max = average_max;
    ga.tol_stall_average = average_tol;
}

// openGA's defaults (L346-349, openGA.pdf Table 1 p. 4), which openGA assist's program keeps
// except best_stall_max, which it sets to 10 (assist/main.js L464): 10 generations for both, at 1e-6
// (best) and 1e-4 (average)
template <typename GA>
void assist_stall(GA &ga) {
    stall(ga, 10, 1e-6, 10, 1e-4);
}

struct RunResult {
    double seconds;
    long long generations;       // solve_next_generation calls, over all attempts
    long long restarts;          // attempts after the first
    long long last_generation;   // the evaluations of the last generation (or initial population)
    double best_cost = std::numeric_limits<double>::infinity();  // the best over all attempts
    std::vector<double> best_genes;                              // its genes
};

// Starts the budget and the clock (rule 4.1: before the first GA and its initial population)
void start_budget(long long max_evaluations, double max_seconds, double target) {
    budget = Budget();
    budget.max_evaluations = max_evaluations;
    budget.max_seconds = max_seconds;
    budget.target = target;
    budget.start = Clock::now();
    budget.deadline = budget.start + std::chrono::duration_cast<Clock::duration>(
                                         std::chrono::duration<double>(max_seconds));
}

// Attempts one after the other until the target is reached or the budget is used up (rule 2.2).
// Each attempt is a new GA, set up by `setup`, from a new random start with attempt_seed:
// solve_init (the initial population), then one generation at a time. An attempt ends when openGA's
// stall criteria detect convergence (the StopReason of solve_next_generation; generation_max never
// fires), and the next one starts; openGA has no restart mechanism. The best of each attempt's last
// generation is kept over all attempts, and every evaluation counts. The clock stops when the last
// attempt ends.
template <typename GA>
RunResult run_ga(long long seed, long long max_evaluations, double max_seconds, double target,
                 const std::function<void(GA &)> &setup) {
    start_budget(max_evaluations, max_seconds, target);
    RunResult result{0.0, 0, 0, 0};
    for (long long attempt = 0;; attempt++) {
        GA ga;
        setup(ga);
        seed_ga(ga, attempt_seed(std::uint64_t(seed), std::uint64_t(attempt)));
        long long before = budget.evaluations;
        ga.solve_init();
        result.last_generation = budget.evaluations - before;
        bool converged = false;
        while (!converged && !budget.reached() && !budget.exhausted()) {
            before = budget.evaluations;
            converged = ga.solve_next_generation() != EA::StopReason::Undefined;
            result.last_generation = budget.evaluations - before;
            result.generations++;
        }
        const auto &generation = ga.last_generation;
        const auto &best = generation.chromosomes[generation.best_chromosome_index];
        if (best.total_cost < result.best_cost) {
            result.best_cost = best.total_cost;
            result.best_genes = numbers(best.genes);
        }
        if (budget.reached() || budget.exhausted()) break;
        result.restarts++;
    }
    result.seconds = std::chrono::duration<double>(Clock::now() - budget.start).count();
    return result;
}

// whether a gene of x lies outside [low, high] (rule 2.4)
bool outside(const std::vector<double> &x, double low, double high) {
    for (double v : x)
        if (v < low || v > high) return true;
    return false;
}

// an integer as an integer, anything else with full precision
std::string number(double value) {
    char buffer[40];
    if (value == std::floor(value) && std::fabs(value) < 1e15)
        std::snprintf(buffer, sizeof buffer, "%lld", (long long)value);
    else
        std::snprintf(buffer, sizeof buffer, "%.17g", value);
    return buffer;
}

std::string real(double value) {
    char buffer[40];
    std::snprintf(buffer, sizeof buffer, "%.17g", value);
    return buffer;
}

std::string json_array(const std::vector<double> &values, bool integers) {
    std::string out = "[";
    for (size_t i = 0; i < values.size(); i++) {
        if (i) out += ", ";
        out += integers ? number(values[i]) : real(values[i]);
    }
    return out + "]";
}

struct Args {
    std::string problem;
    int size;
    std::string mode;
    long long seed_from, seed_to, max_evaluations;
    double max_seconds;
};

// best: the value of the solution, in the problem's own direction (OneMax maximized, the others
// minimized), recomputed from it after the clock (not counted as an evaluation)
void print_single(const Args &args, const char *solver, long long seed, const RunResult &run,
                  double best, double target, bool success, bool integers) {
    std::printf(
        "{\"library\": \"openga\", \"solver\": \"%s\", \"problem\": \"%s\", \"size\": %d, "
        "\"mode\": \"%s\", \"seed\": %lld, \"time_s\": %.6f, \"generations\": %lld, "
        "\"evaluations\": %lld, \"restarts\": %lld, \"last_generation\": %lld",
        solver, args.problem.c_str(), args.size, args.mode.c_str(), seed, run.seconds,
        run.generations, budget.evaluations, run.restarts, run.last_generation);
    // rule 2.4: the continuous runs report the solutions evaluated outside the bounds
    if (args.problem != "onemax" && args.problem != "nqueens") std::printf(", \"outside\": %lld", budget.outside);
    if (budget.reached())
        std::printf(", \"first_hit\": {\"evaluations\": %lld, \"time_s\": %.6f}", budget.hit_evaluations,
                    budget.hit_seconds);
    else
        std::printf(", \"first_hit\": null");
    std::printf(", \"best\": %s, \"target\": %s, \"success\": %s, \"solution\": %s}\n",
                (integers ? number(best) : real(best)).c_str(), number(target).c_str(),
                success ? "true" : "false", json_array(run.best_genes, integers).c_str());
    std::fflush(stdout);
}

// -------------------------------------------------------------------------------------------------
// OneMax
// -------------------------------------------------------------------------------------------------

void solve_onemax(const Args &args, long long seed) {
    // matched OneMax isn't run: openGA has no two-point crossover or bit flip of its own (see the
    // top of the file). Print nothing.
    if (args.mode == "matched") return;
    const int size = args.size;
    const std::function<void(BitsGA &)> setup = [size](BitsGA &ga) {
        configure(ga);
        // random 0/1 genes, as the examples' init_genes draws each gene uniformly in its range
        // (examples/so-rastrigin/so-rastrigin.cpp L36-40)
        ga.init_genes = [size](Bits &b, const Rnd01 &rnd01) {
            b.x.resize(size);
            for (auto &v : b.x) v = rnd01() < 0.5 ? 1 : 0;
        };
        ga.eval_solution = [](const Bits &b, Cost &c) {
            c.cost = -onemax(b.x);  // openGA minimizes
            budget.record(c.cost);
            return true;
        };
        ga.calculate_SO_total_fitness = [](const BitsGA::thisChromosomeType &X) { return X.middle_costs.cost; };
        ga.SO_report_generation = [](int, const EA::GenerationType<Bits, Cost> &, const Bits &) {};
        // bit flip with probability 1 / size per gene: openGA has no binary operators, and the
        // generated real-valued mutation (assist/main.js L259-279) doesn't apply to 0/1 genes
        // (openGA.pdf p. 6: "the clients should edit these operators according to their need")
        ga.mutate = [size](const Bits &base, const Rnd01 &rnd01, double) {
            Bits child = base;
            for (auto &v : child.x)
                if (rnd01() < 1.0 / size) v ^= 1;
            return child;
        };
        // openGA assist's generated settings (assist/main.js): population "medium" 200
        // (L435-436, the default selection, assist/index.html L24), crossover_fraction 0.7 and
        // mutation_rate 0.2 (L460-461), elite_count 10 (L465), best_stall_max 10 (L464) with the
        // other stall settings at openGA's defaults. Its generation_max 1000 (L447) is lifted.
        // With 0/1 genes, the generated crossover, a random mix of the parents per gene
        // (L282-295), becomes uniform crossover.
        ga.population = 200;
        ga.crossover_fraction = 0.7;
        ga.mutation_rate = 0.2;
        ga.elite_count = 10;
        assist_stall(ga);
        ga.crossover = [](const Bits &a, const Bits &b, const Rnd01 &rnd01) {
            Bits child = a;
            for (size_t i = 0; i < child.x.size(); i++)
                if (rnd01() < 0.5) child.x[i] = b.x[i];
            return child;
        };
    };

    const RunResult run = run_ga<BitsGA>(seed, args.max_evaluations, args.max_seconds, -double(size), setup);
    const std::vector<std::uint8_t> bits(run.best_genes.begin(), run.best_genes.end());
    const double best = onemax(bits);
    print_single(args, "ga", seed, run, best, size, best >= size, true);
}

// -------------------------------------------------------------------------------------------------
// N-Queens
// -------------------------------------------------------------------------------------------------

void solve_nqueens(const Args &args, long long seed) {
    const int size = args.size;
    const std::function<void(PermutationGA &)> setup = [size](PermutationGA &ga) {
        configure(ga);
        // openGA assist's generated settings, as for OneMax: population 200, crossover_fraction
        // 0.7, mutation_rate 0.2, elite_count 10, best_stall_max 10 (assist/main.js L435-436,
        // L460-465) and openGA's other stall defaults. openGA has no permutation operators and no
        // permutation example, so the usual ones: a random permutation, order crossover (OX1, one
        // child, as openGA's crossover returns one) and a swap of two genes.
        ga.population = 200;
        ga.crossover_fraction = 0.7;
        ga.mutation_rate = 0.2;
        ga.elite_count = 10;
        assist_stall(ga);
        ga.init_genes = [size](Permutation &q, const Rnd01 &rnd01) {
            q.p.resize(size);
            for (int i = 0; i < size; i++) q.p[i] = i;
            for (int i = size - 1; i > 0; i--) std::swap(q.p[i], q.p[random_index(rnd01, i + 1)]);
        };
        ga.eval_solution = [](const Permutation &q, Cost &c) {
            c.cost = nqueens(q.p);
            budget.record(c.cost);
            return true;
        };
        ga.calculate_SO_total_fitness = [](const PermutationGA::thisChromosomeType &X) { return X.middle_costs.cost; };
        ga.SO_report_generation = [](int, const EA::GenerationType<Permutation, Cost> &, const Permutation &) {};
        ga.crossover = [size](const Permutation &a, const Permutation &b, const Rnd01 &rnd01) {
            int first = random_index(rnd01, size), second = random_index(rnd01, size);
            if (first > second) std::swap(first, second);
            Permutation child;
            child.p.assign(size, -1);
            std::vector<char> used(size, 0);
            for (int i = first; i <= second; i++) {
                child.p[i] = a.p[i];
                used[a.p[i]] = 1;
            }
            int position = (second + 1) % size;
            for (int k = 0; k < size; k++) {
                const int gene = b.p[(second + 1 + k) % size];
                if (used[gene]) continue;
                child.p[position] = gene;
                position = (position + 1) % size;
            }
            return child;
        };
        ga.mutate = [size](const Permutation &base, const Rnd01 &rnd01, double) {
            Permutation child = base;
            const int i = random_index(rnd01, size);
            int j = random_index(rnd01, size - 1);
            if (j >= i) j++;
            std::swap(child.p[i], child.p[j]);
            return child;
        };
    };

    const RunResult run = run_ga<PermutationGA>(seed, args.max_evaluations, args.max_seconds, 0.0, setup);
    const std::vector<int> order(run.best_genes.begin(), run.best_genes.end());
    const double best = nqueens(order);
    print_single(args, "ga", seed, run, best, 0, best <= 0.0, true);
}

// -------------------------------------------------------------------------------------------------
// Rastrigin, Rosenbrock, Ackley
// -------------------------------------------------------------------------------------------------

// The settings of one of openGA's real-valued programs
struct RealSettings {
    const char *solver;
    unsigned population;
    double crossover_fraction, mutation_rate;
    int elite_count;
    // the mutation moves each gene by mu * (rnd01() - rnd01()), mu = radius * shrink_scale, times
    // rnd01() per gene if random_radius
    double radius;
    bool random_radius;
    // the stall criteria: generations and tolerance, of the best and the average cost
    int best_stall_max;
    double tol_stall_best;
    int average_stall_max;
    double tol_stall_average;
};

void solve_real(const Args &args, long long seed, const RealProblem &problem, const RealSettings &settings) {
    const int size = args.size;
    const double low = problem.low, high = problem.high;
    const RealFunction function = problem.function;
    const std::function<void(RealGA &)> setup = [&settings, size, low, high, function](RealGA &ga) {
        configure(ga);
        ga.population = settings.population;
        ga.crossover_fraction = settings.crossover_fraction;
        ga.mutation_rate = settings.mutation_rate;
        ga.elite_count = settings.elite_count;
        stall(ga, settings.best_stall_max, settings.tol_stall_best, settings.average_stall_max,
              settings.tol_stall_average);
        // uniform in the bounds (so-rastrigin.cpp L36-40, assist/main.js L234)
        ga.init_genes = [size, low, high](Reals &r, const Rnd01 &rnd01) {
            r.x.resize(size);
            for (auto &v : r.x) v = low + (high - low) * rnd01();
        };
        // the solutions evaluated outside the bounds are counted as the library proposed them
        ga.eval_solution = [function, low, high](const Reals &r, Cost &c) {
            if (outside(r.x, low, high)) budget.outside++;
            c.cost = function(r.x);
            budget.record(c.cost);
            return true;
        };
        ga.calculate_SO_total_fitness = [](const RealGA::thisChromosomeType &X) { return X.middle_costs.cost; };
        ga.SO_report_generation = [](int, const EA::GenerationType<Reals, Cost> &, const Reals &) {};
        // so-rastrigin.cpp L53-73 and assist/main.js L259-279: every gene moves by
        // mu * (rnd01() - rnd01()), and the whole child is drawn again while a gene is out of range:
        // the examples' bound handling (rule 2.4)
        const double radius = settings.radius;
        const bool random_radius = settings.random_radius;
        ga.mutate = [radius, random_radius, low, high](const Reals &base, const Rnd01 &rnd01, double shrink_scale) {
            Reals child;
            bool out_of_range;
            do {
                out_of_range = false;
                child = base;
                for (auto &v : child.x) {
                    const double mu = random_radius ? radius * rnd01() * shrink_scale : radius * shrink_scale;
                    v += mu * (rnd01() - rnd01());
                    if (v < low || v > high) out_of_range = true;
                }
            } while (out_of_range);
            return child;
        };
        // so-rastrigin.cpp L75-87 and assist/main.js L282-295: a random mix of the parents per gene,
        // which stays between the parents, so inside the bounds
        ga.crossover = [](const Reals &a, const Reals &b, const Rnd01 &rnd01) {
            Reals child;
            child.x.resize(a.x.size());
            for (size_t i = 0; i < a.x.size(); i++) {
                const double r = rnd01();
                child.x[i] = r * a.x[i] + (1.0 - r) * b.x[i];
            }
            return child;
        };
    };

    const RunResult run = run_ga<RealGA>(seed, args.max_evaluations, args.max_seconds, REAL_TARGET, setup);
    const double best = function(run.best_genes);
    print_single(args, settings.solver, seed, run, best, REAL_TARGET, best <= REAL_TARGET, false);
}

void solve_real_all(const Args &args, long long seed, const RealProblem &problem) {
    // examples/so-rastrigin/so-rastrigin.cpp, openGA's example for an n-dimensional real vector on
    // a multimodal function (Rastrigin in [-5.12, 5.12]): population 10000 (L145), elite_count 10
    // (L157), crossover_fraction 0.7 (L158), mutation_rate 0.1 (L159), the mutation radius
    // 1.7 * rnd01() * shrink_scale (L66), as written, and its stall criteria, 20 generations at 1e-6
    // for the best and the average cost (L153-156). Its generation_max 1000 (L146) is lifted.
    // Multimodal problems only: Rosenbrock (unimodal) runs only the assist program.
    if (args.problem != "rosenbrock") {
        const RealSettings example{"ga", 10000, 0.7, 0.1, 10, 1.7, true, 20, 1e-6, 20, 1e-6};
        solve_real(args, seed, problem, example);
    }
    // openGA assist's generated program for real variables (assist/main.js): population 200
    // (L435-436), crossover_fraction 0.7, mutation_rate 0.2 (L460-461), elite_count 10 (L465),
    // mutation radius 0.2 * shrink_scale (L265), best_stall_max 10 (L464) and openGA's other stall
    // defaults (L346-349): 10 generations at 1e-6 (best) and 1e-4 (average). Its generation_max 1000
    // (L447) is lifted.
    const RealSettings assist{"ga_assist", 200, 0.7, 0.2, 10, 0.2, false, 10, 1e-6, 10, 1e-4};
    solve_real(args, seed, problem, assist);
}

// -------------------------------------------------------------------------------------------------
// values <problem> <size>: one JSON array per line on stdin, its value (or objectives) per line,
// with the fitness functions of the runs
// -------------------------------------------------------------------------------------------------

std::vector<double> parse_array(const std::string &line) {
    std::vector<double> values;
    std::string token;
    auto flush = [&]() {
        size_t begin = token.find_first_not_of(" \t\r");
        size_t end = token.find_last_not_of(" \t\r");
        if (begin != std::string::npos) {
            const std::string t = token.substr(begin, end - begin + 1);
            values.push_back(t == "true" ? 1.0 : t == "false" ? 0.0 : std::strtod(t.c_str(), nullptr));
        }
        token.clear();
    };
    for (char c : line) {
        if (c == '[' || c == ']') continue;
        if (c == ',')
            flush();
        else
            token += c;
    }
    flush();
    return values;
}

int values_command(const std::string &problem, int size) {
    RealProblem real_one;
    ObjectiveFunction function;
    int variables, objectives;
    const bool is_real = real_problem(problem, real_one);
    const bool is_front = front_problem(problem, size, function, variables, objectives);
    if (!is_real && !is_front && problem != "onemax" && problem != "nqueens") {
        std::fprintf(stderr, "unknown problem: %s\n", problem.c_str());
        return 2;
    }
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        const std::vector<double> x = parse_array(line);
        if (problem == "onemax") {
            std::vector<std::uint8_t> bits(x.begin(), x.end());
            std::printf("%s\n", number(onemax(bits)).c_str());
        } else if (problem == "nqueens") {
            std::vector<int> order(x.begin(), x.end());
            std::printf("%s\n", number(nqueens(order)).c_str());
        } else if (is_real) {
            std::printf("%s\n", real(real_one.function(x)).c_str());
        } else {
            std::printf("%s\n", json_array(function(x, objectives), false).c_str());
        }
    }
    std::fflush(stdout);
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
        std::printf("%s\n", OPENGA_VERSION);
        return 0;
    }
    // the shifts are computed before any run
    rastrigin_shift = shifts(5.12);
    ackley_shift = shifts(32.768);
    if (argc == 4 && std::strcmp(argv[1], "values") == 0) return values_command(argv[2], std::atoi(argv[3]));
    if (argc != 8) {
        std::fprintf(stderr,
                     "usage: ga_bench_openga <problem> <size> <mode> <seed_from> <seed_to> "
                     "<max_evaluations> <max_seconds>\n"
                     "       ga_bench_openga values <problem> <size>\n");
        return 2;
    }
    Args args{argv[1], std::atoi(argv[2]), argv[3], std::atoll(argv[4]), std::atoll(argv[5]),
              std::atoll(argv[6]), std::atof(argv[7])};
    if (size_t(args.size) > MAX_GENES) {
        std::fprintf(stderr, "at most %zu genes\n", MAX_GENES);
        return 2;
    }
    const std::string &p = args.problem;
    RealProblem real_one;
    for (long long seed = args.seed_from; seed <= args.seed_to; seed++) {
        if (p == "onemax")
            solve_onemax(args, seed);
        else if (p == "nqueens")
            solve_nqueens(args, seed);
        else if (real_problem(p, real_one))
            solve_real_all(args, seed, real_one);
        else
            return 0;  // the multi-objective scenarios (matched) and anything else: print nothing
    }
    return 0;
}
