// Benchmark adapter for openGA (https://github.com/Arash-codedev/openGA), a header-only C++ GA.
//
// Usage: ga_bench_openga <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
//        ga_bench_openga --version
// Prints one JSON line per solver per seed, see ../../README.md for the fields.
//
// openGA's loop (EA::Genetic, SOGA and NSGA_III modes), which the adapter can't change:
// - every generation keeps the whole population and adds round(population * crossover_fraction)
//   children; each child comes from two distinct parents chosen by a rank-based roulette
//   (chance 1 / sqrt(rank + 1), the rank being the front index for NSGA-III), the user's
//   crossover (one child per call) and, with probability mutation_rate, the user's mutation;
// - SOGA survival: the elite_count best of parents + children, then a rank-based roulette for the
//   rest. The roulette is normalized by the cumulative chance at index population - 1, so it can
//   only pick the previous population (indices < population): a new child survives only through
//   an elite slot;
// - NSGA_III survival: non-dominated fronts, then niching on Das-Dennis reference directions.
// All randomness goes through the library's own std::mt19937_64 (also passed to the operators as
// rnd01), which the adapter seeds with the run's seed. The library has no seed setter, so the
// adapter reaches the private generator with the standard explicit-instantiation access idiom.
// multi_threading is off: the population is evaluated sequentially on one core.
//
// The adapter drives the generations itself (solve_init, then solve_next_generation) and stops
// after the generation in which the target is reached or the evaluation budget or the time limit
// is used up; openGA's own stall and generation limits are disabled.

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
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

struct Objectives {
    std::vector<double> f;  // minimized
};

using BitsGA = EA::Genetic<Bits, Cost>;
using PermutationGA = EA::Genetic<Permutation, Cost>;
using RealGA = EA::Genetic<Reals, Cost>;
using FrontGA = EA::Genetic<Reals, Objectives>;

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
template struct RngAccess<RngTag<FrontGA>, &FrontGA::rng>;

template <typename GA>
void seed_ga(GA &ga, std::uint64_t seed) {
    (ga.*rng_member(RngTag<GA>())).seed(seed);
}

// -------------------------------------------------------------------------------------------------
// Budget: counts evaluations in the fitness function and keeps the best cost
// -------------------------------------------------------------------------------------------------

using Clock = std::chrono::steady_clock;

struct Budget {
    long long evaluations = 0;
    long long max_evaluations = 0;
    double best = std::numeric_limits<double>::infinity();
    Clock::time_point deadline;

    void record(double cost) {
        evaluations++;
        if (cost < best) best = cost;
    }

    bool exhausted() const { return evaluations >= max_evaluations || Clock::now() >= deadline; }
};

Budget budget;

// -------------------------------------------------------------------------------------------------
// Fitness functions, identical to the other adapters
// -------------------------------------------------------------------------------------------------

double onemax(const std::vector<std::uint8_t> &x) {
    int ones = 0;
    for (std::uint8_t v : x) ones += v;
    return ones;
}

// Number of diagonal conflicts, O(n) (as NQueens.fitness in adapters/pymoo/bench.py)
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

// Rastrigin and Ackley are shifted, so an optimum at the origin can't favour operators that
// drift towards 0: gene i is measured from s_i = 2 ((37 i + 11) mod 101) / 101 - 1, in [-1, 1]
double shift(size_t i) { return 2.0 * double((37 * i + 11) % 101) / 101.0 - 1.0; }

double rastrigin(const std::vector<double> &x) {
    double sum = 10.0 * double(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        const double v = x[i] - shift(i);
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
        const double v = x[i] - shift(i);
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

// -------------------------------------------------------------------------------------------------
// Running a GA and printing the result
// -------------------------------------------------------------------------------------------------

using Rnd01 = std::function<double(void)>;

int random_index(const Rnd01 &rnd01, int n) {
    int i = int(rnd01() * n);
    return i < n ? i : n - 1;
}

template <typename GA>
void configure(GA &ga, EA::GA_MODE mode) {
    ga.problem_mode = mode;
    ga.multi_threading = false;  // single-threaded: sequential evaluation
    ga.verbose = false;
    // the adapter stops the run itself, see the top of the file
    ga.generation_max = INT_MAX;
    ga.best_stall_max = INT_MAX;
    ga.average_stall_max = INT_MAX;
}

struct RunResult {
    double seconds;
    long long generations;
};

// solve_init, then one generation at a time until done() or the budget is used up
template <typename GA>
RunResult run_ga(GA &ga, const std::function<bool()> &done) {
    const auto start = Clock::now();
    ga.solve_init();
    long long generations = 0;
    while (!done() && !budget.exhausted()) {
        ga.solve_next_generation();
        generations++;
    }
    const double seconds = std::chrono::duration<double>(Clock::now() - start).count();
    return {seconds, generations};
}

void start_budget(long long max_evaluations, double max_seconds) {
    budget = Budget();
    budget.max_evaluations = max_evaluations;
    budget.deadline = Clock::now() + std::chrono::duration_cast<Clock::duration>(
                                         std::chrono::duration<double>(max_seconds));
}

std::string number(double value) {
    if (value == std::floor(value) && std::fabs(value) < 1e15) {
        char buffer[32];
        std::snprintf(buffer, sizeof buffer, "%lld", (long long)value);
        return buffer;
    }
    char buffer[40];
    std::snprintf(buffer, sizeof buffer, "%.17g", value);
    return buffer;
}

struct Args {
    std::string problem;
    int size;
    std::string mode;
    long long seed_from, seed_to, max_evaluations;
    double max_seconds;
};

void print_head(const Args &args, const char *solver, long long seed, const RunResult &run) {
    std::printf(
        "{\"library\": \"openga\", \"solver\": \"%s\", \"problem\": \"%s\", \"size\": %d, "
        "\"mode\": \"%s\", \"seed\": %lld, \"time_s\": %.6f, \"generations\": %lld, "
        "\"evaluations\": %lld",
        solver, args.problem.c_str(), args.size, args.mode.c_str(), seed, run.seconds,
        run.generations, budget.evaluations);
}

void print_single(const Args &args, const char *solver, long long seed, const RunResult &run,
                  double best, double target, bool success) {
    print_head(args, solver, seed, run);
    std::printf(", \"best\": %s, \"target\": %s, \"success\": %s}\n", number(best).c_str(),
                number(target).c_str(), success ? "true" : "false");
    std::fflush(stdout);
}

// -------------------------------------------------------------------------------------------------
// OneMax
// -------------------------------------------------------------------------------------------------

void solve_onemax(const Args &args, long long seed) {
    const int size = args.size;
    BitsGA ga;
    configure(ga, EA::GA_MODE::SOGA);
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
    // bit flip with probability 1 / size per gene
    ga.mutate = [size](const Bits &base, const Rnd01 &rnd01, double) {
        Bits child = base;
        for (auto &v : child.x)
            if (rnd01() < 1.0 / size) v ^= 1;
        return child;
    };

    if (args.mode == "matched") {
        // As DEAP's eaSimple (population 300, tournament 3, two-point crossover 0.5, bit flip 1/size
        // on 20% of the children, no elitism), as close as openGA allows. Differences:
        // - parents: openGA's rank-based roulette (two distinct parents), not a tournament of 3;
        // - 300 children per generation (crossover_fraction 1), each from a two-point crossover with
        //   probability 0.5 (else a copy of the first parent); crossover gives one child per call;
        // - openGA evaluates every child, also the unchanged copies (DEAP only the changed ones);
        // - survival: the best 300 of parents + children (elite_count = population), as pymoo's
        //   matched GA. "No elitism" can't be expressed: a new child enters openGA's next
        //   population only through an elite slot (see the top of the file).
        ga.population = 300;
        ga.crossover_fraction = 1.0;
        ga.mutation_rate = 0.2;
        ga.elite_count = 300;
        ga.crossover = [size](const Bits &a, const Bits &b, const Rnd01 &rnd01) {
            Bits child = a;
            if (rnd01() < 0.5) {
                // DEAP's cxTwoPoint: cut points in 1..size-1, the middle from the second parent
                int first = 1 + random_index(rnd01, size);
                int second = 1 + random_index(rnd01, size - 1);
                if (second >= first)
                    second++;
                else
                    std::swap(first, second);
                for (int i = first; i < second; i++) child.x[i] = b.x[i];
            }
            return child;
        };
    } else {
        // openGA assist's generated defaults (assist/main.js: population "medium" 200,
        // crossover_fraction 0.7, mutation_rate 0.2, elite_count 10), with 0/1 genes: uniform
        // crossover (the per-gene random mix of the examples' crossover) and a bit flip.
        ga.population = 200;
        ga.crossover_fraction = 0.7;
        ga.mutation_rate = 0.2;
        ga.elite_count = 10;
        ga.crossover = [](const Bits &a, const Bits &b, const Rnd01 &rnd01) {
            Bits child = a;
            for (size_t i = 0; i < child.x.size(); i++)
                if (rnd01() < 0.5) child.x[i] = b.x[i];
            return child;
        };
    }

    seed_ga(ga, std::uint64_t(seed));
    start_budget(args.max_evaluations, args.max_seconds);
    const RunResult run = run_ga(ga, [size] { return -budget.best >= size; });
    const double best = -budget.best;
    print_single(args, "ga", seed, run, best, size, best >= size);
}

// -------------------------------------------------------------------------------------------------
// N-Queens
// -------------------------------------------------------------------------------------------------

void solve_nqueens(const Args &args, long long seed) {
    const int size = args.size;
    PermutationGA ga;
    configure(ga, EA::GA_MODE::SOGA);
    // openGA assist's generated defaults (population 200, crossover_fraction 0.7, mutation_rate
    // 0.2, elite_count 10); openGA has no permutation operators, so the usual ones: order crossover
    // (OX1, one child) and a swap of two genes
    ga.population = 200;
    ga.crossover_fraction = 0.7;
    ga.mutation_rate = 0.2;
    ga.elite_count = 10;
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

    seed_ga(ga, std::uint64_t(seed));
    start_budget(args.max_evaluations, args.max_seconds);
    const RunResult run = run_ga(ga, [] { return budget.best <= 0.0; });
    print_single(args, "ga", seed, run, budget.best, 0, budget.best <= 0.0);
}

// -------------------------------------------------------------------------------------------------
// Rastrigin, Rosenbrock, Ackley
// -------------------------------------------------------------------------------------------------

void solve_real(const Args &args, long long seed, double (*function)(const std::vector<double> &),
                double low, double high) {
    const int size = args.size;
    RealGA ga;
    configure(ga, EA::GA_MODE::SOGA);
    // examples/so-rastrigin/so-rastrigin.cpp: population 10000, elite_count 10,
    // crossover_fraction 0.7, mutation_rate 0.1, the per-gene random mix crossover and the
    // mutation of radius 1.7 * rnd01() * shrink_scale, redrawn while out of range. The radius is
    // 1.7 for [-5.12, 5.12] and scaled to the width of the other domains.
    ga.population = 10000;
    ga.crossover_fraction = 0.7;
    ga.mutation_rate = 0.1;
    ga.elite_count = 10;
    const double radius = 1.7 * (high - low) / 10.24;
    ga.init_genes = [size, low, high](Reals &r, const Rnd01 &rnd01) {
        r.x.resize(size);
        for (auto &v : r.x) v = low + (high - low) * rnd01();
    };
    ga.eval_solution = [function](const Reals &r, Cost &c) {
        c.cost = function(r.x);
        budget.record(c.cost);
        return true;
    };
    ga.calculate_SO_total_fitness = [](const RealGA::thisChromosomeType &X) { return X.middle_costs.cost; };
    ga.SO_report_generation = [](int, const EA::GenerationType<Reals, Cost> &, const Reals &) {};
    ga.mutate = [radius, low, high](const Reals &base, const Rnd01 &rnd01, double shrink_scale) {
        Reals child;
        bool out_of_range;
        do {
            out_of_range = false;
            child = base;
            for (auto &v : child.x) {
                const double mu = radius * rnd01() * shrink_scale;
                v += mu * (rnd01() - rnd01());
                if (v < low || v > high) out_of_range = true;
            }
        } while (out_of_range);
        return child;
    };
    ga.crossover = [](const Reals &a, const Reals &b, const Rnd01 &rnd01) {
        Reals child;
        child.x.resize(a.x.size());
        for (size_t i = 0; i < a.x.size(); i++) {
            const double r = rnd01();
            child.x[i] = r * a.x[i] + (1.0 - r) * b.x[i];
        }
        return child;
    };

    seed_ga(ga, std::uint64_t(seed));
    start_budget(args.max_evaluations, args.max_seconds);
    const RunResult run = run_ga(ga, [] { return budget.best <= REAL_TARGET; });
    print_single(args, "ga", seed, run, budget.best, REAL_TARGET, budget.best <= REAL_TARGET);
}

// -------------------------------------------------------------------------------------------------
// Multi-objective: openGA's NSGA-III
// -------------------------------------------------------------------------------------------------

// SBX on [0, 1] (as DEAP's cxSimulatedBinaryBounded: each variable with probability 0.5); openGA's
// crossover returns one child, so one of the two at random
Reals sbx(const Reals &a, const Reals &b, double eta, const Rnd01 &rnd01) {
    Reals c1 = a, c2 = b;
    const double low = 0.0, high = 1.0;
    for (size_t i = 0; i < a.x.size(); i++) {
        if (rnd01() > 0.5) continue;
        if (std::fabs(a.x[i] - b.x[i]) <= 1e-14) continue;
        const double x1 = std::min(a.x[i], b.x[i]), x2 = std::max(a.x[i], b.x[i]);
        const double rand = rnd01();

        double beta = 1.0 + 2.0 * (x1 - low) / (x2 - x1);
        double alpha = 2.0 - std::pow(beta, -(eta + 1.0));
        double beta_q = rand <= 1.0 / alpha ? std::pow(rand * alpha, 1.0 / (eta + 1.0))
                                            : std::pow(1.0 / (2.0 - rand * alpha), 1.0 / (eta + 1.0));
        double v1 = 0.5 * (x1 + x2 - beta_q * (x2 - x1));

        beta = 1.0 + 2.0 * (high - x2) / (x2 - x1);
        alpha = 2.0 - std::pow(beta, -(eta + 1.0));
        beta_q = rand <= 1.0 / alpha ? std::pow(rand * alpha, 1.0 / (eta + 1.0))
                                     : std::pow(1.0 / (2.0 - rand * alpha), 1.0 / (eta + 1.0));
        double v2 = 0.5 * (x1 + x2 + beta_q * (x2 - x1));

        v1 = std::min(std::max(v1, low), high);
        v2 = std::min(std::max(v2, low), high);
        if (rnd01() <= 0.5) std::swap(v1, v2);
        c1.x[i] = v1;
        c2.x[i] = v2;
    }
    return rnd01() < 0.5 ? c1 : c2;
}

// polynomial mutation on [0, 1] (as DEAP's mutPolynomialBounded)
Reals polynomial_mutation(const Reals &base, double eta, double rate, const Rnd01 &rnd01) {
    Reals child = base;
    const double low = 0.0, high = 1.0;
    const double power = 1.0 / (eta + 1.0);
    for (auto &x : child.x) {
        if (rnd01() > rate) continue;
        const double delta1 = (x - low) / (high - low), delta2 = (high - x) / (high - low);
        const double rand = rnd01();
        double delta_q;
        if (rand < 0.5) {
            const double xy = 1.0 - delta1;
            const double val = 2.0 * rand + (1.0 - 2.0 * rand) * std::pow(xy, eta + 1.0);
            delta_q = std::pow(val, power) - 1.0;
        } else {
            const double xy = 1.0 - delta2;
            const double val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * std::pow(xy, eta + 1.0);
            delta_q = 1.0 - std::pow(val, power);
        }
        x = std::min(std::max(x + delta_q * (high - low), low), high);
    }
    return child;
}

void solve_front(const Args &args, long long seed) {
    using ObjectiveFunction = std::vector<double> (*)(const std::vector<double> &, int);
    ObjectiveFunction function;
    int variables, objectives;
    if (args.problem == "zdt1" || args.problem == "zdt2" || args.problem == "zdt3") {
        function = args.problem == "zdt1" ? zdt1 : args.problem == "zdt2" ? zdt2 : zdt3;
        variables = args.size;
        objectives = 2;
    } else {
        // size: the number of objectives, with k = 10 (DTLZ2) or 5 (DTLZ1)
        function = args.problem == "dtlz2" ? dtlz2 : dtlz1;
        objectives = args.size;
        variables = args.problem == "dtlz2" ? objectives + 9 : objectives + 4;
    }
    // matched NSGA-III: 100 directions (99 divisions) and population 100 for 2 objectives, 91
    // directions (12 divisions) and population 92 for 3
    unsigned population, divisions;
    if (objectives == 2) {
        population = 100;
        divisions = 99;
    } else if (objectives == 3) {
        population = 92;
        divisions = 12;
    } else {
        return;  // unsupported
    }
    const double rate = 1.0 / variables;

    FrontGA ga;
    configure(ga, EA::GA_MODE::NSGA_III);
    // matched: SBX eta 30 at 1 and polynomial mutation eta 20 at 1 / n per variable on every child
    // (mutation_rate 1), population children per generation (crossover_fraction 1). Differences:
    // parents come from openGA's rank-based roulette on the front index, not at random, and the
    // crossover gives one of the two SBX children.
    ga.population = population;
    ga.reference_vector_divisions = divisions;
    ga.crossover_fraction = 1.0;
    ga.mutation_rate = 1.0;
    ga.init_genes = [variables](Reals &r, const Rnd01 &rnd01) {
        r.x.resize(variables);
        for (auto &v : r.x) v = rnd01();
    };
    ga.eval_solution = [function, objectives](const Reals &r, Objectives &o) {
        o.f = function(r.x, objectives);
        budget.evaluations++;
        return true;
    };
    ga.calculate_MO_objectives = [](FrontGA::thisChromosomeType &X) { return X.middle_costs.f; };
    ga.MO_report_generation = [](int, const EA::GenerationType<Reals, Objectives> &,
                                 const std::vector<unsigned int> &) {};
    ga.crossover = [](const Reals &a, const Reals &b, const Rnd01 &rnd01) { return sbx(a, b, 30.0, rnd01); };
    ga.mutate = [rate](const Reals &base, const Rnd01 &rnd01, double) {
        return polynomial_mutation(base, 20.0, rate, rnd01);
    };

    seed_ga(ga, std::uint64_t(seed));
    start_budget(args.max_evaluations, args.max_seconds);
    const RunResult run = run_ga(ga, [] { return false; });

    // the final non-dominated front of the population
    print_head(args, "nsga3", seed, run);
    std::printf(", \"front\": [");
    const auto &generation = ga.last_generation;
    bool first = true;
    if (!generation.fronts.empty()) {
        for (unsigned index : generation.fronts[0]) {
            const auto &f = generation.chromosomes[index].objectives;
            std::printf("%s[", first ? "" : ", ");
            for (size_t m = 0; m < f.size(); m++) std::printf("%s%.17g", m ? ", " : "", f[m]);
            std::printf("]");
            first = false;
        }
    }
    std::printf("]}\n");
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char **argv) {
    if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
        std::printf("%s\n", OPENGA_VERSION);
        return 0;
    }
    if (argc != 8) {
        std::fprintf(stderr,
                     "usage: ga_bench_openga <problem> <size> <mode> <seed_from> <seed_to> "
                     "<max_evaluations> <max_seconds>\n");
        return 2;
    }
    Args args{argv[1], std::atoi(argv[2]), argv[3], std::atoll(argv[4]), std::atoll(argv[5]),
              std::atoll(argv[6]), std::atof(argv[7])};
    const std::string &p = args.problem;
    for (long long seed = args.seed_from; seed <= args.seed_to; seed++) {
        if (p == "onemax")
            solve_onemax(args, seed);
        else if (p == "nqueens")
            solve_nqueens(args, seed);
        else if (p == "rastrigin")
            solve_real(args, seed, rastrigin, -5.12, 5.12);
        else if (p == "rosenbrock")
            solve_real(args, seed, rosenbrock, -5.0, 10.0);
        else if (p == "ackley")
            solve_real(args, seed, ackley, -32.768, 32.768);
        else if (p == "zdt1" || p == "zdt2" || p == "zdt3" || p == "dtlz1" || p == "dtlz2")
            solve_front(args, seed);
        else
            return 0;  // unsupported: print nothing
    }
    return 0;
}
