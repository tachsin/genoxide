/*
 * Benchmark adapter for Jenetics (io.jenetics:jenetics and jenetics.ext), following
 * docs/benchmarks/rules.md. Its page, docs/benchmarks/libraries/jenetics.md, gives every method,
 * where Jenetics recommends it, what was left out and why, and the separate test runs.
 *
 * Usage: java -cp ... Bench <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
 *        java -cp ... Bench values <problem> <size>     (one JSON solution per line on stdin)
 *        java -cp ... Bench --version
 * Prints one JSON line per solver per seed, see ../../README.md for the fields.
 *
 * - Evaluations (rule 3) are counted by the adapter, in the fitness function the engine calls.
 *   Jenetics evaluates only individuals it hasn't evaluated yet: survivors and offspring left
 *   unchanged by the alterers keep their fitness.
 * - One thread (rule 4.3): the engine gets `.executor(Runnable::run)`, which also evaluates the
 *   fitness on the calling thread (Jenetics uses the ForkJoinPool common pool by default); the
 *   manual's "Reproducibility" section (2.7) configures it the same way. run.sh runs the JVM with
 *   the serial garbage collector and with -Xbatch, so the JIT compiles on the calling thread's time.
 * - Seeds (rule 5.2): the engine's random generator is RandomRegistry's L64X256MixRandom (its
 *   default algorithm), created from the seed (manual 1.4.2 and 2.7).
 * - Time (rule 4.2): before the timed runs, every solver runs once untimed, with 1,000 evaluations
 *   and the seed 1,000,003, so the JIT has compiled the fitness function and the engine.
 * - Keeping going (rule 2.2): an evolution stream has no end of its own (its examples end it with
 *   Limits.bySteadyFitness and a generation limit, which the budget replaces); every run ends at the
 *   target, the budget or the time cap, checked after each generation.
 */

import io.jenetics.Alterer;
import io.jenetics.AltererResult;
import io.jenetics.BitChromosome;
import io.jenetics.BitGene;
import io.jenetics.DoubleGene;
import io.jenetics.Gene;
import io.jenetics.Genotype;
import io.jenetics.MeanAlterer;
import io.jenetics.MultiPointCrossover;
import io.jenetics.Mutator;
import io.jenetics.Optimize;
import io.jenetics.PartiallyMatchedCrossover;
import io.jenetics.Phenotype;
import io.jenetics.Selector;
import io.jenetics.SwapMutator;
import io.jenetics.TournamentSelector;
import io.jenetics.engine.Codecs;
import io.jenetics.engine.Engine;
import io.jenetics.engine.EvolutionResult;
import io.jenetics.ext.SimulatedBinaryCrossover;
import io.jenetics.ext.moea.NSGA2Selector;
import io.jenetics.ext.moea.UFTournamentSelector;
import io.jenetics.ext.moea.Vec;
import io.jenetics.ext.moea.VecFactory;
import io.jenetics.util.DoubleRange;
import io.jenetics.util.MSeq;
import io.jenetics.util.RandomRegistry;
import io.jenetics.util.Seq;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Locale;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public final class Bench {

    static final String LIBRARY = "jenetics";
    // the untimed JIT warm-up run of every solver (rule 4.2)
    static final long WARM_UP_SEED = 1_000_003L;
    static final long WARM_UP_EVALUATIONS = 1_000L;

    // ---------------------------------------------------------------------------------------------
    // Fitness functions, identical to problems.py
    // ---------------------------------------------------------------------------------------------

    static double onemax(BitChromosome chromosome) {
        return chromosome.bitCount();
    }

    /** Diagonal conflicts of queens at (i, order[i]): for each diagonal, its queens minus one. */
    static double nqueens(int[] order) {
        int n = order.length;
        int[] left = new int[2 * n - 1];
        int[] right = new int[2 * n - 1];
        for (int i = 0; i < n; i++) {
            left[i + order[i]]++;
            right[n - 1 - i + order[i]]++;
        }
        int conflicts = 0;
        for (int i = 0; i < 2 * n - 1; i++) {
            if (left[i] > 1) conflicts += left[i] - 1;
            if (right[i] > 1) conflicts += right[i] - 1;
        }
        return conflicts;
    }

    /** The shift of Rastrigin and Ackley: s_i = 2 ((37 i + 11) mod 101) / 101 - 1. */
    static double[] shift(int n) {
        double[] s = new double[n];
        for (int i = 0; i < n; i++) {
            s[i] = (2 * ((37 * i + 11) % 101)) / 101.0 - 1.0;
        }
        return s;
    }

    static double rastrigin(double[] x, double[] s) {
        double sum = 0.0;
        for (int i = 0; i < x.length; i++) {
            double z = x[i] - s[i];
            sum += z * z - 10.0 * Math.cos(2.0 * Math.PI * z);
        }
        return 10.0 * x.length + sum;
    }

    static double rosenbrock(double[] x) {
        double sum = 0.0;
        for (int i = 0; i < x.length - 1; i++) {
            double a = x[i + 1] - x[i] * x[i];
            double b = 1.0 - x[i];
            sum += 100.0 * a * a + b * b;
        }
        return sum;
    }

    static double ackley(double[] x, double[] s) {
        int n = x.length;
        double squares = 0.0;
        double cosines = 0.0;
        for (int i = 0; i < n; i++) {
            double z = x[i] - s[i];
            squares += z * z;
            cosines += Math.cos(2.0 * Math.PI * z);
        }
        return -20.0 * Math.exp(-0.2 * Math.sqrt(squares / n)) - Math.exp(cosines / n) + 20.0 + Math.E;
    }

    static double zdtG(double[] x) {
        double sum = 0.0;
        for (int i = 1; i < x.length; i++) sum += x[i];
        return 1.0 + 9.0 * sum / (x.length - 1);
    }

    static double[] zdt1(double[] x) {
        double g = zdtG(x);
        return new double[] {x[0], g * (1.0 - Math.sqrt(x[0] / g))};
    }

    static double[] zdt2(double[] x) {
        double g = zdtG(x);
        double r = x[0] / g;
        return new double[] {x[0], g * (1.0 - r * r)};
    }

    static double[] zdt3(double[] x) {
        double g = zdtG(x);
        double r = x[0] / g;
        return new double[] {x[0], g * (1.0 - Math.sqrt(r) - r * Math.sin(10.0 * Math.PI * x[0]))};
    }

    /** DTLZ1 with 3 objectives (7 variables, k = 5). */
    static double[] dtlz1(double[] x) {
        double sum = 0.0;
        for (int i = 2; i < x.length; i++) {
            double d = x[i] - 0.5;
            sum += d * d - Math.cos(20.0 * Math.PI * d);
        }
        double g = 100.0 * ((x.length - 2) + sum);
        return new double[] {
            0.5 * x[0] * x[1] * (1.0 + g),
            0.5 * x[0] * (1.0 - x[1]) * (1.0 + g),
            0.5 * (1.0 - x[0]) * (1.0 + g),
        };
    }

    /** DTLZ2 with 3 objectives (12 variables, k = 10). */
    static double[] dtlz2(double[] x) {
        double g = 0.0;
        for (int i = 2; i < x.length; i++) {
            double d = x[i] - 0.5;
            g += d * d;
        }
        double a = x[0] * Math.PI / 2.0;
        double b = x[1] * Math.PI / 2.0;
        return new double[] {
            (1.0 + g) * Math.cos(a) * Math.cos(b),
            (1.0 + g) * Math.cos(a) * Math.sin(b),
            (1.0 + g) * Math.sin(a),
        };
    }

    /** A single-objective real-valued problem: its function and bounds. */
    record RealProblem(ToDoubleFunction<double[]> function, double lower, double upper) {}

    static RealProblem realProblem(String name, int size) {
        double[] s = shift(size);
        return switch (name) {
            case "rastrigin" -> new RealProblem(x -> rastrigin(x, s), -5.12, 5.12);
            case "rosenbrock" -> new RealProblem(Bench::rosenbrock, -5.0, 10.0);
            case "ackley" -> new RealProblem(x -> ackley(x, s), -32.768, 32.768);
            default -> null;
        };
    }

    /** A multi-objective problem: its function, variables, objectives and population size. */
    record FrontProblem(Function<double[], double[]> function, int variables, int objectives, int population) {}

    static FrontProblem frontProblem(String name, int size) {
        return switch (name) {
            case "zdt1" -> new FrontProblem(Bench::zdt1, size, 2, 100);
            case "zdt2" -> new FrontProblem(Bench::zdt2, size, 2, 100);
            case "zdt3" -> new FrontProblem(Bench::zdt3, size, 2, 100);
            // size: the number of objectives (3); the variable counts are fixed
            case "dtlz1" -> size == 3 ? new FrontProblem(Bench::dtlz1, 7, 3, 92) : null;
            case "dtlz2" -> size == 3 ? new FrontProblem(Bench::dtlz2, 12, 3, 92) : null;
            default -> null;
        };
    }

    // ---------------------------------------------------------------------------------------------
    // The budget: counts the evaluations, keeps the best value and its solution
    // ---------------------------------------------------------------------------------------------

    static final class Budget {
        final long maxEvaluations;
        final long deadline;
        final boolean minimize;
        final double target;
        long evaluations;
        double best;
        // the best solution: boolean[], int[] or double[]
        Object solution;

        Budget(long maxEvaluations, double maxSeconds, boolean minimize, double target) {
            this.maxEvaluations = maxEvaluations;
            this.deadline = System.nanoTime() + (long) (maxSeconds * 1e9);
            this.minimize = minimize;
            this.target = target;
            this.best = minimize ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
        }

        /** Counts one evaluation; returns whether its value is the best so far. */
        boolean record(double value) {
            evaluations++;
            if (minimize ? value < best : value > best) {
                best = value;
                return true;
            }
            return false;
        }

        boolean reached() {
            return minimize ? best <= target : best >= target;
        }

        /** The end of a run: the target, the budget or the time cap (rule 2.1). */
        boolean done() {
            return reached() || evaluations >= maxEvaluations || System.nanoTime() >= deadline;
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Operators Jenetics doesn't have (for the matched scenarios only)
    // ---------------------------------------------------------------------------------------------

    /**
     * DEAP's mutation in eaSimple: with probability `individualRate` a child is mutated, and then
     * each bit flips with probability `geneRate`; a mutated child is evaluated again. (Jenetics'
     * Mutator selects individuals, chromosomes and genes each with p^(1/3), and its BitGene
     * mutation draws a new random bit instead of flipping it.)
     */
    static final class MatchedBitFlip implements Alterer<BitGene, Double> {
        final double individualRate;
        final double geneRate;

        MatchedBitFlip(double individualRate, double geneRate) {
            this.individualRate = individualRate;
            this.geneRate = geneRate;
        }

        @Override
        public AltererResult<BitGene, Double> alter(Seq<Phenotype<BitGene, Double>> population, long generation) {
            RandomGenerator random = RandomRegistry.random();
            MSeq<Phenotype<BitGene, Double>> result = MSeq.of(population);
            int mutations = 0;
            for (int i = 0; i < result.length(); i++) {
                if (random.nextDouble() >= individualRate) continue;
                BitChromosome chromosome = result.get(i).genotype().chromosome().as(BitChromosome.class);
                int length = chromosome.length();
                BitSet bits = chromosome.toBitSet();
                for (int j = 0; j < length; j++) {
                    if (random.nextDouble() < geneRate) {
                        bits.flip(j);
                        mutations++;
                    }
                }
                result.set(i, Phenotype.of(Genotype.of(BitChromosome.of(bits, length)), generation));
            }
            return new AltererResult<>(result.toISeq(), mutations);
        }
    }

    /**
     * Polynomial mutation (Deb), with probability `rate` per gene and distribution index `eta`, as
     * in jMetal's PolynomialMutation. Jenetics has no polynomial mutation.
     */
    static final class PolynomialMutator<C extends Comparable<? super C>> implements Alterer<DoubleGene, C> {
        final double rate;
        final double eta;

        PolynomialMutator(double rate, double eta) {
            this.rate = rate;
            this.eta = eta;
        }

        @Override
        public AltererResult<DoubleGene, C> alter(Seq<Phenotype<DoubleGene, C>> population, long generation) {
            RandomGenerator random = RandomRegistry.random();
            MSeq<Phenotype<DoubleGene, C>> result = MSeq.of(population);
            int mutations = 0;
            for (int i = 0; i < result.length(); i++) {
                var chromosome = result.get(i).genotype().chromosome();
                MSeq<DoubleGene> genes = null;
                for (int j = 0; j < chromosome.length(); j++) {
                    if (random.nextDouble() >= rate) continue;
                    if (genes == null) genes = MSeq.of(chromosome);
                    DoubleGene gene = genes.get(j);
                    genes.set(j, gene.newInstance(mutate(gene.doubleValue(), gene.min(), gene.max(), random)));
                    mutations++;
                }
                if (genes != null) {
                    result.set(i, Phenotype.of(Genotype.of(chromosome.newInstance(genes.toISeq())), generation));
                }
            }
            return new AltererResult<>(result.toISeq(), mutations);
        }

        double mutate(double y, double lower, double upper, RandomGenerator random) {
            double delta1 = (y - lower) / (upper - lower);
            double delta2 = (upper - y) / (upper - lower);
            double rnd = random.nextDouble();
            double power = 1.0 / (eta + 1.0);
            double deltaq;
            if (rnd <= 0.5) {
                double xy = 1.0 - delta1;
                double value = 2.0 * rnd + (1.0 - 2.0 * rnd) * Math.pow(xy, eta + 1.0);
                deltaq = Math.pow(value, power) - 1.0;
            } else {
                double xy = 1.0 - delta2;
                double value = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * Math.pow(xy, eta + 1.0);
                deltaq = 1.0 - Math.pow(value, power);
            }
            y += deltaq * (upper - lower);
            // a DoubleGene's upper bound is exclusive
            return Math.min(Math.max(y, lower), Math.nextDown(upper));
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Single-objective runs
    // ---------------------------------------------------------------------------------------------

    record Args(String problem, int size, String mode, long seedFrom, long seedTo, long maxEvaluations,
                double maxSeconds) {}

    static void seed(long seed) {
        // the engine's random generator (manual 1.4.2): RandomRegistry's default algorithm, seeded
        RandomRegistry.random(RandomGeneratorFactory.of("L64X256MixRandom").create(seed));
    }

    /** Runs an evolution stream until the budget ends the run; returns the generations. */
    static <G extends Gene<?, G>, C extends Comparable<? super C>> long evolve(
            Engine<G, C> engine, Budget budget, List<EvolutionResult<G, C>> last) {
        long[] generations = {0};
        engine.stream()
            .limit(result -> {
                generations[0] = result.generation();
                if (last != null) {
                    last.clear();
                    last.add(result);
                }
                return !budget.done();
            })
            .forEach(result -> {});
        return generations[0];
    }

    interface SingleSolver {
        long run(Budget budget);
    }

    record Solver(String name, SingleSolver solver) {}

    /** The OneMax fitness through the budget, keeping the best bits. */
    static double countOnes(Genotype<BitGene> genotype, Budget budget) {
        BitChromosome chromosome = genotype.chromosome().as(BitChromosome.class);
        double value = onemax(chromosome);
        if (budget.record(value)) {
            boolean[] bits = new boolean[chromosome.length()];
            for (int i = 0; i < bits.length; i++) bits[i] = chromosome.get(i).bit();
            budget.solution = bits;
        }
        return value;
    }

    static List<Solver> singleSolvers(Args args, boolean[] minimize, double[] target) {
        int size = args.size();
        List<Solver> solvers = new ArrayList<>();
        switch (args.problem()) {
            case "onemax" -> {
                minimize[0] = false;
                target[0] = size;
                if (args.mode().equals("matched")) {
                    // as DEAP's eaSimple: population 300, tournament of 3, two-point crossover,
                    // bit-flip with probability 1 / size on 20% of the children, no elitism.
                    // Differences: Jenetics' crossover picks each individual with probability p
                    // and mates it with a random other one (DEAP: consecutive pairs with
                    // probability 0.5); p = 0.25 gives DEAP's expected number of crossovers
                    // (N / 2 pairs * 0.5). The mutation is the custom MatchedBitFlip above.
                    solvers.add(new Solver("ga", budget -> evolve(
                        Engine.builder((Genotype<BitGene> gt) -> countOnes(gt, budget),
                                Genotype.of(BitChromosome.of(size, 0.5)))
                            .populationSize(300)
                            .offspringFraction(1.0)
                            .offspringSelector(new TournamentSelector<>(3))
                            .alterers(new MultiPointCrossover<>(0.25, 2), new MatchedBitFlip(0.2, 1.0 / size))
                            .maximalPhenotypeAge(Long.MAX_VALUE / 2)
                            .executor(Runnable::run)
                            .build(),
                        budget, null)));
                } else {
                    // The README's "Hello World (Ones counting)" (BitChromosome.of(n, 0.5)) and
                    // jenetics.example/OnesCounting.java: the engine defaults, population 50,
                    // TournamentSelector(3) for offspring and survivors, SinglePointCrossover(0.2),
                    // Mutator(0.15), 60% offspring, maximal age 70. (The manual's "Ones counting"
                    // example, section 6.1, sets population 500, RouletteWheelSelector,
                    // Mutator(0.55) and SinglePointCrossover(0.06); left out, see the page.)
                    solvers.add(new Solver("ga", budget -> evolve(
                        Engine.builder((Genotype<BitGene> gt) -> countOnes(gt, budget),
                                Genotype.of(BitChromosome.of(size, 0.5)))
                            .executor(Runnable::run)
                            .build(),
                        budget, null)));
                }
            }
            case "nqueens" -> {
                minimize[0] = true;
                target[0] = 0;
                // jenetics.example/TravelingSalesman.java, Jenetics' permutation example:
                // Codecs.ofPermutation, SwapMutator(0.15), PartiallyMatchedCrossover(0.15), the
                // other settings the engine defaults (population 50, TournamentSelector(3), 60%
                // offspring, maximal age 70). (The manual's version of it, section 6.5, sets
                // population 500, maximal age 11, SwapMutator(0.2) and
                // PartiallyMatchedCrossover(0.35); left out, see the page.)
                solvers.add(new Solver("ga", budget -> evolve(
                    Engine.builder((int[] order) -> {
                                double value = nqueens(order);
                                if (budget.record(value)) budget.solution = order.clone();
                                return value;
                            }, Codecs.ofPermutation(size))
                        .optimize(Optimize.MINIMUM)
                        .alterers(new SwapMutator<>(0.15), new PartiallyMatchedCrossover<>(0.15))
                        .executor(Runnable::run)
                        .build(),
                    budget, null)));
            }
            case "rastrigin", "rosenbrock", "ackley" -> {
                minimize[0] = true;
                target[0] = 0.01;
                RealProblem problem = realProblem(args.problem(), size);
                // The manual's "Rastrigin function" example (section 6.3), with the settings of its
                // "Real function" example (6.2): Codecs.ofVector, population 500, Mutator(0.03),
                // MeanAlterer(0.6), other settings the defaults
                solvers.add(new Solver("ga", budget -> evolve(
                    Engine.builder((double[] x) -> {
                                double value = problem.function().applyAsDouble(x);
                                if (budget.record(value)) budget.solution = x.clone();
                                return value;
                            }, Codecs.ofVector(new DoubleRange(problem.lower(), problem.upper()), size))
                        .populationSize(500)
                        .optimize(Optimize.MINIMUM)
                        .alterers(new Mutator<>(0.03), new MeanAlterer<>(0.6))
                        .executor(Runnable::run)
                        .build(),
                    budget, null)));
            }
            default -> {
                return null;
            }
        }
        return solvers;
    }

    static void runSingle(Args args, boolean print) {
        boolean[] minimize = new boolean[1];
        double[] target = new double[1];
        List<Solver> solvers = singleSolvers(args, minimize, target);
        if (solvers == null) return;
        for (long seed = args.seedFrom(); seed <= args.seedTo(); seed++) {
            for (Solver solver : solvers) {
                seed(seed);
                Budget budget = new Budget(args.maxEvaluations(), args.maxSeconds(), minimize[0], target[0]);
                long start = System.nanoTime();
                long generations = solver.solver().run(budget);
                double time = (System.nanoTime() - start) / 1e9;
                if (!print) continue;
                System.out.println("{\"library\":\"" + LIBRARY + "\",\"solver\":\"" + solver.name()
                    + "\",\"problem\":\"" + args.problem() + "\",\"size\":" + args.size()
                    + ",\"mode\":\"" + args.mode() + "\",\"seed\":" + seed
                    + ",\"time_s\":" + String.format(Locale.ROOT, "%.6f", time)
                    + ",\"generations\":" + generations + ",\"evaluations\":" + budget.evaluations
                    + ",\"best\":" + number(budget.best) + ",\"target\":" + number(target[0])
                    + ",\"success\":" + budget.reached() + ",\"solution\":" + json(budget.solution) + "}");
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Multi-objective runs
    // ---------------------------------------------------------------------------------------------

    /** The indexes of the non-dominated points (minimized), without duplicates. */
    static List<Integer> nonDominated(List<double[]> points) {
        List<Integer> front = new ArrayList<>();
        for (int i = 0; i < points.size(); i++) {
            double[] p = points.get(i);
            boolean keep = true;
            for (int j = 0; j < points.size() && keep; j++) {
                double[] q = points.get(j);
                if (j == i) continue;
                boolean notWorse = true;
                boolean better = false;
                for (int k = 0; k < p.length; k++) {
                    if (q[k] > p[k]) notWorse = false;
                    if (q[k] < p[k]) better = true;
                }
                if (notWorse && (better || j < i)) keep = false; // dominated, or a duplicate seen before
            }
            if (keep) front.add(i);
        }
        return front;
    }

    static void runFront(Args args, boolean print) {
        FrontProblem problem = frontProblem(args.problem(), args.size());
        if (problem == null) return;
        int n = problem.variables();
        int population = problem.population();
        Optimize[] directions = new Optimize[problem.objectives()];
        Arrays.fill(directions, Optimize.MINIMUM);
        VecFactory<double[]> vectors = VecFactory.ofDoubleVec(directions);

        for (long seed = args.seedFrom(); seed <= args.seedTo(); seed++) {
            for (String solver : new String[] {"nsga2", "moea"}) {
                seed(seed);
                Budget budget = new Budget(args.maxEvaluations(), args.maxSeconds(), true, Double.NEGATIVE_INFINITY);
                // Every objective minimized through VecFactory.ofDoubleVec(MINIMUM, ...), the
                // manual's way to set each objective's direction (3.1.7.4), with the engine's
                // default direction. Not Vec.of with the engine minimizing, as the manual's DTLZ1
                // example does: then the crowding distance is always 0 inside the fronts (a
                // Jenetics bug, see the library's page), and both solvers lose their diversity.
                Function<double[], Vec<double[]>> fitness = x -> {
                    budget.evaluations++;
                    return vectors.newVec(problem.function().apply(x));
                };
                List<EvolutionResult<DoubleGene, Vec<double[]>>> last = new ArrayList<>();
                long start = System.nanoTime();
                long generations;
                Seq<Phenotype<DoubleGene, Vec<double[]>>> result;
                if (solver.equals("nsga2")) {
                    // NSGA-II with the matched settings, built from the Jenetics engine and the
                    // jenetics.ext.moea selectors (manual 3.1.7.2: "the implementation doesn't
                    // exactly follow an established algorithm, like NSGA2"). The engine keeps
                    // "survivors" and adds "offspring" to them. With a population of 2N, N
                    // survivors chosen by NSGA2Selector (rank, then crowding distance) and N
                    // offspring, the population is NSGA-II's parents + children, and the survivors
                    // are NSGA-II's selection of N from them. The parents of the offspring are
                    // drawn from those same N by UFTournamentSelector, the crowded binary
                    // tournament (with unique fitnesses) of Fortin & Parizeau 2013.
                    // Differences: the initial population has 2N random members; Jenetics'
                    // SimulatedBinaryCrossover (η 15) changes one of the two parents only, so
                    // probability 0.45 per individual gives NSGA-II's expected number of
                    // crossovers (N / 2 pairs * 0.9); polynomial mutation (η 20, 1 / n) is the
                    // custom PolynomialMutator above.
                    NSGA2Selector<DoubleGene, Vec<double[]>> survival = NSGA2Selector.ofVec();
                    UFTournamentSelector<DoubleGene, Vec<double[]>> tournament = UFTournamentSelector.ofVec();
                    Selector<DoubleGene, Vec<double[]>> parents =
                        (candidates, count, optimize) ->
                            tournament.select(survival.select(candidates, population, optimize), count, optimize);
                    var engine = Engine.builder(fitness, Codecs.ofVector(new DoubleRange(0.0, 1.0), n))
                        .populationSize(2 * population)
                        .offspringSize(population)
                        .offspringSelector(parents)
                        .survivorsSelector(survival)
                        .alterers(new SimulatedBinaryCrossover<>(0.45, 15.0), new PolynomialMutator<>(1.0 / n, 20.0))
                        .maximalPhenotypeAge(Long.MAX_VALUE / 2)
                        .executor(Runnable::run)
                        .build();
                    generations = evolve(engine, budget, last);
                    // the last population holds N survivors and N children: NSGA-II's final
                    // selection of N from them is the final population
                    var end = last.get(0);
                    result = survival.select(end.population(), population, end.optimize());
                } else {
                    // The manual's "DTLZ1" example (section 6.9), Jenetics' own multi-objective
                    // setup: SimulatedBinaryCrossover(1) (contiguity 2.5, its default),
                    // Mutator(1 / n), TournamentSelector(5) for the offspring, NSGA2Selector for
                    // the survivors, the other settings the defaults (60% offspring, maximal age
                    // 70). The population is the scenario's (the example's is 100), and the
                    // objectives are minimized as above (the example: Vec.of and .minimizing()).
                    // Its final population is the result (the example collects MOEA.toParetoSet
                    // over all generations, an archive that rule 7.2 excludes).
                    var engine = Engine.builder(fitness, Codecs.ofVector(new DoubleRange(0.0, 1.0), n))
                        .populationSize(population)
                        .alterers(new SimulatedBinaryCrossover<>(1), new Mutator<>(1.0 / n))
                        .offspringSelector(new TournamentSelector<>(5))
                        .survivorsSelector(NSGA2Selector.ofVec())
                        .executor(Runnable::run)
                        .build();
                    generations = evolve(engine, budget, last);
                    result = last.get(0).population();
                }
                List<double[]> points = new ArrayList<>();
                List<double[]> solutions = new ArrayList<>();
                for (var phenotype : result) {
                    points.add(phenotype.fitness().data());
                    var chromosome = phenotype.genotype().chromosome();
                    double[] x = new double[chromosome.length()];
                    for (int i = 0; i < x.length; i++) x[i] = chromosome.get(i).doubleValue();
                    solutions.add(x);
                }
                List<Integer> front = nonDominated(points);
                double time = (System.nanoTime() - start) / 1e9;
                if (!print) continue;
                StringBuilder frontJson = new StringBuilder("[");
                StringBuilder solutionsJson = new StringBuilder("[");
                for (int i = 0; i < front.size(); i++) {
                    if (i > 0) {
                        frontJson.append(',');
                        solutionsJson.append(',');
                    }
                    frontJson.append(json(points.get(front.get(i))));
                    solutionsJson.append(json(solutions.get(front.get(i))));
                }
                System.out.println("{\"library\":\"" + LIBRARY + "\",\"solver\":\"" + solver
                    + "\",\"problem\":\"" + args.problem() + "\",\"size\":" + args.size()
                    + ",\"mode\":\"" + args.mode() + "\",\"seed\":" + seed
                    + ",\"time_s\":" + String.format(Locale.ROOT, "%.6f", time)
                    + ",\"generations\":" + generations + ",\"evaluations\":" + budget.evaluations
                    + ",\"front\":" + frontJson.append(']') + ",\"solutions\":" + solutionsJson.append(']') + "}");
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // `values`: the adapter's fitness functions at given solutions (rule 1.2)
    // ---------------------------------------------------------------------------------------------

    static double[] parse(String line) {
        String body = line.trim();
        body = body.substring(1, body.length() - 1).trim();
        if (body.isEmpty()) return new double[0];
        String[] parts = body.split(",");
        double[] x = new double[parts.length];
        for (int i = 0; i < parts.length; i++) {
            String part = parts[i].trim();
            x[i] = part.equals("true") ? 1 : part.equals("false") ? 0 : Double.parseDouble(part);
        }
        return x;
    }

    static void values(String problem, int size) throws IOException {
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        FrontProblem front = frontProblem(problem, size);
        RealProblem real = realProblem(problem, size);
        StringBuilder out = new StringBuilder();
        String line;
        while ((line = in.readLine()) != null) {
            if (line.isBlank()) continue;
            double[] x = parse(line);
            if (front != null) {
                out.append(json(front.function().apply(x)));
            } else if (real != null) {
                out.append(number(real.function().applyAsDouble(x)));
            } else if (problem.equals("onemax")) {
                BitSet bits = new BitSet(x.length);
                for (int i = 0; i < x.length; i++) bits.set(i, x[i] != 0);
                out.append(number(onemax(BitChromosome.of(bits, x.length))));
            } else if (problem.equals("nqueens")) {
                int[] order = new int[x.length];
                for (int i = 0; i < x.length; i++) order[i] = (int) x[i];
                out.append(number(nqueens(order)));
            } else {
                throw new IllegalArgumentException("unknown problem " + problem);
            }
            out.append('\n');
        }
        System.out.print(out);
    }

    // ---------------------------------------------------------------------------------------------

    static String number(double value) {
        if (value == Math.rint(value) && Math.abs(value) < 1e15) return Long.toString((long) value);
        return Double.toString(value);
    }

    static String json(Object solution) {
        StringBuilder text = new StringBuilder("[");
        if (solution instanceof boolean[] bits) {
            for (int i = 0; i < bits.length; i++) text.append(i > 0 ? "," : "").append(bits[i] ? 1 : 0);
        } else if (solution instanceof int[] order) {
            for (int i = 0; i < order.length; i++) text.append(i > 0 ? "," : "").append(order[i]);
        } else if (solution instanceof double[] x) {
            for (int i = 0; i < x.length; i++) text.append(i > 0 ? "," : "").append(x[i]);
        }
        return text.append(']').toString();
    }

    static void run(Args args, boolean print) {
        if (frontProblem(args.problem(), args.size()) != null) {
            runFront(args, print);
        } else {
            runSingle(args, print);
        }
    }

    static String version() {
        String version = Engine.class.getPackage().getImplementationVersion();
        return version != null ? version : "unknown";
    }

    public static void main(String[] argv) throws IOException {
        if (argv.length == 1 && argv[0].equals("--version")) {
            System.out.println(version());
            return;
        }
        if (argv.length == 3 && argv[0].equals("values")) {
            values(argv[1], Integer.parseInt(argv[2]));
            return;
        }
        if (argv.length != 7) {
            System.err.println("usage: Bench <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>");
            System.exit(2);
        }
        Args args = new Args(argv[0], Integer.parseInt(argv[1]), argv[2], Long.parseLong(argv[3]),
            Long.parseLong(argv[4]), Long.parseLong(argv[5]), Double.parseDouble(argv[6]));
        // JIT warm-up (rule 4.2): every solver once on the same problem, untimed and unprinted
        run(new Args(args.problem(), args.size(), args.mode(), WARM_UP_SEED, WARM_UP_SEED,
            WARM_UP_EVALUATIONS, args.maxSeconds()), false);
        run(args, true);
    }
}
