/*
 * Benchmark adapter for Jenetics (io.jenetics:jenetics and jenetics.ext).
 *
 * Usage: java -cp ... Bench <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
 *        java -cp ... Bench --version
 * Prints one JSON line per solver per seed, see ../../README.md for the fields.
 *
 * Every run is single-threaded: the engine gets `.executor(Runnable::run)`, which also evaluates
 * the fitness on the calling thread (Jenetics uses the ForkJoinPool common pool by default).
 * Before the timed runs of a problem, every solver runs once untimed with a small budget, so the
 * JIT has compiled the fitness function and the engine.
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
import io.jenetics.SwapMutator;
import io.jenetics.TournamentSelector;
import io.jenetics.engine.Codecs;
import io.jenetics.engine.Engine;
import io.jenetics.engine.EvolutionResult;
import io.jenetics.ext.SimulatedBinaryCrossover;
import io.jenetics.ext.moea.MOEA;
import io.jenetics.ext.moea.NSGA2Selector;
import io.jenetics.ext.moea.UFTournamentSelector;
import io.jenetics.ext.moea.Vec;
import io.jenetics.ext.moea.VecFactory;
import io.jenetics.util.DoubleRange;
import io.jenetics.util.ISeq;
import io.jenetics.util.MSeq;
import io.jenetics.util.RandomRegistry;
import io.jenetics.util.Seq;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

public final class Bench {

    static final String LIBRARY = "jenetics";

    // ---------------------------------------------------------------------------------------------
    // Fitness functions, identical to the ones in the other adapters
    // ---------------------------------------------------------------------------------------------

    /** Counts the evaluations and keeps the best value found, stops at the budget. */
    static final class Budget {
        final long maxEvaluations;
        final long deadline;
        final boolean minimize;
        final double target;
        long evaluations;
        double best;

        Budget(long maxEvaluations, double maxSeconds, boolean minimize, double target) {
            this.maxEvaluations = maxEvaluations;
            this.deadline = System.nanoTime() + (long) (maxSeconds * 1e9);
            this.minimize = minimize;
            this.target = target;
            this.best = minimize ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
        }

        double record(double value) {
            evaluations++;
            if (minimize ? value < best : value > best) {
                best = value;
            }
            return value;
        }

        boolean reached() {
            return minimize ? best <= target : best >= target;
        }

        boolean done() {
            return reached() || evaluations >= maxEvaluations || System.nanoTime() >= deadline;
        }
    }

    static double onemax(BitChromosome chromosome) {
        return chromosome.bitCount();
    }

    /** Number of diagonal conflicts, O(n) (same as NQueens.fitness in adapters/pymoo/bench.py). */
    static double nqueens(int[] individual) {
        int size = individual.length;
        int[] left = new int[2 * size - 1];
        int[] right = new int[2 * size - 1];
        for (int i = 0; i < size; i++) {
            left[i + individual[i]]++;
            right[size - 1 - i + individual[i]]++;
        }
        int conflicts = 0;
        for (int i = 0; i < 2 * size - 1; i++) {
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

    /** DTLZ1 with 3 objectives and 7 variables (k = 5). */
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

    /** DTLZ2 with 3 objectives and 12 variables (k = 10). */
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

    // (function, variables, objectives, population size, Das-Dennis divisions)
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
    // Operators Jenetics doesn't have
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
    // Runs
    // ---------------------------------------------------------------------------------------------

    record Args(String problem, int size, String mode, long seedFrom, long seedTo, long maxEvaluations,
                double maxSeconds) {}

    static void seed(long seed) {
        // the engine's random generator, per thread (the runs are on one thread)
        RandomRegistry.random(() -> RandomGeneratorFactory.of("L64X256MixRandom").create(seed));
    }

    /** Runs the evolution stream until the budget is used up; returns the generations. */
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

    static String number(double value) {
        if (value == Math.rint(value) && Math.abs(value) < 1e15) return Long.toString((long) value);
        return Double.toString(value);
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
                        Engine.builder(
                                (Genotype<BitGene> gt) -> budget.record(onemax(gt.chromosome().as(BitChromosome.class))),
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
                    // the README's "Hello World" and jenetics.example/OnesCounting.java: the engine
                    // defaults (population 50, tournament of 3, SinglePointCrossover(0.2),
                    // Mutator(0.15), 60% offspring, maximal age 70), with random bits (p = 0.5)
                    solvers.add(new Solver("ga", budget -> evolve(
                        Engine.builder(
                                (Genotype<BitGene> gt) -> budget.record(onemax(gt.chromosome().as(BitChromosome.class))),
                                Genotype.of(BitChromosome.of(size, 0.5)))
                            .executor(Runnable::run)
                            .build(),
                        budget, null)));
                }
            }
            case "nqueens" -> {
                minimize[0] = true;
                target[0] = 0;
                // the permutation codec with the settings of jenetics.example/TravelingSalesman.java:
                // engine defaults, SwapMutator(0.15) and PartiallyMatchedCrossover(0.15)
                solvers.add(new Solver("ga", budget -> evolve(
                    Engine.builder((int[] p) -> budget.record(nqueens(p)), Codecs.ofPermutation(size))
                        .optimize(Optimize.MINIMUM)
                        .alterers(new SwapMutator<>(0.15), new PartiallyMatchedCrossover<>(0.15))
                        .executor(Runnable::run)
                        .build(),
                    budget, null)));
            }
            case "rastrigin", "rosenbrock", "ackley" -> {
                minimize[0] = true;
                target[0] = 0.01;
                double[] s = shift(size);
                double lower;
                double upper;
                ToDoubleFunction<double[]> function;
                switch (args.problem()) {
                    case "rastrigin" -> { lower = -5.12; upper = 5.12; function = x -> rastrigin(x, s); }
                    case "rosenbrock" -> { lower = -5.0; upper = 10.0; function = Bench::rosenbrock; }
                    default -> { lower = -32.768; upper = 32.768; function = x -> ackley(x, s); }
                }
                // jenetics.example/RealFunction.java (the real-valued example of the manual):
                // population 500, Mutator(0.03), MeanAlterer(0.6), other settings the defaults
                solvers.add(new Solver("ga", budget -> evolve(
                    Engine.builder((double[] x) -> budget.record(function.applyAsDouble(x)),
                            Codecs.ofVector(new DoubleRange(lower, upper), size))
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
                    + ",\"time_s\":" + String.format(java.util.Locale.ROOT, "%.6f", time)
                    + ",\"generations\":" + generations + ",\"evaluations\":" + budget.evaluations
                    + ",\"best\":" + number(budget.best) + ",\"target\":" + number(target[0])
                    + ",\"success\":" + budget.reached() + "}");
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Multi-objective
    // ---------------------------------------------------------------------------------------------

    /** The non-dominated objective vectors (minimized), without duplicates. */
    static List<double[]> nonDominated(List<double[]> points) {
        List<double[]> front = new ArrayList<>();
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
            if (keep) front.add(p);
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
                Function<double[], Vec<double[]>> fitness = x -> {
                    budget.evaluations++;
                    return vectors.newVec(problem.function().apply(x));
                };
                List<EvolutionResult<DoubleGene, Vec<double[]>>> last = new ArrayList<>();
                List<double[]> points = new ArrayList<>();
                long start = System.nanoTime();
                long generations;
                if (solver.equals("nsga2")) {
                    // NSGA-II with the matched settings, as close as the Jenetics engine allows
                    // (jenetics.ext.moea): the engine keeps "survivors" and adds "offspring".
                    // With a population of 2N, N survivors chosen by NSGA2Selector (rank and
                    // crowding distance) and N offspring, the survivors are NSGA-II's selection of
                    // N from parents + children. Differences: the initial population has 2N
                    // members; parents come from all 2N (not the N survivors), by UFTournamentSelector
                    // (the crowded binary tournament of Fortin & Parizeau 2013); Jenetics'
                    // SimulatedBinaryCrossover (η 15) changes one of the two children only, so
                    // probability 0.45 per individual gives NSGA-II's expected number of
                    // crossovers (N / 2 pairs * 0.9); polynomial mutation (η 20, 1 / n) is the
                    // custom PolynomialMutator above.
                    var engine = Engine.builder(fitness, Codecs.ofVector(new DoubleRange(0.0, 1.0), n))
                        .populationSize(2 * population)
                        .offspringSize(population)
                        .offspringSelector(UFTournamentSelector.ofVec())
                        .survivorsSelector(NSGA2Selector.ofVec())
                        .alterers(new SimulatedBinaryCrossover<>(0.45, 15.0), new PolynomialMutator<>(1.0 / n, 20.0))
                        .maximalPhenotypeAge(Long.MAX_VALUE / 2)
                        .executor(Runnable::run)
                        .build();
                    generations = evolve(engine, budget, last);
                    // the last population holds N survivors and N children: NSGA-II's final
                    // selection of N from them, as the result
                    var result = last.get(0);
                    var selected = NSGA2Selector.<DoubleGene, double[], Vec<double[]>>ofVec()
                        .select(result.population(), population, result.optimize());
                    for (var phenotype : selected) points.add(phenotype.fitness().data());
                } else {
                    // the documented MOEA setup (javadoc of io.jenetics.ext.moea.MOEA and the
                    // manual): Mutator(0.1), MeanAlterer, TournamentSelector(2) for the offspring,
                    // UFTournamentSelector for the survivors, the Pareto set collected with
                    // MOEA.toParetoSet(); population N instead of the default 50
                    var engine = Engine.builder(fitness, Codecs.ofVector(new DoubleRange(0.0, 1.0), n))
                        .populationSize(population)
                        .alterers(new Mutator<>(0.1), new MeanAlterer<>())
                        .offspringSelector(new TournamentSelector<>(2))
                        .survivorsSelector(UFTournamentSelector.ofVec())
                        .executor(Runnable::run)
                        .build();
                    long[] count = {0};
                    ISeq<Phenotype<DoubleGene, Vec<double[]>>> set = engine.stream()
                        .limit(result -> {
                            count[0] = result.generation();
                            return !budget.done();
                        })
                        .collect(MOEA.toParetoSet());
                    generations = count[0];
                    for (var phenotype : set) points.add(phenotype.fitness().data());
                }
                List<double[]> front = nonDominated(points);
                double time = (System.nanoTime() - start) / 1e9;
                if (!print) continue;
                StringBuilder json = new StringBuilder();
                json.append("{\"library\":\"").append(LIBRARY).append("\",\"solver\":\"").append(solver)
                    .append("\",\"problem\":\"").append(args.problem()).append("\",\"size\":").append(args.size())
                    .append(",\"mode\":\"").append(args.mode()).append("\",\"seed\":").append(seed)
                    .append(",\"time_s\":").append(String.format(java.util.Locale.ROOT, "%.6f", time))
                    .append(",\"generations\":").append(generations)
                    .append(",\"evaluations\":").append(budget.evaluations).append(",\"front\":[");
                for (int i = 0; i < front.size(); i++) {
                    if (i > 0) json.append(',');
                    json.append('[');
                    double[] point = front.get(i);
                    for (int k = 0; k < point.length; k++) {
                        if (k > 0) json.append(',');
                        json.append(point[k]);
                    }
                    json.append(']');
                }
                System.out.println(json.append("]}"));
            }
        }
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

    public static void main(String[] argv) {
        if (argv.length == 1 && argv[0].equals("--version")) {
            System.out.println(version());
            return;
        }
        if (argv.length != 7) {
            System.err.println("usage: Bench <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>");
            System.exit(2);
        }
        Args args = new Args(argv[0], Integer.parseInt(argv[1]), argv[2], Long.parseLong(argv[3]),
            Long.parseLong(argv[4]), Long.parseLong(argv[5]), Double.parseDouble(argv[6]));
        // JIT warm-up: every solver once on the same problem, untimed and unprinted
        run(new Args(args.problem(), args.size(), args.mode(), 1_000_003L, 1_000_003L,
            Math.min(args.maxEvaluations(), 10_000L), Math.min(args.maxSeconds(), 2.0)), false);
        run(args, true);
    }
}
