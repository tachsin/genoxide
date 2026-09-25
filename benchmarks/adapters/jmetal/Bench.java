/*
 * Benchmark adapter for jMetal (org.uma.jmetal: jmetal-core, jmetal-algorithm, jmetal-component),
 * following docs/benchmarks/rules.md. Its page, docs/benchmarks/libraries/jmetal.md, gives every
 * method, where jMetal presents it, what was left out and why, and the separate test runs.
 *
 * Usage: java -cp ... Bench <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
 *        java -cp ... Bench values <problem> <size>     (one JSON solution per line on stdin)
 *        java -cp ... Bench --version
 * Prints one JSON line per solver per seed, see ../../README.md for the fields.
 *
 * - The problems are jMetal problems (Abstract*Problem) whose evaluate() calls the fitness
 *   functions below and counts the evaluations (rule 3), not jMetal's own problem classes. A
 *   single-objective run ends inside evaluate(), at the target, the budget or the time cap (rule
 *   2.1), by an exception the adapter catches; a multi-objective run ends through a Termination (or
 *   SPEA2's stopping condition) checked after each generation.
 * - One thread (rule 4.3): jMetal evaluates sequentially (SequentialEvaluation and
 *   SequentialSolutionListEvaluator, its defaults); run.sh runs the JVM with the serial garbage
 *   collector and with -Xbatch, so the JIT compiles on the calling thread's time.
 * - Seeds (rule 5.2): JMetalRandom's seed, before every run. Two parts of jMetal don't use it: its
 *   CMA-ES draws its samples from its own java.util.Random, seeded with System.currentTimeMillis(),
 *   which it doesn't let a user set (the adapter replaces it by a seeded one, see cmaes() below),
 *   and its initial permutations are shuffled by Collections.shuffle's own generator (the adapter
 *   shuffles them with JMetalRandom, see NQueensProblem.createSolution()).
 * - Time (rule 4.2): before the timed runs, every solver runs once untimed, with 1,000 evaluations
 *   and the seed 1,000,003, so the JIT has compiled the fitness function and the algorithm.
 */

import org.uma.jmetal.algorithm.multiobjective.spea2.SPEA2;
import org.uma.jmetal.algorithm.singleobjective.differentialevolution.DifferentialEvolution;
import org.uma.jmetal.algorithm.singleobjective.evolutionstrategy.CovarianceMatrixAdaptationEvolutionStrategy;
import org.uma.jmetal.algorithm.singleobjective.evolutionstrategy.EvolutionStrategyBuilder;
import org.uma.jmetal.component.algorithm.EvolutionaryAlgorithm;
import org.uma.jmetal.component.algorithm.multiobjective.MOEADBuilder;
import org.uma.jmetal.component.algorithm.multiobjective.NSGAIIBuilder;
import org.uma.jmetal.component.algorithm.multiobjective.NSGAIIIBuilder;
import org.uma.jmetal.component.algorithm.multiobjective.SMSEMOABuilder;
import org.uma.jmetal.component.algorithm.singleobjective.GeneticAlgorithmBuilder;
import org.uma.jmetal.component.catalogue.common.termination.Termination;
import org.uma.jmetal.component.catalogue.ea.selection.impl.NaryTournamentSelection;
import org.uma.jmetal.operator.crossover.impl.DifferentialEvolutionCrossover;
import org.uma.jmetal.operator.crossover.impl.PMXCrossover;
import org.uma.jmetal.operator.crossover.impl.SBXCrossover;
import org.uma.jmetal.operator.crossover.impl.SinglePointCrossover;
import org.uma.jmetal.operator.mutation.impl.BitFlipMutation;
import org.uma.jmetal.operator.mutation.impl.PermutationSwapMutation;
import org.uma.jmetal.operator.mutation.impl.PolynomialMutation;
import org.uma.jmetal.operator.selection.impl.BinaryTournamentSelection;
import org.uma.jmetal.operator.selection.impl.DifferentialEvolutionSelection;
import org.uma.jmetal.problem.binaryproblem.impl.AbstractBinaryProblem;
import org.uma.jmetal.problem.doubleproblem.impl.AbstractDoubleProblem;
import org.uma.jmetal.problem.permutationproblem.impl.AbstractIntegerPermutationProblem;
import org.uma.jmetal.solution.Solution;
import org.uma.jmetal.solution.binarysolution.BinarySolution;
import org.uma.jmetal.solution.doublesolution.DoubleSolution;
import org.uma.jmetal.solution.permutationsolution.PermutationSolution;
import org.uma.jmetal.solution.permutationsolution.impl.IntegerPermutationSolution;
import org.uma.jmetal.util.aggregationfunction.impl.PenaltyBoundaryIntersection;
import org.uma.jmetal.util.aggregationfunction.impl.Tschebyscheff;
import org.uma.jmetal.util.comparator.ObjectiveComparator;
import org.uma.jmetal.util.evaluator.impl.SequentialSolutionListEvaluator;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;
import org.uma.jmetal.util.referencepoint.ReferencePointGenerator;
import org.uma.jmetal.util.sequencegenerator.impl.RandomPermutationCycle;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.logging.Level;
import java.util.logging.LogManager;

public final class Bench {

    static final String LIBRARY = "jmetal";
    // the untimed JIT warm-up run of every solver (rule 4.2)
    static final long WARM_UP_SEED = 1_000_003L;
    static final long WARM_UP_EVALUATIONS = 1_000L;

    // ---------------------------------------------------------------------------------------------
    // Fitness functions, identical to problems.py
    // ---------------------------------------------------------------------------------------------

    static double onemax(BitSet bits) {
        return bits.cardinality();
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
    record RealFunction(ToDoubleFunction<double[]> function, double lower, double upper) {}

    static RealFunction realFunction(String name, int size) {
        double[] s = shift(size);
        return switch (name) {
            case "rastrigin" -> new RealFunction(x -> rastrigin(x, s), -5.12, 5.12);
            case "rosenbrock" -> new RealFunction(Bench::rosenbrock, -5.0, 10.0);
            case "ackley" -> new RealFunction(x -> ackley(x, s), -32.768, 32.768);
            default -> null;
        };
    }

    // (function, variables, objectives, population size, Das-Dennis divisions)
    record FrontFunction(Function<double[], double[]> function, int variables, int objectives, int population,
                         int divisions) {}

    static FrontFunction frontFunction(String name, int size) {
        return switch (name) {
            case "zdt1" -> new FrontFunction(Bench::zdt1, size, 2, 100, 99);
            case "zdt2" -> new FrontFunction(Bench::zdt2, size, 2, 100, 99);
            case "zdt3" -> new FrontFunction(Bench::zdt3, size, 2, 100, 99);
            // size: the number of objectives (3); the variable counts are fixed
            case "dtlz1" -> size == 3 ? new FrontFunction(Bench::dtlz1, 7, 3, 92, 12) : null;
            case "dtlz2" -> size == 3 ? new FrontFunction(Bench::dtlz2, 12, 3, 92, 12) : null;
            default -> null;
        };
    }

    // ---------------------------------------------------------------------------------------------
    // The budget: counts the evaluations, keeps the best value and its solution
    // ---------------------------------------------------------------------------------------------

    /** Thrown by a single-objective problem to end a run at the target, the budget or the time cap. */
    static final class Stop extends RuntimeException {
        Stop() {
            super(null, null, false, false);
        }
    }

    static final Stop STOP = new Stop();

    static final class Budget {
        final long maxEvaluations;
        final long deadline;
        final boolean minimize;
        final double target;
        long evaluations;
        double best;
        boolean timeUp;
        // the best solution: boolean[], int[] or double[]
        Object solution;
        // evaluated solutions outside the problem's bounds (rule 2.4)
        long outside;

        Budget(long maxEvaluations, double maxSeconds, boolean minimize, double target) {
            this.maxEvaluations = maxEvaluations;
            this.deadline = System.nanoTime() + (long) (maxSeconds * 1e9);
            this.minimize = minimize;
            this.target = target;
            this.best = minimize ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
        }

        /** Single-objective: called before an evaluation, ends the run at the budget or the cap. */
        void before() {
            if (evaluations >= maxEvaluations) throw STOP;
            // the clock every 64 evaluations
            if ((evaluations & 63) == 0 && System.nanoTime() >= deadline) {
                timeUp = true;
            }
            if (timeUp) throw STOP;
        }

        boolean better(double value) {
            return minimize ? value < best : value > best;
        }

        /** Single-objective: counts an evaluation (its solution is kept first), ends at the target. */
        void record(double value) {
            evaluations++;
            if (better(value)) best = value;
            if (reached()) throw STOP;
        }

        boolean reached() {
            return minimize ? best <= target : best >= target;
        }

        boolean done() {
            return reached() || evaluations >= maxEvaluations || timeUp || System.nanoTime() >= deadline;
        }

        /** Multi-objective: checked between generations. */
        boolean exhausted() {
            return evaluations >= maxEvaluations || System.nanoTime() >= deadline;
        }

        /** Counts a solution the library evaluates outside [lower, upper] (rule 2.4). */
        void bounds(double[] x, double lower, double upper) {
            for (double v : x) {
                if (!(v >= lower && v <= upper)) {
                    outside++;
                    return;
                }
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // The problems as jMetal problems (jMetal minimizes)
    // ---------------------------------------------------------------------------------------------

    static double[] values(DoubleSolution solution) {
        List<Double> variables = solution.variables();
        double[] x = new double[variables.size()];
        for (int i = 0; i < x.length; i++) x[i] = variables.get(i);
        return x;
    }

    static final class OneMaxProblem extends AbstractBinaryProblem {
        final int bits;
        final Budget budget;

        OneMaxProblem(int bits, Budget budget) {
            this.bits = bits;
            this.budget = budget;
        }

        @Override public List<Integer> numberOfBitsPerVariable() { return List.of(bits); }
        @Override public int numberOfVariables() { return 1; }
        @Override public int numberOfObjectives() { return 1; }
        @Override public int numberOfConstraints() { return 0; }
        @Override public String name() { return "OneMax"; }

        @Override
        public BinarySolution evaluate(BinarySolution solution) {
            budget.before();
            BitSet variable = solution.variables().get(0);
            double ones = onemax(variable);
            solution.objectives()[0] = -ones; // maximize the number of ones
            if (budget.better(ones)) {
                boolean[] copy = new boolean[bits];
                for (int i = 0; i < bits; i++) copy[i] = variable.get(i);
                budget.solution = copy;
            }
            budget.record(ones);
            return solution;
        }
    }

    static final class NQueensProblem extends AbstractIntegerPermutationProblem {
        final int size;
        final Budget budget;

        NQueensProblem(int size, Budget budget) {
            this.size = size;
            this.budget = budget;
        }

        @Override public int numberOfVariables() { return size; }
        @Override public int numberOfObjectives() { return 1; }
        @Override public int numberOfConstraints() { return 0; }
        @Override public String name() { return "NQueens"; }

        /**
         * A random permutation, as jMetal's own createSolution() makes, but drawn with JMetalRandom:
         * jMetal's IntegerPermutationSolution shuffles with Collections.shuffle(list), whose
         * generator ignores JMetalRandom's seed, so the same seed wouldn't give the same run (rule
         * 5.2). The same uniform shuffle (Fisher-Yates), with the seeded generator.
         */
        @Override
        public PermutationSolution<Integer> createSolution() {
            var solution = new IntegerPermutationSolution(size, 1, 0);
            List<Integer> order = new ArrayList<>(size);
            for (int i = 0; i < size; i++) order.add(i);
            for (int i = size - 1; i > 0; i--) {
                Collections.swap(order, i, JMetalRandom.getInstance().nextInt(0, i));
            }
            for (int i = 0; i < size; i++) solution.variables().set(i, order.get(i));
            return solution;
        }

        @Override
        public PermutationSolution<Integer> evaluate(PermutationSolution<Integer> solution) {
            budget.before();
            List<Integer> variables = solution.variables();
            int[] order = new int[size];
            for (int i = 0; i < size; i++) order[i] = variables.get(i);
            double conflicts = nqueens(order);
            solution.objectives()[0] = conflicts;
            if (budget.better(conflicts)) budget.solution = order;
            budget.record(conflicts);
            return solution;
        }
    }

    static final class RealProblem extends AbstractDoubleProblem {
        final ToDoubleFunction<double[]> function;
        final Budget budget;
        final double lower;
        final double upper;

        RealProblem(String name, int size, RealFunction real, Budget budget) {
            this.function = real.function();
            this.budget = budget;
            this.lower = real.lower();
            this.upper = real.upper();
            numberOfObjectives(1);
            numberOfConstraints(0);
            name(name);
            variableBounds(Collections.nCopies(size, real.lower()), Collections.nCopies(size, real.upper()));
        }

        @Override
        public DoubleSolution evaluate(DoubleSolution solution) {
            budget.before();
            double[] x = values(solution);
            budget.bounds(x, lower, upper);
            double value = function.applyAsDouble(x);
            solution.objectives()[0] = value;
            if (budget.better(value)) budget.solution = x;
            budget.record(value);
            return solution;
        }
    }

    static final class FrontProblem extends AbstractDoubleProblem {
        final Function<double[], double[]> function;
        final Budget budget;

        FrontProblem(String name, FrontFunction front, Budget budget) {
            this.function = front.function();
            this.budget = budget;
            numberOfObjectives(front.objectives());
            numberOfConstraints(0);
            name(name);
            variableBounds(Collections.nCopies(front.variables(), 0.0), Collections.nCopies(front.variables(), 1.0));
        }

        @Override
        public DoubleSolution evaluate(DoubleSolution solution) {
            budget.evaluations++;
            double[] x = values(solution);
            budget.bounds(x, 0.0, 1.0);
            double[] f = function.apply(x);
            System.arraycopy(f, 0, solution.objectives(), 0, f.length);
            return solution;
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Single-objective runs
    // ---------------------------------------------------------------------------------------------

    record Args(String problem, int size, String mode, long seedFrom, long seedTo, long maxEvaluations,
                double maxSeconds) {}

    /** A solver: runs until the budget ends it; returns the generations. */
    interface SingleSolver {
        long run(Budget budget, long seed);
    }

    record Solver(String name, SingleSolver solver) {}

    /** jMetal's sequential evaluator, counting the generations. */
    static final class CountingEvaluator<S extends Solution<?>> extends SequentialSolutionListEvaluator<S> {
        long calls;

        @Override
        public List<S> evaluate(List<S> solutionList, org.uma.jmetal.problem.Problem<S> problem) {
            calls++;
            return super.evaluate(solutionList, problem);
        }
    }

    /** Runs a component algorithm; the termination only counts the generations (Stop ends it). */
    static <S extends Solution<?>> long runComponent(Function<Termination, EvolutionaryAlgorithm<S>> build) {
        long[] generations = {0};
        EvolutionaryAlgorithm<S> algorithm = build.apply(status -> {
            generations[0]++;
            return false;
        });
        try {
            algorithm.run();
        } catch (Stop stop) {
            // the target, the budget or the time cap
        }
        return generations[0];
    }

    /** Runs a classic (jmetal-algorithm) algorithm whose own evaluation limit is out of reach. */
    static void runClassic(Runnable run) {
        try {
            run.run();
        } catch (Stop stop) {
            // the target, the budget or the time cap
        }
    }

    /**
     * jMetal's CMA-ES with its defaults (λ 10, σ 0.3). Its maximum of evaluations, a budget, is
     * lifted (rule 2.2). It also ends an attempt when its covariance matrix degenerates: when the
     * eigendecomposition fails its check (checkEigenCorrectness sets the evaluations to the
     * maximum), or when CMAESUtils.tql2 throws ArrayIndexOutOfBoundsException (NaN in the matrix).
     * jMetal has no restart mechanism, so the adapter then starts it again from a new random point,
     * with JMetalRandom seeded with seed * 1000 + restart (rule 2.2); the budget keeps the best and
     * counts every evaluation. Bounds (rule 2.4): jMetal clips every sample to the bounds
     * (Bounds.restrict in sampleSolution).
     * jMetal's CMA-ES has a bug: once it has converged, its σ grows without bound, every sample
     * lands on the bounds and the run stalls without ending, often for 200,000 evaluations before
     * tql2 throws. The adapter doesn't restart it then (rule 8.4), so the results show the bug.
     */
    static long cmaes(Budget budget, long seed, RealProblem problem, int size, double lower, double upper) {
        long generations = 0;
        for (int restart = 0; ; restart++) {
            if (restart > 0) JMetalRandom.getInstance().setSeed(seed * 1000 + restart);
            // jMetal starts the mean at a random point in [0, 1)^n whatever the bounds, which is next
            // to the shifted optimum of these problems (the shift lies in [-1, 1]); the start here is
            // a random point within the bounds, like the other libraries' CMA-ES
            double[] start = new double[size];
            for (int i = 0; i < size; i++) start[i] = JMetalRandom.getInstance().nextDouble(lower, upper);
            var algorithm = new CovarianceMatrixAdaptationEvolutionStrategy.Builder(problem)
                .setMaxEvaluations(Integer.MAX_VALUE)
                .setTypicalX(start)
                .build();
            // its sampling generator is `new Random(System.currentTimeMillis())`, which a user can't
            // set: replaced by one seeded from JMetalRandom, so the same seed gives the same run
            seedCmaes(algorithm, JMetalRandom.getInstance().nextInt(0, Integer.MAX_VALUE - 1));
            long before = budget.evaluations;
            try {
                algorithm.run();
            } catch (Stop stop) {
                return generations + (budget.evaluations - before) / algorithm.getLambda();
            } catch (ArrayIndexOutOfBoundsException error) {
                // NaN in the covariance matrix: the attempt ends, as when it ends by itself
            }
            generations += (budget.evaluations - before) / algorithm.getLambda();
            if (budget.done()) return generations;
        }
    }

    static void seedCmaes(CovarianceMatrixAdaptationEvolutionStrategy algorithm, long seed) {
        try {
            Field field = CovarianceMatrixAdaptationEvolutionStrategy.class.getDeclaredField("rand");
            field.setAccessible(true);
            field.set(algorithm, new Random(seed));
        } catch (ReflectiveOperationException error) {
            throw new IllegalStateException(error);
        }
    }

    static List<Solver> singleSolvers(Args args, boolean[] minimize, double[] target) {
        int size = args.size();
        List<Solver> solvers = new ArrayList<>();
        switch (args.problem()) {
            case "onemax" -> {
                minimize[0] = false;
                target[0] = size;
                if (args.mode().equals("matched")) {
                    // as close to DEAP's eaSimple as jMetal allows: population 300, tournament of
                    // 3, crossover with probability 0.5, generational replacement without elitism
                    // (a Replacement lambda: the children replace the parents).
                    // Differences: jMetal's binary crossovers are single-point, HUX and uniform
                    // (its TwoPointCrossover is for numbers), so single-point; its variation mutates
                    // every child, so bit-flip with 0.2 / size per bit (DEAP: 1 / size on 20% of
                    // the children, the same expected number of flips); every child is evaluated,
                    // changed or not.
                    solvers.add(new Solver("ga", (budget, seed) -> runComponent(termination ->
                        new GeneticAlgorithmBuilder<>("GGA", new OneMaxProblem(size, budget), 300, 300,
                                new SinglePointCrossover<>(0.5), new BitFlipMutation<>(0.2 / size))
                            .setSelection(new NaryTournamentSelection<>(3, 300, new ObjectiveComparator<>(0)))
                            .setReplacement((population, offspring) -> offspring)
                            .setTermination(termination)
                            .build())));
                } else {
                    // jmetal-component examples/singleobjective/geneticalgorithm/
                    // GenerationalGeneticAlgorithmBinaryExample.java (on OneMax): population 100,
                    // 100 children, binary tournament, SinglePointCrossover(0.9),
                    // BitFlipMutation(1 / bits), (μ + λ) replacement (GeneticAlgorithmBuilder's)
                    solvers.add(new Solver("ga", (budget, seed) -> runComponent(termination ->
                        new GeneticAlgorithmBuilder<>("GGA", new OneMaxProblem(size, budget), 100, 100,
                                new SinglePointCrossover<>(0.9), new BitFlipMutation<>(1.0 / size))
                            .setTermination(termination)
                            .build())));
                    // jmetal-algorithm examples/singleobjective/ElitistEvolutionStrategyRunner.java
                    // (on OneMax): the elitist (μ + λ) evolution strategy with μ 1, λ 10,
                    // BitFlipMutation(1 / bits)
                    solvers.add(new Solver("es", (budget, seed) -> {
                        var es = new EvolutionStrategyBuilder<BinarySolution>(new OneMaxProblem(size, budget),
                                new BitFlipMutation<>(1.0 / size), EvolutionStrategyBuilder.EvolutionStrategyVariant.ELITIST)
                            .setMaxEvaluations(Integer.MAX_VALUE)
                            .setMu(1)
                            .setLambda(10)
                            .build();
                        runClassic(es::run);
                        return Math.max(0, budget.evaluations - 1) / 10;
                    }));
                }
            }
            case "nqueens" -> {
                minimize[0] = true;
                target[0] = 0;
                // jmetal-component examples/singleobjective/geneticalgorithm/GeneticAlgorithmTSPExample.java:
                // population 100, 100 children, binary tournament, PMXCrossover(0.9),
                // PermutationSwapMutation(1 / n), (μ + λ) replacement
                solvers.add(new Solver("ga", (budget, seed) -> runComponent(termination ->
                    new GeneticAlgorithmBuilder<>("GGA", new NQueensProblem(size, budget), 100, 100,
                            new PMXCrossover(0.9), new PermutationSwapMutation<Integer>(1.0 / size))
                        .setTermination(termination)
                        .build())));
            }
            case "rastrigin", "rosenbrock", "ackley" -> {
                minimize[0] = true;
                target[0] = 0.01;
                RealFunction real = realFunction(args.problem(), size);
                Function<Budget, RealProblem> problem = budget -> new RealProblem(args.problem(), size, real, budget);
                // Bounds (rule 2.4): SBXCrossover, PolynomialMutation and
                // DifferentialEvolutionCrossover repair a value outside the bounds to the bound
                // (RepairDoubleSolutionWithBoundValue, their default); CMA-ES clips its samples
                // jmetal-component examples/singleobjective/geneticalgorithm/GenerationalGeneticAlgorithmExample.java:
                // population 100, 100 children, binary tournament, SBXCrossover(0.9, η 20),
                // PolynomialMutation(1 / n, η 20), (μ + λ) replacement
                solvers.add(new Solver("ga", (budget, seed) -> runComponent(termination ->
                    new GeneticAlgorithmBuilder<>("GGA", problem.apply(budget), 100, 100,
                            new SBXCrossover(0.9, 20.0), new PolynomialMutation(1.0 / size, 20.0))
                        .setTermination(termination)
                        .build())));
                // jmetal-algorithm examples/singleobjective/DifferentialEvolutionRunner.java:
                // DE/rand/1/bin with CR 0.5 and F 0.5, population 100
                solvers.add(new Solver("de", (budget, seed) -> {
                    var evaluator = new CountingEvaluator<DoubleSolution>();
                    var de = new DifferentialEvolution(problem.apply(budget), Integer.MAX_VALUE, 100,
                        new DifferentialEvolutionCrossover(0.5, 0.5, DifferentialEvolutionCrossover.DE_VARIANT.RAND_1_BIN),
                        new DifferentialEvolutionSelection(), evaluator);
                    runClassic(de::run);
                    return Math.max(0, evaluator.calls - 1);
                }));
                // jmetal-algorithm examples/singleobjective/CovarianceMatrixAdaptationEvolutionStrategyRunner.java:
                // the builder's defaults (λ 10, σ 0.3), with restarts (see cmaes() above)
                solvers.add(new Solver("cma_es", (budget, seed) ->
                    cmaes(budget, seed, problem.apply(budget), size, real.lower(), real.upper())));
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
                JMetalRandom.getInstance().setSeed(seed);
                Budget budget = new Budget(args.maxEvaluations(), args.maxSeconds(), minimize[0], target[0]);
                long start = System.nanoTime();
                long generations = solver.solver().run(budget, seed);
                double time = (System.nanoTime() - start) / 1e9;
                if (!print) continue;
                // the continuous problems report the solutions evaluated outside the bounds
                String outside = realFunction(args.problem(), args.size()) != null
                    ? ",\"outside\":" + budget.outside : "";
                System.out.println("{\"library\":\"" + LIBRARY + "\",\"solver\":\"" + solver.name()
                    + "\",\"problem\":\"" + args.problem() + "\",\"size\":" + args.size()
                    + ",\"mode\":\"" + args.mode() + "\",\"seed\":" + seed
                    + ",\"time_s\":" + String.format(Locale.ROOT, "%.6f", time)
                    + ",\"generations\":" + generations + ",\"evaluations\":" + budget.evaluations
                    + ",\"best\":" + number(budget.best) + ",\"target\":" + number(target[0])
                    + ",\"success\":" + budget.reached() + outside
                    + ",\"solution\":" + json(budget.solution) + "}");
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

    /** SPEA2 (classic implementation) stopping at the budget, with its iterations. */
    static final class BudgetSPEA2 extends SPEA2<DoubleSolution> {
        final Budget budget;

        BudgetSPEA2(FrontProblem problem, int population, SBXCrossover crossover, PolynomialMutation mutation,
                    Budget budget) {
            super(problem, Integer.MAX_VALUE, population, crossover, mutation, new BinaryTournamentSelection<>(),
                new SequentialSolutionListEvaluator<>(), 1);
            this.budget = budget;
        }

        @Override
        protected boolean isStoppingConditionReached() {
            return budget.exhausted();
        }

        long generations() {
            return iterations - 1;
        }
    }

    /** Das-Dennis weights for MOEA/D, as the weight file jMetal reads for 3 objectives. */
    static Path weightDirectory(int objectives, int divisions) throws IOException {
        Path directory = Files.createTempDirectory("jmetal-weights");
        directory.toFile().deleteOnExit();
        List<double[]> weights = ReferencePointGenerator.generateSingleLayer(objectives, divisions);
        Path file = directory.resolve("W" + objectives + "D_" + weights.size() + ".dat");
        try (PrintWriter out = new PrintWriter(Files.newBufferedWriter(file))) {
            for (double[] w : weights) {
                StringBuilder line = new StringBuilder();
                for (int k = 0; k < w.length; k++) {
                    if (k > 0) line.append(' ');
                    line.append(w[k]);
                }
                out.println(line);
            }
        }
        file.toFile().deleteOnExit();
        return directory;
    }

    // the matched multi-objective algorithms (rule 6.1); jMetal's others, such as SMPSO, aren't run.
    // Bounds (rule 2.4): SBXCrossover and PolynomialMutation repair a value outside the bounds to
    // the bound (RepairDoubleSolutionWithBoundValue, their default).
    static final String[] FRONT_SOLVERS = {"nsga2", "nsga3", "spea2", "moead", "sms_emoa"};

    static void runFront(Args args, boolean print) throws IOException {
        FrontFunction front = frontFunction(args.problem(), args.size());
        if (front == null) return;
        int n = front.variables();
        int m = front.objectives();
        int population = front.population();
        String weights = weightDirectory(m, front.divisions()).toString();

        for (long seed = args.seedFrom(); seed <= args.seedTo(); seed++) {
            for (String solver : FRONT_SOLVERS) {
                JMetalRandom.getInstance().setSeed(seed);
                Budget budget = new Budget(args.maxEvaluations(), args.maxSeconds(), true, Double.NEGATIVE_INFINITY);
                FrontProblem problem = new FrontProblem(args.problem(), front, budget);
                long[] generations = {0};
                Termination termination = status -> {
                    if (budget.exhausted()) return true;
                    generations[0]++;
                    return false;
                };
                PolynomialMutation mutation = new PolynomialMutation(1.0 / n, 20.0);
                long start = System.nanoTime();
                List<DoubleSolution> result;
                switch (solver) {
                    case "nsga2" -> {
                        // N parents, N children, SBX η 15 at 0.9, PM η 20 at 1 / n
                        var algorithm = new NSGAIIBuilder<>(problem, population, population,
                                new SBXCrossover(0.9, 15.0), mutation)
                            .setTermination(termination)
                            .build();
                        algorithm.run();
                        result = algorithm.result();
                    }
                    case "nsga3" -> {
                        // Das-Dennis directions (99 divisions: 100 for 2 objectives; 12: 91 for 3),
                        // the population rounded up to a multiple of 4 by jMetal (100, 92),
                        // SBX η 30 at 1, PM η 20 at 1 / n
                        var algorithm = new NSGAIIIBuilder<>(problem, front.divisions(),
                                new SBXCrossover(1.0, 30.0), mutation)
                            .setTermination(termination)
                            .build();
                        algorithm.run();
                        result = algorithm.result();
                    }
                    case "spea2" -> {
                        // the classic implementation (no component version): population and archive
                        // N, binary tournament, SBX η 15 at 0.9, PM η 20 at 1 / n, k = 1; its result
                        // is the non-dominated part of its archive of N
                        var algorithm = new BudgetSPEA2(problem, population, new SBXCrossover(0.9, 15.0), mutation,
                            budget);
                        algorithm.run();
                        generations[0] = algorithm.generations();
                        result = algorithm.result();
                    }
                    case "moead" -> {
                        // 100 weights (91 Das-Dennis weights for 3 objectives), 20 neighbors,
                        // parents from the neighborhood with probability 0.9, at most 2
                        // replacements, Tchebycheff (PBI θ 5 for 3 objectives), SBX η 20 at 1,
                        // PM η 20 at 1 / n; one child per step (jMetal's MOEA/D is steady-state)
                        int weightCount = m == 2 ? 100 : ReferencePointGenerator.calculateNumberOfReferencePoints(m, front.divisions());
                        var algorithm = new MOEADBuilder<>(problem, weightCount, new SBXCrossover(1.0, 20.0), mutation,
                                weights, new RandomPermutationCycle(weightCount), false)
                            .setAggregationFunction(m == 2 ? new Tschebyscheff(false)
                                : new PenaltyBoundaryIntersection(5.0, false))
                            .setTermination(termination)
                            .build();
                        algorithm.run();
                        result = algorithm.result();
                    }
                    default -> {
                        // SMS-EMOA: population N, one child per step, SBX η 15 at 0.9, PM η 20 at 1 / n
                        var algorithm = new SMSEMOABuilder<>(problem, population, new SBXCrossover(0.9, 15.0), mutation)
                            .setTermination(termination)
                            .build();
                        algorithm.run();
                        result = algorithm.result();
                    }
                }
                List<double[]> points = new ArrayList<>();
                for (DoubleSolution solution : result) points.add(solution.objectives().clone());
                List<Integer> set = nonDominated(points);
                double time = (System.nanoTime() - start) / 1e9;
                if (!print) continue;
                StringBuilder frontJson = new StringBuilder("[");
                StringBuilder solutionsJson = new StringBuilder("[");
                for (int i = 0; i < set.size(); i++) {
                    if (i > 0) {
                        frontJson.append(',');
                        solutionsJson.append(',');
                    }
                    frontJson.append(json(points.get(set.get(i))));
                    solutionsJson.append(json(values(result.get(set.get(i)))));
                }
                System.out.println("{\"library\":\"" + LIBRARY + "\",\"solver\":\"" + solver
                    + "\",\"problem\":\"" + args.problem() + "\",\"size\":" + args.size()
                    + ",\"mode\":\"" + args.mode() + "\",\"seed\":" + seed
                    + ",\"time_s\":" + String.format(Locale.ROOT, "%.6f", time)
                    + ",\"generations\":" + generations[0] + ",\"evaluations\":" + budget.evaluations
                    + ",\"outside\":" + budget.outside
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
        FrontFunction front = frontFunction(problem, size);
        RealFunction real = realFunction(problem, size);
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
                out.append(number(onemax(bits)));
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

    static void run(Args args, boolean print) throws IOException {
        if (frontFunction(args.problem(), args.size()) != null) {
            runFront(args, print);
        } else {
            runSingle(args, print);
        }
    }

    static String version() {
        String version = EvolutionaryAlgorithm.class.getPackage().getImplementationVersion();
        return version != null ? version : "7.5";
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
        // jMetal logs to stderr through java.util.logging; keep it quiet
        LogManager.getLogManager().reset();
        java.util.logging.Logger.getLogger("").setLevel(Level.OFF);
        Args args = new Args(argv[0], Integer.parseInt(argv[1]), argv[2], Long.parseLong(argv[3]),
            Long.parseLong(argv[4]), Long.parseLong(argv[5]), Double.parseDouble(argv[6]));
        // JIT warm-up (rule 4.2): every solver once on the same problem, untimed and unprinted
        run(new Args(args.problem(), args.size(), args.mode(), WARM_UP_SEED, WARM_UP_SEED,
            WARM_UP_EVALUATIONS, args.maxSeconds()), false);
        run(args, true);
    }
}
