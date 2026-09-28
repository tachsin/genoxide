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
 *   run ends inside evaluate(), at the target, the budget or the time cap (rule 2.1), by an
 *   exception the adapter catches.
 * - One thread (rule 4.3): jMetal evaluates sequentially (SequentialEvaluation and
 *   SequentialSolutionListEvaluator, its defaults); run.sh runs the JVM with the serial garbage
 *   collector and with -Xbatch, so the JIT compiles on the calling thread's time.
 * - Seeds (rule 5.2): JMetalRandom's seed, before every run. Two parts of jMetal don't use it: its
 *   CMA-ES draws its samples from its own java.util.Random, seeded with System.currentTimeMillis(),
 *   which it doesn't let a user set (the adapter replaces it by a seeded one, see cmaes() below),
 *   and its initial permutations are shuffled by Collections.shuffle's own generator (the adapter
 *   shuffles them with JMetalRandom, see NQueensProblem.createSolution()).
 * - Time (rules 4.1 and 4.2): the clock starts before the algorithm creates its initial population
 *   and stops when the run ends. Before the timed runs, every solver runs once untimed and
 *   unprinted, with the seed 999,999, 50,000 evaluations and the scenario's time cap, so the JIT has
 *   compiled the fitness function and the algorithm.
 * - First hit: a single-objective run records the evaluation (and the time) at which the best value
 *   first reaches the target, in evaluate(), and prints it as "first_hit".
 */

import org.uma.jmetal.algorithm.singleobjective.differentialevolution.DifferentialEvolution;
import org.uma.jmetal.algorithm.singleobjective.evolutionstrategy.CovarianceMatrixAdaptationEvolutionStrategy;
import org.uma.jmetal.algorithm.singleobjective.evolutionstrategy.EvolutionStrategyBuilder;
import org.uma.jmetal.component.algorithm.EvolutionaryAlgorithm;
import org.uma.jmetal.component.algorithm.singleobjective.GeneticAlgorithmBuilder;
import org.uma.jmetal.component.catalogue.common.termination.Termination;
import org.uma.jmetal.operator.crossover.impl.DifferentialEvolutionCrossover;
import org.uma.jmetal.operator.crossover.impl.PMXCrossover;
import org.uma.jmetal.operator.crossover.impl.SBXCrossover;
import org.uma.jmetal.operator.crossover.impl.SinglePointCrossover;
import org.uma.jmetal.operator.mutation.impl.BitFlipMutation;
import org.uma.jmetal.operator.mutation.impl.PermutationSwapMutation;
import org.uma.jmetal.operator.mutation.impl.PolynomialMutation;
import org.uma.jmetal.operator.selection.impl.DifferentialEvolutionSelection;
import org.uma.jmetal.problem.binaryproblem.impl.AbstractBinaryProblem;
import org.uma.jmetal.problem.doubleproblem.impl.AbstractDoubleProblem;
import org.uma.jmetal.problem.permutationproblem.impl.AbstractIntegerPermutationProblem;
import org.uma.jmetal.solution.Solution;
import org.uma.jmetal.solution.binarysolution.BinarySolution;
import org.uma.jmetal.solution.doublesolution.DoubleSolution;
import org.uma.jmetal.solution.permutationsolution.PermutationSolution;
import org.uma.jmetal.solution.permutationsolution.impl.IntegerPermutationSolution;
import org.uma.jmetal.util.evaluator.impl.SequentialSolutionListEvaluator;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
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
    static final long WARM_UP_SEED = 999_999L;
    static final long WARM_UP_EVALUATIONS = 50_000L;
    // JMETAL_CMAES_AS_IS=1: CMA-ES without the workaround of its crash (see cmaes()), for the page
    static final boolean CMAES_AS_IS = "1".equals(System.getenv("JMETAL_CMAES_AS_IS"));

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

    /**
     * The shift of Rastrigin and Ackley: s_i = 0.8 upper (2 ((37 i + 11) mod 101) / 101 - 1), with
     * `upper` the box's upper bound, computed in this order (problems.py).
     */
    static double[] shift(int n, double upper) {
        double[] s = new double[n];
        for (int i = 0; i < n; i++) {
            s[i] = 0.8 * upper * ((2 * ((37 * i + 11) % 101)) / 101.0 - 1.0);
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

    /** A single-objective real-valued problem: its function and bounds. */
    record RealFunction(ToDoubleFunction<double[]> function, double lower, double upper) {}

    static RealFunction realFunction(String name, int size) {
        return switch (name) {
            case "rastrigin" -> {
                double[] s = shift(size, 5.12);
                yield new RealFunction(x -> rastrigin(x, s), -5.12, 5.12);
            }
            case "rosenbrock" -> new RealFunction(Bench::rosenbrock, -5.0, 10.0);
            case "ackley" -> {
                double[] s = shift(size, 32.768);
                yield new RealFunction(x -> ackley(x, s), -32.768, 32.768);
            }
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
        // the clock: started when the budget is created, just before the run
        final long start;
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
        // the first evaluation whose value reaches the target, and the clock then (-1: not yet)
        long firstHitEvaluations = -1;
        double firstHitSeconds;
        // the evaluations at the end of the last generation, and that generation's (rule 2.3)
        long generationEnd;
        long lastGeneration;

        Budget(long maxEvaluations, double maxSeconds, boolean minimize, double target) {
            this.maxEvaluations = maxEvaluations;
            this.start = System.nanoTime();
            this.deadline = start + (long) (maxSeconds * 1e9);
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
            if (reached()) {
                if (firstHitEvaluations < 0) {
                    firstHitEvaluations = evaluations;
                    firstHitSeconds = (System.nanoTime() - start) / 1e9;
                }
                throw STOP;
            }
        }

        /** Marks the end of a generation. */
        void endGeneration() {
            lastGeneration = evaluations - generationEnd;
            generationEnd = evaluations;
        }

        /**
         * The evaluations since the start of the last generation (rule 2.3): of one cut short, or
         * of the last one that ended.
         */
        long lastGeneration() {
            return evaluations > generationEnd ? evaluations - generationEnd : lastGeneration;
        }

        /**
         * The generations of a method that doesn't call back between them: after `from`
         * evaluations, generations of `size` (rule 2.3).
         */
        void fixedGenerations(long from, long size) {
            if (evaluations > from) generationEnd = from + (evaluations - from - 1) / size * size;
        }

        /** The "first_hit" field of a single-objective run. */
        String firstHit() {
            if (firstHitEvaluations < 0) return "null";
            return "{\"evaluations\":" + firstHitEvaluations + ",\"time_s\":"
                + String.format(Locale.ROOT, "%.6f", firstHitSeconds) + "}";
        }

        boolean reached() {
            return minimize ? best <= target : best >= target;
        }

        boolean done() {
            return reached() || evaluations >= maxEvaluations || timeUp || System.nanoTime() >= deadline;
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

    /**
     * jMetal's sequential evaluator, counting the generations: a call evaluates the initial
     * population or a generation, and ends the previous one.
     */
    static final class CountingEvaluator<S extends Solution<?>> extends SequentialSolutionListEvaluator<S> {
        final Budget budget;
        long calls;

        CountingEvaluator(Budget budget) {
            this.budget = budget;
        }

        @Override
        public List<S> evaluate(List<S> solutionList, org.uma.jmetal.problem.Problem<S> problem) {
            calls++;
            budget.endGeneration();
            return super.evaluate(solutionList, problem);
        }
    }

    /**
     * Runs a component algorithm; the termination, checked after the initial population and after
     * every generation, only counts the generations and marks their ends (Stop ends the run).
     */
    static <S extends Solution<?>> long runComponent(Budget budget,
                                                     Function<Termination, EvolutionaryAlgorithm<S>> build) {
        long[] generations = {0};
        EvolutionaryAlgorithm<S> algorithm = build.apply(status -> {
            budget.endGeneration();
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
     * lifted (rule 2.2). It ends an attempt by itself when its covariance matrix degenerates and
     * the eigendecomposition fails its check (checkEigenCorrectness sets the evaluations to the
     * maximum): a convergence criterion, so the run starts again (rule 2.2).
     * Workaround of a crash (rule 8.4; the page shows both results): when NaN reaches the
     * covariance matrix, CMAESUtils.tql2 throws ArrayIndexOutOfBoundsException and the run
     * aborts. The adapter catches it and starts again, as after a convergence. Without the
     * workaround (JMETAL_CMAES_AS_IS=1), the run ends at the crash with the best value found.
     * jMetal has no restart mechanism, so the adapter starts it again from a new random point, with
     * JMetalRandom seeded with (seed + 1) * 1,000,000 + restart (rule 2.2); the budget keeps the
     * best and counts every evaluation. Bounds (rule 2.4): jMetal clips every sample to the bounds
     * (Bounds.restrict in sampleSolution).
     * jMetal's CMA-ES has a bug: once it has converged, its σ grows without bound, every sample
     * lands on the bounds and the run stalls without ending, often for 200,000 evaluations before
     * tql2 throws. The adapter doesn't restart it then (rule 8.4), so the results show the bug.
     */
    static long cmaes(Budget budget, long seed, RealProblem problem, int size, double lower, double upper) {
        long generations = 0;
        for (int restart = 0; ; restart++) {
            if (restart > 0) JMetalRandom.getInstance().setSeed((seed + 1) * 1_000_000 + restart);
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
                budget.fixedGenerations(before, algorithm.getLambda());
                return generations + (budget.evaluations - before) / algorithm.getLambda();
            } catch (ArrayIndexOutOfBoundsException error) {
                // the crash of tql2 on NaN in the covariance matrix: the workaround goes on with a
                // new attempt (see above); as is, the run ends here
                if (CMAES_AS_IS) {
                    budget.fixedGenerations(before, algorithm.getLambda());
                    return generations + (budget.evaluations - before) / algorithm.getLambda();
                }
            }
            // λ evaluations a generation, the initial population included
            budget.fixedGenerations(before, algorithm.getLambda());
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
                    // Not run (rule 6.1): jMetal has no generational replacement without elitism
                    // (its replacements are (μ + λ), (μ, λ) with μ < λ, pairwise, random and the
                    // multi-objective ones; the classic GenerationalGeneticAlgorithm keeps 2
                    // elites) and no two-point crossover for bits (TwoPointCrossover is for numbers)
                    return null;
                } else {
                    // jmetal-component examples/singleobjective/geneticalgorithm/
                    // GenerationalGeneticAlgorithmBinaryExample.java (on OneMax): population 100,
                    // 100 children, binary tournament, SinglePointCrossover(0.9),
                    // BitFlipMutation(1 / bits), (μ + λ) replacement (GeneticAlgorithmBuilder's)
                    solvers.add(new Solver("ga", (budget, seed) -> runComponent(budget, termination ->
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
                        // μ = 1 initial solution, then λ = 10 a generation
                        budget.fixedGenerations(1, 10);
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
                solvers.add(new Solver("ga", (budget, seed) -> runComponent(budget, termination ->
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
                solvers.add(new Solver("ga", (budget, seed) -> runComponent(budget, termination ->
                    new GeneticAlgorithmBuilder<>("GGA", problem.apply(budget), 100, 100,
                            new SBXCrossover(0.9, 20.0), new PolynomialMutation(1.0 / size, 20.0))
                        .setTermination(termination)
                        .build())));
                // jmetal-algorithm examples/singleobjective/DifferentialEvolutionRunner.java:
                // DE/rand/1/bin with CR 0.5 and F 0.5, population 100
                solvers.add(new Solver("de", (budget, seed) -> {
                    var evaluator = new CountingEvaluator<DoubleSolution>(budget);
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
                // the clock starts here (Budget.start), before the algorithm creates its population
                Budget budget = new Budget(args.maxEvaluations(), args.maxSeconds(), minimize[0], target[0]);
                long generations = solver.solver().run(budget, seed);
                double time = (System.nanoTime() - budget.start) / 1e9;
                if (!print) continue;
                // the continuous problems report the solutions evaluated outside the bounds
                String outside = realFunction(args.problem(), args.size()) != null
                    ? ",\"outside\":" + budget.outside : "";
                System.out.println("{\"library\":\"" + LIBRARY + "\",\"solver\":\"" + solver.name()
                    + "\",\"problem\":\"" + args.problem() + "\",\"size\":" + args.size()
                    + ",\"mode\":\"" + args.mode() + "\",\"seed\":" + seed
                    + ",\"time_s\":" + String.format(Locale.ROOT, "%.6f", time)
                    + ",\"generations\":" + generations + ",\"evaluations\":" + budget.evaluations
                    + ",\"last_generation\":" + budget.lastGeneration()
                    + ",\"best\":" + number(budget.best) + ",\"target\":" + number(target[0])
                    + ",\"success\":" + budget.reached() + ",\"first_hit\":" + budget.firstHit() + outside
                    + ",\"solution\":" + json(budget.solution) + "}");
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
        RealFunction real = realFunction(problem, size);
        StringBuilder out = new StringBuilder();
        String line;
        while ((line = in.readLine()) != null) {
            if (line.isBlank()) continue;
            double[] x = parse(line);
            if (real != null) {
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
        runSingle(new Args(args.problem(), args.size(), args.mode(), WARM_UP_SEED, WARM_UP_SEED,
            WARM_UP_EVALUATIONS, args.maxSeconds()), false);
        runSingle(args, true);
    }
}
