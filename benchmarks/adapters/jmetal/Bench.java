/*
 * Benchmark adapter for jMetal (org.uma.jmetal: jmetal-core, jmetal-algorithm, jmetal-component).
 *
 * Usage: java -cp ... Bench <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
 *        java -cp ... Bench --version
 * Prints one JSON line per solver per seed, see ../../README.md for the fields.
 *
 * The problems are implemented here with the formulas of the other adapters (not jMetal's own
 * problem classes), so every library has the same fitness code path and the evaluations are
 * counted in the fitness function. Everything is sequential (SequentialEvaluation and
 * SequentialSolutionListEvaluator, jMetal's defaults). Before the timed runs of a problem, every
 * solver runs once untimed with a small budget, so the JIT has compiled everything.
 *
 * Multi-objective algorithms use jMetal's component-based builders (org.uma.jmetal.component, the
 * API its documentation recommends since jMetal 6) where they exist, and the classic
 * implementation (org.uma.jmetal.algorithm) otherwise (SPEA2, and the single-objective DE, CMA-ES
 * and PSO).
 */

import org.uma.jmetal.algorithm.multiobjective.spea2.SPEA2;
import org.uma.jmetal.algorithm.singleobjective.differentialevolution.DifferentialEvolution;
import org.uma.jmetal.algorithm.singleobjective.evolutionstrategy.CovarianceMatrixAdaptationEvolutionStrategy;
import org.uma.jmetal.algorithm.singleobjective.particleswarmoptimization.StandardPSO2011;
import org.uma.jmetal.component.algorithm.EvolutionaryAlgorithm;
import org.uma.jmetal.component.algorithm.ParticleSwarmOptimizationAlgorithm;
import org.uma.jmetal.component.algorithm.multiobjective.MOEADBuilder;
import org.uma.jmetal.component.algorithm.multiobjective.NSGAIIBuilder;
import org.uma.jmetal.component.algorithm.multiobjective.NSGAIIIBuilder;
import org.uma.jmetal.component.algorithm.multiobjective.SMPSOBuilder;
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
import org.uma.jmetal.util.aggregationfunction.impl.PenaltyBoundaryIntersection;
import org.uma.jmetal.util.aggregationfunction.impl.Tschebyscheff;
import org.uma.jmetal.util.comparator.ObjectiveComparator;
import org.uma.jmetal.util.evaluator.SolutionListEvaluator;
import org.uma.jmetal.util.evaluator.impl.SequentialSolutionListEvaluator;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;
import org.uma.jmetal.util.referencepoint.ReferencePointGenerator;
import org.uma.jmetal.util.sequencegenerator.impl.RandomPermutationCycle;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.function.Function;
import java.util.function.ToDoubleFunction;
import java.util.logging.Level;
import java.util.logging.LogManager;

public final class Bench {

    static final String LIBRARY = "jmetal";

    // ---------------------------------------------------------------------------------------------
    // Fitness functions, identical to the ones in the other adapters
    // ---------------------------------------------------------------------------------------------

    /** Thrown by a single-objective problem to stop a run at the target or at the budget. */
    static final class Stop extends RuntimeException {
        Stop() {
            super(null, null, false, false);
        }
    }

    static final Stop STOP = new Stop();

    /** Counts the evaluations and keeps the best value found, stops at the budget. */
    static final class Budget {
        final long maxEvaluations;
        final long deadline;
        final boolean minimize;
        final double target;
        long evaluations;
        double best;
        boolean timeUp;

        Budget(long maxEvaluations, double maxSeconds, boolean minimize, double target) {
            this.maxEvaluations = maxEvaluations;
            this.deadline = System.nanoTime() + (long) (maxSeconds * 1e9);
            this.minimize = minimize;
            this.target = target;
            this.best = minimize ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
        }

        /** Single-objective: called before an evaluation, stops at the budget. */
        void before() {
            if (evaluations >= maxEvaluations) throw STOP;
            // the clock every 64 evaluations
            if ((evaluations & 63) == 0 && System.nanoTime() >= deadline) {
                timeUp = true;
            }
            if (timeUp) throw STOP;
        }

        /** Single-objective: records a fitness value, stops at the target. */
        double record(double value) {
            evaluations++;
            if (minimize ? value < best : value > best) {
                best = value;
            }
            if (reached()) throw STOP;
            return value;
        }

        boolean reached() {
            return minimize ? best <= target : best >= target;
        }

        /** Multi-objective: checked between generations. */
        boolean exhausted() {
            return evaluations >= maxEvaluations || System.nanoTime() >= deadline;
        }
    }

    static double onemax(java.util.BitSet bits) {
        return bits.cardinality();
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
            double ones = onemax(solution.variables().get(0));
            solution.objectives()[0] = -ones; // maximize the number of ones
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

        @Override
        public PermutationSolution<Integer> evaluate(PermutationSolution<Integer> solution) {
            budget.before();
            List<Integer> variables = solution.variables();
            int[] p = new int[size];
            for (int i = 0; i < size; i++) p[i] = variables.get(i);
            double conflicts = nqueens(p);
            solution.objectives()[0] = conflicts;
            budget.record(conflicts);
            return solution;
        }
    }

    static final class RealProblem extends AbstractDoubleProblem {
        final ToDoubleFunction<double[]> function;
        final Budget budget;

        RealProblem(String name, int size, double lower, double upper, ToDoubleFunction<double[]> function,
                    Budget budget) {
            this.function = function;
            this.budget = budget;
            numberOfObjectives(1);
            numberOfConstraints(0);
            name(name);
            variableBounds(Collections.nCopies(size, lower), Collections.nCopies(size, upper));
        }

        @Override
        public DoubleSolution evaluate(DoubleSolution solution) {
            budget.before();
            double value = function.applyAsDouble(values(solution));
            solution.objectives()[0] = value;
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
            double[] f = function.apply(values(solution));
            System.arraycopy(f, 0, solution.objectives(), 0, f.length);
            return solution;
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Single-objective runs
    // ---------------------------------------------------------------------------------------------

    record Args(String problem, int size, String mode, long seedFrom, long seedTo, long maxEvaluations,
                double maxSeconds) {}

    /** A solver: runs until Stop (or its own end), returns the generations. */
    interface SingleSolver {
        long run(Budget budget);
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
            // the target or the budget
        }
        return generations[0];
    }

    static long runClassic(Runnable run, CountingEvaluator<?> evaluator) {
        try {
            run.run();
        } catch (Stop stop) {
            // the target or the budget
        }
        return Math.max(0, evaluator.calls - 1);
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
                    solvers.add(new Solver("ga", budget -> runComponent(termination ->
                        new GeneticAlgorithmBuilder<>("GGA", new OneMaxProblem(size, budget), 300, 300,
                                new SinglePointCrossover<>(0.5), new BitFlipMutation<>(0.2 / size))
                            .setSelection(new NaryTournamentSelection<>(3, 300, new ObjectiveComparator<>(0)))
                            .setReplacement((population, offspring) -> offspring)
                            .setTermination(termination)
                            .build())));
                } else {
                    // jmetal-component examples/singleobjective/geneticalgorithm/
                    // GenerationalGeneticAlgorithmBinaryExample.java: population 100, 100 children,
                    // binary tournament, SinglePointCrossover(0.9), BitFlipMutation(1 / bits),
                    // (μ + λ) replacement
                    solvers.add(new Solver("ga", budget -> runComponent(termination ->
                        new GeneticAlgorithmBuilder<>("GGA", new OneMaxProblem(size, budget), 100, 100,
                                new SinglePointCrossover<>(0.9), new BitFlipMutation<>(1.0 / size))
                            .setTermination(termination)
                            .build())));
                }
            }
            case "nqueens" -> {
                minimize[0] = true;
                target[0] = 0;
                // jmetal-component examples/singleobjective/geneticalgorithm/GeneticAlgorithmTSPExample.java:
                // population 100, 100 children, binary tournament, PMXCrossover(0.9),
                // PermutationSwapMutation(1 / n), (μ + λ) replacement
                solvers.add(new Solver("ga", budget -> runComponent(termination ->
                    new GeneticAlgorithmBuilder<>("GGA", new NQueensProblem(size, budget), 100, 100,
                            new PMXCrossover(0.9), new PermutationSwapMutation<Integer>(1.0 / size))
                        .setTermination(termination)
                        .build())));
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
                Function<Budget, RealProblem> problem =
                    budget -> new RealProblem(args.problem(), size, lower, upper, function, budget);
                // jmetal-component examples/singleobjective/geneticalgorithm/GenerationalGeneticAlgorithmExample.java:
                // population 100, 100 children, binary tournament, SBXCrossover(0.9, η 20),
                // PolynomialMutation(1 / n, η 20), (μ + λ) replacement
                solvers.add(new Solver("ga", budget -> runComponent(termination ->
                    new GeneticAlgorithmBuilder<>("GGA", problem.apply(budget), 100, 100,
                            new SBXCrossover(0.9, 20.0), new PolynomialMutation(1.0 / size, 20.0))
                        .setTermination(termination)
                        .build())));
                // jmetal-algorithm examples/singleobjective/DifferentialEvolutionRunner.java:
                // DE/rand/1/bin with CR 0.5 and F 0.5, population 100
                solvers.add(new Solver("de", budget -> {
                    var evaluator = new CountingEvaluator<DoubleSolution>();
                    var de = new DifferentialEvolution(problem.apply(budget), Integer.MAX_VALUE, 100,
                        new DifferentialEvolutionCrossover(0.5, 0.5, DifferentialEvolutionCrossover.DE_VARIANT.RAND_1_BIN),
                        new DifferentialEvolutionSelection(), evaluator);
                    return runClassic(de::run, evaluator);
                }));
                // jmetal-algorithm examples/singleobjective/CovarianceMatrixAdaptationEvolutionStrategyRunner.java:
                // the defaults (λ 10, σ 0.3), no restarts; it ends by itself when its covariance
                // matrix degenerates. Difference: jMetal's default start is a random point in
                // [0, 1)^n whatever the bounds, which is next to the optimum of these problems
                // (the shift lies in [-1, 1]); the start here is a random point within the bounds,
                // like the other libraries' CMA-ES
                solvers.add(new Solver("cma_es", budget -> {
                    double[] start = new double[size];
                    for (int i = 0; i < size; i++) start[i] = JMetalRandom.getInstance().nextDouble(lower, upper);
                    var cmaes = new CovarianceMatrixAdaptationEvolutionStrategy.Builder(problem.apply(budget))
                        .setMaxEvaluations(Integer.MAX_VALUE)
                        .setTypicalX(start)
                        .build();
                    try {
                        cmaes.run();
                    } catch (Stop stop) {
                        // the target or the budget
                    } catch (RuntimeException error) {
                        // once the covariance matrix degenerates (converged, e.g. in a local
                        // optimum), jMetal's eigendecomposition (CMAESUtils.tql2) can fail with an
                        // ArrayIndexOutOfBoundsException: the run ends there, with what it found
                        System.err.println("jmetal cma_es ended with " + error + " after "
                            + budget.evaluations + " evaluations");
                    }
                    return budget.evaluations / cmaes.getLambda();
                }));
                // jmetal-algorithm examples/singleobjective/StandardPSO2011Runner.java: SPSO 2011,
                // swarm 10 + 2 sqrt(n), 3 informants
                solvers.add(new Solver("pso", budget -> {
                    var evaluator = new CountingEvaluator<DoubleSolution>();
                    var pso = new StandardPSO2011(problem.apply(budget), 10 + (int) (2 * Math.sqrt(size)),
                        Integer.MAX_VALUE, 3, evaluator);
                    return runClassic(pso::run, evaluator);
                }));
            }
            default -> {
                return null;
            }
        }
        return solvers;
    }

    static String number(double value) {
        if (value == Math.rint(value) && Math.abs(value) < 1e15) return Long.toString((long) value);
        return Double.toString(value);
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
    // Multi-objective runs
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

    static final String[] FRONT_SOLVERS = {"nsga2", "nsga3", "spea2", "moead", "sms_emoa", "smpso"};

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
                        // N, binary tournament, SBX η 15 at 0.9, PM η 20 at 1 / n, k = 1
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
                    case "sms_emoa" -> {
                        // population N, one child per step, SBX η 15 at 0.9, PM η 20 at 1 / n
                        var algorithm = new SMSEMOABuilder<>(problem, population, new SBXCrossover(0.9, 15.0), mutation)
                            .setTermination(termination)
                            .build();
                        algorithm.run();
                        result = algorithm.result();
                    }
                    default -> {
                        // SMPSO (jMetal's own algorithm), the defaults of the builder: swarm and
                        // leader archive N, PM η 20 at 1 / n as the turbulence on every 6th particle
                        ParticleSwarmOptimizationAlgorithm algorithm = new SMPSOBuilder(problem, population)
                            .setTermination(termination)
                            .build();
                        algorithm.run();
                        result = algorithm.result();
                    }
                }
                List<double[]> points = new ArrayList<>();
                for (DoubleSolution solution : result) points.add(solution.objectives().clone());
                List<double[]> set = nonDominated(points);
                double time = (System.nanoTime() - start) / 1e9;
                if (!print) continue;
                StringBuilder json = new StringBuilder();
                json.append("{\"library\":\"").append(LIBRARY).append("\",\"solver\":\"").append(solver)
                    .append("\",\"problem\":\"").append(args.problem()).append("\",\"size\":").append(args.size())
                    .append(",\"mode\":\"").append(args.mode()).append("\",\"seed\":").append(seed)
                    .append(",\"time_s\":").append(String.format(java.util.Locale.ROOT, "%.6f", time))
                    .append(",\"generations\":").append(generations[0])
                    .append(",\"evaluations\":").append(budget.evaluations).append(",\"front\":[");
                for (int i = 0; i < set.size(); i++) {
                    if (i > 0) json.append(',');
                    json.append('[');
                    double[] point = set.get(i);
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
        if (argv.length != 7) {
            System.err.println("usage: Bench <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>");
            System.exit(2);
        }
        // jMetal logs to stderr through java.util.logging; keep it quiet
        LogManager.getLogManager().reset();
        java.util.logging.Logger.getLogger("").setLevel(Level.OFF);
        Args args = new Args(argv[0], Integer.parseInt(argv[1]), argv[2], Long.parseLong(argv[3]),
            Long.parseLong(argv[4]), Long.parseLong(argv[5]), Double.parseDouble(argv[6]));
        // JIT warm-up: every solver once on the same problem, untimed and unprinted
        run(new Args(args.problem(), args.size(), args.mode(), 1_000_003L, 1_000_003L,
            Math.min(args.maxEvaluations(), 10_000L), Math.min(args.maxSeconds(), 2.0)), false);
        run(args, true);
    }
}
