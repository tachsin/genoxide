/*
 * Benchmark adapter for jMetal 7.5 (org.uma.jmetal: jmetal-core, jmetal-algorithm), following
 * docs/benchmarks/rules.md. It runs the matched suite only: DE/rand/1/bin on Rastrigin 30 ("de")
 * and CMA-ES on Rosenbrock 10 ("cma_es"), with jMetal's own classes; it prints nothing for any
 * other problem, size or mode. Its page, docs/benchmarks/libraries/jmetal.md, maps every setting
 * to the definition, lists the differences and jMetal's bugs, and gives the separate test runs.
 *
 * Usage: java -cp ... Bench <problem> <size> <mode> <seed_from> <seed_to> <max_evaluations> <max_seconds>
 *        java -cp ... Bench values <problem> <size>     (one JSON solution per line on stdin)
 *        java -cp ... Bench --version
 * Prints one JSON line per seed, see ../../README.md for the fields.
 *
 * - The problems are jMetal problems (AbstractDoubleProblem) whose evaluate() calls the fitness
 *   functions below and counts the evaluations (rule 3). A run ends inside evaluate(), at the
 *   target (Rosenbrock 10 only: Rastrigin 30 has none), the budget or the time cap, by an
 *   exception the adapter catches.
 * - One thread (rule 4.3): jMetal evaluates sequentially; run.sh runs the JVM with the serial
 *   garbage collector and -Xbatch, so the JIT compiles on the calling thread's time.
 * - Seeds (rule 5.2): JMetalRandom's seed, before every run. jMetal's CMA-ES draws its samples from
 *   a java.util.Random of its own, seeded with the clock; the adapter seeds it (see cmaes()).
 * - Time (rules 4.1 and 4.2): the clock starts before the algorithm creates its initial population
 *   and stops when the run ends. Before the timed runs, the solver runs once untimed and unprinted,
 *   with the seed 999,999, 50,000 evaluations and the scenario's time cap (the JIT warm-up).
 */

import org.uma.jmetal.algorithm.singleobjective.differentialevolution.DifferentialEvolution;
import org.uma.jmetal.algorithm.singleobjective.evolutionstrategy.CovarianceMatrixAdaptationEvolutionStrategy;
import org.uma.jmetal.operator.crossover.impl.DifferentialEvolutionCrossover;
import org.uma.jmetal.operator.selection.impl.DifferentialEvolutionSelection;
import org.uma.jmetal.problem.Problem;
import org.uma.jmetal.problem.doubleproblem.impl.AbstractDoubleProblem;
import org.uma.jmetal.solution.doublesolution.DoubleSolution;
import org.uma.jmetal.util.evaluator.impl.SequentialSolutionListEvaluator;
import org.uma.jmetal.util.pseudorandom.JMetalRandom;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.lang.reflect.Field;
import java.nio.charset.StandardCharsets;
import java.util.Collections;
import java.util.List;
import java.util.Locale;
import java.util.Random;
import java.util.function.ToDoubleFunction;
import java.util.logging.Level;
import java.util.logging.LogManager;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

public final class Bench {

    static final String LIBRARY = "jmetal";
    // the untimed JIT warm-up run (rule 4.2)
    static final long WARM_UP_SEED = 999_999L;
    static final long WARM_UP_EVALUATIONS = 50_000L;

    // ---------------------------------------------------------------------------------------------
    // Fitness functions, identical to problems.py
    // ---------------------------------------------------------------------------------------------

    /**
     * The shift of Rastrigin: s_i = 0.8 upper (2 ((37 i + 11) mod 101) / 101 - 1), with `upper`
     * the box's upper bound, computed in this order (problems.py).
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

    /** A problem: its function and bounds. */
    record RealFunction(ToDoubleFunction<double[]> function, double lower, double upper) {}

    static RealFunction realFunction(String name, int size) {
        return switch (name) {
            case "rastrigin" -> {
                double[] s = shift(size, 5.12);
                yield new RealFunction(x -> rastrigin(x, s), -5.12, 5.12);
            }
            case "rosenbrock" -> new RealFunction(Bench::rosenbrock, -5.0, 10.0);
            default -> null;
        };
    }

    // ---------------------------------------------------------------------------------------------
    // The budget: counts the evaluations, keeps the best value and its solution
    // ---------------------------------------------------------------------------------------------

    /** Thrown by the problem to end a run at the target, the budget or the time cap. */
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
        // NaN: no target, the run ends only at the budget or the time cap (best <= NaN is false)
        final double target;
        long evaluations;
        double best = Double.POSITIVE_INFINITY;
        boolean timeUp;
        double[] solution;
        // evaluated solutions outside the problem's bounds (rule 2.4)
        long outside;
        // the first evaluation whose value reaches the target, and the clock then (-1: not yet)
        long firstHitEvaluations = -1;
        double firstHitSeconds;
        // the evaluations at the end of the last generation, and that generation's (rule 2.3)
        long generationEnd;
        long lastGeneration;

        Budget(long maxEvaluations, double maxSeconds, double target) {
            this.maxEvaluations = maxEvaluations;
            this.start = System.nanoTime();
            this.deadline = start + (long) (maxSeconds * 1e9);
            this.target = target;
        }

        /** Called before an evaluation: ends the run at the budget or the time cap. */
        void before() {
            if (evaluations >= maxEvaluations) throw STOP;
            // the clock every 64 evaluations
            if ((evaluations & 63) == 0 && System.nanoTime() >= deadline) {
                timeUp = true;
            }
            if (timeUp) throw STOP;
        }

        /** Counts an evaluation (its solution is kept first), ends the run at the target. */
        void record(double value) {
            evaluations++;
            if (value < best) best = value;
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

        /** The generations of a method that doesn't call back between them: all of `size` (rule 2.3). */
        void fixedGenerations(long size) {
            if (evaluations > 0) generationEnd = (evaluations - 1) / size * size;
        }

        /** The "first_hit" field. */
        String firstHit() {
            if (firstHitEvaluations < 0) return "null";
            return "{\"evaluations\":" + firstHitEvaluations + ",\"time_s\":"
                + String.format(Locale.ROOT, "%.6f", firstHitSeconds) + "}";
        }

        boolean reached() {
            return best <= target;
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
    // The problem as a jMetal problem (jMetal minimizes)
    // ---------------------------------------------------------------------------------------------

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
            List<Double> variables = solution.variables();
            double[] x = new double[variables.size()];
            for (int i = 0; i < x.length; i++) x[i] = variables.get(i);
            budget.bounds(x, lower, upper);
            double value = function.applyAsDouble(x);
            solution.objectives()[0] = value;
            if (value < budget.best) budget.solution = x;
            budget.record(value);
            return solution;
        }
    }

    /**
     * jMetal's sequential evaluator (DifferentialEvolution's default), counting the generations: a
     * call evaluates the initial population or a generation's trials, and ends the previous one.
     */
    static final class CountingEvaluator extends SequentialSolutionListEvaluator<DoubleSolution> {
        final Budget budget;
        long calls;

        CountingEvaluator(Budget budget) {
            this.budget = budget;
        }

        @Override
        public List<DoubleSolution> evaluate(List<DoubleSolution> solutionList, Problem<DoubleSolution> problem) {
            calls++;
            budget.endGeneration();
            return super.evaluate(solutionList, problem);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // The methods: each returns the generations of its run
    // ---------------------------------------------------------------------------------------------

    /**
     * DE/rand/1/bin, matched (docs/benchmarks/libraries/jmetal.md, "Rastrigin 30"): jmetal-algorithm's
     * DifferentialEvolution, as examples/singleobjective/DifferentialEvolutionRunner.java builds it,
     * with the definition's settings: population 100 (uniform in the box, the problem's
     * createSolution()), DifferentialEvolutionSelection (3 distinct, other than the target),
     * DifferentialEvolutionCrossover with CR 0.9, F 0.5 and RAND_1_BIN. It builds all the trials
     * from the last generation, and a trial replaces its target unless the target is better.
     * Bounds: the crossover sets a trial's gene outside the box to the bound
     * (RepairDoubleSolutionWithBoundValue, its default). Its only stop, its maximum of evaluations,
     * is out of reach. Rastrigin 30 has no target: the run ends at the budget or the time cap.
     */
    static long de(Budget budget, RealProblem problem) {
        var evaluator = new CountingEvaluator(budget);
        var de = new DifferentialEvolution(problem, Integer.MAX_VALUE, 100,
            new DifferentialEvolutionCrossover(0.9, 0.5, DifferentialEvolutionCrossover.DE_VARIANT.RAND_1_BIN),
            new DifferentialEvolutionSelection(), evaluator);
        try {
            de.run();
        } catch (Stop stop) {
            // the budget or the time cap
        }
        // the generations after the initial population
        return Math.max(0, evaluator.calls - 1);
    }

    /**
     * CMA-ES, matched (docs/benchmarks/libraries/jmetal.md, "Rosenbrock 10"): jmetal-algorithm's
     * CovarianceMatrixAdaptationEvolutionStrategy, as
     * examples/singleobjective/CovarianceMatrixAdaptationEvolutionStrategyRunner.java builds it,
     * with the Builder's options set to the definition: λ 10 (its default; μ 5 and the positive
     * log weights follow from it), σ 4.5, and the initial mean (setTypicalX) a point drawn uniformly
     * in the box with JMetalRandom. Its learning rates are Hansen's. It first evaluates an initial
     * population of λ uniform points that its updates don't use. Bounds: it clips every sample to
     * the box (Bounds.restrict). Its maximum of evaluations is out of reach.
     * Seed (rule 5.2): jMetal draws the samples from its own `new Random(System.currentTimeMillis())`,
     * which a user can't set; the adapter replaces it by a java.util.Random seeded from JMetalRandom,
     * by reflection, before the run. The algorithm is unchanged.
     * The run can end by itself, as jMetal's own code has it, in two cases (see the page): when
     * the eigendecomposition fails its check (checkEigenCorrectness sets the evaluations to the
     * maximum), and when NaN reaches the covariance matrix and CMAESUtils.tql2 throws
     * ArrayIndexOutOfBoundsException. Neither is worked around: the run ends there, with the best
     * value found, and prints why in "ended_by".
     */
    static long cmaes(Budget budget, RealProblem problem, int size, String[] endedBy) {
        double[] mean = new double[size];
        for (int i = 0; i < size; i++) mean[i] = JMetalRandom.getInstance().nextDouble(problem.lower, problem.upper);
        var cmaes = new CovarianceMatrixAdaptationEvolutionStrategy.Builder(problem)
            .setLambda(10)
            .setSigma(4.5)
            .setTypicalX(mean)
            .setMaxEvaluations(Integer.MAX_VALUE)
            .build();
        seedCmaes(cmaes, JMetalRandom.getInstance().nextInt(0, Integer.MAX_VALUE - 1));
        try {
            cmaes.run();
            endedBy[0] = "checkEigenCorrectness";
        } catch (Stop stop) {
            // the target, the budget or the time cap
        } catch (ArrayIndexOutOfBoundsException crash) {
            endedBy[0] = "tql2 ArrayIndexOutOfBoundsException";
        }
        // λ evaluations a generation, the initial population included
        budget.fixedGenerations(10);
        return budget.evaluations / 10;
    }

    static void seedCmaes(CovarianceMatrixAdaptationEvolutionStrategy cmaes, long seed) {
        try {
            Field field = CovarianceMatrixAdaptationEvolutionStrategy.class.getDeclaredField("rand");
            field.setAccessible(true);
            field.set(cmaes, new Random(seed));
        } catch (ReflectiveOperationException error) {
            throw new IllegalStateException(error);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // The runs
    // ---------------------------------------------------------------------------------------------

    record Args(String problem, int size, String mode, long seedFrom, long seedTo, long maxEvaluations,
                double maxSeconds) {}

    /** The solver of the matched suite for a scenario: "de", "cma_es", or null (nothing to run). */
    static String solver(Args args) {
        if (!args.mode().equals("matched")) return null;
        if (args.problem().equals("rastrigin") && args.size() == 30) return "de";
        if (args.problem().equals("rosenbrock") && args.size() == 10) return "cma_es";
        // not run: matched OneMax 1000 (no generational replacement without elitism, no two-point
        // crossover for bits; see the page)
        return null;
    }

    static void run(Args args, boolean print) {
        String solver = solver(args);
        if (solver == null) return;
        RealFunction real = realFunction(args.problem(), args.size());
        for (long seed = args.seedFrom(); seed <= args.seedTo(); seed++) {
            JMetalRandom.getInstance().setSeed(seed);
            String[] endedBy = {null};
            // Rastrigin 30 has no target (a fixed budget, measured by the error at its end);
            // Rosenbrock 10's is 0.01
            double target = solver.equals("de") ? Double.NaN : 0.01;
            // the clock starts here (Budget.start), before the algorithm creates its population
            Budget budget = new Budget(args.maxEvaluations(), args.maxSeconds(), target);
            RealProblem problem = new RealProblem(args.problem(), args.size(), real, budget);
            long generations = solver.equals("de") ? de(budget, problem) : cmaes(budget, problem, args.size(), endedBy);
            double time = (System.nanoTime() - budget.start) / 1e9;
            if (!print) continue;
            System.out.println("{\"library\":\"" + LIBRARY + "\",\"solver\":\"" + solver
                + "\",\"problem\":\"" + args.problem() + "\",\"size\":" + args.size()
                + ",\"mode\":\"" + args.mode() + "\",\"seed\":" + seed
                + ",\"time_s\":" + String.format(Locale.ROOT, "%.6f", time)
                + ",\"generations\":" + generations + ",\"evaluations\":" + budget.evaluations
                + ",\"last_generation\":" + budget.lastGeneration()
                + ",\"best\":" + number(budget.best) + ",\"target\":" + (Double.isNaN(target) ? "null" : number(target))
                + ",\"success\":" + budget.reached() + ",\"first_hit\":" + budget.firstHit()
                + ",\"outside\":" + budget.outside
                + (endedBy[0] != null ? ",\"ended_by\":\"" + endedBy[0] + "\"" : "")
                + ",\"solution\":" + json(budget.solution) + "}");
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
        for (int i = 0; i < parts.length; i++) x[i] = Double.parseDouble(parts[i].trim());
        return x;
    }

    static void values(String problem, int size) throws IOException {
        RealFunction real = realFunction(problem, size);
        if (real == null) throw new IllegalArgumentException("unknown problem " + problem);
        BufferedReader in = new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8));
        StringBuilder out = new StringBuilder();
        String line;
        while ((line = in.readLine()) != null) {
            if (line.isBlank()) continue;
            out.append(number(real.function().applyAsDouble(parse(line)))).append('\n');
        }
        System.out.print(out);
    }

    // ---------------------------------------------------------------------------------------------

    static String number(double value) {
        if (value == Math.rint(value) && Math.abs(value) < 1e15) return Long.toString((long) value);
        return Double.toString(value);
    }

    static String json(double[] x) {
        StringBuilder text = new StringBuilder("[");
        if (x != null) {
            for (int i = 0; i < x.length; i++) text.append(i > 0 ? "," : "").append(x[i]);
        }
        return text.append(']').toString();
    }

    /** The version of the jmetal-algorithm jar on the class path (its manifest has none). */
    static String version() {
        String jar = DifferentialEvolution.class.getProtectionDomain().getCodeSource().getLocation().getPath();
        Matcher matcher = Pattern.compile("jmetal-algorithm-([^/]+)\\.jar$").matcher(jar);
        return matcher.find() ? matcher.group(1) : "unknown";
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
        // JIT warm-up (rule 4.2): the solver once on the same problem, untimed and unprinted
        run(new Args(args.problem(), args.size(), args.mode(), WARM_UP_SEED, WARM_UP_SEED,
            WARM_UP_EVALUATIONS, args.maxSeconds()), false);
        run(args, true);
    }
}
