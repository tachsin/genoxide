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
 * - Time (rules 4.1 and 4.2): the clock starts before the engine creates its initial population and
 *   stops when the run ends. Before the timed runs, every solver runs once untimed and unprinted,
 *   with the seed 999,999, 50,000 evaluations and the scenario's time cap, so the JIT has compiled
 *   the fitness function and the engine.
 * - First hit: a single-objective run records the evaluation (and the time) at which the best value
 *   first reaches the target, in the fitness function, and prints it as "first_hit".
 * - Keeping going (rule 2.2): an evolution stream has no end of its own; the examples end it with a
 *   generation limit (a budget, replaced by the scenario's) or with Limits.bySteadyFitness (a
 *   convergence criterion: the attempt ends and the run restarts from a new random population, see
 *   evolve()). Every run ends at the target, the budget or the time cap, checked after each
 *   generation.
 * - Bounds (rule 2.4): the continuous runs count the evaluated solutions outside the problem's
 *   bounds, in the fitness function, and print them as "outside".
 * - The multi-objective scenarios aren't run (rule 6.1): they use the library's own SBX and
 *   polynomial mutation, and Jenetics has no polynomial mutation (nor NSGA-III, SPEA2, MOEA/D or
 *   SMS-EMOA). The adapter prints nothing for them.
 */

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
import io.jenetics.engine.Limits;
import io.jenetics.util.DoubleRange;
import io.jenetics.util.RandomRegistry;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Locale;
import java.util.function.ToDoubleFunction;
import java.util.random.RandomGeneratorFactory;

public final class Bench {

    static final String LIBRARY = "jenetics";
    // the untimed JIT warm-up run of every solver (rule 4.2)
    static final long WARM_UP_SEED = 999_999L;
    static final long WARM_UP_EVALUATIONS = 50_000L;

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
    record RealProblem(ToDoubleFunction<double[]> function, double lower, double upper) {}

    static RealProblem realProblem(String name, int size) {
        return switch (name) {
            case "rastrigin" -> {
                double[] s = shift(size, 5.12);
                yield new RealProblem(x -> rastrigin(x, s), -5.12, 5.12);
            }
            case "rosenbrock" -> new RealProblem(Bench::rosenbrock, -5.0, 10.0);
            case "ackley" -> {
                double[] s = shift(size, 32.768);
                yield new RealProblem(x -> ackley(x, s), -32.768, 32.768);
            }
            default -> null;
        };
    }

    // ---------------------------------------------------------------------------------------------
    // The budget: counts the evaluations, keeps the best value and its solution
    // ---------------------------------------------------------------------------------------------

    static final class Budget {
        final long maxEvaluations;
        // the clock: started when the budget is created, just before the run
        final long start;
        final long deadline;
        final boolean minimize;
        final double target;
        long evaluations;
        double best;
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

        /** Counts one evaluation; returns whether its value is the best so far. */
        boolean record(double value) {
            evaluations++;
            if (minimize ? value < best : value > best) {
                best = value;
                if (firstHitEvaluations < 0 && reached()) {
                    firstHitEvaluations = evaluations;
                    firstHitSeconds = (System.nanoTime() - start) / 1e9;
                }
                return true;
            }
            return false;
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

        /** The "first_hit" field of a single-objective run. */
        String firstHit() {
            if (firstHitEvaluations < 0) return "null";
            return "{\"evaluations\":" + firstHitEvaluations + ",\"time_s\":"
                + String.format(Locale.ROOT, "%.6f", firstHitSeconds) + "}";
        }

        boolean reached() {
            return minimize ? best <= target : best >= target;
        }

        /** The end of a run: the target, the budget or the time cap (rule 2.1). */
        boolean done() {
            return reached() || evaluations >= maxEvaluations || System.nanoTime() >= deadline;
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
            Engine<G, C> engine, Budget budget) {
        return evolve(engine, budget, 0, -1);
    }

    /**
     * Runs evolution streams until the budget ends the run; returns the generations. With
     * `steadyGenerations` > 0, an attempt also ends after that many generations without a better
     * best fitness (Limits.bySteadyFitness, the convergence criterion of the documented example),
     * and the run starts again from a new random population, the generator seeded with
     * (seed + 1) * 1,000,000 + restart (rule 2.2); the budget keeps the best and counts every
     * evaluation.
     */
    static <G extends Gene<?, G>, C extends Comparable<? super C>> long evolve(
            Engine<G, C> engine, Budget budget, int steadyGenerations, long seed) {
        long generations = 0;
        for (int restart = 0; ; restart++) {
            if (restart > 0) seed((seed + 1) * 1_000_000 + restart);
            long[] attempt = {0};
            // every generation's end, the first of each attempt (its initial population and its
            // first offspring) included: this limit comes first, so it sees every result, also
            // the one that the steady-fitness limit ends the attempt at
            var stream = engine.stream().limit(result -> {
                budget.endGeneration();
                return true;
            });
            if (steadyGenerations > 0) stream = stream.limit(Limits.bySteadyFitness(steadyGenerations));
            stream
                .limit(result -> {
                    attempt[0] = result.generation();
                    return !budget.done();
                })
                .forEach(result -> {});
            generations += attempt[0];
            if (steadyGenerations <= 0 || budget.done()) return generations;
        }
    }

    interface SingleSolver {
        long run(Budget budget, long seed);
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
                    // as DEAP's eaSimple, with Jenetics' own components: population 300, every
                    // individual an offspring (no survivors, so generational without elitism),
                    // TournamentSelector(3) (with replacement), two-point crossover
                    // (MultiPointCrossover with 2 points), Mutator, no maximal age.
                    // Differences: Jenetics' crossover picks each individual with probability p
                    // and mates it with a random other one (DEAP: consecutive pairs with
                    // probability 0.5); p = 0.25 gives DEAP's expected number of crossovers
                    // (N / 2 pairs * 0.5). Jenetics has no bit-flip mutation: its Mutator(p)
                    // gives a bit a new random value (a flip half the time), picking the
                    // individual, the chromosome and the bit each with p^(1/3); p = 0.4 / size
                    // gives DEAP's expected number of flipped bits per child (0.2 * size * 1 / size).
                    solvers.add(new Solver("ga", (budget, seed) -> evolve(
                        Engine.builder((Genotype<BitGene> gt) -> countOnes(gt, budget),
                                Genotype.of(BitChromosome.of(size, 0.5)))
                            .populationSize(300)
                            .offspringFraction(1.0)
                            .offspringSelector(new TournamentSelector<>(3))
                            .alterers(new MultiPointCrossover<>(0.25, 2), new Mutator<>(0.4 / size))
                            .maximalPhenotypeAge(Long.MAX_VALUE / 2)
                            .executor(Runnable::run)
                            .build(),
                        budget)));
                } else {
                    // "Hello World (Ones counting)", the first example of jenetics.io and of the
                    // README (BitChromosome.of(n, 0.5)), also the delivered program
                    // jenetics.example/OnesCounting.java: the engine defaults, population 50,
                    // TournamentSelector(3) for offspring and survivors, SinglePointCrossover(0.2),
                    // Mutator(0.15), 60% offspring, maximal age 70. Both end the stream with a
                    // generation limit only (limit(100), limit(10)), which the budget replaces.
                    // (The manual's listing, section 6.1, sets other values; see the page.)
                    solvers.add(new Solver("ga", (budget, seed) -> evolve(
                        Engine.builder((Genotype<BitGene> gt) -> countOnes(gt, budget),
                                Genotype.of(BitChromosome.of(size, 0.5)))
                            .executor(Runnable::run)
                            .build(),
                        budget)));
                }
            }
            case "nqueens" -> {
                minimize[0] = true;
                target[0] = 0;
                // The manual's "Traveling salesman" example (section 6.5), Jenetics' permutation
                // example: Codecs.ofPermutation, population 500, maximal age 11, SwapMutator(0.2),
                // PartiallyMatchedCrossover(0.35), the other settings the engine defaults
                // (TournamentSelector(3), 60% offspring). It ends the stream with
                // Limits.bySteadyFitness(25), a convergence criterion: an attempt ends after 25
                // generations without a better best, and the run restarts (rule 2.2); its
                // limit(250) is a budget, lifted. (The delivered program
                // jenetics.example/TravelingSalesman.java sets other values; see the page.)
                solvers.add(new Solver("ga", (budget, seed) -> evolve(
                    Engine.builder((int[] order) -> {
                                double value = nqueens(order);
                                if (budget.record(value)) budget.solution = order.clone();
                                return value;
                            }, Codecs.ofPermutation(size))
                        .optimize(Optimize.MINIMUM)
                        .maximalPhenotypeAge(11)
                        .populationSize(500)
                        .alterers(new SwapMutator<>(0.2), new PartiallyMatchedCrossover<>(0.35))
                        .executor(Runnable::run)
                        .build(),
                    budget, 25, seed)));
            }
            case "rastrigin", "rosenbrock", "ackley" -> {
                minimize[0] = true;
                target[0] = 0.01;
                RealProblem problem = realProblem(args.problem(), size);
                // The manual's "Rastrigin function" example (section 6.3), whose engine is the one
                // of its "Real function" example (6.2, and the delivered program
                // jenetics.example/RealFunction.java): Codecs.ofVector, population 500,
                // Mutator(0.03), MeanAlterer(0.6), other settings the defaults. Both end the
                // stream with Limits.bySteadyFitness(7), a convergence criterion: an attempt ends
                // after 7 generations without a better best, and the run restarts (rule 2.2; the
                // "Real function" example's limit(100) is a budget, lifted). A DoubleGene stays in
                // its range: Mutator draws a new value in the range, MeanAlterer takes the mean of
                // two values in it (rule 2.4).
                solvers.add(new Solver("ga", (budget, seed) -> evolve(
                    Engine.builder((double[] x) -> {
                                budget.bounds(x, problem.lower(), problem.upper());
                                double value = problem.function().applyAsDouble(x);
                                if (budget.record(value)) budget.solution = x.clone();
                                return value;
                            }, Codecs.ofVector(new DoubleRange(problem.lower(), problem.upper()), size))
                        .populationSize(500)
                        .optimize(Optimize.MINIMUM)
                        .alterers(new Mutator<>(0.03), new MeanAlterer<>(0.6))
                        .executor(Runnable::run)
                        .build(),
                    budget, 7, seed)));
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
                // the clock starts here (Budget.start), before the engine creates its population
                Budget budget = new Budget(args.maxEvaluations(), args.maxSeconds(), minimize[0], target[0]);
                long generations = solver.solver().run(budget, seed);
                double time = (System.nanoTime() - budget.start) / 1e9;
                if (!print) continue;
                // the continuous problems report the solutions evaluated outside the bounds
                String outside = realProblem(args.problem(), args.size()) != null
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
        RealProblem real = realProblem(problem, size);
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
        runSingle(args, print);
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
