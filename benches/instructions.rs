//! Instruction counts of the hot paths, with gungraun (Callgrind). Unlike wall time, they are
//! exact and noise-free, so CI can show the effect of every change.
//!
//! Needs Linux, Valgrind and the runner of the same version as the gungraun dependency:
//!
//! ```text
//! cargo install gungraun-runner --version 0.19.4
//! cargo bench --bench instructions
//! ```

use genoxide::Objective::Minimize;
use genoxide::gp::{
    Columns, ConstantMutation, Gp, HoistMutation, Mutations, PointMutation, PrimitiveSet,
    ShrinkMutation, SubtreeCrossover, Tree, TreeMutation,
};
use genoxide::multi::problems::{Dtlz2, MultiProblem, Zdt1};
use genoxide::multi::{self, MultiFitnessFunction};
use genoxide::prelude::*;
use gungraun::prelude::*;
use rand::RngExt;
use std::hint::black_box;

type OneMaxGa = Ga<Binary, Tournament, UniformCrossover, BitFlip>;

fn one_max(genome: &Bits) -> f64 {
    genome.count_ones() as f64
}

fn one_max_ga(len: usize) -> OneMaxGa {
    Ga::builder(Binary::new(len).unwrap())
        .population_size(100)
        .select(Tournament::new(3).unwrap())
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::per_gene(1.0 / len as f64).unwrap())
        .seed(0)
        .build()
        .unwrap()
}

fn tell_one_max(ga: &mut OneMaxGa) {
    let fitness: Vec<Fitness> = ga
        .ask()
        .iter()
        .map(|genome| Fitness::new(one_max(genome)))
        .collect();
    ga.tell(&fitness).unwrap();
}

// an algorithm with its initial population evaluated
fn evaluated_ga(len: usize) -> OneMaxGa {
    let mut ga = one_max_ga(len);
    tell_one_max(&mut ga);
    ga
}

fn sphere(x: &Reals) -> f64 {
    x.iter().map(|xi| xi * xi).sum::<f64>()
}

fn tell_sphere<A: Algorithm<Genome = Reals>>(algorithm: &mut A) {
    let fitness: Vec<Fitness> = algorithm
        .ask()
        .iter()
        .map(|x| Fitness::new(sphere(x)))
        .collect();
    algorithm.tell(&fitness).unwrap();
}

// SHADE (the defaults) on 30 genes, with its initial population evaluated
fn evaluated_de(len: usize) -> De {
    let mut de = De::builder(Real::uniform(len, -5.0..=5.0).unwrap())
        .minimize()
        .seed(0)
        .build()
        .unwrap();
    tell_sphere(&mut de);
    de
}

// a (5/5_I, 35)-ES with a step size per gene, with its initial parents evaluated
fn evaluated_es(len: usize) -> Es {
    let mut es = Es::builder(Real::uniform(len, -5.0..=5.0).unwrap())
        .parents(5)
        .offspring(35)
        .minimize()
        .seed(0)
        .build()
        .unwrap();
    tell_sphere(&mut es);
    es
}

// CMA-ES with its default population (10 samples for 10 genes, with an eigendecomposition every
// generation; 17 for 100 genes), from the center with a step of 0.1, so that samples are rarely
// drawn again at the bounds, with its initial samples evaluated
fn evaluated_cmaes((len, covariance): (usize, cmaes::Covariance)) -> Cmaes {
    let mut cmaes = Cmaes::builder(Real::uniform(len, -5.0..=5.0).unwrap())
        .covariance(covariance)
        .initial_mean(Reals::from(vec![0.0; len]))
        .initial_step(0.1)
        .minimize()
        .seed(0)
        .build()
        .unwrap();
    tell_sphere(&mut cmaes);
    cmaes
}

// a swarm of 40 particles, with its initial positions evaluated
fn evaluated_pso(len: usize) -> Pso {
    let mut pso = Pso::builder(Real::uniform(len, -5.0..=5.0).unwrap())
        .population_size(40)
        .minimize()
        .seed(0)
        .build()
        .unwrap();
    tell_sphere(&mut pso);
    pso
}

// the length of a tour of cities on a line, at their numbers
fn line_tour(order: &Order) -> f64 {
    let n = order.len();
    (0..n)
        .map(|i| order[i].abs_diff(order[(i + 1) % n]) as f64)
        .sum()
}

type TourSearch = LocalSearch<Permutation, InversionMutation>;

// 2-opt hill climbing on a tour, 4 neighbors per step, with its initial tour evaluated
fn evaluated_local_search(len: usize) -> TourSearch {
    let mut search = LocalSearch::builder(Permutation::new(len).unwrap())
        .neighbor(InversionMutation)
        .neighbors(4)
        .minimize()
        .seed(0)
        .build()
        .unwrap();
    let fitness: Vec<Fitness> = search
        .ask()
        .iter()
        .map(|order| Fitness::new(line_tour(order)))
        .collect();
    search.tell(&fitness).unwrap();
    search
}

fn two_genomes(len: usize) -> (Binary, Bits, Bits, StreamRng) {
    let binary = Binary::new(len).unwrap();
    let mut rng = StreamRng::seed_from_u64(0);
    let a = binary.random_genome(&mut rng);
    let b = binary.random_genome(&mut rng);
    (binary, a, b, rng)
}

fn two_reals(len: usize) -> (Real, Reals, Reals, StreamRng) {
    let real = Real::uniform(len, -5.12..=5.12).unwrap();
    let mut rng = StreamRng::seed_from_u64(0);
    let a = real.random_genome(&mut rng);
    let b = real.random_genome(&mut rng);
    (real, a, b, rng)
}

fn two_orders(len: usize) -> (Permutation, Order, Order, StreamRng) {
    let permutation = Permutation::new(len).unwrap();
    let mut rng = StreamRng::seed_from_u64(0);
    let a = permutation.random_genome(&mut rng);
    let b = permutation.random_genome(&mut rng);
    (permutation, a, b, rng)
}

// every operator bench below applies its operator 100 times
const REPEATS: usize = 100;

fn evaluated_population(size: usize) -> (Population<Bits>, StreamRng) {
    let binary = Binary::new(100).unwrap();
    let mut rng = StreamRng::seed_from_u64(0);
    let population = (0..size)
        .map(|_| {
            let mut individual = Individual::new(binary.random_genome(&mut rng));
            individual.set_fitness(Fitness::new(one_max(individual.genome())));
            individual
        })
        .collect();
    (population, rng)
}

type OneMaxSteady = SteadyGa<Binary, Tournament, UniformCrossover, BitFlip>;

// a steady-state GA for asynchronous evaluation, with its population of `size` evaluated
fn evaluated_steady(size: usize) -> OneMaxSteady {
    let mut steady = Ga::builder(Binary::new(100).unwrap())
        .population_size(size)
        .select(Tournament::new(3).unwrap())
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::per_gene(0.01).unwrap())
        .seed(0)
        .build_steady()
        .unwrap();
    for _ in 0..size {
        let genome = steady.propose();
        let fitness = Fitness::new(one_max(&genome));
        steady.receive(genome, fitness).unwrap();
    }
    steady
}

type OneMaxIslands = Islands<OneMaxGa>;

// four islands of 25, with their initial populations evaluated
fn evaluated_islands(len: usize) -> OneMaxIslands {
    let islands = (0..4)
        .map(|seed| {
            Ga::builder(Binary::new(len).unwrap())
                .population_size(25)
                .select(Tournament::new(3).unwrap())
                .crossover(UniformCrossover::new())
                .mutate(BitFlip::per_gene(1.0 / len as f64).unwrap())
                .seed(seed)
                .build()
                .unwrap()
        })
        .collect();
    let mut islands = Islands::builder(islands).interval(1).build().unwrap();
    let fitness: Vec<Fitness> = islands
        .ask()
        .iter()
        .map(|genome| Fitness::new(one_max(genome)))
        .collect();
    islands.tell(&fitness).unwrap();
    islands
}

#[library_benchmark]
#[bench::binary_1000(1_000)]
fn random_binary_genome(len: usize) -> Bits {
    let binary = Binary::new(len).unwrap();
    black_box(binary.random_genome(&mut StreamRng::seed_from_u64(0)))
}

#[library_benchmark]
#[bench::permutation_100(100)]
fn random_permutation(len: usize) -> Order {
    let permutation = Permutation::new(len).unwrap();
    black_box(permutation.random_genome(&mut StreamRng::seed_from_u64(0)))
}

#[library_benchmark]
#[bench::binary_1000(setup = two_genomes, args = (1_000))]
fn uniform_crossover((binary, mut a, mut b, mut rng): (Binary, Bits, Bits, StreamRng)) -> Bits {
    UniformCrossover::new().crossover(&binary, &mut a, &mut b, &mut rng);
    black_box(a)
}

#[library_benchmark]
#[bench::binary_1000(setup = two_genomes, args = (1_000))]
fn bit_flip((binary, mut a, _, mut rng): (Binary, Bits, Bits, StreamRng)) -> Bits {
    BitFlip::per_gene(0.001)
        .unwrap()
        .mutate(&binary, &mut a, &mut rng);
    black_box(a)
}

#[library_benchmark]
#[bench::of_100(setup = evaluated_population, args = (100))]
fn tournament((population, mut rng): (Population<Bits>, StreamRng)) -> Vec<usize> {
    black_box(
        Tournament::new(3)
            .unwrap()
            .select(&population, Objective::Maximize, 100, &mut rng),
    )
}

#[library_benchmark]
#[bench::binary_1000(setup = two_genomes, args = (1_000))]
fn two_point_crossover((binary, mut a, mut b, mut rng): (Binary, Bits, Bits, StreamRng)) -> Bits {
    for _ in 0..REPEATS {
        PointCrossover::two_point().crossover(&binary, &mut a, &mut b, &mut rng);
    }
    black_box(a)
}

#[library_benchmark]
#[bench::real_30(setup = two_reals, args = (30))]
fn simulated_binary_crossover(
    (real, mut a, mut b, mut rng): (Real, Reals, Reals, StreamRng),
) -> Reals {
    let sbx = SimulatedBinaryCrossover::new(15.0).unwrap();
    for _ in 0..REPEATS {
        sbx.crossover(&real, &mut a, &mut b, &mut rng);
    }
    black_box(a)
}

#[library_benchmark]
#[bench::real_30(setup = two_reals, args = (30))]
fn polynomial_mutation((real, mut a, _, mut rng): (Real, Reals, Reals, StreamRng)) -> Reals {
    let mutation = PolynomialMutation::per_gene(0.1, 20.0).unwrap();
    for _ in 0..REPEATS {
        mutation.mutate(&real, &mut a, &mut rng);
    }
    black_box(a)
}

#[library_benchmark]
#[bench::real_30(setup = two_reals, args = (30))]
fn gaussian_mutation((real, mut a, _, mut rng): (Real, Reals, Reals, StreamRng)) -> Reals {
    let mutation = GaussianMutation::per_gene(0.1, 0.05).unwrap();
    for _ in 0..REPEATS {
        mutation.mutate(&real, &mut a, &mut rng);
    }
    black_box(a)
}

fn integers(len: usize) -> (Integer, Integers, StreamRng) {
    let integer = Integer::uniform(len, -10..=10).unwrap();
    let mut rng = StreamRng::seed_from_u64(0);
    let genome = integer.random_genome(&mut rng);
    (integer, genome, rng)
}

#[library_benchmark]
#[bench::integer_50(setup = integers, args = (50))]
fn uniform_mutation((integer, mut genome, mut rng): (Integer, Integers, StreamRng)) -> Integers {
    let mutation = UniformMutation::count(1).unwrap();
    for _ in 0..REPEATS {
        mutation.mutate(&integer, &mut genome, &mut rng);
    }
    black_box(genome)
}

#[library_benchmark]
#[bench::permutation_100(setup = two_orders, args = (100))]
fn order_crossover(
    (permutation, mut a, mut b, mut rng): (Permutation, Order, Order, StreamRng),
) -> Order {
    for _ in 0..REPEATS {
        OrderCrossover.crossover(&permutation, &mut a, &mut b, &mut rng);
    }
    black_box(a)
}

#[library_benchmark]
#[bench::permutation_100(setup = two_orders, args = (100))]
fn edge_recombination_crossover(
    (permutation, mut a, mut b, mut rng): (Permutation, Order, Order, StreamRng),
) -> Order {
    for _ in 0..REPEATS {
        EdgeRecombinationCrossover.crossover(&permutation, &mut a, &mut b, &mut rng);
    }
    black_box(a)
}

#[library_benchmark]
#[bench::permutation_100(setup = two_orders, args = (100))]
fn inversion_mutation(
    (permutation, mut a, _, mut rng): (Permutation, Order, Order, StreamRng),
) -> Order {
    for _ in 0..REPEATS {
        InversionMutation.mutate(&permutation, &mut a, &mut rng);
    }
    black_box(a)
}

#[library_benchmark]
#[bench::of_100(setup = evaluated_population, args = (100))]
fn rank((population, mut rng): (Population<Bits>, StreamRng)) -> Vec<usize> {
    black_box(Rank::default().select(&population, Objective::Maximize, 100, &mut rng))
}

// the portable math of fitness functions, over 1000 arguments
#[library_benchmark]
#[bench::of_1000(setup = two_reals, args = (1_000))]
fn portable_math((_, a, b, _): (Real, Reals, Reals, StreamRng)) -> f64 {
    let mut sum = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        sum += genoxide::math::sin(x) + genoxide::math::cos(y) + genoxide::math::exp(x);
        sum += genoxide::math::ln(y.abs()) + genoxide::math::powf(x.abs(), y);
    }
    black_box(sum)
}

// one generation: breeding, evaluation and survival
#[library_benchmark]
#[bench::one_max_1000(setup = evaluated_ga, args = (1_000))]
fn generation(mut ga: OneMaxGa) -> OneMaxGa {
    tell_one_max(&mut ga);
    black_box(ga)
}

// one generation of differential evolution: trials, evaluation and selection
#[library_benchmark]
#[bench::sphere_30(setup = evaluated_de, args = (30))]
fn de_generation(mut de: De) -> De {
    tell_sphere(&mut de);
    black_box(de)
}

// one generation of an evolution strategy: offspring, evaluation and selection
#[library_benchmark]
#[bench::sphere_30(setup = evaluated_es, args = (30))]
fn es_generation(mut es: Es) -> Es {
    tell_sphere(&mut es);
    black_box(es)
}

// one generation of four islands, with a migration
#[library_benchmark]
#[bench::one_max_1000(setup = evaluated_islands, args = (1_000))]
fn islands_generation(mut islands: OneMaxIslands) -> OneMaxIslands {
    let fitness: Vec<Fitness> = islands
        .ask()
        .iter()
        .map(|genome| Fitness::new(one_max(genome)))
        .collect();
    islands.tell(&fitness).unwrap();
    black_box(islands)
}

// 100 results of a steady-state GA of 100, each one proposed and received in turn
#[library_benchmark]
#[bench::one_max_100(setup = evaluated_steady, args = (100))]
fn steady_results(mut steady: OneMaxSteady) -> OneMaxSteady {
    for _ in 0..100 {
        let genome = steady.propose();
        let fitness = Fitness::new(one_max(&genome));
        black_box(steady.receive(genome, fitness).unwrap());
    }
    steady
}

// the CI's `run`, with statistics and a hall of fame
#[library_benchmark]
#[bench::one_max_100_50_generations(100)]
fn run_observed(len: usize) -> (Outcome<Bits>, Statistics, HallOfFame<Bits>) {
    let mut statistics = Statistics::new();
    let mut hall_of_fame = HallOfFame::new(10).unwrap();
    let outcome = Engine::new(one_max_ga(len), one_max)
        .stop_when(Stop::generations(50))
        .observe(&mut statistics)
        .observe(&mut hall_of_fame)
        .run()
        .unwrap();
    black_box((outcome, statistics, hall_of_fame))
}

// one generation of CMA-ES: samples, evaluation and the update of the distribution
#[library_benchmark]
#[bench::sphere_10(setup = evaluated_cmaes, args = ((10, cmaes::Covariance::Full)))]
#[bench::sphere_100_diagonal(setup = evaluated_cmaes, args = ((100, cmaes::Covariance::Diagonal)))]
fn cmaes_generation(mut cmaes: Cmaes) -> Cmaes {
    tell_sphere(&mut cmaes);
    black_box(cmaes)
}

// one generation of particle swarm optimization: flight, evaluation and the personal bests
#[library_benchmark]
#[bench::sphere_30(setup = evaluated_pso, args = (30))]
fn pso_generation(mut pso: Pso) -> Pso {
    tell_sphere(&mut pso);
    black_box(pso)
}

// 100 steps of local search: neighbors, evaluation and acceptance
#[library_benchmark]
#[bench::tour_100(setup = evaluated_local_search, args = (100))]
fn local_search_steps(mut search: TourSearch) -> TourSearch {
    for _ in 0..100 {
        let fitness: Vec<Fitness> = search
            .ask()
            .iter()
            .map(|order| Fitness::new(line_tour(order)))
            .collect();
        search.tell(&fitness).unwrap();
    }
    black_box(search)
}

type ZdtNsga2 = Nsga2<Real, SimulatedBinaryCrossover, PolynomialMutation, 2>;

// NSGA-II on ZDT1 (30 genes, 100 individuals), with its initial population evaluated
fn evaluated_nsga2(len: usize) -> ZdtNsga2 {
    let mut nsga2 = Nsga2::builder(Real::uniform(len, 0.0..=1.0).unwrap(), [Minimize; 2])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
        .mutate(PolynomialMutation::per_gene(1.0 / len as f64, 20.0).unwrap())
        .seed(0)
        .build()
        .unwrap();
    tell_multi(&mut nsga2, Zdt1::new(len));
    nsga2
}

type DtlzNsga3 = Nsga3<Real, SimulatedBinaryCrossover, PolynomialMutation, 3>;

// NSGA-III on DTLZ2 with 3 objectives (12 genes, 91 directions), with its initial population
// evaluated
fn evaluated_nsga3(divisions: usize) -> DtlzNsga3 {
    let problem = Dtlz2::<3>::new(12);
    let mut nsga3 = Nsga3::builder(
        problem.representation(),
        [Minimize; 3],
        multi::das_dennis::<3>(divisions),
    )
    .crossover(SimulatedBinaryCrossover::new(30.0).unwrap())
    .mutate(PolynomialMutation::per_gene(1.0 / 12.0, 20.0).unwrap())
    .seed(0)
    .build()
    .unwrap();
    tell_multi(&mut nsga3, problem);
    nsga3
}

fn tell_multi<A, F, const M: usize>(algorithm: &mut A, problem: F)
where
    A: MultiObjectiveAlgorithm<M, Genome = Reals>,
    F: MultiFitnessFunction<Reals, M, Output = [f64; M]>,
{
    let scores: Vec<multi::Scores<M>> = algorithm
        .ask()
        .iter()
        .map(|x| multi::Scores::new(problem.evaluate(x)))
        .collect();
    algorithm.tell(&scores).unwrap();
}

// the scores of `size` random points on and behind the positive eighth of the unit sphere
fn sphere_scores(size: usize) -> Vec<multi::Scores<3>> {
    let problem = Dtlz2::<3>::new(12);
    let real = problem.representation();
    let mut rng = StreamRng::seed_from_u64(0);
    (0..size)
        .map(|_| multi::Scores::new(problem.evaluate(&real.random_genome(&mut rng))))
        .collect()
}

// one generation of NSGA-II: breeding, evaluation, non-dominated sorting and crowding
#[library_benchmark]
#[bench::zdt1_30(setup = evaluated_nsga2, args = (30))]
fn nsga2_generation(mut nsga2: ZdtNsga2) -> ZdtNsga2 {
    tell_multi(&mut nsga2, Zdt1::new(30));
    black_box(nsga2)
}

// one generation of NSGA-III: breeding, evaluation, sorting, normalization and niching
#[library_benchmark]
#[bench::dtlz2_91(setup = evaluated_nsga3, args = (12))]
fn nsga3_generation(mut nsga3: DtlzNsga3) -> DtlzNsga3 {
    tell_multi(&mut nsga3, Dtlz2::<3>::new(12));
    black_box(nsga3)
}

#[library_benchmark]
#[bench::three_objectives_200(setup = sphere_scores, args = (200))]
fn non_dominated_sort(scores: Vec<multi::Scores<3>>) -> Vec<Vec<usize>> {
    black_box(multi::non_dominated_sort(&scores, &[Minimize; 3]))
}

#[library_benchmark]
#[bench::three_objectives_200(setup = sphere_scores, args = (200))]
fn hypervolume(scores: Vec<multi::Scores<3>>) -> f64 {
    let front: Vec<[f64; 3]> = scores.iter().filter_map(multi::Scores::values).collect();
    black_box(multi::indicator::hypervolume(
        &front,
        &[4.0; 3],
        &[Minimize; 3],
    ))
}

#[library_benchmark]
#[bench::one_max_100_50_generations(100)]
fn run(len: usize) -> Outcome<Bits> {
    black_box(
        Engine::new(one_max_ga(len), one_max)
            .stop_when(Stop::generations(50))
            .run()
            .unwrap(),
    )
}

// genetic programming: {+, −, ×, analytic quotient, sin, cos, x, constants}, as in the plan's
// measurements
#[derive(Clone, Copy, Debug)]
enum Op {
    Add,
    Sub,
    Mul,
    Aq,
    Sin,
    Cos,
    X,
}

fn formula_gp() -> Gp<Op> {
    let mut set = PrimitiveSet::builder();
    let real = set.new_type("real");
    set.function("add", Op::Add, [real, real], real)
        .function("sub", Op::Sub, [real, real], real)
        .function("mul", Op::Mul, [real, real], real)
        .function("aq", Op::Aq, [real, real], real)
        .function("sin", Op::Sin, [real], real)
        .function("cos", Op::Cos, [real], real)
        .terminal("x", Op::X, real)
        .constants(real, genoxide::gp::Constants::uniform(-1.0..=1.0).unwrap());
    Gp::builder(set.build(real).unwrap()).build().unwrap()
}

// `count` random trees of 45 to 55 nodes
fn formula_trees(count: usize) -> (Gp<Op>, Vec<Tree>, StreamRng) {
    let gp = formula_gp();
    let mut rng = StreamRng::seed_from_u64(0);
    let mut trees = Vec::with_capacity(count);
    while trees.len() < count {
        let tree = gp.grow(8, &mut rng).unwrap();
        if (45..=55).contains(&tree.len()) {
            trees.push(tree);
        }
    }
    (gp, trees, rng)
}

fn formula_columns(op: Op, args: &[&[f64]], out: &mut [f64], xs: &[f64]) {
    let binary = |out: &mut [f64], f: fn(f64, f64) -> f64| {
        for ((out, &a), &b) in out.iter_mut().zip(args[0]).zip(args[1]) {
            *out = f(a, b);
        }
    };
    let unary = |out: &mut [f64], f: fn(f64) -> f64| {
        for (out, &a) in out.iter_mut().zip(args[0]) {
            *out = f(a);
        }
    };
    match op {
        Op::Add => binary(out, |a, b| a + b),
        Op::Sub => binary(out, |a, b| a - b),
        Op::Mul => binary(out, |a, b| a * b),
        Op::Aq => binary(out, |a, b| a / (1.0 + b * b).sqrt()),
        Op::Sin => unary(out, genoxide::math::sin),
        Op::Cos => unary(out, genoxide::math::cos),
        Op::X => out.copy_from_slice(xs),
    }
}

#[library_benchmark]
#[bench::pairs_of_50_nodes(setup = formula_trees, args = (2 * REPEATS))]
fn subtree_crossover((gp, mut trees, mut rng): (Gp<Op>, Vec<Tree>, StreamRng)) -> Vec<Tree> {
    let crossover = SubtreeCrossover::new();
    for [a, b] in trees.as_chunks_mut::<2>().0 {
        crossover.crossover(&gp, a, b, &mut rng);
    }
    black_box(trees)
}

// every tree on 100 points; outside the benchmark, whose closures Callgrind leaves out (it toggles
// its collection on the benchmark's name)
fn evaluate_trees(gp: &Gp<Op>, trees: &[Tree]) -> f64 {
    let xs: Vec<f64> = (0..100).map(|i| f64::from(i) / 49.5 - 1.0).collect();
    let mut columns = Columns::new(xs.len());
    let mut sum = 0.0;
    for tree in trees {
        let values = tree.evaluate_columns(gp.primitives(), &mut columns, |op, args, out| {
            formula_columns(op, args, out, &xs)
        });
        sum += values[0];
    }
    sum
}

#[library_benchmark]
#[bench::trees_of_50_nodes_100_points(setup = formula_trees, args = (REPEATS))]
fn tree_columns((gp, trees, _): (Gp<Op>, Vec<Tree>, StreamRng)) -> f64 {
    black_box(evaluate_trees(&gp, &trees))
}

// `REPEATS` trees of 45 to 55 nodes, and a tree mutation: point, hoist, shrink, constant, or a
// mix of subtree, point, hoist and shrink
fn mutation_trees(kind: usize) -> (Gp<Op>, Vec<Tree>, StreamRng, TreeMutation) {
    let (gp, trees, rng) = formula_trees(REPEATS);
    let mutation = match kind {
        0 => PointMutation::count(1).unwrap().into(),
        1 => HoistMutation.into(),
        2 => ShrinkMutation.into(),
        3 => ConstantMutation::gaussian(0.1).unwrap().into(),
        _ => TreeMutation::Subtree(genoxide::gp::SubtreeMutation::new()),
    };
    (gp, trees, rng, mutation)
}

#[library_benchmark]
#[bench::point(setup = mutation_trees, args = (0))]
#[bench::hoist(setup = mutation_trees, args = (1))]
#[bench::shrink(setup = mutation_trees, args = (2))]
#[bench::constant(setup = mutation_trees, args = (3))]
#[bench::subtree(setup = mutation_trees, args = (4))]
fn tree_mutation(
    (gp, mut trees, mut rng, mutation): (Gp<Op>, Vec<Tree>, StreamRng, TreeMutation),
) -> Vec<Tree> {
    for tree in &mut trees {
        mutation.mutate(&gp, tree, &mut rng);
    }
    black_box(trees)
}

fn mixed_mutation_trees(_: usize) -> (Gp<Op>, Vec<Tree>, StreamRng, Mutations) {
    let (gp, trees, rng) = formula_trees(REPEATS);
    let mutations = Mutations::builder()
        .subtree(0.5)
        .point(0.3)
        .hoist(0.1)
        .shrink(0.1)
        .build()
        .unwrap();
    (gp, trees, rng, mutations)
}

#[library_benchmark]
#[bench::trees_of_50_nodes(setup = mixed_mutation_trees, args = (0))]
fn tree_mutations(
    (gp, mut trees, mut rng, mutations): (Gp<Op>, Vec<Tree>, StreamRng, Mutations),
) -> Vec<Tree> {
    for tree in &mut trees {
        mutations.mutate(&gp, tree, &mut rng);
    }
    black_box(trees)
}

// a population of 100 trees of 45 to 55 nodes with random scores
fn evaluated_trees(size: usize) -> (Population<Tree>, StreamRng) {
    let (_, trees, mut rng) = formula_trees(size);
    let population = trees
        .into_iter()
        .map(|tree| {
            let mut individual = Individual::new(tree);
            individual.set_fitness(Fitness::new(rng.random_range(0.0..1.0)));
            individual
        })
        .collect();
    (population, rng)
}

#[library_benchmark]
#[bench::of_100(setup = evaluated_trees, args = (100))]
fn double_tournament((population, mut rng): (Population<Tree>, StreamRng)) -> Vec<usize> {
    black_box(DoubleTournament::new(7, 1.4).unwrap().select(
        &population,
        Objective::Minimize,
        100,
        &mut rng,
    ))
}

library_benchmark_group!(
    name = hot_paths,
    benchmarks = [
        random_binary_genome,
        random_permutation,
        uniform_crossover,
        bit_flip,
        two_point_crossover,
        simulated_binary_crossover,
        polynomial_mutation,
        gaussian_mutation,
        uniform_mutation,
        order_crossover,
        edge_recombination_crossover,
        inversion_mutation,
        tournament,
        rank,
        portable_math,
        generation,
        de_generation,
        es_generation,
        cmaes_generation,
        pso_generation,
        local_search_steps,
        run,
        islands_generation,
        steady_results,
        run_observed,
        nsga2_generation,
        nsga3_generation,
        non_dominated_sort,
        hypervolume,
        subtree_crossover,
        tree_columns,
        tree_mutation,
        tree_mutations,
        double_tournament
    ]
);

main!(library_benchmark_groups = hot_paths);
