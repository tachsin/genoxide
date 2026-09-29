//! Instruction counts of the hot paths, with gungraun (Callgrind). Unlike wall time, they are
//! exact and noise-free, so CI can show the effect of every change.
//!
//! Needs Linux, Valgrind and the runner of the same version as the gungraun dependency:
//!
//! ```text
//! cargo install gungraun-runner --version 0.19.4
//! cargo bench --bench instructions
//! ```

use genoxide::prelude::*;
use gungraun::prelude::*;
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
        run
    ]
);

main!(library_benchmark_groups = hot_paths);
