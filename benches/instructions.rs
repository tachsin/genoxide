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
        tournament,
        generation,
        de_generation,
        es_generation,
        cmaes_generation,
        pso_generation,
        local_search_steps,
        run
    ]
);

main!(library_benchmark_groups = hot_paths);
