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
        run
    ]
);

main!(library_benchmark_groups = hot_paths);
