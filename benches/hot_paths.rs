//! Wall-time benchmarks of the hot paths.
//!
//! ```text
//! cargo bench --bench hot_paths
//! ```

use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use genoxide::prelude::*;

fn one_max(genome: &Bits) -> f64 {
    genome.count_ones() as f64
}

fn evaluated(population: Vec<Bits>) -> Population<Bits> {
    population
        .into_iter()
        .map(|genome| {
            let mut individual = Individual::new(genome);
            individual.set_fitness(Fitness::new(one_max(individual.genome())));
            individual
        })
        .collect()
}

fn genomes(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_genome");
    let mut rng = StreamRng::seed_from_u64(0);
    let binary = Binary::new(1_000).unwrap();
    group.bench_function("binary_1000", |b| b.iter(|| binary.random_genome(&mut rng)));
    let integer = Integer::uniform(100, -1_000..=1_000).unwrap();
    group.bench_function("integer_100", |b| {
        b.iter(|| integer.random_genome(&mut rng))
    });
    let real = Real::uniform(100, -5.12..=5.12).unwrap();
    group.bench_function("real_100", |b| b.iter(|| real.random_genome(&mut rng)));
    let permutation = Permutation::new(100).unwrap();
    group.bench_function("permutation_100", |b| {
        b.iter(|| permutation.random_genome(&mut rng))
    });
    group.finish();
}

fn crossover(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossover");
    let mut rng = StreamRng::seed_from_u64(0);
    let binary = Binary::new(1_000).unwrap();
    let (mut a, mut b) = (
        binary.random_genome(&mut rng),
        binary.random_genome(&mut rng),
    );
    group.bench_function("uniform_binary_1000", |bench| {
        bench.iter(|| UniformCrossover::new().crossover(&binary, &mut a, &mut b, &mut rng))
    });
    group.bench_function("two_point_binary_1000", |bench| {
        bench.iter(|| PointCrossover::two_point().crossover(&binary, &mut a, &mut b, &mut rng))
    });
    let real = Real::uniform(100, -5.12..=5.12).unwrap();
    let (mut x, mut y) = (real.random_genome(&mut rng), real.random_genome(&mut rng));
    group.bench_function("uniform_real_100", |bench| {
        bench.iter(|| UniformCrossover::new().crossover(&real, &mut x, &mut y, &mut rng))
    });
    group.finish();
}

fn mutation(c: &mut Criterion) {
    let mut group = c.benchmark_group("mutation");
    let mut rng = StreamRng::seed_from_u64(0);
    let binary = Binary::new(1_000).unwrap();
    let mut bits = binary.random_genome(&mut rng);
    let bit_flip = BitFlip::per_gene(0.001).unwrap();
    group.bench_function("bit_flip_binary_1000", |b| {
        b.iter(|| bit_flip.mutate(&binary, &mut bits, &mut rng))
    });
    let real = Real::uniform(100, -5.12..=5.12).unwrap();
    let mut reals = real.random_genome(&mut rng);
    let uniform = UniformMutation::per_gene(0.01).unwrap();
    group.bench_function("uniform_real_100", |b| {
        b.iter(|| uniform.mutate(&real, &mut reals, &mut rng))
    });
    let permutation = Permutation::new(100).unwrap();
    let mut order = permutation.random_genome(&mut rng);
    group.bench_function("swap_permutation_100", |b| {
        b.iter(|| SwapMutation::new().mutate(&permutation, &mut order, &mut rng))
    });
    group.finish();
}

fn selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("selection");
    let mut rng = StreamRng::seed_from_u64(0);
    let binary = Binary::new(100).unwrap();
    let population = evaluated((0..100).map(|_| binary.random_genome(&mut rng)).collect());
    let objective = Objective::Maximize;
    let tournament = Tournament::new(3).unwrap();
    group.bench_function("tournament_3_of_100", |b| {
        b.iter(|| tournament.select(&population, objective, 100, &mut rng))
    });
    group.bench_function("roulette_100", |b| {
        b.iter(|| Roulette.select(&population, objective, 100, &mut rng))
    });
    group.bench_function("rank_100", |b| {
        b.iter(|| Rank::default().select(&population, objective, 100, &mut rng))
    });
    group.finish();
}

fn one_max_ga(len: usize, seed: u64) -> Ga<Binary, Tournament, UniformCrossover, BitFlip> {
    Ga::builder(Binary::new(len).unwrap())
        .population_size(100)
        .select(Tournament::new(3).unwrap())
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::per_gene(1.0 / len as f64).unwrap())
        .seed(seed)
        .build()
        .unwrap()
}

fn tell_one_max(ga: &mut Ga<Binary, Tournament, UniformCrossover, BitFlip>) {
    let fitness: Vec<Fitness> = ga
        .ask()
        .iter()
        .map(|genome| Fitness::new(one_max(genome)))
        .collect();
    ga.tell(&fitness).unwrap();
}

fn algorithm(c: &mut Criterion) {
    let mut group = c.benchmark_group("ga");
    // one generation: breeding, evaluation and survival
    group.bench_function("generation_one_max_1000_population_100", |b| {
        b.iter_batched(
            || {
                let mut ga = one_max_ga(1_000, 0);
                tell_one_max(&mut ga);
                ga
            },
            |mut ga| {
                tell_one_max(&mut ga);
                ga
            },
            BatchSize::SmallInput,
        )
    });
    group.bench_function("run_one_max_100_50_generations", |b| {
        b.iter(|| {
            Engine::new(one_max_ga(100, 0), one_max)
                .stop_when(Stop::generations(50))
                .run()
                .unwrap()
        })
    });
    group.finish();
}

criterion_group!(benches, genomes, crossover, mutation, selection, algorithm);
criterion_main!(benches);
