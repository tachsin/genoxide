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
use genoxide::multi::problems::{Dtlz2, MultiProblem, Zdt1};
use genoxide::multi::{self, MultiFitnessFunction};
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
        run,
        nsga2_generation,
        nsga3_generation,
        non_dominated_sort,
        hypervolume
    ]
);

main!(library_benchmark_groups = hot_paths);
