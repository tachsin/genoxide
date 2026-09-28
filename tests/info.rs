//! Info returned with the fitness in an `Evaluated`: kept by genome, never used by the search.

use genoxide::Objective::Minimize;
use genoxide::algorithm::islands::Topology;
use genoxide::multi::{MultiObjectiveAlgorithm, MultiOutcome};
use genoxide::observer::{GenerationStatistics, Observer, Snapshot};
use genoxide::prelude::*;
use std::cell::Cell;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

// what the fitness function computed: the genome it saw, so the info of a genome can be checked
#[derive(Clone, Debug, PartialEq)]
struct Seen {
    genome: String,
    ones: u64,
}

fn seen(genome: &Bits) -> Seen {
    Seen {
        genome: genome.to_string(),
        ones: genome.count_ones() as u64,
    }
}

fn one_max(genome: &Bits) -> f64 {
    genome.count_ones() as f64
}

fn one_max_seen(genome: &Bits) -> Evaluated<f64, Seen> {
    Evaluated::new(one_max(genome), seen(genome))
}

type OneMaxGa = Ga<Binary, Tournament, UniformCrossover, BitFlip>;

fn ga(scheme: Scheme, memetic: bool, seed: u64) -> OneMaxGa {
    let builder = Ga::builder(Binary::new(40).unwrap())
        .population_size(30)
        .select(Tournament::new(3).unwrap())
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::per_gene(1.0 / 40.0).unwrap())
        .scheme(scheme)
        .seed(seed);
    let builder = if memetic {
        builder.memetic(1, 3)
    } else {
        builder
    };
    builder.build().unwrap()
}

fn schemes() -> [(Scheme, bool); 5] {
    [
        (Scheme::Generational { elitism: 1 }, false),
        (Scheme::Generational { elitism: 2 }, true),
        (Scheme::SteadyState { replacements: 5 }, false),
        (Scheme::MuPlusLambda { lambda: 30 }, false),
        (Scheme::MuCommaLambda { lambda: 40 }, false),
    ]
}

// statistics without the timings, which differ between runs
fn records(statistics: &Statistics) -> Vec<GenerationStatistics> {
    statistics
        .records()
        .iter()
        .cloned()
        .map(|mut record| {
            record.elapsed = Duration::ZERO;
            record
        })
        .collect()
}

#[test]
fn every_evaluated_genome_has_its_own_info() {
    for (scheme, memetic) in schemes() {
        let mut hall_of_fame = HallOfFame::new(5).unwrap();
        let checked = Cell::new(0);
        let outcome = Engine::new(ga(scheme, memetic, 1), one_max_seen)
            .stop_when(Stop::generations(40))
            .observe(&mut hall_of_fame)
            .on_generation(|snapshot| {
                // copies of parents and survivors too: the info is their genome's
                let individuals = snapshot.population().iter().chain(snapshot.discarded());
                for individual in individuals.chain([snapshot.best()]) {
                    let info = snapshot.info::<Seen>(individual.genome());
                    assert_eq!(info, Some(&seen(individual.genome())), "{scheme:?}");
                    checked.set(checked.get() + 1);
                }
                assert_eq!(
                    snapshot.best_info::<Seen>(),
                    Some(&seen(snapshot.best().genome()))
                );
            })
            .run()
            .unwrap();
        assert!(checked.get() > 40 * 30);
        assert_eq!(
            outcome.best_info::<Seen>(),
            Some(&seen(outcome.best_genome()))
        );
        assert_eq!(hall_of_fame.individuals().len(), 5);
        for individual in hall_of_fame.individuals() {
            let info = hall_of_fame.info::<Seen>(individual.genome());
            assert_eq!(info, Some(&seen(individual.genome())));
        }
    }
}

#[test]
fn another_type_or_no_info_is_none() {
    let mut hall_of_fame = HallOfFame::new(3).unwrap();
    let outcome = Engine::new(ga(Scheme::default(), false, 2), one_max_seen)
        .stop_when(Stop::generations(5))
        .observe(&mut hall_of_fame)
        .on_generation(|snapshot| {
            assert!(snapshot.best_info::<String>().is_none());
            assert!(snapshot.info::<Seen>(&Bits::zeros(3)).is_none());
        })
        .run()
        .unwrap();
    assert!(outcome.best_info::<String>().is_none());
    assert!(outcome.best_info::<Seen>().is_some());
    let best = hall_of_fame.best().unwrap().genome();
    assert!(hall_of_fame.info::<u64>(best).is_none());
    assert!(hall_of_fame.info::<Seen>(&Bits::zeros(40)).is_none());

    // no info at all
    let mut hall_of_fame = HallOfFame::new(3).unwrap();
    let outcome = Engine::new(ga(Scheme::default(), false, 2), one_max)
        .stop_when(Stop::generations(5))
        .observe(&mut hall_of_fame)
        .on_generation(|snapshot| assert!(snapshot.best_info::<Seen>().is_none()))
        .run()
        .unwrap();
    assert!(outcome.best_info::<Seen>().is_none());
    let best = hall_of_fame.best().unwrap().genome();
    assert!(hall_of_fame.info::<Seen>(best).is_none());
}

// runs `algorithm` with and without info: the same outcome, statistics and hall of fame
fn same_run_with_and_without_info<A, F, T>(make: impl Fn() -> A, fitness: F, info: T)
where
    A: Algorithm,
    F: Fn(&A::Genome) -> f64 + Sync + Copy,
    T: Fn(&A::Genome) -> String + Sync + Copy,
{
    let run = |with_info: bool| {
        let mut statistics = Statistics::new();
        let mut hall_of_fame = HallOfFame::new(5).unwrap();
        let outcome = if with_info {
            Engine::new(make(), move |genome: &A::Genome| {
                Evaluated::new(fitness(genome), info(genome))
            })
            .stop_when(Stop::generations(30))
            .observe(&mut statistics)
            .observe(&mut hall_of_fame)
            .run()
            .unwrap()
        } else {
            Engine::new(make(), fitness)
                .stop_when(Stop::generations(30))
                .observe(&mut statistics)
                .observe(&mut hall_of_fame)
                .run()
                .unwrap()
        };
        (outcome, records(&statistics), hall_of_fame)
    };
    let (with, with_statistics, with_hall) = run(true);
    let (without, without_statistics, without_hall) = run(false);
    assert_eq!(with.best(), without.best());
    assert_eq!(with.evaluations(), without.evaluations());
    assert_eq!(with.generations(), without.generations());
    assert_eq!(with_statistics, without_statistics);
    assert_eq!(with_hall, without_hall);
    assert_eq!(with.best_info::<String>(), Some(&info(with.best_genome())));
    assert!(without.best_info::<String>().is_none());
}

fn sphere(x: &Reals) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

fn genes(x: &Reals) -> String {
    format!("{:?}", &x[..])
}

#[test]
fn info_never_changes_the_search() {
    for (scheme, memetic) in schemes() {
        same_run_with_and_without_info(|| ga(scheme, memetic, 3), one_max, |g| g.to_string());
    }
    let real = || Real::uniform(5, -5.0..=5.0).unwrap();
    same_run_with_and_without_info(
        || De::builder(real()).minimize().seed(1).build().unwrap(),
        sphere,
        genes,
    );
    same_run_with_and_without_info(
        || {
            Pso::builder(real())
                .population_size(20)
                .minimize()
                .seed(1)
                .build()
                .unwrap()
        },
        sphere,
        genes,
    );
    same_run_with_and_without_info(
        || Cmaes::builder(real()).minimize().seed(1).build().unwrap(),
        sphere,
        genes,
    );
    same_run_with_and_without_info(
        || {
            let islands = (0..3)
                .map(|seed| {
                    Ga::builder(real())
                        .population_size(10)
                        .select(Tournament::new(2).unwrap())
                        .crossover(UniformCrossover::new())
                        .mutate(PolynomialMutation::per_gene(0.2, 20.0).unwrap())
                        .minimize()
                        .seed(seed)
                        .build()
                        .unwrap()
                })
                .collect::<Vec<_>>();
            Islands::builder(islands)
                .topology(Topology::Ring)
                .interval(3)
                .migrants(1)
                .build()
                .unwrap()
        },
        sphere,
        genes,
    );
}

// equality and serialization ignore the info
#[cfg(feature = "serde")]
#[test]
fn outcomes_compare_and_serialize_without_the_info() {
    let mut hall_of_fame = HallOfFame::new(3).unwrap();
    let with = Engine::new(ga(Scheme::default(), false, 4), one_max_seen)
        .stop_when(Stop::generations(10))
        .observe(&mut hall_of_fame)
        .run()
        .unwrap();
    let cloned = with.clone();
    assert_eq!(cloned, with);
    assert_eq!(cloned.best_info::<Seen>(), with.best_info::<Seen>());
    let json = serde_json::to_string(&with).unwrap();
    let without: Outcome<Bits> = serde_json::from_str(&json).unwrap();
    assert!(without.best_info::<Seen>().is_none());
    assert_eq!(without, with);

    let json = serde_json::to_string(&hall_of_fame).unwrap();
    let without: HallOfFame<Bits> = serde_json::from_str(&json).unwrap();
    assert_eq!(without, hall_of_fame);
    let best = hall_of_fame.best().unwrap().genome();
    assert!(hall_of_fame.info::<Seen>(best).is_some());
    assert!(without.info::<Seen>(best).is_none());
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_info_matches_sequential() {
    let run = |parallel: bool| {
        let infos = std::cell::RefCell::new(Vec::new());
        let outcome = Engine::new(
            ga(Scheme::MuPlusLambda { lambda: 30 }, true, 5),
            one_max_seen,
        )
        .stop_when(Stop::generations(20))
        .parallel(parallel)
        .on_generation(|snapshot| {
            let population = snapshot.population().iter();
            infos.borrow_mut().extend(
                population
                    .chain(snapshot.discarded())
                    .map(|individual| snapshot.info::<Seen>(individual.genome()).cloned()),
            );
        })
        .run()
        .unwrap();
        let best_info = outcome.best_info::<Seen>().cloned();
        (outcome.into_best(), best_info, infos.into_inner())
    };
    let (best, best_info, infos) = run(true);
    assert_eq!(run(false), (best, best_info, infos.clone()));
    assert!(infos.iter().all(Option::is_some));
}

#[test]
fn batch_info() {
    let calls = AtomicUsize::new(0);
    let batch = Batch(|genomes: &[&Bits]| {
        calls.fetch_add(1, Ordering::Relaxed);
        genomes
            .iter()
            .map(|genome| one_max_seen(genome))
            .collect::<Vec<_>>()
    });
    let mut hall_of_fame = HallOfFame::new(4).unwrap();
    let outcome = Engine::new(ga(Scheme::default(), false, 6), batch)
        .stop_when(Stop::generations(15))
        .observe(&mut hall_of_fame)
        .on_generation(|snapshot| {
            for individual in snapshot.population().iter() {
                let info = snapshot.info::<Seen>(individual.genome());
                assert_eq!(info, Some(&seen(individual.genome())));
            }
        })
        .run()
        .unwrap();
    assert_eq!(calls.into_inner(), 16);
    let one_at_a_time = Engine::new(ga(Scheme::default(), false, 6), one_max_seen)
        .stop_when(Stop::generations(15))
        .run()
        .unwrap();
    assert_eq!(outcome.best(), one_at_a_time.best());
    assert_eq!(
        outcome.best_info::<Seen>(),
        Some(&seen(outcome.best_genome()))
    );
    for individual in hall_of_fame.individuals() {
        assert!(hall_of_fame.info::<Seen>(individual.genome()).is_some());
    }
}

#[test]
fn a_reevaluation_replaces_the_info() {
    // the fitness function changes at generation 5, and says which version scored a genome
    let version = AtomicU64::new(0);
    let fitness = |bits: &Bits| {
        let version = version.load(Ordering::Relaxed);
        let score = bits.count_ones() as f64 * if version == 0 { 1.0 } else { -1.0 };
        Evaluated::new(score, version)
    };
    let reevaluated = Cell::new(false);
    let outcome = Engine::new(ga(Scheme::default(), false, 7), &fitness)
        .stop_when(Stop::generations(10))
        .on_generation(|snapshot| {
            let expected = u64::from(reevaluated.get());
            for individual in snapshot.population().iter() {
                assert_eq!(snapshot.info::<u64>(individual.genome()), Some(&expected));
            }
            assert_eq!(snapshot.best_info::<u64>(), Some(&expected));
        })
        .control(|ga, progress| {
            if progress.generation() == 5 && !reevaluated.get() {
                version.store(1, Ordering::Relaxed);
                reevaluated.set(true);
                ga.reevaluate()?;
            }
            Ok(())
        })
        .run()
        .unwrap();
    assert!(reevaluated.get());
    assert_eq!(outcome.best_info::<u64>(), Some(&1));
}

#[test]
fn a_continued_run_keeps_the_info() {
    let mut engine =
        Engine::new(ga(Scheme::default(), false, 8), one_max_seen).stop_when(Stop::generations(5));
    engine.run().unwrap();
    // already met: the outcome at once, with the info
    let outcome = engine.run().unwrap();
    assert_eq!(outcome.generations(), 5);
    assert_eq!(
        outcome.best_info::<Seen>(),
        Some(&seen(outcome.best_genome()))
    );
    // another engine has no info for what the first evaluated
    let outcome = Engine::new(engine.into_algorithm(), one_max_seen)
        .stop_when(Stop::generations(5))
        .run()
        .unwrap();
    assert!(outcome.best_info::<Seen>().is_none());
}

// checks that the population, the discarded individuals and the best have their info
struct EveryInfo(usize);

impl Observer<Bits> for EveryInfo {
    fn observe(&mut self, snapshot: &Snapshot<'_, Bits>) {
        let individuals = snapshot.population().iter().chain(snapshot.discarded());
        for individual in individuals.chain([snapshot.best()]) {
            let info = snapshot.info::<Seen>(individual.genome());
            assert_eq!(info, Some(&seen(individual.genome())));
            self.0 += 1;
        }
    }
}

#[test]
fn asynchronous_info() {
    let steady = || {
        Ga::builder(Binary::new(40).unwrap())
            .population_size(20)
            .select(Tournament::new(3).unwrap())
            .crossover(UniformCrossover::new())
            .mutate(BitFlip::per_gene(1.0 / 40.0).unwrap())
            .seed(9)
            .build_steady()
            .unwrap()
    };
    let mut checked = EveryInfo(0);
    let mut hall_of_fame = HallOfFame::new(4).unwrap();
    let with = AsyncEngine::new(steady(), one_max_seen)
        .workers(1)
        .stop_when(Stop::evaluations(600))
        .observe(&mut hall_of_fame)
        .observe(&mut checked)
        .run()
        .unwrap();
    assert!(checked.0 > 600);
    assert_eq!(with.best_info::<Seen>(), Some(&seen(with.best_genome())));
    for individual in hall_of_fame.individuals() {
        let info = hall_of_fame.info::<Seen>(individual.genome());
        assert_eq!(info, Some(&seen(individual.genome())));
    }
    // with one worker, the same run as without info; with several, info on every result
    let without = AsyncEngine::new(steady(), one_max)
        .workers(1)
        .stop_when(Stop::evaluations(600))
        .run()
        .unwrap();
    assert_eq!(with.best(), without.best());
    assert_eq!(with.evaluations(), without.evaluations());
    let batch = Batch(|genomes: &[&Bits]| genomes.iter().map(|g| one_max_seen(g)).collect());
    let outcome = AsyncEngine::new(steady(), batch)
        .workers(4)
        .stop_when(Stop::evaluations(600))
        .run()
        .unwrap();
    assert_eq!(
        outcome.best_info::<Seen>(),
        Some(&seen(outcome.best_genome()))
    );
}

// Schaffer's problem, and the genes as info
fn schaffer(x: &Reals) -> [f64; 2] {
    [x[0] * x[0], (x[0] - 2.0) * (x[0] - 2.0)]
}

fn nsga2() -> impl MultiObjectiveAlgorithm<2, Genome = Reals> {
    Nsga2::builder(
        Real::uniform(2, -10.0..=10.0).unwrap(),
        [Minimize, Minimize],
    )
    .population_size(20)
    .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
    .mutate(PolynomialMutation::per_gene(0.5, 20.0).unwrap())
    .seed(1)
    .build()
    .unwrap()
}

#[test]
fn multi_objective_info() {
    let checked = Cell::new(0);
    let with: MultiOutcome<Reals, 2> =
        MultiEngine::new(nsga2(), |x: &Reals| Evaluated::new(schaffer(x), genes(x)))
            .stop_when(Stop::generations(30))
            .on_generation(|snapshot| {
                let individuals = snapshot.population().iter().chain(snapshot.discarded());
                for individual in individuals.chain(snapshot.front()) {
                    let info = snapshot.info::<String>(individual.genome());
                    assert_eq!(info, Some(&genes(individual.genome())));
                    checked.set(checked.get() + 1);
                }
                assert!(snapshot.info::<u8>(snapshot.front()[0].genome()).is_none());
            })
            .run()
            .unwrap();
    assert!(checked.get() > 30 * 20);
    for individual in with.front() {
        let info = with.info::<String>(individual.genome());
        assert_eq!(info, Some(&genes(individual.genome())));
    }
    let without = MultiEngine::new(nsga2(), schaffer)
        .stop_when(Stop::generations(30))
        .run()
        .unwrap();
    assert_eq!(with.front(), without.front());
    assert_eq!(with.evaluations(), without.evaluations());
    assert!(
        without
            .info::<String>(without.front()[0].genome())
            .is_none()
    );

    // a batch, in parallel or not
    let batch = Batch(|genomes: &[&Reals]| {
        genomes
            .iter()
            .map(|x| Evaluated::new(schaffer(x), genes(x)))
            .collect::<Vec<_>>()
    });
    let outcome = MultiEngine::new(nsga2(), batch)
        .stop_when(Stop::generations(30))
        .run()
        .unwrap();
    assert_eq!(outcome.front(), with.front());
    for individual in outcome.front() {
        assert!(outcome.info::<String>(individual.genome()).is_some());
    }
    #[cfg(feature = "parallel")]
    {
        let parallel = MultiEngine::new(nsga2(), |x: &Reals| Evaluated::new(schaffer(x), genes(x)))
            .stop_when(Stop::generations(30))
            .parallel(true)
            .run()
            .unwrap();
        assert_eq!(parallel.front(), with.front());
        for individual in parallel.front() {
            let info = parallel.info::<String>(individual.genome());
            assert_eq!(info, Some(&genes(individual.genome())));
        }
    }
}
