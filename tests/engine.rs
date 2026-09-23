use genoxide::observer::GenerationStatistics;
use genoxide::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

type OneMaxGa = Ga<Binary, Tournament, UniformCrossover, BitFlip>;

fn ga(len: usize, scheme: Scheme, seed: u64) -> OneMaxGa {
    Ga::builder(Binary::new(len).unwrap())
        .population_size(50)
        .select(Tournament::new(3).unwrap())
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::per_gene(1.0 / len as f64).unwrap())
        .scheme(scheme)
        .seed(seed)
        .build()
        .unwrap()
}

fn one_max(genome: &Bits) -> f64 {
    genome.count_ones() as f64
}

#[test]
fn every_scheme_solves_one_max() {
    for scheme in [
        Scheme::Generational { elitism: 1 },
        Scheme::SteadyState { replacements: 10 },
        Scheme::MuPlusLambda { lambda: 50 },
        Scheme::MuCommaLambda { lambda: 100 },
    ] {
        let outcome = Engine::new(ga(64, scheme, 1), one_max)
            .stop_when(Stop::target(64.0).or(Stop::generations(2_000)))
            .run()
            .unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Target, "{scheme:?}");
        assert_eq!(outcome.best_genome(), &Bits::ones(64));
    }
}

#[test]
fn minimize() {
    let ga = Ga::builder(Binary::new(32).unwrap())
        .population_size(30)
        .select(Tournament::new(3).unwrap())
        .crossover(PointCrossover::two_point())
        .mutate(BitFlip::count(1).unwrap())
        .minimize()
        .seed(5)
        .build()
        .unwrap();
    let outcome = Engine::new(ga, one_max)
        .stop_when(Stop::target(0.0).or(Stop::generations(1_000)))
        .run()
        .unwrap();
    assert_eq!(outcome.best_fitness(), Fitness::new(0.0));
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
fn engine_matches_ask_tell_by_hand() {
    let scheme = Scheme::MuPlusLambda { lambda: 20 };
    let outcome = Engine::new(ga(40, scheme, 3), one_max)
        .stop_when(Stop::generations(30))
        .run()
        .unwrap();

    let mut by_hand = ga(40, scheme, 3);
    while by_hand.generation() < 30 || by_hand.best().is_none() {
        let fitness: Vec<Fitness> = by_hand
            .ask()
            .iter()
            .map(|genome| Fitness::new(one_max(genome)))
            .collect();
        by_hand.tell(&fitness).unwrap();
    }
    assert_eq!(outcome.best(), by_hand.best().unwrap());
    assert_eq!(outcome.evaluations(), by_hand.evaluations());
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_matches_sequential() {
    let run = |parallel: bool| {
        let mut statistics = Statistics::new();
        let mut hall_of_fame = HallOfFame::new(5).unwrap();
        let outcome = Engine::new(ga(200, Scheme::default(), 9), one_max)
            .stop_when(Stop::generations(40))
            .parallel(parallel)
            .observe(&mut statistics)
            .observe(&mut hall_of_fame)
            .run()
            .unwrap();
        (
            outcome.into_best(),
            records(&statistics),
            hall_of_fame.individuals().to_vec(),
        )
    };
    assert_eq!(run(true), run(false));
    // and on a dedicated pool with a single thread
    let single = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| run(true));
    assert_eq!(single, run(false));
}

#[test]
fn stop_reasons() {
    let run = |stop: Stop| {
        Engine::new(ga(32, Scheme::default(), 0), one_max)
            .stop_when(stop)
            .run()
            .unwrap()
    };
    let outcome = run(Stop::generations(7));
    assert_eq!(
        (outcome.stop_reason(), outcome.generations()),
        (StopReason::Generations, 7)
    );
    let outcome = run(Stop::evaluations(200));
    assert_eq!(outcome.stop_reason(), StopReason::Evaluations);
    assert!((200..249).contains(&outcome.evaluations()));
    let outcome = run(Stop::time(Duration::ZERO));
    assert_eq!(
        (outcome.stop_reason(), outcome.generations()),
        (StopReason::Time, 0)
    );
    let outcome = run(Stop::custom(|progress| progress.generation() == 4));
    assert_eq!(
        (outcome.stop_reason(), outcome.generations()),
        (StopReason::Custom, 4)
    );

    // a constant fitness never improves
    let outcome = Engine::new(ga(32, Scheme::default(), 0), |_: &Bits| 1.0)
        .stop_when(Stop::stagnation(5))
        .run()
        .unwrap();
    assert_eq!(
        (outcome.stop_reason(), outcome.generations()),
        (StopReason::Stagnation, 5)
    );
}

#[test]
fn run_continues() {
    let mut engine =
        Engine::new(ga(32, Scheme::default(), 0), one_max).stop_when(Stop::generations(3));
    assert_eq!(engine.run().unwrap().generations(), 3);
    let mut engine = Engine::new(engine.into_algorithm(), one_max).stop_when(Stop::generations(8));
    let outcome = engine.run().unwrap();
    assert_eq!(outcome.generations(), 8);
    assert_eq!(engine.algorithm().generation(), 8);
}

#[test]
fn abort_flag() {
    let flag = Arc::new(AtomicBool::new(false));
    let setter = Arc::clone(&flag);
    let outcome = Engine::new(ga(32, Scheme::default(), 0), one_max)
        .abort_flag(Arc::clone(&flag))
        .on_generation(move |snapshot| {
            if snapshot.progress().generation() == 3 {
                setter.store(true, Ordering::Relaxed);
            }
        })
        .run()
        .unwrap();
    assert_eq!(
        (outcome.stop_reason(), outcome.generations()),
        (StopReason::Aborted, 3)
    );
}

#[test]
fn settings_errors() {
    assert_eq!(
        Engine::new(ga(8, Scheme::default(), 0), one_max).run(),
        Err(Error::MissingSetting {
            setting: "stop_when"
        })
    );
    assert!(matches!(
        Engine::new(ga(8, Scheme::default(), 0), one_max)
            .stop_when(Stop::target(f64::NAN))
            .run(),
        Err(Error::InvalidSetting { .. })
    ));
}

#[test]
fn nan_policy() {
    let nan = |genome: &Bits| {
        if genome.get(0) == Some(true) {
            f64::NAN
        } else {
            1.0
        }
    };
    let mut statistics = Statistics::new();
    Engine::new(ga(8, Scheme::default(), 0), nan)
        .stop_when(Stop::generations(0))
        .observe(&mut statistics)
        .run()
        .unwrap();
    assert!(statistics.last().unwrap().invalid > 0);

    let result = Engine::new(ga(8, Scheme::default(), 0), nan)
        .stop_when(Stop::generations(0))
        .nan_policy(NanPolicy::Error)
        .run();
    assert_eq!(result, Err(Error::NanFitness));
}

#[test]
fn fitness_types() {
    // Option<f64>: None is invalid
    let outcome = Engine::new(ga(8, Scheme::default(), 0), |genome: &Bits| {
        (genome.count_ones() > 2).then_some(1.0)
    })
    .stop_when(Stop::generations(2))
    .run()
    .unwrap();
    assert_eq!(outcome.best_fitness(), Fitness::new(1.0));

    // a type implementing FitnessFunction
    struct Weighted(Vec<f64>);
    impl FitnessFunction<Bits> for Weighted {
        type Output = Fitness;
        fn evaluate(&self, genome: &Bits) -> Fitness {
            Fitness::new(
                genome
                    .iter()
                    .zip(&self.0)
                    .filter(|(bit, _)| *bit)
                    .map(|(_, w)| w)
                    .sum(),
            )
        }
    }
    let outcome = Engine::new(
        ga(4, Scheme::default(), 0),
        Weighted(vec![1.0, 2.0, 3.0, 4.0]),
    )
    .stop_when(Stop::target(10.0).or(Stop::generations(100)))
    .run()
    .unwrap();
    assert_eq!(outcome.best_fitness(), Fitness::new(10.0));
}

#[test]
fn observers_see_every_generation() {
    let mut statistics = Statistics::new();
    let mut hall_of_fame = HallOfFame::new(3).unwrap();
    let mut generations = Vec::new();
    let outcome = Engine::new(ga(32, Scheme::default(), 0), one_max)
        .stop_when(Stop::generations(5))
        .observe(&mut statistics)
        .observe(&mut hall_of_fame)
        .on_generation(|snapshot| generations.push(snapshot.progress().generation()))
        .run()
        .unwrap();
    assert_eq!(generations, [0, 1, 2, 3, 4, 5]);
    let records = statistics.records();
    assert_eq!(records.len(), 6);
    assert_eq!(records[0].evaluations, 50);
    assert!(records.windows(2).all(|pair| {
        !Objective::Maximize.is_better(pair[0].best_so_far.unwrap(), pair[1].best_so_far.unwrap())
    }));
    assert_eq!(records[5].best_so_far, Some(outcome.best_fitness()));
    assert_eq!(hall_of_fame.individuals().len(), 3);
    assert_eq!(
        hall_of_fame.best().unwrap().fitness(),
        outcome.best().fitness()
    );
}

#[test]
fn hall_of_fame_sees_every_evaluated_individual() {
    // the bits as a number: every genome has its own fitness
    fn number(genome: &Bits) -> f64 {
        genome
            .iter()
            .fold(0.0, |value, bit| 2.0 * value + f64::from(u8::from(bit)))
    }
    // (μ,λ) and (μ+λ) with a tiny μ reject most offspring, including ones in the top k
    for scheme in [
        Scheme::MuCommaLambda { lambda: 20 },
        Scheme::MuPlusLambda { lambda: 20 },
    ] {
        let ga = Ga::builder(Binary::new(24).unwrap())
            .population_size(2)
            .select(Tournament::new(2).unwrap())
            .crossover(UniformCrossover::new())
            .mutate(BitFlip::per_gene(0.1).unwrap())
            .scheme(scheme)
            .seed(0)
            .build()
            .unwrap();
        let evaluated = std::sync::Mutex::new(Vec::new());
        let mut hall_of_fame = HallOfFame::new(10).unwrap();
        Engine::new(ga, |genome: &Bits| {
            let value = number(genome);
            evaluated.lock().unwrap().push(value);
            value
        })
        .stop_when(Stop::generations(10))
        .observe(&mut hall_of_fame)
        .run()
        .unwrap();

        let mut expected = evaluated.into_inner().unwrap();
        expected.sort_by(|a, b| b.total_cmp(a));
        expected.dedup();
        expected.truncate(10);
        let found: Vec<f64> = hall_of_fame
            .individuals()
            .iter()
            .map(|individual| individual.fitness().unwrap().score().unwrap())
            .collect();
        assert_eq!(found, expected, "{scheme:?}");
    }
}

#[test]
fn constrained_fitness_functions() {
    // (score, violation): the NaN policy applies to the violation too
    let nan_violation = |_: &Bits| (1.0, f64::NAN);
    let result = Engine::new(ga(8, Scheme::default(), 0), nan_violation)
        .stop_when(Stop::generations(0))
        .nan_policy(NanPolicy::Error)
        .run();
    assert_eq!(result, Err(Error::NanFitness));
    // a negative violation is a bug in the fitness function
    let negative = |_: &Bits| (1.0, -1.0);
    let result = Engine::new(ga(8, Scheme::default(), 0), negative)
        .stop_when(Stop::generations(0))
        .run();
    assert!(matches!(result, Err(Error::InvalidFitness { .. })));
    // a target is only reached by a feasible solution
    let infeasible = |genome: &Bits| (genome.count_ones() as f64, 1.0);
    let outcome = Engine::new(ga(8, Scheme::default(), 0), infeasible)
        .stop_when(Stop::target(0.0).or(Stop::generations(5)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Generations);
}
