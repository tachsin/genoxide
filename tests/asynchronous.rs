//! Asynchronous evaluation with a steady-state GA.

use genoxide::algorithm::Incremental;
use genoxide::prelude::*;
use std::f64::consts::TAU;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

type SteadyOneMax = SteadyGa<Binary, Tournament, UniformCrossover, BitFlip>;

fn steady(len: usize, size: usize, seed: u64) -> SteadyOneMax {
    Ga::builder(Binary::new(len).unwrap())
        .population_size(size)
        .select(Tournament::new(3).unwrap())
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::per_gene(1.0 / len as f64).unwrap())
        .seed(seed)
        .build_steady()
        .unwrap()
}

fn one_max(genome: &Bits) -> f64 {
    genome.count_ones() as f64
}

#[test]
fn one_worker_gives_the_same_run_every_time() {
    let run = || {
        let mut statistics = Statistics::new();
        let mut engine = AsyncEngine::new(steady(64, 20, 1), one_max)
            .workers(1)
            .stop_when(Stop::evaluations(2_000))
            .observe(&mut statistics);
        let outcome = engine.run().unwrap();
        let population = engine.algorithm().population().clone();
        drop(engine);
        let records: Vec<_> = statistics
            .records()
            .iter()
            .map(|record| (record.generation, record.evaluations, record.best))
            .collect();
        (outcome.into_best(), population, records)
    };
    let first = run();
    assert_eq!(first, run());
    // a generation per 20 evaluations: generation 0 after 20, 99 after 2000
    assert_eq!(first.2.len(), 100);
    assert_eq!(first.2[0].0, 0);
    assert_eq!(first.2[0].1, 20);
    assert_eq!(first.2[99], (99, 2_000, first.2[99].2));
}

#[test]
fn workers_solve_problems() {
    for workers in [1, 2, 8] {
        let outcome = AsyncEngine::new(steady(64, 30, 2), one_max)
            .workers(workers)
            .stop_when(Stop::target(64.0).or(Stop::evaluations(50_000)))
            .run()
            .unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Target, "{workers}");
        assert_eq!(outcome.best_genome(), &Bits::ones(64));
    }
    let rastrigin = |x: &Reals| {
        10.0 * x.len() as f64
            + x.iter()
                .map(|xi| xi * xi - 10.0 * (TAU * xi).cos())
                .sum::<f64>()
    };
    let ga = Ga::builder(Real::uniform(5, -5.12..=5.12).unwrap())
        .population_size(40)
        .select(Tournament::new(3).unwrap())
        .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
        .mutate(PolynomialMutation::per_gene(0.2, 20.0).unwrap())
        .minimize()
        .seed(3)
        .build_steady()
        .unwrap();
    let outcome = AsyncEngine::new(ga, rastrigin)
        .workers(4)
        .stop_when(Stop::target(0.01).or(Stop::evaluations(200_000)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
}

#[test]
fn workers_evaluate_at_the_same_time_and_the_budget_is_exact() {
    // evaluations that take different times: the workers stay busy
    let running = AtomicUsize::new(0);
    let most = AtomicUsize::new(0);
    let slow = |genome: &Bits| {
        let now = running.fetch_add(1, Ordering::SeqCst) + 1;
        most.fetch_max(now, Ordering::SeqCst);
        std::thread::sleep(Duration::from_micros(50 * (genome.count_ones() as u64 % 7)));
        running.fetch_sub(1, Ordering::SeqCst);
        genome.count_ones() as f64
    };
    let outcome = AsyncEngine::new(steady(32, 10, 4), slow)
        .workers(4)
        .stop_when(Stop::evaluations(300))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Evaluations);
    assert_eq!(outcome.evaluations(), 300);
    assert!(most.load(Ordering::SeqCst) > 1);
    assert!(most.load(Ordering::SeqCst) <= 4);
    // the initial population comes first, as in a generational run
    let outcome = AsyncEngine::new(steady(8, 10, 4), one_max)
        .workers(1)
        .stop_when(Stop::generations(0))
        .run()
        .unwrap();
    assert_eq!(outcome.evaluations(), 10);
    assert_eq!(outcome.generations(), 0);
    // at least one evaluation, even with a budget of none
    let outcome = AsyncEngine::new(steady(8, 10, 4), one_max)
        .workers(3)
        .stop_when(Stop::evaluations(0))
        .run()
        .unwrap();
    assert_eq!(outcome.evaluations(), 1);
}

#[test]
fn results_in_flight_count_when_stopping() {
    // the target is met by one result; the other workers' results still arrive
    let outcome = AsyncEngine::new(steady(16, 10, 5), one_max)
        .workers(4)
        .stop_when(Stop::target(16.0))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    assert_eq!(outcome.best_fitness(), Fitness::new(16.0));
    let abort = Arc::new(AtomicBool::new(true));
    let outcome = AsyncEngine::new(steady(16, 10, 5), one_max)
        .workers(4)
        .abort_flag(abort)
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Aborted);
    assert!((1..=4).contains(&outcome.evaluations()));
}

#[test]
fn a_panic_in_the_fitness_function_ends_the_run() {
    let evaluations = AtomicUsize::new(0);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        AsyncEngine::new(steady(16, 10, 6), |genome: &Bits| {
            if evaluations.fetch_add(1, Ordering::SeqCst) == 25 {
                panic!("the simulation crashed");
            }
            genome.count_ones() as f64
        })
        .workers(4)
        .stop_when(Stop::evaluations(10_000))
        .run()
    }));
    let payload = result.unwrap_err();
    assert_eq!(
        payload.downcast_ref::<&str>(),
        Some(&"the simulation crashed")
    );
    // the run ended early, without waiting for its budget
    assert!(evaluations.load(Ordering::SeqCst) < 10_000);
}

#[test]
fn errors() {
    let nan = |_: &Bits| f64::NAN;
    let error = AsyncEngine::new(steady(8, 10, 7), nan)
        .workers(2)
        .nan_policy(NanPolicy::Error)
        .stop_when(Stop::evaluations(100))
        .run()
        .unwrap_err();
    assert_eq!(error, Error::NanFitness);
    // NaN is invalid by default
    let outcome = AsyncEngine::new(steady(8, 10, 7), nan)
        .workers(2)
        .stop_when(Stop::evaluations(100))
        .run()
        .unwrap();
    assert_eq!(outcome.best_fitness(), Fitness::invalid());

    let error = AsyncEngine::new(steady(8, 10, 7), one_max)
        .run()
        .unwrap_err();
    assert_eq!(
        error,
        Error::MissingSetting {
            setting: "stop_when"
        }
    );
    let error = AsyncEngine::new(steady(8, 10, 7), one_max)
        .workers(0)
        .stop_when(Stop::evaluations(100))
        .run()
        .unwrap_err();
    assert!(matches!(
        error,
        Error::InvalidSetting {
            setting: "workers",
            ..
        }
    ));
    let error = AsyncEngine::new(steady(8, 10, 7), one_max)
        .stop_when(Stop::evaluations(100))
        .checkpoint_every(0, |_| Ok(()))
        .run()
        .unwrap_err();
    assert!(matches!(
        error,
        Error::InvalidSetting {
            setting: "checkpoint_every",
            ..
        }
    ));

    // settings that don't apply to a steady-state GA
    let builder = || {
        Ga::builder(Binary::new(8).unwrap())
            .population_size(10)
            .select(Tournament::new(2).unwrap())
            .crossover(UniformCrossover::new())
            .mutate(BitFlip::count(1).unwrap())
    };
    let setting = |result: genoxide::Result<SteadyOneMax>| match result {
        Err(Error::InvalidSetting { setting, .. }) => setting,
        other => panic!("{other:?}"),
    };
    assert_eq!(
        setting(
            builder()
                .scheme(Scheme::SteadyState { replacements: 2 })
                .build_steady()
        ),
        "scheme"
    );
    assert_eq!(setting(builder().memetic(1, 2).build_steady()), "memetic");
    assert_eq!(
        setting(builder().population_size(0).build_steady()),
        "population_size"
    );
}

#[test]
fn observers_and_checkpoints_follow_generations() {
    let mut hall_of_fame = HallOfFame::new(3).unwrap();
    let saved = Mutex::new(Vec::new());
    let outcome = AsyncEngine::new(steady(32, 10, 8), one_max)
        .workers(1)
        .stop_when(Stop::evaluations(95))
        .observe(&mut hall_of_fame)
        .checkpoint_every(3, |ga| {
            saved.lock().unwrap().push(ga.evaluations());
            Ok(())
        })
        .run()
        .unwrap();
    // generations 0, 3 and 6 (after 10, 40 and 70 evaluations), and the end
    assert_eq!(saved.into_inner().unwrap(), [10, 40, 70, 95]);
    assert_eq!(outcome.generations(), 8);
    assert_eq!(hall_of_fame.individuals()[0], *outcome.best());
}

#[test]
fn the_steady_ga_protocol() {
    let initial = [Bits::ones(8), Bits::zeros(8)];
    let mut ga = Ga::builder(Binary::new(8).unwrap())
        .population_size(3)
        .select(Tournament::new(2).unwrap())
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::count(1).unwrap())
        .initial_genomes(initial.clone())
        .seed(9)
        .build_steady()
        .unwrap();
    // the initial genomes first, in order, then random ones
    assert_eq!(ga.propose(), initial[0]);
    assert_eq!(ga.propose(), initial[1]);
    let third = ga.propose();
    // no results yet: more random genomes
    let fourth = ga.propose();
    assert!(ga.population().is_empty());

    let score = |genome: &Bits| Fitness::new(genome.count_ones() as f64);
    // results in any order fill the population
    assert_eq!(ga.receive(third.clone(), score(&third)).unwrap(), None);
    assert_eq!(
        ga.receive(initial[1].clone(), score(&initial[1])).unwrap(),
        None
    );
    assert_eq!(
        ga.receive(initial[0].clone(), score(&initial[0])).unwrap(),
        None
    );
    assert_eq!(ga.best().unwrap().genome(), &initial[0]);
    assert_eq!(ga.best_evaluation(), 3);
    // then each result replaces the worst when it's not worse
    let replaced = ga.receive(fourth.clone(), score(&fourth)).unwrap().unwrap();
    assert_eq!(replaced.genome(), &initial[1]);
    let worse = Bits::zeros(8);
    let rejected = ga
        .receive(worse.clone(), Fitness::new(-1.0))
        .unwrap()
        .unwrap();
    assert_eq!(rejected.genome(), &worse);
    // a genome already in the population isn't added twice
    let twice = ga
        .receive(initial[0].clone(), score(&initial[0]))
        .unwrap()
        .unwrap();
    assert_eq!(twice.genome(), &initial[0]);
    assert_eq!(ga.population().len(), 3);
    assert_eq!(ga.evaluations(), 6);
    // children aren't copies of the population
    for _ in 0..50 {
        let child = ga.propose();
        assert!(
            ga.population()
                .iter()
                .all(|individual| individual.genome() != &child)
        );
    }
    // a genome that doesn't fit is an error, and changes nothing
    assert!(ga.receive(Bits::zeros(9), Fitness::new(0.0)).is_err());
    assert_eq!(ga.evaluations(), 6);
}

#[cfg(feature = "serde")]
#[test]
fn one_worker_resumes_exactly() {
    use genoxide::checkpoint;
    let saved = Mutex::new(Vec::new());
    let mut whole = AsyncEngine::new(steady(64, 20, 10), one_max)
        .workers(1)
        .stop_when(Stop::evaluations(1_000))
        .checkpoint_every(10, |ga| {
            let mut bytes = Vec::new();
            checkpoint::save(ga, &mut bytes)?;
            saved.lock().unwrap().push(bytes);
            Ok(())
        });
    let expected = whole.run().unwrap();
    let final_state = {
        let mut bytes = Vec::new();
        checkpoint::save(whole.algorithm(), &mut bytes).unwrap();
        bytes
    };
    drop(whole);
    // generation 20, after 420 evaluations
    let saved = saved.into_inner().unwrap();
    let ga: SteadyOneMax = checkpoint::load(saved[2].as_slice()).unwrap();
    assert_eq!(ga.evaluations(), 420);
    let mut resumed = AsyncEngine::new(ga, one_max)
        .workers(1)
        .stop_when(Stop::evaluations(1_000));
    let outcome = resumed.run().unwrap();
    assert_eq!(outcome.best(), expected.best());
    let mut bytes = Vec::new();
    checkpoint::save(resumed.algorithm(), &mut bytes).unwrap();
    assert_eq!(bytes, final_state);
}
