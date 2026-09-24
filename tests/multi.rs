//! Multi-objective optimization on standard problems, and the multi-objective engine.

use genoxide::Objective::{Maximize, Minimize};
use genoxide::engine::NanPolicy;
use genoxide::multi::indicator::{hypervolume, igd_plus};
use genoxide::operator::{PolynomialMutation, SimulatedBinaryCrossover};
use genoxide::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

type RealNsga2<const M: usize> =
    genoxide::multi::Nsga2<Real, SimulatedBinaryCrossover, PolynomialMutation, M>;

fn nsga2<const M: usize>(
    real: Real,
    objectives: [Objective; M],
    size: usize,
    seed: u64,
) -> RealNsga2<M> {
    let rate = 1.0 / real.bounds().len() as f64;
    Nsga2::builder(real, objectives)
        .population_size(size)
        .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
        .mutate(PolynomialMutation::per_gene(rate, 20.0).unwrap())
        .seed(seed)
        .build()
        .unwrap()
}

fn zdt1(x: &Reals) -> [f64; 2] {
    let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (x.len() - 1) as f64;
    [x[0], g * (1.0 - (x[0] / g).sqrt())]
}

#[test]
fn nsga2_approximates_the_zdt1_front() {
    // the optimal front has a hypervolume of 0.8716 (reference point 1.1, 1.1); pymoo's NSGA-II
    // reaches 0.8696 to 0.8699 with these settings, genoxide 0.8689 to 0.8700 (5 seeds)
    let algorithm = nsga2(
        Real::uniform(30, 0.0..=1.0).unwrap(),
        [Minimize, Minimize],
        100,
        0,
    );
    let outcome = MultiEngine::new(algorithm, zdt1)
        .stop_when(Stop::generations(249))
        .run()
        .unwrap();
    // at most 250 generations of 100; children that copy a parent aren't evaluated
    assert!(outcome.evaluations() <= 25_000);
    let front = outcome.front_values();
    let volume = hypervolume(&front, &[1.1, 1.1], &[Minimize, Minimize]);
    assert!(volume > 0.868, "{volume}");
    // 1000 points of the optimal front, f2 = 1 - √f1
    let optimal: Vec<[f64; 2]> = (0..1000)
        .map(|i| {
            let f1 = i as f64 / 999.0;
            [f1, 1.0 - f1.sqrt()]
        })
        .collect();
    let distance = igd_plus(&front, &optimal, &[Minimize, Minimize]);
    assert!(distance < 0.01, "{distance}");
}

#[test]
fn nsga2_finds_a_feasible_front_of_a_constrained_problem() {
    // CONSTR (Deb): minimize x1 and (1 + x2) / x1, with 9 x1 + x2 >= 6 and 9 x1 - x2 >= 1
    let constr = |x: &Reals| {
        let violation = (6.0 - 9.0 * x[0] - x[1]).max(0.0) + (1.0 - 9.0 * x[0] + x[1]).max(0.0);
        ([x[0], (1.0 + x[1]) / x[0]], violation)
    };
    let real = Real::new([0.1..=1.0, 0.0..=5.0]).unwrap();
    let outcome = MultiEngine::new(nsga2(real, [Minimize, Minimize], 40, 1), constr)
        .stop_when(Stop::generations(100))
        .run()
        .unwrap();
    assert_eq!(outcome.front().len(), 40);
    for individual in outcome.front() {
        let scores = individual.fitness().unwrap();
        assert!(scores.is_feasible(), "{scores}");
    }
    // the front spans x1 from about 0.39 to 1
    let x1: Vec<f64> = outcome.front().iter().map(|x| x.genome()[0]).collect();
    let (low, high) = (
        x1.iter().copied().fold(1.0, f64::min),
        x1.iter().copied().fold(0.0, f64::max),
    );
    assert!(low < 0.45 && high > 0.95, "{low} {high}");
}

#[test]
fn objectives_can_be_maximized() {
    // maximize -x² and -(x - 2)²: the same front as minimizing x² and (x - 2)²
    let schaffer = |x: &Reals| [-x[0] * x[0], -(x[0] - 2.0) * (x[0] - 2.0)];
    let algorithm = nsga2(
        Real::uniform(1, -10.0..=10.0).unwrap(),
        [Maximize, Maximize],
        20,
        2,
    );
    let outcome = MultiEngine::new(algorithm, schaffer)
        .stop_when(Stop::generations(60))
        .run()
        .unwrap();
    assert!(
        outcome
            .front()
            .iter()
            .all(|x| (-0.01..=2.01).contains(&x.genome()[0]))
    );
}

#[test]
fn fitness_functions_return_any_kind_of_scores() {
    let run = |result: fn(&Reals) -> Option<[f64; 2]>| {
        let algorithm = nsga2(
            Real::uniform(2, 0.0..=1.0).unwrap(),
            [Minimize, Minimize],
            8,
            3,
        );
        MultiEngine::new(algorithm, result)
            .stop_when(Stop::generations(3))
            .run()
            .unwrap()
    };
    // None is invalid: an all-invalid population has an all-invalid front
    let outcome = run(|_| None);
    assert!(
        outcome
            .front()
            .iter()
            .all(|x| !x.fitness().unwrap().is_valid())
    );
    assert!(outcome.front_values().is_empty());
    let outcome = run(|x| Some([x[0], x[1]]));
    assert!(!outcome.front_values().is_empty());
    let scores = |x: &Reals| Scores::new([x[0], 1.0 - x[0]]);
    let algorithm = nsga2(
        Real::uniform(2, 0.0..=1.0).unwrap(),
        [Minimize, Minimize],
        8,
        3,
    );
    let outcome = MultiEngine::new(algorithm, scores)
        .stop_when(Stop::generations(3))
        .run()
        .unwrap();
    assert_eq!(outcome.front().len(), 8);
}

#[test]
fn nan_scores_are_invalid_or_errors() {
    let nan = |x: &Reals| [x[0], f64::NAN];
    let algorithm = || {
        nsga2(
            Real::uniform(2, 0.0..=1.0).unwrap(),
            [Minimize, Minimize],
            8,
            4,
        )
    };
    let outcome = MultiEngine::new(algorithm(), nan)
        .stop_when(Stop::generations(2))
        .run()
        .unwrap();
    assert!(outcome.front_values().is_empty());
    let error = MultiEngine::new(algorithm(), nan)
        .nan_policy(NanPolicy::Error)
        .stop_when(Stop::generations(2))
        .run();
    assert_eq!(error, Err(Error::NanFitness));
    let negative = |x: &Reals| ([x[0], x[1]], -1.0);
    let error = MultiEngine::new(algorithm(), negative)
        .stop_when(Stop::generations(2))
        .run();
    assert!(matches!(error, Err(Error::InvalidFitness { .. })));
}

#[test]
fn engine_settings_and_stop_conditions() {
    let algorithm = || {
        nsga2(
            Real::uniform(2, 0.0..=1.0).unwrap(),
            [Minimize, Minimize],
            8,
            5,
        )
    };
    let f = |x: &Reals| [x[0], 1.0 - x[0] + x[1]];
    let error = MultiEngine::new(algorithm(), f).run();
    assert!(matches!(
        error,
        Err(Error::MissingSetting {
            setting: "stop_when"
        })
    ));
    // a target needs a single objective
    let error = MultiEngine::new(algorithm(), f)
        .stop_when(Stop::generations(10).or(Stop::target(0.0)))
        .run();
    assert!(matches!(
        error,
        Err(Error::InvalidSetting {
            setting: "stop_when",
            ..
        })
    ));
    // a front that stops improving: a flat function
    let outcome = MultiEngine::new(algorithm(), |_: &Reals| [1.0, 1.0])
        .stop_when(Stop::stagnation(5).or(Stop::generations(100)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Stagnation);
    assert_eq!(outcome.generations(), 5);
    // an abort flag that is already set stops after the first generation
    let outcome = MultiEngine::new(algorithm(), f)
        .abort_flag(Arc::new(AtomicBool::new(true)))
        .run()
        .unwrap();
    assert_eq!(
        (outcome.stop_reason(), outcome.generations()),
        (StopReason::Aborted, 0)
    );
    // the callback sees every generation
    let mut seen = Vec::new();
    let outcome = MultiEngine::new(algorithm(), f)
        .on_generation(|snapshot| {
            assert_eq!(snapshot.population().len(), 8);
            assert!(!snapshot.front().is_empty());
            seen.push(snapshot.progress().generation());
        })
        .stop_when(Stop::generations(4))
        .run()
        .unwrap();
    assert_eq!(seen, [0, 1, 2, 3, 4]);
    assert!(outcome.evaluations() <= 40);
}

#[test]
fn nsga2_runs_in_parallel_with_the_same_results() {
    let run = |parallel: bool| {
        let algorithm = nsga2(
            Real::uniform(30, 0.0..=1.0).unwrap(),
            [Minimize, Minimize],
            40,
            7,
        );
        let engine = MultiEngine::new(algorithm, zdt1).stop_when(Stop::generations(30));
        #[cfg(feature = "parallel")]
        let engine = engine.parallel(parallel);
        #[cfg(not(feature = "parallel"))]
        let _ = parallel;
        let mut engine = engine;
        engine.run().unwrap().into_front()
    };
    assert_eq!(run(true), run(false));
}

#[test]
fn nsga3_spreads_a_three_objective_front() {
    use genoxide::multi::indicator::igd;
    use genoxide::multi::problems::{Dtlz2, TestProblem};
    use genoxide::multi::{Nsga3, das_dennis};
    // DTLZ2 with 91 reference directions (Deb and Jain's settings): an IGD to the 91 optimal
    // points of 0.0009 to 0.0015 for pymoo, 0.0012 to 0.0016 for genoxide (5 seeds)
    let problem = Dtlz2::<3>::default();
    let nsga3 = Nsga3::builder(problem.real(), [Minimize; 3], das_dennis::<3>(12))
        .population_size(92)
        .crossover(SimulatedBinaryCrossover::new(30.0).unwrap())
        .mutate(PolynomialMutation::per_gene(1.0 / 12.0, 20.0).unwrap())
        .seed(0)
        .build()
        .unwrap();
    let outcome = MultiEngine::new(nsga3, problem)
        .stop_when(Stop::generations(249))
        .run()
        .unwrap();
    let distance = igd(&outcome.front_values(), &problem.optimal_front(91));
    assert!(distance < 0.002, "{distance}");
}

#[test]
fn spea2_approximates_the_zdt1_front() {
    use genoxide::multi::Spea2;
    use genoxide::multi::problems::{TestProblem, Zdt1};
    // a hypervolume of 0.8697 to 0.8704 over 5 seeds; pymoo's SPEA2 reaches 0.8703 to 0.8706
    let problem = Zdt1::new(30);
    let spea2 = Spea2::builder(problem.real(), [Minimize, Minimize])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
        .mutate(PolynomialMutation::per_gene(1.0 / 30.0, 20.0).unwrap())
        .seed(0)
        .build()
        .unwrap();
    let outcome = MultiEngine::new(spea2, problem)
        .stop_when(Stop::generations(249))
        .run()
        .unwrap();
    let volume = hypervolume(&outcome.front_values(), &[1.1, 1.1], &[Minimize, Minimize]);
    assert!(volume > 0.869, "{volume}");
}

#[test]
fn moead_approximates_two_and_three_objective_fronts() {
    use genoxide::multi::indicator::igd;
    use genoxide::multi::problems::{Dtlz2, TestProblem, Zdt1};
    use genoxide::multi::{Decomposition, Moead, das_dennis};
    // ZDT1 with 100 weights and Tchebycheff: a hypervolume of 0.8683 to 0.8688 over 5 seeds
    // (pymoo's sequential MOEA/D 0.8693 to 0.8705, its ParallelMOEAD 0.78 to 0.83)
    let problem = Zdt1::new(30);
    let moead = Moead::builder(problem.real(), [Minimize; 2], das_dennis::<2>(99))
        .crossover(SimulatedBinaryCrossover::new(20.0).unwrap())
        .mutate(PolynomialMutation::per_gene(1.0 / 30.0, 20.0).unwrap())
        .seed(0)
        .build()
        .unwrap();
    let outcome = MultiEngine::new(moead, problem)
        .stop_when(Stop::generations(249))
        .run()
        .unwrap();
    let volume = hypervolume(&outcome.front_values(), &[1.1, 1.1], &[Minimize; 2]);
    assert!(volume > 0.867, "{volume}");
    // DTLZ2 with 91 weights and PBI: an IGD of 0.0008 to 0.0010 (pymoo 0.0005 to 0.0006)
    let problem = Dtlz2::<3>::default();
    let moead = Moead::builder(problem.real(), [Minimize; 3], das_dennis::<3>(12))
        .decomposition(Decomposition::Pbi { theta: 5.0 })
        .crossover(SimulatedBinaryCrossover::new(20.0).unwrap())
        .mutate(PolynomialMutation::per_gene(1.0 / 12.0, 20.0).unwrap())
        .seed(0)
        .build()
        .unwrap();
    let outcome = MultiEngine::new(moead, problem)
        .stop_when(Stop::generations(249))
        .run()
        .unwrap();
    let distance = igd(&outcome.front_values(), &problem.optimal_front(91));
    assert!(distance < 0.0012, "{distance}");
}

#[test]
fn sms_emoa_reaches_the_optimal_zdt1_hypervolume() {
    use genoxide::multi::SmsEmoa;
    use genoxide::multi::problems::{TestProblem, Zdt1};
    // the optimal front's hypervolume is 0.8716: 0.8713 to 0.8716 over 5 seeds, like pymoo's
    // SMS-EMOA (0.8715 to 0.8718)
    let problem = Zdt1::new(30);
    let sms_emoa = SmsEmoa::builder(problem.real(), [Minimize, Minimize])
        .population_size(100)
        .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
        .mutate(PolynomialMutation::per_gene(1.0 / 30.0, 20.0).unwrap())
        .seed(0)
        .build()
        .unwrap();
    let outcome = MultiEngine::new(sms_emoa, problem)
        .stop_when(Stop::generations(249))
        .run()
        .unwrap();
    let volume = hypervolume(&outcome.front_values(), &[1.1, 1.1], &[Minimize, Minimize]);
    assert!(volume > 0.871, "{volume}");
}

#[test]
fn batch_matches_one_genome_at_a_time() {
    let algorithm = || {
        nsga2(
            Real::uniform(10, 0.0..=1.0).unwrap(),
            [Minimize, Minimize],
            20,
            7,
        )
    };
    let one = MultiEngine::new(algorithm(), zdt1)
        .stop_when(Stop::generations(20))
        .run()
        .unwrap();
    let calls = AtomicUsize::new(0);
    let batch = MultiEngine::new(
        algorithm(),
        Batch(|genomes: &[&Reals]| {
            calls.fetch_add(1, Ordering::Relaxed);
            genomes.iter().map(|x| zdt1(x)).collect::<Vec<_>>()
        }),
    )
    .stop_when(Stop::generations(20))
    .run()
    .unwrap();
    assert_eq!(one.front(), batch.front());
    assert_eq!(one.evaluations(), batch.evaluations());
    assert_eq!(calls.into_inner(), 21);
    // a wrong number of scores stops the run
    let error = MultiEngine::new(algorithm(), Batch(|_: &[&Reals]| vec![[0.0, 0.0]]))
        .stop_when(Stop::generations(20))
        .run()
        .unwrap_err();
    assert_eq!(
        error,
        Error::FitnessCount {
            expected: 20,
            got: 1
        }
    );
}
