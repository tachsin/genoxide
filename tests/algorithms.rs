//! The algorithms of 0.3 on standard problems.

use genoxide::prelude::*;
use std::f64::consts::TAU;

fn rastrigin(x: &Reals) -> f64 {
    10.0 * x.len() as f64
        + x.iter()
            .map(|xi| xi * xi - 10.0 * (TAU * xi).cos())
            .sum::<f64>()
}

#[test]
fn differential_evolution_solves_rastrigin() {
    // a small crossover rate suits a separable function like Rastrigin
    for strategy in [
        de::Strategy::Rand1,
        de::Strategy::CurrentToPBest {
            p: 0.1,
            archive: 1.0,
        },
    ] {
        let de = De::builder(Real::uniform(10, -5.12..=5.12).unwrap())
            .population_size(100)
            .strategy(strategy)
            .control(de::Control::Fixed { f: 0.5, cr: 0.1 })
            .minimize()
            .seed(0)
            .build()
            .unwrap();
        let outcome = Engine::new(de, rastrigin)
            .stop_when(Stop::target(0.01).or(Stop::evaluations(100_000)))
            .run()
            .unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Target, "{strategy:?}");
    }
}

#[test]
fn differential_evolution_runs_in_parallel_with_the_same_results() {
    let run = |parallel: bool| {
        let de = De::builder(Real::uniform(10, -5.12..=5.12).unwrap())
            .population_size(50)
            .strategy(de::Strategy::CurrentToPBest {
                p: 0.1,
                archive: 1.0,
            })
            .minimize()
            .seed(7)
            .build()
            .unwrap();
        let engine = Engine::new(de, rastrigin).stop_when(Stop::generations(50));
        #[cfg(feature = "parallel")]
        let engine = engine.parallel(parallel);
        #[cfg(not(feature = "parallel"))]
        let _ = parallel;
        let mut engine = engine;
        engine.run().unwrap().into_best()
    };
    assert_eq!(run(true), run(false));
}

#[test]
fn adaptive_differential_evolution_solves_rastrigin_without_tuning() {
    // JADE and SHADE learn a crossover rate that suits the problem
    for control in [
        de::Control::Jade { c: 0.1 },
        de::Control::Shade { memory: 6 },
    ] {
        let de = De::builder(Real::uniform(10, -5.12..=5.12).unwrap())
            .population_size(100)
            .strategy(de::Strategy::CurrentToPBest {
                p: 0.1,
                archive: 1.0,
            })
            .control(control)
            .minimize()
            .seed(0)
            .build()
            .unwrap();
        let outcome = Engine::new(de, rastrigin)
            .stop_when(Stop::target(0.01).or(Stop::evaluations(100_000)))
            .run()
            .unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Target, "{control:?}");
    }
    let budget = 100_000;
    let de = De::l_shade(Real::uniform(10, -5.12..=5.12).unwrap(), budget)
        .minimize()
        .seed(0)
        .build()
        .unwrap();
    let outcome = Engine::new(de, rastrigin)
        .stop_when(Stop::target(0.01).or(Stop::evaluations(budget)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
}
