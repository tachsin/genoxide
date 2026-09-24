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

fn rosenbrock(x: &Reals) -> f64 {
    x.windows(2)
        .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
        .sum()
}

#[test]
fn particle_swarm_solves_rosenbrock() {
    let pso = Pso::builder(Real::uniform(10, -5.0..=10.0).unwrap())
        .population_size(40)
        .minimize()
        .seed(0)
        .build()
        .unwrap();
    let outcome = Engine::new(pso, rosenbrock)
        .stop_when(Stop::target(0.01).or(Stop::evaluations(200_000)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
}

#[test]
fn particle_swarm_ring_escapes_more_local_optima_than_global() {
    // the ring spreads good positions slowly and keeps exploring
    let median = |topology| {
        let mut results: Vec<f64> = (0..5)
            .map(|seed| {
                let pso = Pso::builder(Real::uniform(10, -5.12..=5.12).unwrap())
                    .population_size(40)
                    .topology(topology)
                    .minimize()
                    .seed(seed)
                    .build()
                    .unwrap();
                let outcome = Engine::new(pso, rastrigin)
                    .stop_when(Stop::evaluations(100_000))
                    .run()
                    .unwrap();
                outcome.best().fitness().unwrap().score().unwrap()
            })
            .collect();
        results.sort_by(f64::total_cmp);
        results[2]
    };
    assert!(median(pso::Topology::Ring { neighbors: 1 }) < median(pso::Topology::Global));
}

#[test]
fn particle_swarm_runs_in_parallel_with_the_same_results() {
    let run = |parallel: bool| {
        let pso = Pso::builder(Real::uniform(10, -5.12..=5.12).unwrap())
            .population_size(30)
            .topology(pso::Topology::Ring { neighbors: 1 })
            .minimize()
            .seed(7)
            .build()
            .unwrap();
        let engine = Engine::new(pso, rastrigin).stop_when(Stop::generations(50));
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
fn cma_es_with_restarts_solves_rastrigin() {
    for restarts in [cmaes::Restarts::Ipop, cmaes::Restarts::Bipop] {
        let cmaes = Cmaes::builder(Real::uniform(10, -5.12..=5.12).unwrap())
            .restarts(restarts)
            .minimize()
            .seed(0)
            .build()
            .unwrap();
        let outcome = Engine::new(cmaes, rastrigin)
            .stop_when(Stop::target(0.01).or(Stop::evaluations(500_000)))
            .run()
            .unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Target, "{restarts:?}");
    }
}

#[test]
fn cma_es_runs_in_parallel_with_the_same_results() {
    let run = |parallel: bool| {
        let cmaes = Cmaes::builder(Real::uniform(10, -5.12..=5.12).unwrap())
            .restarts(cmaes::Restarts::Bipop)
            .minimize()
            .seed(7)
            .build()
            .unwrap();
        let engine = Engine::new(cmaes, rastrigin).stop_when(Stop::generations(300));
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
fn evolution_strategy_runs_in_parallel_with_the_same_results() {
    let run = |parallel: bool| {
        let es = Es::builder(Real::uniform(10, -5.12..=5.12).unwrap())
            .parents(5)
            .offspring(35)
            .recombination(es::Recombination::Dominant { rho: 2 })
            .minimize()
            .seed(7)
            .build()
            .unwrap();
        let engine = Engine::new(es, rastrigin).stop_when(Stop::generations(100));
        #[cfg(feature = "parallel")]
        let engine = engine.parallel(parallel);
        #[cfg(not(feature = "parallel"))]
        let _ = parallel;
        let mut engine = engine;
        engine.run().unwrap().into_best()
    };
    assert_eq!(run(true), run(false));
}

// the best genome after a short run of each algorithm on a fitness function with only +, − and ×
// (unlike Rastrigin's cosine or `powi`, which can differ between platforms); these must never
// change for the same major version, on any platform
fn portable_run<A: Algorithm<Genome = Reals>>(algorithm: A) -> Vec<f64> {
    let rosenbrock = |x: &Reals| {
        x.windows(2)
            .map(|w| {
                let (a, b) = (w[1] - w[0] * w[0], 1.0 - w[0]);
                100.0 * a * a + b * b
            })
            .sum::<f64>()
    };
    Engine::new(algorithm, rosenbrock)
        .stop_when(Stop::generations(30))
        .run()
        .unwrap()
        .into_best()
        .into_genome()
        .into_vec()
}

#[test]
fn portable_runs() {
    let real = || Real::uniform(4, -5.12..=5.12).unwrap();
    let runs = [
        portable_run(
            De::l_shade(real(), 2_000)
                .minimize()
                .seed(1)
                .build()
                .unwrap(),
        ),
        portable_run(
            Pso::builder(real())
                .population_size(10)
                .topology(pso::Topology::Ring { neighbors: 1 })
                .minimize()
                .seed(1)
                .build()
                .unwrap(),
        ),
        portable_run(
            Cmaes::builder(real())
                .restarts(cmaes::Restarts::Bipop)
                .minimize()
                .seed(1)
                .build()
                .unwrap(),
        ),
        portable_run(
            Cmaes::builder(real())
                .covariance(cmaes::Covariance::Diagonal)
                .minimize()
                .seed(1)
                .build()
                .unwrap(),
        ),
        portable_run(
            Es::builder(real())
                .parents(3)
                .offspring(12)
                .minimize()
                .seed(1)
                .build()
                .unwrap(),
        ),
    ];
    let expected: [[f64; 4]; 5] = [
        // L-SHADE
        [
            1.06352535213699,
            1.169859968810408,
            1.4252844985565545,
            1.795976790704828,
        ],
        // PSO, ring
        [
            -0.1799238193223966,
            -0.05058389355291476,
            -0.0612702176085896,
            -0.06988516362853431,
        ],
        // CMA-ES, BIPOP
        [
            0.5279924007248189,
            0.25175040581128894,
            0.04444586199689393,
            -0.025219443384489004,
        ],
        // sep-CMA-ES
        [
            0.8459394832109357,
            0.7058663406676571,
            0.5424466612325327,
            0.3428791600536547,
        ],
        // (3/3_I, 12)-ES
        [
            0.5294149601094941,
            0.27231802269909716,
            0.07182313884499765,
            -0.001476731001038885,
        ],
    ];
    for (run, expected) in runs.iter().zip(expected) {
        assert_eq!(run[..], expected);
    }
}

#[test]
fn islands_run_in_parallel_with_the_same_results() {
    use genoxide::algorithm::islands::Topology;
    let run = |parallel: bool| {
        let islands = (0..4)
            .map(|seed| {
                Ga::builder(Real::uniform(10, -5.12..=5.12).unwrap())
                    .population_size(20)
                    .select(Tournament::new(3).unwrap())
                    .crossover(UniformCrossover::new())
                    .mutate(PolynomialMutation::per_gene(0.1, 20.0).unwrap())
                    .minimize()
                    .seed(seed)
                    .build()
                    .unwrap()
            })
            .collect();
        let islands = Islands::builder(islands)
            .topology(Topology::Random)
            .interval(3)
            .seed(9)
            .build()
            .unwrap();
        let engine = Engine::new(islands, rastrigin).stop_when(Stop::generations(40));
        #[cfg(feature = "parallel")]
        let engine = engine.parallel(parallel);
        #[cfg(not(feature = "parallel"))]
        let _ = parallel;
        let mut engine = engine;
        let outcome = engine.run().unwrap();
        (outcome.into_best(), engine.algorithm().population().clone())
    };
    assert_eq!(run(true), run(false));
}
