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
        portable_run(
            Ga::builder(real())
                .population_size(20)
                .select(Tournament::new(3).unwrap())
                .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
                .mutate(PolynomialMutation::per_gene(0.25, 20.0).unwrap())
                .minimize()
                .seed(1)
                .build()
                .unwrap(),
        ),
        portable_run(
            OpenEs::builder(real())
                .population_size(20)
                .sigma(0.01)
                .evaluate_mean(true)
                .minimize()
                .seed(1)
                .build()
                .unwrap(),
        ),
        portable_run(
            NelderMead::builder(real())
                .minimize()
                .seed(1)
                .build()
                .unwrap(),
        ),
        // forward differences: the trial points and their stencils
        portable_run(Lbfgsb::builder(real()).minimize().seed(1).build().unwrap()),
    ];
    let expected: [[f64; 4]; 9] = [
        // L-SHADE
        [
            0.5886518163542276,
            0.48299024536671004,
            0.16754416091898863,
            0.004140261403254408,
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
            -0.15790629965439074,
            0.022828940946501675,
            0.053509603593612454,
            -0.02819260775544219,
        ],
        // sep-CMA-ES
        [
            0.6531798520603047,
            0.37779440689264376,
            0.08078259643747288,
            0.009117858390141897,
        ],
        // (3/3_I, 12)-ES
        [
            0.25872566707661576,
            0.0749499276289265,
            0.010651887442370292,
            0.005722583060712822,
        ],
        // GA, SBX and polynomial mutation, sequential breeding
        [
            0.481469094537807,
            0.24876448694741246,
            0.04545026095464834,
            -0.016979733128666127,
        ],
        // OpenAI's ES, Adam, the mean evaluated
        [
            -0.4723413661463646,
            -1.2524725186772865,
            0.7656393855501439,
            -2.217604529838133,
        ],
        // Nelder-Mead, adaptive coefficients
        [
            -0.6425949189348894,
            -0.867373269448164,
            -0.27176836817948113,
            -1.087568298282683,
        ],
        // L-BFGS-B, forward differences
        [
            0.5988088458914269,
            0.28669497148242135,
            0.07699164601971455,
            0.012765658807016292,
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

#[test]
fn de_trials_change_a_gene_that_can_change() {
    // with parallel breeding too, where each trial chooses its gene on its own stream
    #[cfg(feature = "parallel")]
    let settings = [false, true];
    #[cfg(not(feature = "parallel"))]
    let settings = [false];
    for parallel_breeding in settings {
        // 9 fixed genes and 1 free one: with CR 0, every trial changes the free gene
        let bounds = (0..9).map(|_| 1.0..=1.0).chain(std::iter::once(-5.0..=5.0));
        let de = De::builder(Real::new(bounds).unwrap())
            .population_size(20)
            .control(de::Control::Fixed { f: 0.5, cr: 0.0 })
            .minimize()
            .seed(0);
        #[cfg(feature = "parallel")]
        let de = de.parallel_breeding(parallel_breeding);
        let mut de = de.build().unwrap();
        assert_eq!(de.parallel_breeding(), parallel_breeding);
        let copies = std::sync::Mutex::new((0, 0));
        // the initial population first
        let fitness: Vec<Fitness> = de.ask().iter().map(|x| Fitness::new(x[9] * x[9])).collect();
        de.tell(&fitness).unwrap();
        for _ in 0..20 {
            let population: Vec<Reals> =
                de.population().iter().map(|x| x.genome().clone()).collect();
            let fitness: Vec<Fitness> = de
                .ask()
                .iter()
                .map(|x| {
                    let mut counts = copies.lock().unwrap();
                    counts.0 += usize::from(population.contains(x));
                    counts.1 += 1;
                    Fitness::new(x[9] * x[9])
                })
                .collect();
            de.tell(&fitness).unwrap();
        }
        let (copied, trials) = copies.into_inner().unwrap();
        // before, about 9 in 10 were copies of their targets
        assert_eq!(copied, 0, "{copied} of {trials} trials are copies");
    }
}

// islands of DEs that build their trials in parallel: the same results with parallel evaluation or
// not, and on any number of threads
#[cfg(feature = "parallel")]
#[test]
fn islands_with_parallel_breeding_are_the_same_on_any_number_of_threads() {
    use genoxide::algorithm::islands::Topology;
    fn run(islands: Vec<De>, parallel: bool) -> (Individual<Reals>, Population<Reals>) {
        let islands = Islands::builder(islands)
            .topology(Topology::Random)
            .interval(3)
            .seed(9)
            .build()
            .unwrap();
        let mut engine = Engine::new(islands, rastrigin)
            .parallel(parallel)
            .stop_when(Stop::generations(20));
        let outcome = engine.run().unwrap();
        (outcome.into_best(), engine.algorithm().population().clone())
    }
    let des = || {
        (0..4)
            .map(|seed| {
                De::builder(Real::uniform(10, -5.12..=5.12).unwrap())
                    .population_size(20)
                    .parallel_breeding(true)
                    .minimize()
                    .seed(seed)
                    .build()
                    .unwrap()
            })
            .collect::<Vec<_>>()
    };
    let on_threads =
        |threads: usize, f: &(dyn Fn() -> (Individual<Reals>, Population<Reals>) + Sync)| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(f)
        };
    let de = run(des(), false);
    assert_eq!(run(des(), true), de);
    for threads in [1, 2, 8] {
        assert_eq!(on_threads(threads, &|| run(des(), true)), de);
    }
}

#[test]
fn a_huge_lambda_is_a_setting_error() {
    // more offspring than fit in memory, on 32 and 64 bits: an individual takes more than 2 bytes
    let huge = [usize::MAX, isize::MAX as usize / 2];
    for scheme in huge.into_iter().flat_map(|lambda| {
        [
            Scheme::MuPlusLambda { lambda },
            Scheme::MuCommaLambda { lambda },
        ]
    }) {
        let result = Ga::builder(Binary::new(8).unwrap())
            .population_size(4)
            .select(Tournament::new(2).unwrap())
            .crossover(UniformCrossover::new())
            .mutate(BitFlip::count(1).unwrap())
            .scheme(scheme)
            .build();
        assert!(matches!(
            result,
            Err(Error::InvalidSetting {
                setting: "scheme",
                ..
            })
        ));
    }
}
