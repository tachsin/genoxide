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
    x.array_windows()
        .map(|&[xi, next]| 100.0 * (next - xi * xi).powi(2) + (1.0 - xi).powi(2))
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
        x.array_windows()
            .map(|&[xi, next]| {
                let (a, b) = (next - xi * xi, 1.0 - xi);
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

// the same for the methods that need a constraint's values and the gradients: Rosenbrock's
// function, with the genes' sum at most 1
fn portable_constrained_run<A: Algorithm<Genome = Reals>>(algorithm: A) -> Vec<f64> {
    let rosenbrock = Constrained::differentiable(
        1,
        |x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]| {
            let mut value = 0.0;
            gradient.fill(0.0);
            for i in 0..x.len() - 1 {
                let (a, b) = (x[i + 1] - x[i] * x[i], 1.0 - x[i]);
                value += 100.0 * a * a + b * b;
                gradient[i] += -400.0 * a * x[i] - 2.0 * b;
                gradient[i + 1] += 200.0 * a;
            }
            g[0] = x.iter().sum::<f64>() - 1.0;
            jacobian.fill(1.0);
            value
        },
    );
    Engine::new(algorithm, rosenbrock)
        .stop_when(Stop::generations(30))
        .run()
        .unwrap()
        .into_best()
        .into_genome()
        .into_vec()
}

// the same for a continuation: Rosenbrock's function with the weight of its valley's walls raised
// from 1 to 10 and 100, in 3 stages of 10 generations, the weight shared through an atomic
fn portable_continuation_run(algorithm: FirstOrder) -> Vec<f64> {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};
    const WEIGHTS: [f64; 3] = [1.0, 10.0, 100.0];
    let weight = Arc::new(AtomicU64::new(0));
    let shared = Arc::clone(&weight);
    let rosenbrock = move |x: &Reals| {
        let w = f64::from_bits(shared.load(Ordering::Relaxed));
        x.array_windows()
            .map(|&[xi, next]| {
                let (a, b) = (next - xi * xi, 1.0 - xi);
                w * a * a + b * b
            })
            .sum::<f64>()
    };
    let continuation = Continuation::builder(algorithm)
        .stages(WEIGHTS.len())
        .generations(10)
        .on_stage(move |stage, _| {
            weight.store(WEIGHTS[stage].to_bits(), Ordering::Relaxed);
            Ok(())
        })
        .build()
        .unwrap();
    let outcome = Engine::new(continuation, rosenbrock)
        .stop_when(Stop::generations(1_000))
        .run()
        .unwrap();
    assert_eq!(outcome.generations(), 30);
    outcome.into_best().into_genome().into_vec()
}

// the same for MOEA/D: two objectives, Rosenbrock's function and the distance from (1, 0, 0, 0),
// with the genes' sum at most 1, for 30 generations; the genome of the fourth of its 10
// subproblems
fn portable_moead_run<C, X>(moead: multi::Moead<Real, C, X, 2>) -> Vec<f64>
where
    C: multi::moead::MoeadCrossover<Real>,
    X: genoxide::operator::Mutate<Real>,
{
    let objectives = |x: &Reals| {
        let rosenbrock = x
            .array_windows()
            .map(|&[xi, next]| {
                let (a, b) = (next - xi * xi, 1.0 - xi);
                100.0 * a * a + b * b
            })
            .sum::<f64>();
        let distance = (x[0] - 1.0) * (x[0] - 1.0) + x[1..].iter().map(|v| v * v).sum::<f64>();
        let sum = x.iter().sum::<f64>();
        ([rosenbrock, distance], constraint::at_most(sum, 1.0))
    };
    let mut engine = MultiEngine::new(moead, objectives).stop_when(Stop::generations(30));
    engine.run().unwrap();
    engine.algorithm().population()[3].genome().to_vec()
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
        portable_run(
            FirstOrder::builder(real())
                .step(first_order::Step::adam(0.05))
                .minimize()
                .seed(1)
                .build()
                .unwrap(),
        ),
        portable_run(
            FirstOrder::builder(real())
                .step(first_order::Step::nesterov(0.0005, 0.9))
                .gradients(genoxide::gradient::Gradients::Central { step: None })
                .minimize()
                .seed(1)
                .build()
                .unwrap(),
        ),
        portable_constrained_run(Mma::builder(real()).minimize().seed(1).build().unwrap()),
        portable_constrained_run(
            Mma::builder(real())
                .method(mma::Method::Gcmma)
                .minimize()
                .seed(1)
                .build()
                .unwrap(),
        ),
        portable_continuation_run(
            FirstOrder::builder(real())
                .step(first_order::Step::adam(0.05))
                .minimize()
                .seed(1)
                .build()
                .unwrap(),
        ),
        // the Gaussian process's fits, the acquisition's maximization, the Latin hypercube
        portable_run(Bo::builder(real()).minimize().seed(1).build().unwrap()),
        portable_run(
            Bo::builder(real())
                .output(bo::Output::Log)
                .acquisition(bo::Acquisition::ProbabilityOfImprovement { xi: 0.1 })
                .kernel(genoxide::model::gp::Kernel::SquaredExponential)
                .minimize()
                .seed(1)
                .build()
                .unwrap(),
        ),
        // batches: the points after the first, chosen with the earlier ones fantasized
        portable_run(
            Bo::builder(real())
                .batch(3)
                .fantasy(bo::Fantasy::KrigingBeliever)
                .minimize()
                .seed(1)
                .build()
                .unwrap(),
        ),
        portable_run(
            Bo::builder(real())
                .batch(2)
                .fantasy(bo::Fantasy::ConstantLiar(bo::Lie::Mean))
                .acquisition(bo::Acquisition::ExpectedImprovement)
                .minimize()
                .seed(1)
                .build()
                .unwrap(),
        ),
        // a model per constraint, the probability of feasibility
        portable_constrained_run(Bo::builder(real()).minimize().seed(1).build().unwrap()),
        // MOEA/D, SBX and polynomial mutation, a constraint
        portable_moead_run(
            Moead::builder(real(), [Objective::Minimize; 2], multi::das_dennis::<2>(9))
                .neighbors(4)
                .crossover(SimulatedBinaryCrossover::new(20.0).unwrap())
                .mutate(PolynomialMutation::per_gene(0.25, 20.0).unwrap())
                .seed(1)
                .build()
                .unwrap(),
        ),
        // MOEA/D-DE: the difference, the bounce at the bounds, children replacing anywhere
        portable_moead_run(
            Moead::builder(real(), [Objective::Minimize; 2], multi::das_dennis::<2>(9))
                .neighbors(4)
                .neighbor_mating(0.5)
                .crossover(multi::DifferentialEvolutionCrossover::new(0.5, 1.0).unwrap())
                .mutate(PolynomialMutation::per_gene(0.25, 20.0).unwrap())
                .seed(1)
                .build()
                .unwrap(),
        ),
    ];

    let expected: [[f64; 4]; 21] = [
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
        // Adam, forward differences
        [
            0.06201319067514992,
            -2.9684459972223216,
            1.3518228963997165,
            -1.3667172812746686,
        ],
        // Nesterov's accelerated gradient, central differences
        [
            -0.9985468040719558,
            -4.29684480105349,
            0.9887762527729134,
            -2.874116295845887,
        ],
        // MMA, the genes' sum at most 1
        [
            -0.4277873319263924,
            0.2310265139269608,
            0.10914038419999017,
            0.0012034094553207975,
        ],
        // GCMMA, the genes' sum at most 1
        [
            0.050640279572221975,
            0.01228299951717058,
            0.009594958407758064,
            0.017415965745941873,
        ],
        // a continuation of Adam, forward differences, the weight raised in 3 stages
        [
            -0.01396419948840609,
            -2.7974966575791904,
            1.1930722176383557,
            -1.3732603836128217,
        ], // Bayesian optimization: log-EI, Matérn 5/2
        [
            0.7497663253697677,
            -0.056696340352855756,
            0.7649350749581538,
            0.7470392106959753,
        ],
        // Bayesian optimization: the log transform, PI, the squared exponential
        [
            0.8989959153523532,
            0.8470798321557886,
            0.7794047534046049,
            0.6827343441068123,
        ],
        // Bayesian optimization in batches of 3, the Kriging believer
        [
            0.5850147650315565,
            0.4831239937078218,
            0.17793827600675627,
            -0.01072745023849997,
        ],
        // Bayesian optimization in batches of 2, EI, the constant liar with the mean
        [
            -0.8772643061279055,
            1.5450479136340656,
            2.0635451232757465,
            4.5196908819175645,
        ],
        // constrained Bayesian optimization, the genes' sum at most 1
        [
            0.6539353361599387,
            0.41472093863190285,
            0.10930157616421621,
            -0.17840095840143988,
        ],
        // MOEA/D, SBX and polynomial mutation, a constraint
        [
            0.281495821273559,
            0.011451771393886454,
            -0.04080469266299676,
            0.026023236154947352,
        ],
        // MOEA/D-DE, the bounce at the bounds, half the parents from the whole population
        [
            -0.13719653731926562,
            0.06751346922755136,
            0.20061038090932337,
            -0.1413363342369776,
        ],
    ];
    assert_eq!(runs.len(), expected.len());
    for (run, expected) in runs.iter().zip(expected) {
        assert_eq!(run[..], expected, "{run:?}");
    }
    // Bayesian optimization of integer genes: the last 4 points of 30 generations
    let quadratic = |x: &Integers| {
        let x: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        let (a, b) = (x[1] - x[0] * x[0] / 8.0, 3.0 - x[0]);
        10.0 * a * a + b * b + 0.5 * (x[2] - x[3]) * (x[2] + 1.0)
    };
    let integer = Bo::builder(Integer::uniform(4, -12..=12).unwrap())
        .minimize()
        .seed(1)
        .build()
        .unwrap();
    let mut engine = Engine::new(integer, quadratic).stop_when(Stop::generations(30));
    engine.run().unwrap();
    let points: Vec<Vec<i64>> = engine.algorithm().population().as_slice()[36..]
        .iter()
        .map(|individual| individual.genome().to_vec())
        .collect();
    let expected = [
        [6, 4, -2, -12],
        [8, 8, -5, -12],
        [7, 6, -5, -12],
        [3, 1, -5, -12],
    ];
    assert_eq!(points, expected, "{points:?}");
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
