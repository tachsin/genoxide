//! Bayesian optimization: the classic problems of Jones, Schonlau and Welch (1998) reached in tens
//! of evaluations, the guarantees of every algorithm, and how invalid points are handled.

use genoxide::algorithm::Incremental;
use genoxide::algorithm::bo::{Acquisition, Fantasy, Lie, Output};
use genoxide::constraint::Constrained;
use genoxide::model::gp::Kernel;
use genoxide::prelude::*;
use genoxide::problems::{Branin, GoldsteinPrice, Hartmann3, Problem, SixHumpCamel};

// the evaluations a run took to come within 1e-3 of `optimum`, or None
fn evaluations_to_target(
    bo: Bo,
    fitness: impl FitnessFunction<Reals, Output = f64>,
    optimum: f64,
    budget: u64,
) -> Option<u64> {
    let outcome = Engine::new(bo, fitness)
        .stop_when(Stop::target(optimum + 1e-3).or(Stop::evaluations(budget)))
        .run()
        .unwrap();
    (outcome.stop_reason() == StopReason::Target).then_some(outcome.evaluations())
}

#[test]
fn branin_in_tens_of_evaluations() {
    // three global minima of 0.397887: a median of 30 evaluations over 20 seeds, all within 50
    for seed in 1..=3 {
        let bo = Bo::builder(Branin.representation())
            .minimize()
            .seed(seed)
            .build()
            .unwrap();
        let evaluations = evaluations_to_target(bo, Branin, 0.397_887_357_729_738, 50);
        assert!(evaluations.is_some(), "seed {seed}");
    }
}

#[test]
fn six_hump_camel_in_tens_of_evaluations() {
    let optimum = SixHumpCamel.optimum().unwrap().value();
    // the usual box for Bayesian optimization, [−3, 3] × [−2, 2]: a median of 50 evaluations
    // over 20 seeds, all within 80
    let narrow = Real::new([-3.0..=3.0, -2.0..=2.0]).unwrap();
    let bo = Bo::builder(narrow).minimize().seed(1).build().unwrap();
    assert!(evaluations_to_target(bo, SixHumpCamel, optimum, 70).is_some());
    // genoxide's box, [−5, 5]², whose corners reach 6,400: the log transform resolves the
    // minima where the standardized values can't (every seed of 20 within 80 evaluations, a
    // median of 31, against 4 of 20)
    let bo = Bo::builder(SixHumpCamel.representation())
        .output(Output::Log)
        .minimize()
        .seed(1)
        .build()
        .unwrap();
    assert!(evaluations_to_target(bo, SixHumpCamel, optimum, 60).is_some());
}

#[test]
fn hartmann_3_in_tens_of_evaluations() {
    // every seed of 20 within 80 evaluations, a median of 26
    let optimum = Hartmann3.optimum().unwrap().value();
    for seed in 1..=3 {
        let bo = Bo::builder(Hartmann3.representation())
            .minimize()
            .seed(seed)
            .build()
            .unwrap();
        let evaluations = evaluations_to_target(bo, Hartmann3, optimum, 60);
        assert!(evaluations.is_some(), "seed {seed}");
    }
}

#[test]
fn goldstein_price_with_the_log_transform() {
    // values from 3 to 10⁶: no seed of 20 reaches the target within 80 evaluations with
    // standardized values, 19 with the log transform, a median of 39
    let bo = Bo::builder(GoldsteinPrice.representation())
        .output(Output::Log)
        .minimize()
        .seed(1)
        .build()
        .unwrap();
    assert!(evaluations_to_target(bo, GoldsteinPrice, 3.0, 70).is_some());
}

#[test]
fn every_acquisition_and_kernel_finds_branin_minima() {
    let settings = [
        (Acquisition::ExpectedImprovement, Kernel::Matern52),
        (
            Acquisition::ProbabilityOfImprovement { xi: 0.01 },
            Kernel::Matern52,
        ),
        (
            Acquisition::UpperConfidenceBound { beta: 4.0 },
            Kernel::Matern52,
        ),
        (
            Acquisition::LogExpectedImprovement,
            Kernel::SquaredExponential,
        ),
    ];
    for (acquisition, kernel) in settings {
        let bo = Bo::builder(Branin.representation())
            .acquisition(acquisition)
            .kernel(kernel)
            .minimize()
            .seed(1)
            .build()
            .unwrap();
        let outcome = Engine::new(bo, Branin)
            .stop_when(Stop::evaluations(40))
            .run()
            .unwrap();
        let best = outcome.best_fitness().score().unwrap();
        assert!(
            best < 0.397_887 + 0.05,
            "{acquisition:?}, {kernel:?}: {best}"
        );
    }
}

// the evaluated genomes of a run, in order, and its best
fn run(bo: Bo, parallel: bool, evaluations: u64) -> (Vec<Vec<u64>>, Individual<Reals>) {
    let engine = Engine::new(bo, Branin).stop_when(Stop::evaluations(evaluations));
    // without the `parallel` feature, both runs are sequential
    #[cfg(feature = "parallel")]
    let engine = engine.parallel(parallel);
    #[cfg(not(feature = "parallel"))]
    let _ = parallel;
    let mut engine = engine;
    let outcome = engine.run().unwrap();
    let genomes = engine
        .algorithm()
        .population()
        .iter()
        .map(|individual| individual.genome().iter().map(|x| x.to_bits()).collect())
        .collect();
    (genomes, outcome.into_best())
}

fn branin_bo(seed: u64) -> Bo {
    Bo::builder(Branin.representation())
        .minimize()
        .seed(seed)
        .build()
        .unwrap()
}

#[test]
fn reproducible_sequentially_and_in_parallel() {
    let (genomes, best) = run(branin_bo(5), false, 20);
    assert_eq!(
        run(branin_bo(5), false, 20),
        (genomes.clone(), best.clone())
    );
    assert_eq!(run(branin_bo(5), true, 20), (genomes.clone(), best));
    assert_ne!(run(branin_bo(6), false, 20).0, genomes);
}

#[test]
fn the_python_package_gives_the_same_run() {
    // python/tests/test_bo.py has the same run, and the bayesian_optimization example prints it
    let outcome = Engine::new(branin_bo(1), Branin)
        .stop_when(Stop::evaluations(30))
        .run()
        .unwrap();
    assert_eq!(outcome.best_fitness().score(), Some(0.39798370755715595));
    assert_eq!(
        outcome.best_genome()[..],
        [9.424508858501198, 2.484571084777484]
    );
}

#[test]
fn no_point_is_asked_twice() {
    let (genomes, _) = run(branin_bo(2), false, 40);
    assert_eq!(genomes.len(), 40);
    for (i, genome) in genomes.iter().enumerate() {
        assert!(!genomes[..i].contains(genome), "point {i} is asked again");
    }
}

#[test]
fn maximizing_mirrors_minimizing() {
    let negated = |x: &Reals| -Branin.evaluate(x);
    let real = Branin.representation();
    let minimized = Engine::new(branin_bo(4), Branin)
        .stop_when(Stop::evaluations(15))
        .run()
        .unwrap();
    let maximized = Bo::builder(real).maximize().seed(4).build().unwrap();
    let maximized = Engine::new(maximized, negated)
        .stop_when(Stop::evaluations(15))
        .run()
        .unwrap();
    assert_eq!(maximized.best_genome(), minimized.best_genome());
    assert_eq!(
        maximized.best_fitness().score().unwrap(),
        -minimized.best_fitness().score().unwrap()
    );
}

#[test]
fn the_design_and_the_protocol() {
    let real = Real::uniform(3, 0.0..=1.0).unwrap();
    let given = vec![
        Reals::from(vec![0.5, 0.5, 0.5]),
        Reals::from(vec![0.1, 0.9, 0.2]),
    ];
    let mut bo = Bo::builder(real)
        .initial_genomes(given.clone())
        .minimize()
        .seed(1)
        .build()
        .unwrap();
    // 2(n + 1) = 8 points, the given ones first
    assert_eq!(bo.initial_points(), 8);
    let design: Vec<Reals> = bo.ask().iter().cloned().collect();
    assert_eq!(design.len(), 8);
    assert_eq!(&design[..2], &given[..]);
    // asking again gives the same; a tell of the wrong count changes nothing
    assert_eq!(bo.ask().iter().cloned().collect::<Vec<_>>(), design);
    assert!(matches!(
        bo.tell(&[Fitness::new(1.0)]),
        Err(Error::FitnessCount {
            expected: 8,
            got: 1
        })
    ));
    let sphere = |x: &Reals| Fitness::new(x.iter().map(|v| (v - 0.3) * (v - 0.3)).sum());
    let fitness: Vec<Fitness> = design.iter().map(sphere).collect();
    bo.tell(&fitness).unwrap();
    assert!(matches!(bo.tell(&fitness), Err(Error::TellWithoutAsk)));
    assert_eq!((bo.generation(), bo.evaluations()), (0, 8));
    assert!(bo.model().is_none());
    // then one point per generation, chosen by a model
    for generation in 1..=3 {
        let asked: Vec<Reals> = bo.ask().iter().cloned().collect();
        assert_eq!(asked.len(), 1);
        assert!(
            bo.model()
                .is_some_and(|model| model.len() == 7 + generation)
        );
        assert!(bo.acquisition_at(&asked[0]).unwrap().is_finite());
        bo.tell(&[sphere(&asked[0])]).unwrap();
        assert_eq!(bo.generation(), generation as u64);
    }
    assert_eq!(bo.population().len(), 11);
}

#[test]
fn invalid_points_enter_the_model_at_the_worst_value() {
    // undefined where x₁ + x₂ > 14: the search goes on around the region
    let partial = |x: &Reals| (x[0] + x[1] <= 14.0).then(|| Branin.evaluate(x));
    let bo = branin_bo(1);
    let mut engine = Engine::new(bo, partial).stop_when(Stop::evaluations(45));
    let outcome = engine.run().unwrap();
    assert!(outcome.best_fitness().score().unwrap() < 0.397_887 + 1e-2);
    let invalid = engine
        .algorithm()
        .population()
        .iter()
        .filter(|individual| individual.fitness().unwrap().score().is_none())
        .count();
    assert!(invalid < 10, "{invalid} invalid points");
    // nothing valid at all: random points, never a panic
    let nothing = |_: &Reals| None::<f64>;
    let outcome = Engine::new(branin_bo(1), nothing)
        .stop_when(Stop::evaluations(10))
        .run()
        .unwrap();
    assert_eq!(outcome.evaluations(), 10);
}

#[test]
fn reevaluation_asks_every_point_again() {
    let mut bo = branin_bo(3);
    for _ in 0..3 {
        let fitness: Vec<Fitness> = bo
            .ask()
            .iter()
            .map(|x| Fitness::new(Branin.evaluate(x)))
            .collect();
        bo.tell(&fitness).unwrap();
    }
    let observed: Vec<Reals> = bo.population().iter().map(|o| o.genome().clone()).collect();
    bo.reevaluate().unwrap();
    let asked: Vec<Reals> = bo.ask().iter().cloned().collect();
    assert_eq!(asked, observed);
    assert!(matches!(bo.reevaluate(), Err(Error::ReevaluationOutOfTurn)));
    // a function shifted by 10: the same order, new values
    let fitness: Vec<Fitness> = asked
        .iter()
        .map(|x| Fitness::new(Branin.evaluate(x) + 10.0))
        .collect();
    let generation = bo.generation();
    bo.tell(&fitness).unwrap();
    assert_eq!(bo.generation(), generation);
    assert_eq!(bo.best_generation(), generation);
    assert!(bo.best().unwrap().fitness().unwrap().score().unwrap() > 10.0);
    assert_eq!(bo.ask().len(), 1);
}

#[test]
fn ucb_beta_changes_during_a_run() {
    let bo = Bo::builder(Branin.representation())
        .acquisition(Acquisition::UpperConfidenceBound { beta: 9.0 })
        .minimize()
        .seed(1)
        .build()
        .unwrap();
    let outcome = Engine::new(bo, Branin)
        .stop_when(Stop::evaluations(30))
        .control(|bo: &mut Bo, progress| {
            let beta = 9.0 / (1.0 + progress.generation() as f64);
            bo.set_acquisition(Acquisition::UpperConfidenceBound { beta })
        })
        .run()
        .unwrap();
    assert!(outcome.best_fitness().score().unwrap() < 0.5);
    let mut bo = branin_bo(1);
    assert!(matches!(
        bo.set_acquisition(Acquisition::UpperConfidenceBound { beta: -1.0 }),
        Err(Error::InvalidSetting { .. })
    ));
    assert_eq!(bo.acquisition(), Acquisition::LogExpectedImprovement);
}

#[test]
fn invalid_settings_are_errors() {
    let real = || Real::uniform(2, 0.0..=1.0).unwrap();
    let setting = |result: Result<Bo>| match result {
        Err(Error::InvalidSetting { setting, .. }) => setting,
        other => panic!("not an invalid setting: {other:?}"),
    };
    let fixed = Real::uniform(2, 1.0..=1.0).unwrap();
    assert_eq!(setting(Bo::builder(fixed).build()), "representation");
    assert_eq!(
        setting(Bo::builder(real()).initial_points(0).build()),
        "initial_points"
    );
    let genomes = vec![Reals::from(vec![0.5, 0.5]); 2];
    assert_eq!(
        setting(Bo::builder(real()).initial_genomes(genomes.clone()).build()),
        "initial_genomes"
    );
    assert_eq!(
        setting(
            Bo::builder(real())
                .initial_points(1)
                .initial_genomes(vec![
                    Reals::from(vec![0.5, 0.5]),
                    Reals::from(vec![0.2, 0.5])
                ])
                .build()
        ),
        "initial_genomes"
    );
    assert!(matches!(
        Bo::builder(real())
            .initial_genomes(vec![Reals::from(vec![2.0, 0.5])])
            .build(),
        Err(Error::InvalidGenome { .. })
    ));
    for acquisition in [
        Acquisition::ProbabilityOfImprovement { xi: -0.1 },
        Acquisition::UpperConfidenceBound { beta: f64::NAN },
    ] {
        let result = Bo::builder(real()).acquisition(acquisition).build();
        assert_eq!(setting(result), "acquisition");
    }
    let noise = genoxide::model::gp::Noise::Learned { min: 0.0 };
    assert_eq!(setting(Bo::builder(real()).noise(noise).build()), "noise");
    assert_eq!(
        setting(Bo::builder(real()).raw_samples(0).build()),
        "raw_samples"
    );
    for starts in [0, 1001] {
        let result = Bo::builder(real()).acquisition_starts(starts).build();
        assert_eq!(setting(result), "acquisition_starts");
    }
    assert_eq!(
        setting(Bo::builder(real()).hyperparameter_starts(0).build()),
        "hyperparameter_starts"
    );
}

#[test]
fn a_fixed_gene_takes_no_part() {
    // Branin with a third gene fixed at 7
    let real = Real::new([-5.0..=10.0, 0.0..=15.0, 7.0..=7.0]).unwrap();
    let fitness = |x: &Reals| Branin.evaluate(&Reals::from(x[..2].to_vec()));
    let bo = Bo::builder(real).minimize().seed(1).build().unwrap();
    assert_eq!(bo.initial_points(), 6);
    let mut engine = Engine::new(bo, fitness).stop_when(Stop::evaluations(40));
    let outcome = engine.run().unwrap();
    assert!(outcome.best_fitness().score().unwrap() < 0.397_887 + 1e-3);
    assert!(
        engine
            .algorithm()
            .population()
            .iter()
            .all(|o| o.genome()[2] == 7.0)
    );
    let length_scales = engine.algorithm().model().unwrap().hyperparameters();
    assert_eq!(length_scales.length_scales()[2], f64::INFINITY);
}

// ---- batches -------------------------------------------------------------------------------------

// a Bo of Branin told its initial design, with a batch of `q` points and the `fantasy`
fn after_the_design(seed: u64, q: usize, fantasy: Fantasy) -> Bo {
    let mut bo = Bo::builder(Branin.representation())
        .batch(q)
        .fantasy(fantasy)
        .minimize()
        .seed(seed)
        .build()
        .unwrap();
    let fitness: Vec<Fitness> = bo
        .ask()
        .iter()
        .map(|x| Fitness::new(Branin.evaluate(x)))
        .collect();
    bo.tell(&fitness).unwrap();
    bo
}

const FANTASIES: [Fantasy; 4] = [
    Fantasy::KrigingBeliever,
    Fantasy::ConstantLiar(Lie::Min),
    Fantasy::ConstantLiar(Lie::Mean),
    Fantasy::ConstantLiar(Lie::Max),
];

// the least distance between two points of a batch, in the unit square
fn least_distance(points: &[Reals]) -> f64 {
    let mut least = f64::INFINITY;
    for (i, a) in points.iter().enumerate() {
        for b in &points[..i] {
            let d = ((a[0] - b[0]) / 15.0).hypot((a[1] - b[1]) / 15.0);
            least = least.min(d);
        }
    }
    least
}

#[test]
fn a_batch_is_new_points_chosen_after_the_fantasies() {
    for seed in 1..=3 {
        let mut single = after_the_design(seed, 1, Fantasy::default());
        let first: Vec<Reals> = single.ask().iter().cloned().collect();
        let mut batches: Vec<Vec<Reals>> = Vec::new();
        for fantasy in FANTASIES {
            let mut bo = after_the_design(seed, 4, fantasy);
            let observed: Vec<Reals> = bo.population().iter().map(|o| o.genome().clone()).collect();
            let batch: Vec<Reals> = bo.ask().iter().cloned().collect();
            assert_eq!(batch.len(), 4);
            // the first point is the one a batch of 1 asks: the same model, the same streams
            assert_eq!(batch[0], first[0], "seed {seed}, {fantasy:?}");
            for (i, point) in batch.iter().enumerate() {
                assert!(!observed.contains(point) && !batch[..i].contains(point));
            }
            // the model is that of the evaluations, without the fantasies
            assert_eq!(bo.model().unwrap().len(), observed.len());
            batches.push(batch);
        }
        // the lies lead the later points elsewhere
        assert_ne!(batches[1][1..], batches[3][1..], "seed {seed}");
    }
}

#[test]
fn a_batch_counts_rounds_and_evaluations() {
    let bo = Bo::builder(Branin.representation())
        .batch(4)
        .minimize()
        .seed(1)
        .build()
        .unwrap();
    assert_eq!(
        (bo.batch(), bo.fantasy()),
        (4, Fantasy::ConstantLiar(Lie::Min))
    );
    let outcome = Engine::new(bo, Branin)
        .stop_when(Stop::generations(10))
        .run()
        .unwrap();
    // the design of 6, then 10 rounds of 4
    assert_eq!(outcome.evaluations(), 46);
    assert!(outcome.best_fitness().score().unwrap() < 0.397_887 + 1e-2);
    // the batch changes between generations
    let outcome = Engine::new(branin_bo(1), Branin)
        .stop_when(Stop::generations(4))
        .control(|bo: &mut Bo, progress| bo.set_batch(2 + progress.generation() as usize))
        .run()
        .unwrap();
    assert_eq!(outcome.evaluations(), 6 + 2 + 3 + 4 + 5);
    let mut bo = branin_bo(1);
    assert!(matches!(bo.set_batch(0), Err(Error::InvalidSetting { .. })));
    assert_eq!(bo.batch(), 1);
}

#[cfg(feature = "parallel")]
#[test]
fn a_batch_gives_the_same_bits_on_any_number_of_threads() {
    let run = |parallel: bool| {
        let bo = Bo::builder(Branin.representation())
            .batch(4)
            .fantasy(Fantasy::KrigingBeliever)
            .minimize()
            .seed(3)
            .build()
            .unwrap();
        let mut engine = Engine::new(bo, Branin)
            .parallel(parallel)
            .stop_when(Stop::evaluations(30));
        engine.run().unwrap();
        let genomes: Vec<Vec<u64>> = engine
            .algorithm()
            .population()
            .iter()
            .map(|o| o.genome().iter().map(|x| x.to_bits()).collect())
            .collect();
        genomes
    };
    let sequential = run(false);
    assert_eq!(sequential.len(), 30);
    for threads in [1, 2, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert_eq!(pool.install(|| run(true)), sequential, "{threads} threads");
    }
}

// ---- asynchronous evaluation -------------------------------------------------------------------

#[test]
fn proposals_fantasize_the_points_being_evaluated() {
    let mut bo = branin_bo(2);
    // the design first, one point at a time
    let design: Vec<Reals> = (0..6).map(|_| bo.propose()).collect();
    assert_eq!(bo.proposed(), &design[..]);
    // results in any order
    for x in design.iter().rev() {
        let fitness = Fitness::new(Branin.evaluate(x));
        assert!(bo.receive(x.clone(), fitness).unwrap().is_none());
    }
    assert!(bo.proposed().is_empty());
    assert_eq!(bo.evaluations(), 6);
    // three points while none of them is evaluated: distinct, and new
    let pending: Vec<Reals> = (0..3).map(|_| bo.propose()).collect();
    for (i, x) in pending.iter().enumerate() {
        assert!(!design.contains(x) && !pending[..i].contains(x));
    }
    assert_eq!(bo.proposed(), &pending[..]);
    // a result that is in already is returned, not added
    let again = bo
        .receive(design[0].clone(), Fitness::new(1.0))
        .unwrap()
        .expect("evaluated already");
    assert_eq!(again.genome(), &design[0]);
    assert_eq!(bo.evaluations(), 6);
    // an invalid genome is an error, and changes nothing
    assert!(matches!(
        bo.receive(Reals::from(vec![20.0, 0.0]), Fitness::new(1.0)),
        Err(Error::InvalidGenome { .. })
    ));
    assert_eq!(bo.proposed().len(), 3);
}

#[test]
fn one_worker_gives_the_same_asynchronous_run_every_time() {
    let run = || {
        let mut engine = AsyncEngine::new(branin_bo(4), Branin)
            .workers(1)
            .stop_when(Stop::evaluations(25));
        let outcome = engine.run().unwrap();
        let population = engine.algorithm().population().clone();
        (outcome.into_best(), population)
    };
    let (best, population) = run();
    assert_eq!(run(), (best.clone(), population.clone()));
    assert_eq!(population.len(), 25);
    assert!(best.fitness().unwrap().score().unwrap() < 0.397_887 + 0.05);
}

#[test]
fn workers_keep_busy_and_reach_the_minimum() {
    // results arrive in the order their evaluations end, so runs differ: each reaches the target
    for workers in [2, 4] {
        let outcome = AsyncEngine::new(branin_bo(1), Branin)
            .workers(workers)
            .stop_when(Stop::target(0.397_887 + 1e-2).or(Stop::evaluations(80)))
            .run()
            .unwrap();
        assert_eq!(
            outcome.stop_reason(),
            StopReason::Target,
            "{workers} workers"
        );
    }
}

// ---- constraints ---------------------------------------------------------------------------------

// Gramacy et al.'s (2016) toy problem: x₁ + x₂ on [0, 1]² with two constraints, the first active
// at the minimum (tests/reference/gramacy_toy.py)
const TOY_MINIMUM: f64 = 0.599_788_052_010_067_6;

fn toy(x: &Reals, g: &mut [f64]) -> f64 {
    let wave = genoxide::math::sin(2.0 * std::f64::consts::PI * (x[0] * x[0] - 2.0 * x[1]));
    g[0] = 1.5 - x[0] - 2.0 * x[1] - 0.5 * wave;
    g[1] = x[0] * x[0] + x[1] * x[1] - 1.5;
    x[0] + x[1]
}

fn toy_bo(seed: u64, batch: usize) -> Bo {
    Bo::builder(Real::uniform(2, 0.0..=1.0).unwrap())
        .batch(batch)
        .minimize()
        .seed(seed)
        .build()
        .unwrap()
}

#[test]
fn constrained_bayesian_optimization_reaches_the_feasible_minimum() {
    // every seed of 20 within 1e-5 in at most 39 evaluations, a median of 27
    for seed in 1..=3 {
        let mut engine = Engine::new(toy_bo(seed, 1), Constrained::new(2, toy))
            .stop_when(Stop::target(TOY_MINIMUM + 1e-5).or(Stop::evaluations(50)));
        let outcome = engine.run().unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Target, "seed {seed}");
        assert!(outcome.best_fitness().is_feasible());
        let bo = engine.algorithm();
        assert_eq!(bo.constraints(), Some(2));
        assert_eq!(bo.constraint_models().len(), 2);
        // the values are kept with each point
        let mut g = [0.0; 2];
        toy(bo.population().as_slice()[3].genome(), &mut g);
        assert_eq!(bo.constraint_values(3), &g[..]);
        // the models interpolate the values: at a point evaluated, clearly feasible or not, the
        // probability of feasibility is about 1 or 0
        for (index, individual) in bo.population().iter().enumerate() {
            let g = bo.constraint_values(index);
            let p = bo
                .probability_of_feasibility_at(individual.genome())
                .unwrap();
            if g.iter().all(|&g| g < -0.05) {
                assert!(p > 0.99, "{g:?}: {p}");
            } else if g.iter().any(|&g| g > 0.05) {
                assert!(p < 0.01, "{g:?}: {p}");
            }
        }
        assert!(
            bo.acquisition_at(outcome.best_genome())
                .unwrap()
                .is_finite()
        );
    }
    // a batch of 3
    let outcome = Engine::new(toy_bo(1, 3), Constrained::new(2, toy))
        .stop_when(Stop::target(TOY_MINIMUM + 1e-4).or(Stop::evaluations(60)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    // and asynchronously, one worker: the values come with each result
    let outcome = AsyncEngine::new(toy_bo(1, 1), Constrained::new(2, toy))
        .workers(1)
        .stop_when(Stop::target(TOY_MINIMUM + 1e-4).or(Stop::evaluations(60)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
}

#[test]
fn without_the_constraints_values_the_search_ignores_them() {
    // the same problem as a (score, violation) closure: the search goes for the infeasible
    // corner, and the best by Deb's rules stays far from the minimum
    let tuple = |x: &Reals| {
        let mut g = [0.0; 2];
        let score = toy(x, &mut g);
        (score, g.iter().map(|&g| g.max(0.0)).sum::<f64>())
    };
    let mut engine = Engine::new(toy_bo(1, 1), tuple).stop_when(Stop::evaluations(40));
    let outcome = engine.run().unwrap();
    assert_eq!(engine.algorithm().constraints(), Some(0));
    assert!(outcome.best_fitness().score().unwrap() > TOY_MINIMUM + 1e-3);
}

#[test]
fn constrained_settings_and_tells() {
    // the upper confidence bound can't be weighed by a probability
    let ucb = Bo::builder(Real::uniform(2, 0.0..=1.0).unwrap())
        .acquisition(Acquisition::UpperConfidenceBound { beta: 2.0 })
        .seed(1)
        .build()
        .unwrap();
    let result = Engine::new(ucb, Constrained::new(2, toy))
        .stop_when(Stop::evaluations(10))
        .run();
    assert!(matches!(
        result,
        Err(Error::InvalidSetting {
            setting: "acquisition",
            ..
        })
    ));
    // after a run with constraints, a tell without their values is an error that changes nothing
    let mut engine =
        Engine::new(toy_bo(1, 1), Constrained::new(2, toy)).stop_when(Stop::evaluations(8));
    engine.run().unwrap();
    let mut bo = engine.into_algorithm();
    assert!(matches!(
        bo.set_acquisition(Acquisition::UpperConfidenceBound { beta: 1.0 }),
        Err(Error::InvalidSetting { .. })
    ));
    let asked = bo.ask().len();
    let fitness = vec![Fitness::new(1.0); asked];
    assert!(matches!(
        bo.tell(&fitness),
        Err(Error::InvalidSetting { .. })
    ));
    assert_eq!(bo.evaluations(), 8);
    let values = vec![-1.0; 2 * asked];
    let evaluations =
        genoxide::engine::Evaluations::with_extras(&fitness, None, Some(&values), None, 2, 2)
            .unwrap();
    bo.tell_evaluations(&evaluations).unwrap();
    assert_eq!(bo.evaluations(), 8 + asked as u64);
}

#[test]
fn a_cec_2006_problem_by_its_constraint_values() {
    use genoxide::problems::cec2006::G24;
    // G24: two genes, two constraints, its minimum on the boundary of the feasible region
    let minimum = G24.optimum().unwrap().value();
    let bo = Bo::builder(G24.representation())
        .minimize()
        .seed(1)
        .build()
        .unwrap();
    let outcome = Engine::new(bo, G24)
        .stop_when(Stop::target(minimum + 1e-4).or(Stop::evaluations(60)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
}

// ---- integer genes -------------------------------------------------------------------------------

#[test]
fn integer_genes_are_searched_on_their_lattice() {
    // a quadratic of 4 integer genes in [−10, 10]⁴, 194,481 points, whose continuous minimum
    // (2.6, −1.3, 0.4, 3.5) isn't on the lattice: its integer minimum, found by enumeration
    let f = |x: &Integers| {
        let x: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        (x[0] - 2.6) * (x[0] - 2.6)
            + 2.0 * (x[1] + 1.3) * (x[1] + 1.3)
            + (x[2] - 0.4) * (x[2] - 0.4)
            + 0.5 * (x[3] - 3.5) * (x[3] - 3.5)
            + 0.3 * (x[0] - 2.6) * (x[1] + 1.3)
    };
    let integer = Integer::uniform(4, -10..=10).unwrap();
    let mut minimum = f64::INFINITY;
    for a in -10..=10 {
        for b in -10..=10 {
            for c in -10..=10 {
                for d in -10..=10 {
                    minimum = minimum.min(f(&Integers::from(vec![a, b, c, d])));
                }
            }
        }
    }
    for seed in 1..=3 {
        let bo = Bo::builder(integer.clone())
            .minimize()
            .seed(seed)
            .build()
            .unwrap();
        let mut engine =
            Engine::new(bo, f).stop_when(Stop::target(minimum).or(Stop::evaluations(50)));
        let outcome = engine.run().unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Target, "seed {seed}");
        let population = engine.algorithm().population().as_slice();
        for (i, individual) in population.iter().enumerate() {
            assert!(integer.validate(individual.genome()).is_ok());
            let earlier = &population[..i];
            assert!(!earlier.iter().any(|o| o.genome() == individual.genome()));
        }
    }
}

#[test]
fn integer_settings() {
    let small = Integer::new([0..=1, 0..=2]).unwrap();
    // the default design, 6 points, fits the 6 points of the lattice; 7 don't
    assert_eq!(
        Bo::builder(small.clone()).build().unwrap().initial_points(),
        6
    );
    let result = Bo::builder(small).initial_points(7).build();
    assert!(matches!(
        result,
        Err(Error::InvalidSetting {
            setting: "initial_points",
            ..
        })
    ));
    // no gene with more than one value is an error
    let fixed = Integer::new([3..=3, 3..=3]).unwrap();
    assert!(Bo::builder(fixed).build().is_err());
}
