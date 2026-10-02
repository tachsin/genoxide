//! Bayesian optimization: the classic problems of Jones, Schonlau and Welch (1998) reached in tens
//! of evaluations, the guarantees of every algorithm, and how invalid points are handled.

use genoxide::algorithm::bo::{Acquisition, Output};
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
