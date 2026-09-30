//! OpenAI's evolution strategy: settings, the ask / tell protocol, mirrored samples, re-evaluation
//! and parallel breeding.

use genoxide::algorithm::open_es::Optimizer;
use genoxide::prelude::*;
use proptest::prelude::*;

fn sphere(x: &Reals) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

fn builder(seed: u64) -> genoxide::algorithm::OpenEsBuilder {
    OpenEs::builder(Real::uniform(20, -5.0..=5.0).unwrap())
        .population_size(10)
        .sigma(0.01)
        .optimizer(Optimizer::adam(0.003))
        .minimize()
        .seed(seed)
}

fn setting(result: Result<OpenEs>) -> &'static str {
    match result {
        Err(Error::InvalidSetting { setting, .. } | Error::MissingSetting { setting }) => setting,
        other => panic!("expected a setting error, got {other:?}"),
    }
}

#[test]
fn validation() {
    let real = || Real::uniform(3, 0.0..=1.0).unwrap();
    let sized = || OpenEs::builder(real()).population_size(10);
    assert_eq!(setting(OpenEs::builder(real()).build()), "population_size");
    for size in [0, 1, 3, 11] {
        assert_eq!(
            setting(OpenEs::builder(real()).population_size(size).build()),
            "population_size"
        );
    }
    for sigma in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert_eq!(setting(sized().sigma(sigma).build()), "sigma");
    }
    for optimizer in [
        Optimizer::adam(0.0),
        Optimizer::adam(f64::NAN),
        Optimizer::sgd(0.1, 1.0),
        Optimizer::sgd(0.1, -0.1),
        Optimizer::Adam {
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 1.0,
        },
    ] {
        assert_eq!(setting(sized().optimizer(optimizer).build()), "optimizer");
    }
    assert_eq!(setting(sized().weight_decay(-0.1).build()), "weight_decay");
    assert!(matches!(
        sized().initial_mean(Reals::from(vec![0.5])).build(),
        Err(Error::InvalidGenome { .. })
    ));
    assert!(matches!(
        sized()
            .initial_mean(Reals::from(vec![0.5, 2.0, 0.5]))
            .build(),
        Err(Error::InvalidGenome { .. })
    ));
    assert!(sized().optimizer(Optimizer::sgd(0.1, 0.0)).build().is_ok());
    let mut es = sized().build().unwrap();
    assert!(es.set_sigma(0.0).is_err());
    assert!(es.set_learning_rate(-1.0).is_err());
    assert_eq!(es.sigma(), 0.02);
    assert_eq!(es.optimizer(), Optimizer::adam(0.01));
    es.set_learning_rate(0.5).unwrap();
    assert_eq!(es.optimizer(), Optimizer::adam(0.5));
}

#[test]
fn ask_tell_protocol() {
    let mut es = builder(1).build().unwrap();
    assert!(matches!(es.tell(&[]), Err(Error::TellWithoutAsk)));
    let first: Vec<Reals> = es.ask().iter().cloned().collect();
    // asking again gives the same genomes
    assert_eq!(es.ask().iter().cloned().collect::<Vec<_>>(), first);
    assert_eq!(first.len(), 10);
    // a wrong count changes nothing
    assert!(matches!(
        es.tell(&[Fitness::new(1.0)]),
        Err(Error::FitnessCount {
            expected: 10,
            got: 1
        })
    ));
    let mean = es.mean();
    let fitness: Vec<Fitness> = first.iter().map(|x| Fitness::new(sphere(x))).collect();
    es.tell(&fitness).unwrap();
    assert_eq!(es.generation(), 0);
    assert_eq!(es.evaluations(), 10);
    assert_ne!(es.mean(), mean);
    let best = first
        .iter()
        .map(sphere)
        .fold(f64::INFINITY, |best, value| best.min(value));
    assert_eq!(es.best().unwrap().fitness(), Some(Fitness::new(best)));
    let second: Vec<Reals> = es.ask().iter().cloned().collect();
    assert_ne!(second, first);
    let fitness: Vec<Fitness> = second.iter().map(|x| Fitness::new(sphere(x))).collect();
    es.tell(&fitness).unwrap();
    assert_eq!(es.generation(), 1);
}

#[test]
fn samples_are_mirrored_around_the_mean() {
    // a mean away from the bounds, where no sample is clamped
    let mut es = builder(2)
        .initial_mean(Reals::from(vec![0.5; 20]))
        .evaluate_mean(true)
        .build()
        .unwrap();
    let mean = es.mean();
    let samples: Vec<Reals> = es.ask().iter().cloned().collect();
    // 5 pairs and the mean, last
    assert_eq!(samples.len(), 11);
    assert_eq!(samples[10], mean);
    for pair in samples[..10].chunks(2) {
        for ((a, b), m) in pair[0].iter().zip(pair[1].iter()).zip(mean.iter()) {
            assert!((a + b - 2.0 * m).abs() < 1e-12);
            // σ is 0.01 of the range of 10
            assert!((a - m).abs() < 0.1 * 6.0);
        }
    }
    let fitness: Vec<Fitness> = samples.iter().map(|x| Fitness::new(sphere(x))).collect();
    es.tell(&fitness).unwrap();
    assert_eq!(es.evaluations(), 11);
}

#[test]
fn samples_stay_within_the_bounds() {
    // a σ of the whole range: most samples would leave [0, 1]
    let mut es = OpenEs::builder(Real::uniform(50, 0.0..=1.0).unwrap())
        .population_size(20)
        .sigma(1.0)
        .optimizer(Optimizer::sgd(10.0, 0.0))
        .seed(3)
        .build()
        .unwrap();
    for _ in 0..5 {
        let samples: Vec<Reals> = es.ask().iter().cloned().collect();
        for sample in &samples {
            assert!(sample.iter().all(|x| (0.0..=1.0).contains(x)));
        }
        let fitness: Vec<Fitness> = samples.iter().map(|x| Fitness::new(x[0])).collect();
        es.tell(&fitness).unwrap();
        assert!(es.mean().iter().all(|x| (0.0..=1.0).contains(x)));
    }
}

#[test]
fn reevaluation_draws_no_random_number() {
    let run = |reevaluate: bool| {
        let mut done = false;
        let mut engine = Engine::new(builder(4).build().unwrap(), sphere)
            .stop_when(Stop::generations(20))
            .control(|es, progress| {
                if reevaluate && !done && progress.generation() == 5 {
                    done = true;
                    es.reevaluate()?;
                }
                Ok(())
            });
        let outcome = engine.run().unwrap();
        (engine.algorithm().mean(), outcome.evaluations())
    };
    let (mean, evaluations) = run(false);
    let (reevaluated, more) = run(true);
    // the same distribution, and one population more evaluated
    assert_eq!(reevaluated, mean);
    assert_eq!(more, evaluations + 10);
}

#[test]
fn reevaluation_out_of_turn() {
    let mut es = builder(5).build().unwrap();
    // before the first tell, nothing to re-evaluate
    es.reevaluate().unwrap();
    es.ask();
    assert!(matches!(es.reevaluate(), Err(Error::ReevaluationOutOfTurn)));
}

#[test]
fn solves_the_sphere() {
    let es = OpenEs::builder(Real::uniform(100, -5.0..=5.0).unwrap())
        .population_size(50)
        .sigma(0.01)
        .optimizer(Optimizer::adam(0.003))
        .evaluate_mean(true)
        .minimize()
        .seed(1)
        .build()
        .unwrap();
    let outcome = Engine::new(es, sphere)
        .stop_when(Stop::target(0.1).or(Stop::generations(2_000)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    // with SGD and momentum too
    let es = OpenEs::builder(Real::uniform(20, -5.0..=5.0).unwrap())
        .population_size(40)
        .sigma(0.01)
        .optimizer(Optimizer::sgd(0.002, 0.9))
        .evaluate_mean(true)
        .minimize()
        .seed(1)
        .build()
        .unwrap();
    let outcome = Engine::new(es, sphere)
        .stop_when(Stop::target(0.01).or(Stop::generations(5_000)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
}

// the mean after 10 generations
fn run(es: OpenEs) -> (Reals, Population<Reals>) {
    let mut engine = Engine::new(es, sphere).stop_when(Stop::generations(10));
    engine.run().unwrap();
    (
        engine.algorithm().mean(),
        engine.algorithm().population().clone(),
    )
}

#[test]
fn parallel_breeding_is_the_same_on_any_thread_count() {
    let make = |parallel| builder(6).parallel_breeding(parallel).build().unwrap();
    let parallel = run(make(true));
    // other samples than without parallel breeding
    assert_ne!(parallel.0, run(make(false)).0);
    #[cfg(feature = "parallel")]
    for threads in [1, 2, 8] {
        let on_threads = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| run(make(true)));
        assert_eq!(on_threads, parallel);
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    // a seeded run repeats itself, and every sample and mean stays within the bounds
    #[test]
    fn seeded_runs_repeat(seed: u64, pairs in 1usize..8, sigma in 0.001..1.0f64) {
        let make = || {
            OpenEs::builder(Real::new([-1.0..=1.0, 0.0..=0.0, 2.0..=10.0]).unwrap())
                .population_size(2 * pairs)
                .sigma(sigma)
                .evaluate_mean(true)
                .minimize()
                .seed(seed)
                .build()
                .unwrap()
        };
        let (mean, population) = run(make());
        prop_assert_eq!(&run(make()), &(mean.clone(), population.clone()));
        prop_assert_eq!(population.len(), 2 * pairs + 1);
        let real = Real::new([-1.0..=1.0, 0.0..=0.0, 2.0..=10.0]).unwrap();
        prop_assert!(real.validate(&mean).is_ok());
        for individual in population.iter() {
            prop_assert!(real.validate(individual.genome()).is_ok());
        }
    }
}
