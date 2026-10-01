//! The Nelder-Mead simplex method: the steps of Lagarias et al. (1998), convergence, bounds,
//! restarts, speculative asks and the ask / tell protocol.

use genoxide::algorithm::nelder_mead::Coefficients;
use genoxide::prelude::*;
use genoxide::problems::{Himmelblau, Problem};

fn rosenbrock(x: &Reals) -> f64 {
    x.windows(2)
        .map(|w| {
            let (a, b) = (w[1] - w[0] * w[0], 1.0 - w[0]);
            100.0 * a * a + b * b
        })
        .sum()
}

fn sphere(x: &Reals) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

// a simplex of (4, 4), (5, 4) and (4, 5) in [0, 8]²: every point of its steps is exact
fn square() -> NelderMead {
    NelderMead::builder(Real::uniform(2, 0.0..=8.0).unwrap())
        .initial_genome(Reals::from(vec![4.0, 4.0]))
        .initial_step(0.125)
        .coefficients(Coefficients::Standard)
        .minimize()
        .seed(1)
        .build()
        .unwrap()
}

fn asked(nelder_mead: &mut NelderMead) -> Vec<Vec<f64>> {
    nelder_mead
        .ask()
        .iter()
        .map(|genome| genome.to_vec())
        .collect()
}

fn tell(nelder_mead: &mut NelderMead, values: &[f64]) {
    let fitness: Vec<Fitness> = values.iter().map(|&value| Fitness::new(value)).collect();
    nelder_mead.tell(&fitness).unwrap();
}

fn simplex(nelder_mead: &NelderMead) -> Vec<(Vec<f64>, f64)> {
    nelder_mead
        .population()
        .iter()
        .map(|vertex| {
            let value = vertex.fitness().unwrap().score().unwrap();
            (vertex.genome().to_vec(), value)
        })
        .collect()
}

// the square's simplex evaluated as 1, 2 and 3: (4, 4) best, (4, 5) worst, so the centroid is
// (4.5, 4), the reflection (5, 3), the expansion (5.5, 2), the outside contraction (4.75, 3.5)
// and the inside one (4.25, 4.5)
fn started() -> NelderMead {
    let mut nelder_mead = square();
    assert_eq!(
        asked(&mut nelder_mead),
        [[4.0, 4.0], [5.0, 4.0], [4.0, 5.0]]
    );
    tell(&mut nelder_mead, &[1.0, 2.0, 3.0]);
    assert_eq!(asked(&mut nelder_mead), [[5.0, 3.0]]);
    nelder_mead
}

#[test]
fn a_reflection_between_the_best_and_the_second_worst_is_accepted() {
    // f1 <= fr < fn: a tie with the best goes after it
    for reflected in [1.5, 1.0] {
        let mut nelder_mead = started();
        tell(&mut nelder_mead, &[reflected]);
        assert_eq!(
            simplex(&nelder_mead),
            [
                (vec![4.0, 4.0], 1.0),
                (vec![5.0, 3.0], reflected),
                (vec![5.0, 4.0], 2.0)
            ]
        );
        assert_eq!(nelder_mead.iterations(), 1);
        assert_eq!(nelder_mead.generation(), 1);
    }
}

#[test]
fn a_reflection_better_than_the_best_expands() {
    // fe < fr accepts the expansion; fe = fr the reflection
    for (expanded, accepted) in [(0.25, vec![5.5, 2.0]), (0.5, vec![5.0, 3.0])] {
        let mut nelder_mead = started();
        tell(&mut nelder_mead, &[0.5]);
        assert_eq!(asked(&mut nelder_mead), [[5.5, 2.0]]);
        tell(&mut nelder_mead, &[expanded]);
        let vertices = simplex(&nelder_mead);
        assert_eq!(vertices[0].0, accepted);
        assert_eq!(vertices[1], (vec![4.0, 4.0], 1.0));
        assert_eq!(vertices[2], (vec![5.0, 4.0], 2.0));
        // the rejected point is discarded, and the best evaluated is the best
        assert_eq!(nelder_mead.discarded().len(), 1);
        assert_eq!(
            nelder_mead.best().unwrap().fitness(),
            Some(Fitness::new(expanded.min(0.5)))
        );
        assert_eq!(nelder_mead.generation(), 2);
        assert_eq!(nelder_mead.iterations(), 1);
    }
}

#[test]
fn a_reflection_between_the_second_worst_and_the_worst_contracts_outside() {
    // fn <= fr < fn+1; fc <= fr accepts the contraction, ties included
    let mut nelder_mead = started();
    tell(&mut nelder_mead, &[2.0]);
    assert_eq!(asked(&mut nelder_mead), [[4.75, 3.5]]);
    tell(&mut nelder_mead, &[2.0]);
    assert_eq!(
        simplex(&nelder_mead),
        [
            (vec![4.0, 4.0], 1.0),
            (vec![5.0, 4.0], 2.0),
            (vec![4.75, 3.5], 2.0)
        ]
    );

    // fc > fr shrinks towards the best: the n other vertices halfway to it
    let mut nelder_mead = started();
    tell(&mut nelder_mead, &[2.5]);
    asked(&mut nelder_mead);
    tell(&mut nelder_mead, &[2.75]);
    assert_eq!(nelder_mead.discarded().len(), 2);
    assert_eq!(asked(&mut nelder_mead), [[4.5, 4.0], [4.0, 4.5]]);
    // a tie with the best keeps the best first
    tell(&mut nelder_mead, &[1.0, 0.5]);
    assert_eq!(
        simplex(&nelder_mead),
        [
            (vec![4.0, 4.5], 0.5),
            (vec![4.0, 4.0], 1.0),
            (vec![4.5, 4.0], 1.0)
        ]
    );
    assert_eq!(nelder_mead.iterations(), 1);
    assert_eq!(nelder_mead.generation(), 3);
}

#[test]
fn a_reflection_no_better_than_the_worst_contracts_inside() {
    // fr >= fn+1; fcc < fn+1 accepts the contraction
    let mut nelder_mead = started();
    tell(&mut nelder_mead, &[3.0]);
    assert_eq!(asked(&mut nelder_mead), [[4.25, 4.5]]);
    tell(&mut nelder_mead, &[2.5]);
    assert_eq!(simplex(&nelder_mead)[2], (vec![4.25, 4.5], 2.5));

    // fcc = fn+1 shrinks
    let mut nelder_mead = started();
    tell(&mut nelder_mead, &[4.0]);
    asked(&mut nelder_mead);
    tell(&mut nelder_mead, &[3.0]);
    assert_eq!(asked(&mut nelder_mead), [[4.5, 4.0], [4.0, 4.5]]);
}

#[test]
fn a_speculative_ask_evaluates_the_four_points_at_once() {
    let mut nelder_mead = NelderMead::builder(Real::uniform(2, 0.0..=8.0).unwrap())
        .initial_genome(Reals::from(vec![4.0, 4.0]))
        .initial_step(0.125)
        .speculative(true)
        .minimize()
        .build()
        .unwrap();
    asked(&mut nelder_mead);
    tell(&mut nelder_mead, &[1.0, 2.0, 3.0]);
    assert_eq!(
        asked(&mut nelder_mead),
        [[5.0, 3.0], [5.5, 2.0], [4.75, 3.5], [4.25, 4.5]]
    );
    // an outside contraction worse than the reflection: a shrink, the four points discarded
    tell(&mut nelder_mead, &[2.5, 0.0, 2.75, 0.0]);
    assert_eq!(nelder_mead.discarded().len(), 4);
    // the expansion wasn't taken, but it's the best evaluated
    assert_eq!(nelder_mead.best().unwrap().genome().to_vec(), [5.5, 2.0]);
    assert_eq!(asked(&mut nelder_mead), [[4.5, 4.0], [4.0, 4.5]]);
}

// the simplexes after every iteration, with the evaluations and generations of the run
fn path(speculative: bool, seed: u64) -> (Vec<Vec<Reals>>, u64, u64) {
    let mut nelder_mead = NelderMead::builder(Real::uniform(3, -5.0..=5.0).unwrap())
        .speculative(speculative)
        .minimize()
        .seed(seed)
        .build()
        .unwrap();
    let mut simplexes = Vec::new();
    let mut iterations = 0;
    while !nelder_mead.is_finished() {
        let fitness: Vec<Fitness> = nelder_mead
            .ask()
            .iter()
            .map(|genome| Fitness::new(rosenbrock(genome)))
            .collect();
        nelder_mead.tell(&fitness).unwrap();
        if nelder_mead.iterations() > iterations {
            iterations = nelder_mead.iterations();
            let vertices = nelder_mead.population().iter();
            simplexes.push(vertices.map(|vertex| vertex.genome().clone()).collect());
        }
    }
    (
        simplexes,
        nelder_mead.evaluations(),
        nelder_mead.generation(),
    )
}

#[test]
fn speculative_asks_take_the_same_path_in_fewer_rounds() {
    for seed in 1..=5 {
        let (plain, plain_evaluations, plain_generations) = path(false, seed);
        let (speculative, speculative_evaluations, speculative_generations) = path(true, seed);
        assert_eq!(plain, speculative);
        assert!(speculative_generations < plain_generations);
        assert!(speculative_evaluations > plain_evaluations);
    }
}

#[test]
fn it_converges_on_rosenbrock_from_the_classic_start() {
    for coefficients in [Coefficients::Adaptive, Coefficients::Standard] {
        let nelder_mead = NelderMead::builder(Real::uniform(2, -5.0..=5.0).unwrap())
            .initial_genome(Reals::from(vec![-1.2, 1.0]))
            .coefficients(coefficients)
            .minimize()
            .build()
            .unwrap();
        let mut engine = Engine::new(nelder_mead, rosenbrock).stop_when(Stop::evaluations(10_000));
        let outcome = engine.run().unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Converged);
        assert!(outcome.best_fitness().score().unwrap() < 1e-15);
        for (x, optimum) in outcome.best_genome().iter().zip([1.0, 1.0]) {
            assert!((x - optimum).abs() < 1e-7);
        }
        assert!(engine.algorithm().size() <= 1e-9);
        // run again, it stops at once
        let again = engine.run().unwrap();
        assert_eq!(again.evaluations(), outcome.evaluations());
        assert_eq!(again.stop_reason(), StopReason::Converged);
    }
}

#[test]
fn the_adaptive_coefficients_converge_in_more_dimensions() {
    // an ellipsoid of 10 genes, from a random start
    let ellipsoid = |x: &Reals| {
        x.iter()
            .enumerate()
            .map(|(i, xi)| (i + 1) as f64 * xi * xi)
            .sum::<f64>()
    };
    let run = |coefficients| {
        let nelder_mead = NelderMead::builder(Real::uniform(10, -5.0..=5.0).unwrap())
            .coefficients(coefficients)
            .minimize()
            .seed(3)
            .build()
            .unwrap();
        Engine::new(nelder_mead, ellipsoid)
            .stop_when(Stop::evaluations(100_000))
            .run()
            .unwrap()
    };
    let adaptive = run(Coefficients::Adaptive);
    assert_eq!(adaptive.stop_reason(), StopReason::Converged);
    assert!(adaptive.best_fitness().score().unwrap() < 1e-12);
    let standard = run(Coefficients::Standard);
    assert!(adaptive.evaluations() < standard.evaluations());
}

#[test]
fn maximizing_takes_the_same_path_as_minimizing_the_negation() {
    let run = |objective, sign: f64| {
        let nelder_mead = NelderMead::builder(Real::uniform(3, -5.0..=5.0).unwrap())
            .objective(objective)
            .seed(4)
            .build()
            .unwrap();
        Engine::new(nelder_mead, move |x: &Reals| sign * rosenbrock(x))
            .stop_when(Stop::evaluations(5_000))
            .run()
            .unwrap()
    };
    let minimized = run(Objective::Minimize, 1.0);
    let maximized = run(Objective::Maximize, -1.0);
    assert_eq!(minimized.best_genome(), maximized.best_genome());
    assert_eq!(minimized.evaluations(), maximized.evaluations());
}

#[test]
fn points_stay_in_the_bounds_and_converge_on_them() {
    // the sphere around (2, -3), outside the box [-1, 1]²: the minimum is the corner (1, -1)
    let shifted = |x: &Reals| (x[0] - 2.0).powi(2) + (x[1] + 3.0).powi(2);
    let mut nelder_mead = NelderMead::builder(Real::uniform(2, -1.0..=1.0).unwrap())
        .minimize()
        .seed(5)
        .build()
        .unwrap();
    while !nelder_mead.is_finished() {
        let candidates = nelder_mead.ask();
        assert!(
            candidates
                .iter()
                .flat_map(|x| x.iter())
                .all(|x| (-1.0..=1.0).contains(x))
        );
        let fitness: Vec<Fitness> = candidates
            .iter()
            .map(|x| Fitness::new(shifted(x)))
            .collect();
        nelder_mead.tell(&fitness).unwrap();
    }
    // mirrored at the bounds, the simplex approaches the corner rather than landing on it
    let best = nelder_mead.best().unwrap();
    for (x, corner) in best.genome().iter().zip([1.0, -1.0]) {
        assert!((x - corner).abs() < 1e-8, "{x}");
    }
    assert!(best.fitness().unwrap().score().unwrap() - 5.0 < 1e-7);
}

#[test]
fn a_simplex_that_meets_a_bound_doesnt_flatten_on_it() {
    // Branin's minima are inside the box; clamped to the bounds, the simplex of seed 2 stopped on
    // the edge x₁ = 10 at f = 1.94, away from any minimum, as did 138 of 1000 seeds. Mirrored,
    // none stops on a bound, and a few stall inside, as Nelder-Mead can (8 of 1000, e.g. seed 20)
    use genoxide::problems::{Branin, Problem};
    let bounds = Branin.representation();
    let mut minima = 0;
    for seed in 1..=50 {
        let nelder_mead = NelderMead::builder(Branin.representation())
            .minimize()
            .seed(seed)
            .build()
            .unwrap();
        let outcome = Engine::new(nelder_mead, Branin)
            .stop_when(Stop::evaluations(10_000))
            .run()
            .unwrap();
        let mut genes = outcome.best_genome().iter().zip(bounds.bounds());
        let on_bound = genes.any(|(x, range)| x == range.start() || x == range.end());
        assert!(!on_bound, "seed {seed}");
        let value = outcome.best_fitness().score().unwrap();
        minima += usize::from(value - 0.397_887_357_729_738 < 1e-9);
    }
    assert_eq!(minima, 49);
}

#[test]
fn an_absolute_step_and_a_relative_tolerance_work_in_any_box() {
    // the same run, bit for bit, in boxes from ±10 to ±1e10: with a fraction of the range, the
    // widest box would start with a step of 2e9 and stop 2 from the minimum
    let run = |width: f64| {
        let nelder_mead = NelderMead::builder(Real::uniform(2, -width..=width).unwrap())
            .initial_genome(Reals::from(vec![-1.2, 1.0]))
            .initial_step_absolute(0.5)
            .minimize()
            .build()
            .unwrap();
        Engine::new(nelder_mead, rosenbrock)
            .stop_when(Stop::evaluations(10_000))
            .run()
            .unwrap()
    };
    let narrow = run(10.0);
    assert_eq!(narrow.stop_reason(), StopReason::Converged);
    assert!(narrow.best_fitness().score().unwrap() < 1e-15);
    for width in [1e3, 1e6, 1e10] {
        let wide = run(width);
        assert_eq!(wide.best(), narrow.best(), "±{width}");
        assert_eq!(wide.evaluations(), narrow.evaluations(), "±{width}");
    }
}

#[test]
fn fixed_genes_are_not_searched() {
    let real = Real::new([-5.0..=5.0, 2.0..=2.0, -5.0..=5.0]).unwrap();
    let mut nelder_mead = NelderMead::builder(real)
        .minimize()
        .seed(6)
        .build()
        .unwrap();
    assert_eq!(nelder_mead.ask().len(), 3);
    let outcome = Engine::new(nelder_mead, sphere)
        .stop_when(Stop::evaluations(10_000))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert_eq!(outcome.best_genome()[1], 2.0);
    assert!(outcome.best_fitness().score().unwrap() < 4.0 + 1e-15);
}

#[test]
fn one_gene() {
    let nelder_mead = NelderMead::builder(Real::uniform(1, -10.0..=10.0).unwrap())
        .minimize()
        .seed(7)
        .build()
        .unwrap();
    let outcome = Engine::new(nelder_mead, |x: &Reals| (x[0] - 3.0).abs())
        .stop_when(Stop::evaluations(10_000))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert!((outcome.best_genome()[0] - 3.0).abs() < 1e-8);
}

#[test]
fn invalid_points_are_worse_than_any_other() {
    // undefined outside the unit disk around (0.5, 0.5); the minimum is inside it
    let defined = |x: &Reals| {
        let (a, b) = (x[0] - 0.5, x[1] - 0.5);
        (a * a + b * b < 1.0).then(|| (x[0] - 0.2).powi(2) + (x[1] - 0.9).powi(2))
    };
    let nelder_mead = NelderMead::builder(Real::uniform(2, -5.0..=5.0).unwrap())
        .initial_genome(Reals::from(vec![0.5, 0.5]))
        .minimize()
        .build()
        .unwrap();
    let outcome = Engine::new(nelder_mead, defined)
        .stop_when(Stop::evaluations(10_000))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert!(outcome.best_fitness().score().unwrap() < 1e-15);
}

#[test]
fn restarts_find_the_minima_of_every_basin() {
    let problem = Himmelblau;
    let nelder_mead = NelderMead::builder(problem.representation())
        .restarts(local::Restarts::Random { times: 19 })
        .minimize()
        .seed(8)
        .build()
        .unwrap();
    let mut ends = Vec::new();
    let mut engine = Engine::new(nelder_mead, problem)
        .stop_when(Stop::evaluations(100_000))
        .control(|nelder_mead: &mut NelderMead, _| {
            if nelder_mead.converged() {
                ends.push(nelder_mead.population()[0].clone());
            }
            Ok(())
        });
    let outcome = engine.run().unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert_eq!(engine.algorithm().restart_count(), 19);
    drop(engine);
    assert_eq!(ends.len(), 20);
    let optimum = problem.optimum().unwrap();
    let mut found = [false; 4];
    for end in &ends {
        assert!(end.fitness().unwrap().score().unwrap() < 1e-15);
        let minimum = optimum
            .solutions()
            .iter()
            .position(|minimum| {
                minimum
                    .iter()
                    .zip(end.genome().iter())
                    .all(|(a, b)| (a - b).abs() < 1e-7)
            })
            .expect("a known minimum");
        found[minimum] = true;
    }
    assert_eq!(found, [true; 4]);
}

#[test]
fn a_stop_condition_met_with_convergence_is_reported() {
    let run = |stop| {
        let nelder_mead = NelderMead::builder(Real::uniform(2, -5.0..=5.0).unwrap())
            .minimize()
            .seed(9)
            .build()
            .unwrap();
        Engine::new(nelder_mead, sphere)
            .stop_when(stop)
            .run()
            .unwrap()
    };
    let converged = run(Stop::evaluations(10_000));
    assert_eq!(converged.stop_reason(), StopReason::Converged);
    let at_the_end = run(Stop::generations(converged.generations()));
    assert_eq!(at_the_end.stop_reason(), StopReason::Generations);
    assert_eq!(at_the_end.best(), converged.best());
}

#[test]
fn a_seed_gives_the_same_run() {
    let run = |seed| {
        let nelder_mead = NelderMead::builder(Real::uniform(4, -5.0..=5.0).unwrap())
            .restarts(local::Restarts::Random { times: 3 })
            .minimize()
            .seed(seed)
            .build()
            .unwrap();
        let outcome = Engine::new(nelder_mead, rosenbrock)
            .stop_when(Stop::evaluations(20_000))
            .run()
            .unwrap();
        (outcome.best().clone(), outcome.evaluations())
    };
    assert_eq!(run(10), run(10));
    assert_ne!(run(10), run(11));
}

#[test]
fn ask_and_tell_protocol() {
    let mut nelder_mead = square();
    assert!(matches!(nelder_mead.tell(&[]), Err(Error::TellWithoutAsk)));
    let first = asked(&mut nelder_mead);
    assert_eq!(asked(&mut nelder_mead), first);
    assert!(matches!(
        nelder_mead.tell(&[Fitness::new(1.0)]),
        Err(Error::FitnessCount {
            expected: 3,
            got: 1
        })
    ));
    assert!(matches!(
        nelder_mead.reevaluate(),
        Err(Error::ReevaluationOutOfTurn)
    ));
    // nothing changed
    assert_eq!(asked(&mut nelder_mead), first);
    tell(&mut nelder_mead, &[1.0, 2.0, 3.0]);
    assert_eq!(nelder_mead.evaluations(), 3);
    assert_eq!(nelder_mead.generation(), 0);
}

#[test]
fn reevaluation_scores_the_simplex_again_without_a_step() {
    let mut nelder_mead = started();
    // under way: the reflection waits for its expansion
    tell(&mut nelder_mead, &[0.5]);
    nelder_mead.reevaluate().unwrap();
    assert_eq!(
        asked(&mut nelder_mead),
        [[4.0, 4.0], [5.0, 4.0], [4.0, 5.0]]
    );
    // the order changes with the new values, and the best is the best of them
    tell(&mut nelder_mead, &[3.0, 2.0, 1.0]);
    assert_eq!(nelder_mead.generation(), 1);
    assert_eq!(nelder_mead.evaluations(), 7);
    assert_eq!(nelder_mead.best().unwrap().genome().to_vec(), [4.0, 5.0]);
    assert_eq!(nelder_mead.best_generation(), 1);
    // the next iteration reflects the new worst, (4, 4), through the centroid (4.5, 4.5)
    assert_eq!(asked(&mut nelder_mead), [[5.0, 5.0]]);
}

#[test]
fn invalid_settings_are_errors() {
    let setting = |builder: genoxide::algorithm::NelderMeadBuilder| match builder.build() {
        Err(Error::InvalidSetting { setting, .. }) => setting,
        other => panic!("{other:?}"),
    };
    let builder = || NelderMead::builder(Real::uniform(2, -1.0..=1.0).unwrap());
    let fixed = NelderMead::builder(Real::new([1.0..=1.0]).unwrap());
    assert_eq!(setting(fixed), "real");
    for step in [0.0, 1.5, f64::NAN] {
        assert_eq!(setting(builder().initial_step(step)), "initial_step");
    }
    for tolerance in [0.0, 1.0, 1.5, f64::NAN] {
        assert_eq!(setting(builder().tolerance(tolerance)), "tolerance");
    }
    for distance in [0.0, -1.0, f64::INFINITY, f64::NAN] {
        assert_eq!(
            setting(builder().initial_step_absolute(distance)),
            "initial_step_absolute"
        );
    }
    // the last of the two initial steps counts
    assert!(
        builder()
            .initial_step_absolute(-1.0)
            .initial_step(0.1)
            .build()
            .is_ok()
    );
    assert_eq!(
        setting(builder().restarts(local::Restarts::Random { times: 0 })),
        "restarts"
    );
    for [reflection, expansion, contraction, shrink] in [
        [0.0, 2.0, 0.5, 0.5],
        [1.0, 1.0, 0.5, 0.5],
        [2.0, 1.5, 0.5, 0.5],
        [1.0, f64::INFINITY, 0.5, 0.5],
        [1.0, 2.0, 1.0, 0.5],
        [1.0, 2.0, 0.5, 0.0],
        [1.0, 2.0, 0.5, f64::NAN],
    ] {
        let coefficients = Coefficients::Custom {
            reflection,
            expansion,
            contraction,
            shrink,
        };
        assert_eq!(
            setting(builder().coefficients(coefficients)),
            "coefficients"
        );
    }
    let outside = builder()
        .initial_genome(Reals::from(vec![0.0, 2.0]))
        .build();
    assert!(matches!(outside, Err(Error::InvalidGenome { .. })));
    let custom = Coefficients::Custom {
        reflection: 1.0,
        expansion: 2.5,
        contraction: 0.4,
        shrink: 0.6,
    };
    assert!(builder().coefficients(custom).build().is_ok());
}

#[test]
fn a_run_as_in_python() {
    // python/tests/test_nelder_mead.py has the same runs through the Python package, with the
    // same results: every setting of the builder, on a problem evaluated in Rust
    use genoxide::problems::Rosenbrock;
    let problem = Rosenbrock::new(4);
    for (speculative, evaluations, generations) in [(false, 1985, 1969), (true, 4815, 1202)] {
        let nelder_mead = NelderMead::builder(problem.representation())
            .coefficients(Coefficients::Custom {
                reflection: 1.0,
                expansion: 2.5,
                contraction: 0.4,
                shrink: 0.6,
            })
            .initial_step(0.2)
            .tolerance(5e-8)
            .restarts(local::Restarts::Random { times: 2 })
            .speculative(speculative)
            .minimize()
            .seed(5)
            .build()
            .unwrap();
        let outcome = Engine::new(nelder_mead, problem)
            .stop_when(Stop::evaluations(20_000))
            .run()
            .unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Converged);
        assert_eq!(outcome.evaluations(), evaluations);
        assert_eq!(outcome.generations(), generations);
        assert_eq!(outcome.best_fitness().score(), Some(6.607094554363915e-14));
    }
}
