//! Continuation: stages that end when the wrapped algorithm finishes or after their budget, the
//! closures called in their order, the state each method keeps between stages, and the run's
//! reproducibility.

use genoxide::algorithm::continuation::{Keep, Stage, StageEnd};
use genoxide::algorithm::first_order::Step;
use genoxide::gradient::Differentiable;
use genoxide::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

const CENTER: [f64; 3] = [0.3, -1.2, 2.0];
// the smoothing of each stage
const EPSILON: [f64; 4] = [1.0, 0.1, 0.01, 0.001];

// Σ √((xᵢ − cᵢ)² + ε²), a smoothed Σ |xᵢ − cᵢ|, with its gradient, and the cell that holds ε
#[expect(clippy::type_complexity)]
fn smoothed() -> (
    Arc<AtomicU64>,
    Differentiable<impl Fn(&Reals, &mut [f64]) -> f64 + Sync>,
) {
    let epsilon = Arc::new(AtomicU64::new(EPSILON[0].to_bits()));
    let shared = Arc::clone(&epsilon);
    let function = Differentiable(move |x: &Reals, gradient: &mut [f64]| {
        let e = f64::from_bits(shared.load(Ordering::Relaxed));
        let mut value = 0.0;
        for i in 0..x.len() {
            let d = x[i] - CENTER[i];
            let root = (d * d + e * e).sqrt();
            gradient[i] = d / root;
            value += root;
        }
        value
    });
    (epsilon, function)
}

// a continuation of `algorithm` through the stages of EPSILON, setting the cell
fn continuation<A: Continue>(
    algorithm: A,
    epsilon: &Arc<AtomicU64>,
    generations: Option<u64>,
    keep: Keep,
) -> Continuation<A> {
    let epsilon = Arc::clone(epsilon);
    let builder = Continuation::builder(algorithm)
        .stages(EPSILON.len())
        .keep(keep)
        .on_stage(move |stage, _| {
            epsilon.store(EPSILON[stage].to_bits(), Ordering::Relaxed);
            Ok(())
        });
    match generations {
        Some(generations) => builder.generations(generations),
        None => builder,
    }
    .build()
    .unwrap()
}

fn real() -> Real {
    Real::uniform(3, -5.0..=5.0).unwrap()
}

fn adam() -> FirstOrder {
    FirstOrder::builder(real())
        .step(Step::adam(0.05))
        .gradient_tolerance(1e-9)
        .minimize()
        .seed(1)
        .build()
        .unwrap()
}

fn lbfgsb(keep_pairs: bool) -> Lbfgsb {
    Lbfgsb::builder(real())
        .gradient_tolerance(1e-10)
        .keep_pairs(keep_pairs)
        .minimize()
        .seed(1)
        .build()
        .unwrap()
}

#[test]
fn stages_run_to_convergence_and_the_run_finishes_after_the_last() {
    let (epsilon, function) = smoothed();
    let mut engine = Engine::new(
        continuation(lbfgsb(false), &epsilon, None, Keep::State),
        function,
    )
    .stop_when(Stop::evaluations(10_000));
    let outcome = engine.run().unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    let continuation = engine.algorithm();
    assert_eq!(continuation.stage(), EPSILON.len() - 1);
    let stages = continuation.stages();
    assert_eq!(stages.len(), EPSILON.len());
    for (index, stage) in stages.iter().enumerate() {
        assert_eq!(stage.index(), index);
        assert_eq!(stage.end(), StageEnd::Finished);
        // each stage's minimum is n ε at x = c
        let expected = 3.0 * EPSILON[index];
        let best = stage.best().score().unwrap();
        assert!((best - expected).abs() < 1e-9, "{index}: {best}");
    }
    // the stages' generations and evaluations add up to the run's
    let generations: u64 = stages.iter().map(Stage::generations).sum();
    let evaluations: u64 = stages.iter().map(Stage::evaluations).sum();
    assert_eq!(generations, outcome.generations());
    assert_eq!(evaluations, outcome.evaluations());
    // the last stage's best is the outcome's, by the last function
    assert_eq!(Some(outcome.best_fitness()), Some(stages[3].best()));
    for (x, c) in outcome.best_genome().iter().zip(CENTER) {
        assert!((x - c).abs() < 1e-8);
    }
}

#[test]
fn a_budget_ends_each_stage_and_the_last_finishes_the_run() {
    let (epsilon, function) = smoothed();
    let mut engine = Engine::new(
        continuation(adam(), &epsilon, Some(7), Keep::State),
        function,
    )
    .stop_when(Stop::generations(1_000));
    let outcome = engine.run().unwrap();
    // 4 stages of 7 generations, each ended by its budget
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert_eq!(outcome.generations(), 28);
    let stages = engine.algorithm().stages();
    assert!(stages.iter().all(|stage| stage.generations() == 7));
    assert!(
        stages
            .iter()
            .all(|stage| stage.end() == StageEnd::Generations)
    );
    // the first stage evaluates the start and 7 points; the others re-evaluate theirs first
    let evaluations: Vec<u64> = stages.iter().map(Stage::evaluations).collect();
    assert_eq!(evaluations, [8, 8, 8, 8]);
    assert_eq!(epsilon.load(Ordering::Relaxed), EPSILON[3].to_bits());

    // the engine's stop conditions end the whole run, in a stage
    let (epsilon, function) = smoothed();
    let mut engine = Engine::new(
        continuation(adam(), &epsilon, Some(7), Keep::State),
        function,
    )
    .stop_when(Stop::generations(10));
    let outcome = engine.run().unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Generations);
    assert_eq!(engine.algorithm().stage(), 1);
    assert_eq!(engine.algorithm().stages().len(), 1);
    assert!(!engine.algorithm().is_finished());
}

#[test]
fn the_closures_are_called_in_order_with_the_stages() {
    let (epsilon, function) = smoothed();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let (stage_calls, finished_calls) = (Arc::clone(&calls), Arc::clone(&calls));
    let cell = Arc::clone(&epsilon);
    let continuation = Continuation::builder(adam())
        .stages(3)
        .generations(4)
        .on_stage(move |stage, adam: &mut FirstOrder| {
            stage_calls.lock().unwrap().push(format!("on {stage}"));
            cell.store(EPSILON[stage].to_bits(), Ordering::Relaxed);
            // the algorithm's settings can change by stage too
            adam.set_learning_rate(0.05 / (1 << stage) as f64)
        })
        .on_stage_finished(move |stage, adam| {
            finished_calls.lock().unwrap().push(format!(
                "finished {} at {}",
                stage.index(),
                adam.generation()
            ));
        })
        .build()
        .unwrap();
    let mut engine = Engine::new(continuation, function).stop_when(Stop::generations(100));
    engine.run().unwrap();
    let calls = calls.lock().unwrap().clone();
    assert_eq!(
        calls,
        [
            "on 0",
            "finished 0 at 4",
            "on 1",
            "finished 1 at 8",
            "on 2",
            "finished 2 at 12"
        ]
    );
    assert_eq!(
        engine.algorithm().algorithm().step().learning_rate(),
        0.0125
    );
}

#[test]
fn an_error_of_the_closure_stops_the_run() {
    let (_, function) = smoothed();
    let continuation = Continuation::builder(adam())
        .stages(2)
        .generations(3)
        .on_stage(|stage, _| match stage {
            0 => Ok(()),
            _ => Err(Error::InvalidSetting {
                setting: "epsilon",
                reason: "no second stage".to_string(),
            }),
        })
        .build()
        .unwrap();
    let mut engine = Engine::new(continuation, function).stop_when(Stop::generations(100));
    let error = engine.run().unwrap_err();
    assert!(matches!(
        error,
        Error::InvalidSetting {
            setting: "epsilon",
            ..
        }
    ));
    assert_eq!(engine.algorithm().stage(), 1);
}

// the steps t of Adam's corrections at the start of each stage, after `next_stage`
fn adam_steps(keep: Keep) -> Vec<u64> {
    let (epsilon, function) = smoothed();
    let steps = Arc::new(Mutex::new(Vec::new()));
    let recorded = Arc::clone(&steps);
    let continuation = Continuation::builder(adam())
        .stages(3)
        .generations(5)
        .keep(keep)
        .on_stage(move |stage, adam: &mut FirstOrder| {
            recorded.lock().unwrap().push(adam.steps());
            epsilon.store(EPSILON[stage].to_bits(), Ordering::Relaxed);
            Ok(())
        })
        .build()
        .unwrap();
    let mut engine = Engine::new(continuation, function).stop_when(Stop::generations(100));
    engine.run().unwrap();
    let mut steps = steps.lock().unwrap().clone();
    steps.push(engine.algorithm().algorithm().steps());
    steps
}

#[test]
fn adam_keeps_its_moments_with_the_state_and_resets_them_with_the_point() {
    // t counts on across the stages, or starts again from 0 in each
    assert_eq!(adam_steps(Keep::State), [0, 5, 10, 15]);
    assert_eq!(adam_steps(Keep::Point), [0, 0, 0, 5]);

    // by hand: the averages survive a next stage that keeps the state, and the first step after
    // one that keeps the point is Adam's first, of the learning rate per gene
    let sphere = |x: &Reals| x.iter().map(|xi| xi * xi).sum::<f64>();
    let run = |keep: Keep| {
        let mut adam = FirstOrder::builder(real())
            .step(Step::adam(0.1))
            .gradients(genoxide::gradient::Gradients::Central { step: None })
            .initial_genome(Reals::from(vec![1.0, 2.0, 3.0]))
            .minimize()
            .build()
            .unwrap();
        for _ in 0..10 {
            let fitness: Vec<Fitness> =
                adam.ask().iter().map(|x| Fitness::new(sphere(x))).collect();
            adam.tell(&fitness).unwrap();
        }
        adam.next_stage(keep).unwrap();
        let before = adam.population()[0].genome().clone();
        for _ in 0..2 {
            let fitness: Vec<Fitness> =
                adam.ask().iter().map(|x| Fitness::new(sphere(x))).collect();
            adam.tell(&fitness).unwrap();
        }
        let after = adam.population()[0].genome().clone();
        (adam.steps(), before, after)
    };
    let (steps, before, after) = run(Keep::State);
    // 9 steps before, the re-evaluation, then 1
    assert_eq!(steps, 10);
    let (reset, before_reset, after_reset) = run(Keep::Point);
    assert_eq!(reset, 1);
    assert_eq!(before, before_reset);
    // Adam's first step moves each gene by the learning rate, to rounding
    for (a, b) in before_reset.iter().zip(after_reset.iter()) {
        assert!(((a - b).abs() - 0.1).abs() < 1e-6, "{a} {b}");
    }
    assert_ne!(after, after_reset);
}

#[test]
fn lbfgsb_drops_its_pairs_unless_built_to_keep_them() {
    for (keep_pairs, keep, kept) in [
        (false, Keep::State, false),
        (true, Keep::State, true),
        (true, Keep::Point, false),
    ] {
        let (epsilon, function) = smoothed();
        let pairs = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&pairs);
        let continuation = Continuation::builder(lbfgsb(keep_pairs))
            .stages(2)
            .generations(4)
            .keep(keep)
            .on_stage(move |stage, lbfgsb: &mut Lbfgsb| {
                recorded.lock().unwrap().push(lbfgsb.pairs());
                epsilon.store(EPSILON[stage].to_bits(), Ordering::Relaxed);
                Ok(())
            })
            .build()
            .unwrap();
        Engine::new(continuation, function)
            .stop_when(Stop::generations(100))
            .run()
            .unwrap();
        let pairs = pairs.lock().unwrap().clone();
        assert_eq!(pairs[0], 0);
        assert_eq!(pairs[1] > 0, kept, "{keep_pairs} {keep:?}: {pairs:?}");
    }
}

#[test]
fn mma_keeps_its_asymptotes_with_the_state() {
    for keep in [Keep::State, Keep::Point] {
        let (epsilon, function) = smoothed();
        let asymptotes = Arc::new(Mutex::new(Vec::new()));
        let (at_end, at_start) = (Arc::clone(&asymptotes), Arc::clone(&asymptotes));
        let mma = Mma::builder(real()).minimize().seed(1).build().unwrap();
        let continuation = Continuation::builder(mma)
            .stages(2)
            .generations(6)
            .keep(keep)
            .on_stage(move |stage, mma: &mut Mma| {
                let x = mma.population()[0].genome().to_vec();
                at_start
                    .lock()
                    .unwrap()
                    .push((x, mma.lower_asymptotes().to_vec()));
                epsilon.store(EPSILON[stage].to_bits(), Ordering::Relaxed);
                Ok(())
            })
            .on_stage_finished(move |_, mma| {
                let x = mma.population()[0].genome().to_vec();
                at_end
                    .lock()
                    .unwrap()
                    .push((x, mma.lower_asymptotes().to_vec()));
            })
            .build()
            .unwrap();
        Engine::new(continuation, function)
            .stop_when(Stop::generations(100))
            .run()
            .unwrap();
        // the start, the end of stage 0, the start of stage 1, the end of stage 1
        let asymptotes = asymptotes.lock().unwrap().clone();
        assert_eq!(asymptotes.len(), 4);
        let (end, start) = (&asymptotes[1], &asymptotes[2]);
        assert_eq!(end.0, start.0);
        match keep {
            Keep::State => assert_eq!(end.1, start.1),
            _ => {
                // half the range of 10 below the point, as at a start
                for (x, l) in start.0.iter().zip(&start.1) {
                    assert!((x - 5.0 - l).abs() < 1e-12);
                }
                assert_ne!(end.1, start.1);
            }
        }
    }
}

#[test]
fn nelder_mead_keeps_its_simplex_with_the_state() {
    for keep in [Keep::State, Keep::Point] {
        let (epsilon, function) = smoothed();
        let simplices = Arc::new(Mutex::new(Vec::new()));
        let (at_end, at_start) = (Arc::clone(&simplices), Arc::clone(&simplices));
        let vertices = |nelder_mead: &NelderMead| -> Vec<Vec<f64>> {
            let mut vertices: Vec<Vec<f64>> = nelder_mead
                .population()
                .iter()
                .map(|vertex| vertex.genome().to_vec())
                .collect();
            vertices.sort_by(|a, b| a.partial_cmp(b).unwrap());
            vertices
        };
        let nelder_mead = NelderMead::builder(real())
            .minimize()
            .seed(1)
            .build()
            .unwrap();
        let continuation = Continuation::builder(nelder_mead)
            .stages(2)
            .generations(20)
            .keep(keep)
            .on_stage(move |stage, nelder_mead: &mut NelderMead| {
                at_start.lock().unwrap().push(vertices(nelder_mead));
                epsilon.store(EPSILON[stage].to_bits(), Ordering::Relaxed);
                Ok(())
            })
            .on_stage_finished(move |_, nelder_mead| {
                at_end.lock().unwrap().push(vertices(nelder_mead));
            })
            .build()
            .unwrap();
        let mut engine = Engine::new(continuation, function).stop_when(Stop::generations(100));
        engine.run().unwrap();
        let simplices = simplices.lock().unwrap().clone();
        let (end, start) = (&simplices[1], &simplices[2]);
        match keep {
            Keep::State => assert_eq!(end, start),
            _ => {
                assert_ne!(end, start);
                // the initial steps, a tenth of the range, around one vertex
                let base = &start[0];
                let moved = start.iter().filter(|vertex| *vertex != base).count();
                assert_eq!(moved, 3);
            }
        }
        assert_eq!(engine.algorithm().stages()[1].generations(), 20);
    }
}

#[test]
fn cma_es_keeps_its_distribution_with_the_state() {
    for keep in [Keep::State, Keep::Point] {
        let (epsilon, function) = smoothed();
        let steps = Arc::new(Mutex::new(Vec::new()));
        let (at_end, at_start) = (Arc::clone(&steps), Arc::clone(&steps));
        let cmaes = Cmaes::builder(real())
            .restarts(cmaes::Restarts::Stop)
            .minimize()
            .seed(1)
            .build()
            .unwrap();
        let continuation = Continuation::builder(cmaes)
            .stages(2)
            .generations(30)
            .keep(keep)
            .on_stage(move |stage, cmaes: &mut Cmaes| {
                at_start
                    .lock()
                    .unwrap()
                    .push((cmaes.step_size(), cmaes.mean().to_vec()));
                epsilon.store(EPSILON[stage].to_bits(), Ordering::Relaxed);
                Ok(())
            })
            .on_stage_finished(move |_, cmaes| {
                at_end
                    .lock()
                    .unwrap()
                    .push((cmaes.step_size(), cmaes.mean().to_vec()));
            })
            .build()
            .unwrap();
        Engine::new(continuation, function)
            .stop_when(Stop::generations(1_000))
            .run()
            .unwrap();
        let steps = steps.lock().unwrap().clone();
        let (end, start) = (&steps[1], &steps[2]);
        assert_eq!(end.1, start.1, "the mean is kept");
        match keep {
            Keep::State => assert_eq!(end.0, start.0),
            _ => {
                assert!(end.0 < 0.3);
                assert_eq!(start.0, 0.3);
            }
        }
    }
}

#[test]
fn next_stage_goes_on_after_convergence() {
    // a stage that converged, by hand: the next stage's run goes on, not finished
    let sphere = |x: &Reals| x.iter().map(|xi| xi * xi).sum::<f64>();
    let mut nelder_mead = NelderMead::builder(real())
        .minimize()
        .seed(1)
        .build()
        .unwrap();
    while !nelder_mead.is_finished() {
        let fitness: Vec<Fitness> = nelder_mead
            .ask()
            .iter()
            .map(|x| Fitness::new(sphere(x)))
            .collect();
        nelder_mead.tell(&fitness).unwrap();
    }
    let generation = nelder_mead.generation();
    nelder_mead.next_stage(Keep::Point).unwrap();
    assert!(!nelder_mead.is_finished());
    let asked = nelder_mead.ask().len();
    assert_eq!(asked, 4);
    // out of turn
    assert!(matches!(
        nelder_mead.next_stage(Keep::State),
        Err(Error::ReevaluationOutOfTurn)
    ));
    let fitness: Vec<Fitness> = nelder_mead
        .ask()
        .iter()
        .map(|x| Fitness::new(sphere(x) + 1.0))
        .collect();
    nelder_mead.tell(&fitness).unwrap();
    assert_eq!(nelder_mead.generation(), generation);
    assert!(!nelder_mead.is_finished());
}

#[test]
fn runs_are_reproducible_sequential_or_parallel() {
    let run = |parallel: bool| {
        let (epsilon, function) = smoothed();
        let cmaes = Cmaes::builder(Real::uniform(3, -5.0..=5.0).unwrap())
            .minimize()
            .seed(3)
            .build()
            .unwrap();
        let engine = Engine::new(
            continuation(cmaes, &epsilon, Some(25), Keep::State),
            function,
        )
        .stop_when(Stop::generations(1_000));
        #[cfg(feature = "parallel")]
        let engine = engine.parallel(parallel);
        #[cfg(not(feature = "parallel"))]
        let _ = parallel;
        let mut engine = engine;
        let outcome = engine.run().unwrap();
        (outcome.into_best(), engine.algorithm().stages().to_vec())
    };
    let first = run(false);
    assert_eq!(first, run(false));
    assert_eq!(first, run(true));
    assert_eq!(first.1.len(), 4);
}

#[test]
fn settings_are_validated() {
    let missing = |result: Result<Continuation<FirstOrder>>| match result {
        Err(Error::MissingSetting { setting }) => setting,
        other => panic!("{other:?}"),
    };
    let invalid = |result: Result<Continuation<FirstOrder>>| match result {
        Err(Error::InvalidSetting { setting, .. }) => setting,
        other => panic!("{other:?}"),
    };
    let on_stage = |_: usize, _: &mut FirstOrder| Ok(());
    assert_eq!(
        missing(Continuation::builder(adam()).on_stage(on_stage).build()),
        "stages"
    );
    assert_eq!(
        missing(Continuation::builder(adam()).stages(2).build()),
        "on_stage"
    );
    assert_eq!(
        invalid(
            Continuation::builder(adam())
                .stages(0)
                .on_stage(on_stage)
                .build()
        ),
        "stages"
    );
    assert_eq!(
        invalid(
            Continuation::builder(adam())
                .stages(2)
                .generations(0)
                .on_stage(on_stage)
                .build()
        ),
        "generations"
    );
    let built = Continuation::builder(adam())
        .stages(2)
        .generations(3)
        .keep(Keep::Point)
        .on_stage(on_stage)
        .build()
        .unwrap();
    assert_eq!(
        (
            built.stage(),
            built.stage_count(),
            built.generations(),
            built.keep()
        ),
        (0, 2, Some(3), Keep::Point)
    );
    assert!(format!("{built:?}").contains("stage_count: 2"));
}

#[test]
fn a_single_stage_is_the_algorithm_alone() {
    let (_, function) = smoothed();
    let alone = Engine::new(lbfgsb(false), function)
        .stop_when(Stop::evaluations(10_000))
        .run()
        .unwrap();
    let (epsilon, function) = smoothed();
    let cell = Arc::clone(&epsilon);
    let continuation = Continuation::builder(lbfgsb(false))
        .stages(1)
        .on_stage(move |_, _| {
            cell.store(EPSILON[0].to_bits(), Ordering::Relaxed);
            Ok(())
        })
        .build()
        .unwrap();
    let staged = Engine::new(continuation, function)
        .stop_when(Stop::evaluations(10_000))
        .run()
        .unwrap();
    assert_eq!(staged.best(), alone.best());
    assert_eq!(staged.evaluations(), alone.evaluations());
    assert_eq!(staged.stop_reason(), StopReason::Converged);
}
