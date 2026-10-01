//! L-BFGS-B: convergence and its rate, bounds and the active set, the sources of gradients and
//! their cost, restarts, re-evaluation and the ask / tell protocol.

use genoxide::engine::{Evaluations, Provided};
use genoxide::gradient::{Differentiable, Gradients};
use genoxide::prelude::*;
use genoxide::problems::{
    AxisParallelEllipsoid, Problem, Rastrigin, Rosenbrock, Schwefel1_2, Sphere,
};
use std::cell::RefCell;

// Rosenbrock's classic start, (−1.2, 1) repeated
fn classic_start(n: usize) -> Reals {
    (0..n)
        .map(|i| if i % 2 == 0 { -1.2 } else { 1.0 })
        .collect()
}

fn run<F: FitnessFunction<Reals>>(
    lbfgsb: Lbfgsb,
    fitness: F,
    evaluations: u64,
) -> (Outcome<Reals>, Lbfgsb) {
    let mut engine = Engine::new(lbfgsb, fitness).stop_when(Stop::evaluations(evaluations));
    let outcome = engine.run().unwrap();
    (outcome, engine.into_algorithm())
}

fn distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt()
}

#[test]
fn rosenbrock_converges_from_the_classic_start() {
    // 2, 100 and 1000 genes; the analytic gradient (Gradients::Auto)
    for (n, budget) in [(2, 200), (100, 2_000), (1000, 20_000)] {
        let problem = Rosenbrock::new(n);
        let lbfgsb = Lbfgsb::builder(problem.representation())
            .initial_genome(classic_start(n))
            .gradient_tolerance(1e-8)
            .function_tolerance(0.0)
            .minimize()
            .build()
            .unwrap();
        let (outcome, lbfgsb) = run(lbfgsb, problem, budget);
        assert_eq!(outcome.stop_reason(), StopReason::Converged, "{n}");
        assert_eq!(
            lbfgsb.converged(),
            Some(lbfgsb::Criterion::ProjectedGradient)
        );
        assert_eq!(lbfgsb.gradients(), Gradients::Supplied);
        assert!(lbfgsb.projected_gradient() <= 1e-8);
        assert!(outcome.best_fitness().score().unwrap() < 1e-16, "{n}");
        assert!(distance(outcome.best_genome(), &vec![1.0; n]) < 1e-7, "{n}");
        // one evaluation per gradient, every trial with one
        assert_eq!(outcome.evaluations(), lbfgsb.gradient_evaluations());
        assert_eq!(lbfgsb.stencil_evaluations(), 0);
    }
}

#[test]
fn smooth_functions_converge_to_their_minimum() {
    let build = |real: Real| {
        Lbfgsb::builder(real)
            .gradient_tolerance(1e-9)
            .function_tolerance(0.0)
            .minimize()
            .seed(3)
            .build()
            .unwrap()
    };
    let n = 30;
    let (outcome, _) = run(
        build(Sphere::new(n).representation()),
        Sphere::new(n),
        1_000,
    );
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert!(outcome.best_fitness().score().unwrap() < 1e-18);
    let problem = AxisParallelEllipsoid::new(n);
    let (outcome, _) = run(build(problem.representation()), problem, 1_000);
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert!(outcome.best_fitness().score().unwrap() < 1e-16);
    // ill-conditioned: the curvature of the prefix sums
    let problem = Schwefel1_2::new(n);
    let (outcome, _) = run(build(problem.representation()), problem, 5_000);
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert!(outcome.best_fitness().score().unwrap() < 1e-14);
}

#[test]
fn the_convergence_near_the_minimum_is_superlinear() {
    // Rosenbrock in 2 genes, whose minimum (1, 1) is exact: the error of each iterate
    let problem = Rosenbrock::new(2);
    let lbfgsb = Lbfgsb::builder(problem.representation())
        .initial_genome(classic_start(2))
        .gradient_tolerance(0.0)
        .function_tolerance(0.0)
        .minimize()
        .build()
        .unwrap();
    let errors = RefCell::new(Vec::new());
    let mut engine = Engine::new(lbfgsb, problem)
        .stop_when(Stop::evaluations(200))
        .on_generation(|snapshot| {
            let x = snapshot.population()[0].genome();
            let error = distance(x, &[1.0, 1.0]);
            let mut errors = errors.borrow_mut();
            if errors.last() != Some(&error) {
                errors.push(error);
            }
        });
    engine.run().unwrap();
    drop(engine);
    let errors = errors.into_inner();
    // the ratios eₖ₊₁ / eₖ of the iterates between 1e-2 and 1e-12 of the minimum
    let ratios: Vec<f64> = errors
        .windows(2)
        .filter(|pair| pair[0] < 1e-2 && pair[1] > 1e-12)
        .map(|pair| pair[1] / pair[0])
        .collect();
    assert!(ratios.len() >= 3, "{errors:?}");
    // they go to 0: the last ones far below any linear rate
    let last = &ratios[ratios.len() - 3..];
    assert!(last.iter().all(|&ratio| ratio < 0.2), "{ratios:?}");
    assert!(ratios.last().unwrap() < &0.05, "{ratios:?}");
}

// a quadratic with A tridiagonal (4 on the diagonal, −1 off it), in [−1, 1]ⁿ, whose solution is
// `quadratic_solution()`: written around it, ½ dᵀAd + g*ᵀd with d = x − x*, so its minimum is 0
// exactly and the values near it keep their precision. g*, the gradient at x*, is 0 where x* is
// free, +2 at a lower bound and −3 at an upper one: the KKT conditions, with multipliers of the
// right sign, so x* is the unique solution
const N: usize = 12;

fn quadratic_solution() -> [f64; N] {
    [
        1.0, 0.25, -1.0, -0.5, 1.0, 1.0, 0.0, -1.0, 0.125, 0.5, -1.0, 1.0,
    ]
}

fn a_times(x: &[f64]) -> Vec<f64> {
    (0..x.len())
        .map(|i| {
            let mut value = 4.0 * x[i];
            if i > 0 {
                value -= x[i - 1];
            }
            if i + 1 < x.len() {
                value -= x[i + 1];
            }
            value
        })
        .collect()
}

fn gradient_at_solution() -> Vec<f64> {
    quadratic_solution()
        .iter()
        .map(|&x| {
            if x == -1.0 {
                2.0
            } else if x == 1.0 {
                -3.0
            } else {
                0.0
            }
        })
        .collect()
}

#[test]
fn a_bound_constrained_quadratic_ends_on_its_active_set() {
    let solution = quadratic_solution();
    let multipliers = gradient_at_solution();
    let outside = std::sync::atomic::AtomicBool::new(false);
    let quadratic = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        if x.iter().any(|xi| !(-1.0..=1.0).contains(xi)) {
            outside.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        let d: Vec<f64> = x.iter().zip(&solution).map(|(x, s)| x - s).collect();
        let ad = a_times(&d);
        let mut value = 0.0;
        for i in 0..x.len() {
            gradient[i] = ad[i] + multipliers[i];
            value += d[i] * (0.5 * ad[i] + multipliers[i]);
        }
        value
    });
    for seed in 0..10 {
        let lbfgsb = Lbfgsb::builder(Real::uniform(N, -1.0..=1.0).unwrap())
            .gradient_tolerance(1e-12)
            .function_tolerance(0.0)
            .minimize()
            .seed(seed)
            .build()
            .unwrap();
        let (outcome, lbfgsb) = run(lbfgsb, quadratic, 1_000);
        assert_eq!(outcome.stop_reason(), StopReason::Converged);
        assert_eq!(
            lbfgsb.converged(),
            Some(lbfgsb::Criterion::ProjectedGradient)
        );
        assert!(lbfgsb.projected_gradient() <= 1e-12);
        let x = outcome.best_genome();
        for (gene, (&xi, &expected)) in x.iter().zip(&solution).enumerate() {
            if expected.abs() == 1.0 {
                // the active set, exactly on its bounds
                assert_eq!(xi, expected, "gene {gene}, seed {seed}");
            } else {
                assert!((xi - expected).abs() < 1e-12, "gene {gene}: {xi}");
            }
        }
    }
    // central differences, whose stencils step inwards from the bounds
    for seed in 0..5 {
        let lbfgsb = Lbfgsb::builder(Real::uniform(N, -1.0..=1.0).unwrap())
            .gradients(Gradients::Central { step: None })
            .gradient_tolerance(1e-8)
            .function_tolerance(0.0)
            .minimize()
            .seed(seed)
            .build()
            .unwrap();
        let (outcome, lbfgsb) = run(lbfgsb, quadratic, 100_000);
        assert_eq!(
            lbfgsb.converged(),
            Some(lbfgsb::Criterion::ProjectedGradient)
        );
        for (&xi, &expected) in outcome.best_genome().iter().zip(&solution) {
            if expected.abs() == 1.0 {
                assert_eq!(xi, expected);
            } else {
                assert!((xi - expected).abs() < 1e-8, "{xi}");
            }
        }
    }
    assert!(
        !outside.into_inner(),
        "a point outside the box was evaluated"
    );
}

#[test]
fn a_minimum_on_the_bound_is_found_from_inside() {
    // the sphere in [1, 5]ⁿ: the minimum at the corner (1, …, 1)
    let n = 5;
    let lbfgsb = Lbfgsb::builder(Real::uniform(n, 1.0..=5.0).unwrap())
        .minimize()
        .seed(1)
        .build()
        .unwrap();
    let (outcome, lbfgsb) = run(lbfgsb, Sphere::new(n), 100);
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert_eq!(outcome.best_genome()[..], [1.0; 5]);
    assert_eq!(outcome.best_fitness(), Fitness::new(5.0));
    assert_eq!(lbfgsb.projected_gradient(), 0.0);
}

#[test]
fn finite_differences_cost_their_stencils() {
    let n = 10;
    let problem = Rosenbrock::new(n);
    // the same function without its gradient
    let plain = move |x: &Reals| problem.evaluate(x);
    let build = |gradients: Gradients| {
        Lbfgsb::builder(problem.representation())
            .initial_genome(classic_start(n))
            .gradients(gradients)
            // forward differences can't meet a much smaller tolerance at the minimum
            .gradient_tolerance(1e-5)
            .function_tolerance(0.0)
            .minimize()
            .build()
            .unwrap()
    };
    let (supplied, supplied_run) = run(build(Gradients::Auto), problem, 100_000);
    let (forward, forward_run) = run(build(Gradients::Auto), plain, 100_000);
    let (central, central_run) = run(build(Gradients::Central { step: None }), problem, 100_000);
    assert_eq!(supplied_run.gradients(), Gradients::Supplied);
    assert_eq!(forward_run.gradients(), Gradients::Forward { step: None });
    assert_eq!(central_run.gradients(), Gradients::Central { step: None });
    for outcome in [&supplied, &forward, &central] {
        assert_eq!(outcome.stop_reason(), StopReason::Converged);
        assert!(
            outcome.best_fitness().score().unwrap() < 1e-10,
            "{:?}",
            outcome.best_fitness()
        );
    }
    // a trial is one round: its point and its stencil
    let n = n as u64;
    assert_eq!(supplied.evaluations(), supplied_run.gradient_evaluations());
    assert_eq!(
        forward.evaluations(),
        forward_run.gradient_evaluations() * (n + 1)
    );
    assert_eq!(
        forward_run.stencil_evaluations(),
        forward_run.gradient_evaluations() * n
    );
    assert_eq!(
        central.evaluations(),
        central_run.gradient_evaluations() * (2 * n + 1)
    );
    assert_eq!(
        forward.generations() + 1,
        forward_run.gradient_evaluations()
    );
    // the supplied gradient is the cheapest by far
    assert!(forward.evaluations() > 5 * supplied.evaluations());
    assert!(central.evaluations() > forward.evaluations());
}

#[test]
fn supplied_gradients_must_be_provided() {
    let lbfgsb = Lbfgsb::builder(Real::uniform(3, -1.0..=1.0).unwrap())
        .gradients(Gradients::Supplied)
        .minimize()
        .build()
        .unwrap();
    let mut engine = Engine::new(lbfgsb, |x: &Reals| x[0]).stop_when(Stop::generations(5));
    assert!(matches!(
        engine.run(),
        Err(Error::InvalidSetting {
            setting: "gradients",
            ..
        })
    ));
    // Auto above 10⁴ genes without a gradient
    let lbfgsb = Lbfgsb::builder(Real::uniform(20_000, -1.0..=1.0).unwrap())
        .minimize()
        .seed(1)
        .build()
        .unwrap();
    let mut engine = Engine::new(lbfgsb, |x: &Reals| x[0]).stop_when(Stop::generations(5));
    assert!(matches!(
        engine.run(),
        Err(Error::InvalidSetting {
            setting: "gradients",
            ..
        })
    ));
}

#[test]
fn maximizing_a_function_is_minimizing_its_negation() {
    let problem = Rosenbrock::new(4);
    let negated = Differentiable(move |x: &Reals, gradient: &mut [f64]| {
        let value =
            problem.evaluate_with(x, &mut genoxide::engine::Extras::with_gradient(gradient));
        for g in gradient.iter_mut() {
            *g = -*g;
        }
        -value
    });
    let build = |objective: Objective| {
        Lbfgsb::builder(problem.representation())
            .objective(objective)
            .seed(9)
            .build()
            .unwrap()
    };
    let (minimized, _) = run(build(Objective::Minimize), problem, 1_000);
    let (maximized, _) = run(build(Objective::Maximize), negated, 1_000);
    assert_eq!(minimized.best_genome(), maximized.best_genome());
    assert_eq!(minimized.evaluations(), maximized.evaluations());
    assert_eq!(
        minimized.best_fitness().score().unwrap(),
        -maximized.best_fitness().score().unwrap()
    );
}

#[test]
fn random_restarts_find_better_minima() {
    // Rastrigin in 2 genes: every run ends in a local minimum, the best of them is kept; about
    // one start in a hundred is in the global minimum's basin
    let problem = Rastrigin::new(2);
    let lbfgsb = Lbfgsb::builder(problem.representation())
        .restarts(local::Restarts::Random { times: 300 })
        .minimize()
        .seed(4)
        .build()
        .unwrap();
    let (outcome, lbfgsb) = run(lbfgsb, problem, 100_000);
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert_eq!(lbfgsb.restart_count(), 300);
    assert!(
        outcome.best_fitness().score().unwrap() < 1e-10,
        "{:?}",
        outcome.best_fitness()
    );
}

#[test]
fn fixed_genes_stay_fixed() {
    let real = Real::new([-5.0..=5.0, 2.0..=2.0, -5.0..=5.0]).unwrap();
    let lbfgsb = Lbfgsb::builder(real).minimize().seed(2).build().unwrap();
    let (outcome, _) = run(lbfgsb, Sphere::new(3), 1_000);
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert_eq!(outcome.best_genome()[1], 2.0);
    assert!(outcome.best_genome()[0].abs() < 1e-8);
    assert!(
        Lbfgsb::builder(Real::new([1.0..=1.0]).unwrap())
            .build()
            .is_err()
    );
}

#[test]
fn invalid_points_are_stepped_back_from() {
    // the sphere, undefined (NaN) beyond x₀ = 2: the minimum (3, …) is out of reach, the run
    // ends near the edge
    let n = 3;
    let shifted = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        if x[0] > 2.0 {
            return f64::NAN;
        }
        let mut value = 0.0;
        for i in 0..x.len() {
            gradient[i] = 2.0 * (x[i] - 3.0);
            value += (x[i] - 3.0) * (x[i] - 3.0);
        }
        value
    });
    let lbfgsb = Lbfgsb::builder(Real::uniform(n, -10.0..=10.0).unwrap())
        .initial_genome(Reals::from(vec![0.0; n]))
        .minimize()
        .build()
        .unwrap();
    let (outcome, lbfgsb) = run(lbfgsb, shifted, 1_000);
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert!(outcome.best_fitness().is_valid());
    assert!(outcome.best_genome()[0] <= 2.0);
    assert!(
        outcome.best_genome()[0] > 1.5,
        "{:?}",
        outcome.best_genome()
    );
    assert!(lbfgsb.converged().is_some());
    // a start where the function isn't defined
    let lbfgsb = Lbfgsb::builder(Real::uniform(n, -10.0..=10.0).unwrap())
        .initial_genome(Reals::from(vec![5.0; n]))
        .minimize()
        .build()
        .unwrap();
    let (outcome, lbfgsb) = run(lbfgsb, shifted, 1_000);
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert_eq!(lbfgsb.converged(), Some(lbfgsb::Criterion::NotFinite));
    assert_eq!(outcome.evaluations(), 1);
}

fn sphere_fitness(genomes: Vec<Vec<f64>>) -> Vec<Fitness> {
    genomes
        .iter()
        .map(|x| Fitness::new(x.iter().map(|xi| xi * xi).sum()))
        .collect()
}

fn asked(lbfgsb: &mut Lbfgsb) -> Vec<Vec<f64>> {
    lbfgsb.ask().iter().map(|genome| genome.to_vec()).collect()
}

#[test]
fn ask_and_tell_by_hand() {
    let mut lbfgsb = Lbfgsb::builder(Real::uniform(3, -2.0..=2.0).unwrap())
        .initial_genome(Reals::from(vec![1.0, -1.0, 0.5]))
        .minimize()
        .build()
        .unwrap();
    // without an engine, Auto is forward differences: the point and its 3 stencil points
    assert_eq!(lbfgsb.gradients(), Gradients::Forward { step: None });
    assert_eq!(lbfgsb.tell(&[]), Err(Error::TellWithoutAsk));
    let first = asked(&mut lbfgsb);
    assert_eq!(first.len(), 4);
    assert_eq!(first[0], [1.0, -1.0, 0.5]);
    assert_eq!(asked(&mut lbfgsb), first);
    assert_eq!(
        lbfgsb.tell(&[Fitness::new(1.0)]),
        Err(Error::FitnessCount {
            expected: 4,
            got: 1
        })
    );
    assert_eq!(asked(&mut lbfgsb), first);
    lbfgsb.tell(&sphere_fitness(first)).unwrap();
    assert_eq!(lbfgsb.generation(), 0);
    assert_eq!(lbfgsb.evaluations(), 4);
    while lbfgsb.converged().is_none() && lbfgsb.generation() < 100 {
        let genomes = asked(&mut lbfgsb);
        lbfgsb.tell(&sphere_fitness(genomes)).unwrap();
    }
    assert_eq!(
        lbfgsb.converged(),
        Some(lbfgsb::Criterion::ProjectedGradient)
    );
    assert!(lbfgsb.is_finished());
    assert!(lbfgsb.best().unwrap().fitness().unwrap().score().unwrap() < 1e-12);
    // a finished run asks for nothing more
    assert!(asked(&mut lbfgsb).is_empty());
    lbfgsb.tell(&[]).unwrap();
}

#[test]
fn supplied_gradients_are_told_with_the_evaluations() {
    let mut lbfgsb = Lbfgsb::builder(Real::uniform(2, -2.0..=2.0).unwrap())
        .initial_genome(Reals::from(vec![1.0, 1.0]))
        .minimize()
        .build()
        .unwrap();
    lbfgsb.prepare(Provided::GRADIENT).unwrap();
    assert_eq!(lbfgsb.gradients(), Gradients::Supplied);
    assert!(lbfgsb.wants().gradient);
    let genomes = asked(&mut lbfgsb);
    assert_eq!(genomes.len(), 1);
    let fitness = sphere_fitness(genomes.clone());
    // a plain tell has no gradient
    assert!(matches!(
        lbfgsb.tell(&fitness),
        Err(Error::InvalidSetting {
            setting: "gradients",
            ..
        })
    ));
    let gradient: Vec<f64> = genomes[0].iter().map(|x| 2.0 * x).collect();
    let evaluations = Evaluations::with_gradients(&fitness, &gradient, 2).unwrap();
    lbfgsb.tell_evaluations(&evaluations).unwrap();
    assert_eq!(lbfgsb.gradient_evaluations(), 1);
    // B = I at the start: the first trial is a unit step towards the minimum
    let trial = asked(&mut lbfgsb)[0].clone();
    let expected = 1.0 - 1.0 / 2f64.sqrt();
    assert!((trial[0] - expected).abs() < 1e-15 && (trial[1] - expected).abs() < 1e-15);
}

#[test]
fn reevaluation_scores_the_current_point_again() {
    let problem = Rosenbrock::new(3);
    let mut lbfgsb = Lbfgsb::builder(problem.representation())
        .initial_genome(classic_start(3))
        .minimize()
        .seed(5)
        .build()
        .unwrap();
    lbfgsb.prepare(Provided::GRADIENT).unwrap();
    // before the first tell it does nothing
    lbfgsb.reevaluate().unwrap();
    let evaluate = |lbfgsb: &mut Lbfgsb, scale: f64| {
        let genomes = asked(lbfgsb);
        let mut fitness = Vec::new();
        let mut gradients = Vec::new();
        for genome in &genomes {
            let mut gradient = vec![0.0; 3];
            let value = problem.evaluate_with(
                &Reals::from(genome.clone()),
                &mut genoxide::engine::Extras::with_gradient(&mut gradient),
            );
            fitness.push(Fitness::new(scale * value));
            gradients.extend(gradient.iter().map(|g| scale * g));
        }
        let evaluations = Evaluations::with_gradients(&fitness, &gradients, 3).unwrap();
        lbfgsb.tell_evaluations(&evaluations).unwrap();
        genomes
    };
    for _ in 0..20 {
        evaluate(&mut lbfgsb, 1.0);
    }
    assert!(lbfgsb.pairs() > 0);
    let _ = lbfgsb.ask();
    assert_eq!(lbfgsb.reevaluate(), Err(Error::ReevaluationOutOfTurn));
    evaluate(&mut lbfgsb, 1.0);
    let generation = lbfgsb.generation();
    // the function changes: twice the old one
    lbfgsb.reevaluate().unwrap();
    assert_eq!(lbfgsb.pairs(), 0);
    let genomes = evaluate(&mut lbfgsb, 2.0);
    assert_eq!(genomes[0][..], lbfgsb.population()[0].genome()[..]);
    assert_eq!(lbfgsb.generation(), generation);
    assert_eq!(lbfgsb.best_generation(), generation);
    assert_eq!(
        lbfgsb.best().unwrap().fitness(),
        lbfgsb.population()[0].fitness()
    );
    // the search goes on on the new function
    for _ in 0..200 {
        if lbfgsb.is_finished() {
            break;
        }
        evaluate(&mut lbfgsb, 2.0);
    }
    assert!(lbfgsb.is_finished());
    assert!(lbfgsb.best().unwrap().fitness().unwrap().score().unwrap() < 1e-10);
}

#[test]
fn seeded_runs_repeat_and_parallel_evaluation_changes_nothing() {
    let problem = Rosenbrock::new(6);
    let plain = move |x: &Reals| problem.evaluate(x);
    let run_once = |seed: u64, parallel: bool| {
        let lbfgsb = Lbfgsb::builder(problem.representation())
            .restarts(local::Restarts::Random { times: 2 })
            .minimize()
            .seed(seed)
            .build()
            .unwrap();
        let engine = Engine::new(lbfgsb, plain).stop_when(Stop::evaluations(20_000));
        #[cfg(feature = "parallel")]
        let engine = engine.parallel(parallel);
        #[cfg(not(feature = "parallel"))]
        let _ = parallel;
        let mut engine = engine;
        let outcome = engine.run().unwrap();
        (outcome.into_best(), engine.algorithm().evaluations())
    };
    assert_eq!(run_once(1, false), run_once(1, false));
    assert_eq!(run_once(1, false), run_once(1, true));
    assert_ne!(run_once(1, false).0, run_once(2, false).0);
}

#[test]
fn the_memory_can_change_during_a_run() {
    let problem = Rosenbrock::new(20);
    let lbfgsb = Lbfgsb::builder(problem.representation())
        .initial_genome(classic_start(20))
        .memory(3)
        .minimize()
        .build()
        .unwrap();
    let mut engine = Engine::new(lbfgsb, problem)
        .stop_when(Stop::evaluations(5_000))
        .control(|lbfgsb: &mut Lbfgsb, progress| {
            if progress.generation() == 20 {
                lbfgsb.set_memory(15)?;
            }
            Ok(())
        });
    let outcome = engine.run().unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert_eq!(engine.algorithm().memory(), 15);
    assert!(engine.algorithm().pairs() > 3);
    let mut lbfgsb = Lbfgsb::builder(problem.representation()).build().unwrap();
    assert!(lbfgsb.set_memory(0).is_err());
}

#[test]
fn settings_are_validated() {
    let real = || Real::uniform(2, -1.0..=1.0).unwrap();
    let setting = |result: Result<Lbfgsb>| match result {
        Err(Error::InvalidSetting { setting, .. }) => setting,
        other => panic!("{other:?}"),
    };
    assert_eq!(setting(Lbfgsb::builder(real()).memory(0).build()), "memory");
    assert_eq!(
        setting(Lbfgsb::builder(real()).gradient_tolerance(-1.0).build()),
        "gradient_tolerance"
    );
    assert_eq!(
        setting(Lbfgsb::builder(real()).function_tolerance(f64::NAN).build()),
        "function_tolerance"
    );
    assert_eq!(
        setting(Lbfgsb::builder(real()).max_line_search(0).build()),
        "max_line_search"
    );
    assert_eq!(
        setting(
            Lbfgsb::builder(real())
                .gradients(Gradients::Forward { step: Some(0.0) })
                .build()
        ),
        "gradients"
    );
    assert_eq!(
        setting(
            Lbfgsb::builder(real())
                .restarts(local::Restarts::Random { times: 0 })
                .build()
        ),
        "restarts"
    );
    assert!(matches!(
        Lbfgsb::builder(real())
            .initial_genome(Reals::from(vec![2.0, 0.0]))
            .build(),
        Err(Error::InvalidGenome { .. })
    ));
}

#[test]
fn a_run_as_in_python() {
    // python/tests/test_lbfgsb.py has the same run, evaluated in Rust, with the same results
    let problem = Rosenbrock::new(4);
    let lbfgsb = Lbfgsb::builder(problem.representation())
        .memory(5)
        .restarts(local::Restarts::Random { times: 2 })
        .minimize()
        .seed(5)
        .build()
        .unwrap();
    let (outcome, _) = run(lbfgsb, problem, 20_000);
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert_eq!((outcome.evaluations(), outcome.generations()), (271, 270));
    assert_eq!(outcome.best_fitness(), Fitness::new(2.059739817072154e-12));
}
