//! MMA and GCMMA: optima and KKT conditions on problems with known solutions, the CEC 2006
//! problems, the papers' test problems, bounds, restoration, reproducibility and the protocol.

use genoxide::algorithm::MmaBuilder;
use genoxide::algorithm::mma::{Convergence, Method};
use genoxide::constraint::Constrained;
use genoxide::engine::{Evaluations, Extras, Provided, Wanted};
use genoxide::prelude::*;
use genoxide::problems::{Problem, Sphere, cec2006, engineering};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

const METHODS: [Method; 2] = [Method::Mma, Method::Gcmma];

// minimize |x − (2, 1, −1)|² subject to x₀ + x₁ + x₂ ≤ 1, 0.8 − x₁ ≤ 0 and x₀ − x₁ − 3 ≤ 0. By
// hand: with x₁ = 0.8 on its limit, (x₀, x₂) is (2, −1) projected onto x₀ + x₂ ≤ 0.2, so x* =
// (1.6, 0.8, −1.4) with f = 0.36; stationarity 2(x − p) + λ₁ (1, 1, 1) + λ₂ (0, −1, 0) = 0 gives λ₁ = 0.8 and
// λ₂ = 0.4, and the third constraint is inactive (−2.2), with λ₃ = 0
fn projection(x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]) -> f64 {
    let p = [2.0, 1.0, -1.0];
    let mut value = 0.0;
    for j in 0..3 {
        value += (x[j] - p[j]) * (x[j] - p[j]);
        gradient[j] = 2.0 * (x[j] - p[j]);
    }
    g[0] = x[0] + x[1] + x[2] - 1.0;
    g[1] = 0.8 - x[1];
    g[2] = x[0] - x[1] - 3.0;
    jacobian.copy_from_slice(&[1.0, 1.0, 1.0, 0.0, -1.0, 0.0, 1.0, -1.0, 0.0]);
    value
}

fn mma(real: Real, method: Method) -> MmaBuilder {
    Mma::builder(real).method(method).minimize()
}

// the gradient of the Lagrangian f₀ + Σ λᵢ gᵢ at x, from the problem's derivatives
fn lagrangian_gradient(
    problem: impl Fn(&Reals, &mut [f64], &mut [f64], &mut [f64]) -> f64,
    x: &Reals,
    multipliers: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let (n, m) = (x.len(), multipliers.len());
    let (mut gradient, mut g, mut jacobian) = (vec![0.0; n], vec![0.0; m], vec![0.0; m * n]);
    problem(x, &mut gradient, &mut g, &mut jacobian);
    for (i, lambda) in multipliers.iter().enumerate() {
        for j in 0..n {
            gradient[j] += lambda * jacobian[i * n + j];
        }
    }
    (gradient, g)
}

#[test]
fn kkt_conditions_at_a_known_optimum() {
    for method in METHODS {
        let mma = mma(Real::uniform(3, -5.0..=5.0).unwrap(), method)
            .initial_genome(Reals::from(vec![-2.0, 3.0, 4.0]))
            .build()
            .unwrap();
        let mut engine = Engine::new(mma, Constrained::differentiable(3, projection))
            .stop_when(Stop::evaluations(1_000));
        let outcome = engine.run().unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Converged, "{method:?}");
        let mma = engine.algorithm();
        // the current point to 1e-8; the best, by score, only to about √ε, where f is flat
        let x = mma.population()[0].genome();
        for (xj, exact) in x.iter().zip([1.6, 0.8, -1.4]) {
            assert!((xj - exact).abs() < 1e-8, "{method:?}: {x:?}");
        }
        assert!(outcome.best_fitness().is_feasible());
        assert!((outcome.best_fitness().score().unwrap() - 0.36).abs() < 1e-12);
        // the multipliers: of the right sign, the two active constraints' as derived
        let multipliers = mma.multipliers();
        assert!(multipliers.iter().all(|&lambda| lambda >= 0.0));
        assert!((multipliers[0] - 0.8).abs() < 1e-6, "{multipliers:?}");
        assert!((multipliers[1] - 0.4).abs() < 1e-6, "{multipliers:?}");
        assert!(multipliers[2] < 1e-9, "{multipliers:?}");
        // stationarity, feasibility and complementarity at the current point
        let current = mma.population()[0].genome();
        let (gradient, g) = lagrangian_gradient(projection, current, multipliers);
        assert!(gradient.iter().all(|d| d.abs() < 1e-6), "{gradient:?}");
        assert!(g.iter().all(|&g| g < 1e-9));
        for (lambda, g) in multipliers.iter().zip(&g) {
            assert!((lambda * g).abs() < 1e-8);
        }
        assert!(mma.kkt_residual() <= 1e-8);
    }
}

#[test]
fn maximizing_is_minimizing_the_negated_score() {
    let negated = |x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]| {
        let value = projection(x, gradient, g, jacobian);
        gradient.iter_mut().for_each(|d| *d = -*d);
        -value
    };
    for method in METHODS {
        let start = Reals::from(vec![-2.0, 3.0, 4.0]);
        let real = Real::uniform(3, -5.0..=5.0).unwrap();
        let minimized = mma(real.clone(), method)
            .initial_genome(start.clone())
            .build()
            .unwrap();
        let maximized = Mma::builder(real)
            .method(method)
            .initial_genome(start)
            .maximize()
            .build()
            .unwrap();
        let mut first = Engine::new(minimized, Constrained::differentiable(3, projection))
            .stop_when(Stop::evaluations(1_000));
        let mut second = Engine::new(maximized, Constrained::differentiable(3, negated))
            .stop_when(Stop::evaluations(1_000));
        let (a, b) = (first.run().unwrap(), second.run().unwrap());
        assert_eq!(a.best_genome(), b.best_genome(), "{method:?}");
        assert_eq!(a.evaluations(), b.evaluations());
        assert_eq!(
            first.algorithm().multipliers(),
            second.algorithm().multipliers()
        );
        assert_eq!(
            b.best_fitness().score(),
            a.best_fitness().score().map(|s| -s)
        );
    }
}

// minimize Σ cⱼ/xⱼ subject to Σ xⱼ ≤ V: the Lagrange conditions cⱼ/xⱼ² = λ give xⱼ = √(cⱼ/λ),
// and the constraint λ = (Σ √cₖ)² / V², so xⱼ = V √cⱼ / Σ √cₖ while the bounds are inactive
fn volume(c: Vec<f64>, volume: f64) -> impl Fn(&Reals, &mut [f64], &mut [f64], &mut [f64]) -> f64 {
    move |x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]| {
        let (mut value, mut sum) = (0.0, 0.0);
        for j in 0..x.len() {
            value += c[j] / x[j];
            gradient[j] = -c[j] / (x[j] * x[j]);
            jacobian[j] = 1.0;
            sum += x[j];
        }
        g[0] = sum - volume;
        value
    }
}

fn costs(n: usize) -> Vec<f64> {
    (0..n).map(|j| 1.0 + (j % 9) as f64).collect()
}

#[test]
fn the_closed_form_optimum_of_many_variables() {
    let n = 1_000;
    let c = costs(n);
    let total: f64 = c.iter().map(|c| c.sqrt()).sum();
    let v = n as f64;
    for method in METHODS {
        let mma = mma(Real::uniform(n, 0.01..=10.0).unwrap(), method)
            .initial_genome(Reals::from(vec![0.5; n]))
            .build()
            .unwrap();
        let problem = Constrained::differentiable(1, volume(c.clone(), v));
        let mut engine = Engine::new(mma, problem).stop_when(Stop::evaluations(500));
        let outcome = engine.run().unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Converged, "{method:?}");
        for (j, xj) in outcome.best_genome().iter().enumerate() {
            let exact = v * c[j].sqrt() / total;
            assert!(
                (xj - exact).abs() <= 1e-7 * exact,
                "{method:?}: x{j} = {xj}, not {exact}"
            );
        }
        let lambda = total * total / (v * v);
        let multiplier = engine.algorithm().multipliers()[0];
        assert!((multiplier - lambda).abs() <= 1e-7 * lambda, "{multiplier}");
        assert!(outcome.best_fitness().is_feasible());
    }
}

// bounds that are active: the volume allows more than the upper bounds of the most expensive
// genes, which stay there; the others share the rest as √cⱼ
#[test]
fn bounds_are_never_violated_and_hold_at_the_optimum() {
    let n = 20;
    let c = costs(n);
    let outside = AtomicUsize::new(0);
    let real = Real::uniform(n, 0.5..=1.5).unwrap();
    let bounds = real.bounds().to_vec();
    let inner = volume(c.clone(), 22.0);
    let checked = Constrained::differentiable(
        1,
        |x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]| {
            if x.iter().zip(&bounds).any(|(xj, range)| !range.contains(xj)) {
                outside.fetch_add(1, Ordering::Relaxed);
            }
            inner(x, gradient, g, jacobian)
        },
    );
    for method in METHODS {
        let mma = mma(real.clone(), method)
            .initial_genome(Reals::from(vec![0.5; n]))
            .build()
            .unwrap();
        let mut engine = Engine::new(mma, checked).stop_when(Stop::evaluations(1_000));
        let outcome = engine.run().unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Converged);
        let x = outcome.best_genome();
        // the genes the volume would take above 1.5 are at 1.5, found one at a time (the most
        // expensive first); the others share the volume left as √cⱼ (none reaches 0.5)
        let mut at_bound: Vec<usize> = Vec::new();
        let (free, left, roots) = loop {
            let free: Vec<usize> = (0..n).filter(|j| !at_bound.contains(j)).collect();
            let left = 22.0 - 1.5 * at_bound.len() as f64;
            let roots: f64 = free.iter().map(|&j| c[j].sqrt()).sum();
            let above: Vec<usize> = free
                .iter()
                .copied()
                .filter(|&j| left * c[j].sqrt() / roots > 1.5)
                .collect();
            if above.is_empty() {
                break (free, left, roots);
            }
            at_bound.extend(above);
        };
        assert_eq!(at_bound.len(), 4, "c = 8 and 9");
        for &j in &at_bound {
            assert_eq!(x[j], 1.5, "{method:?}: x{j}");
        }
        for &j in &free {
            let exact = left * c[j].sqrt() / roots;
            assert!(
                (x[j] - exact).abs() < 1e-7,
                "{method:?}: x{j} = {}, not {exact}",
                x[j]
            );
        }
    }
    assert_eq!(outside.load(Ordering::Relaxed), 0);
}

// GCMMA, which converges from any start; MMA's approximations of a function whose gradient
// vanishes inside the bounds have almost no curvature, and with the asymptotes at their nearest it
// can cycle around such a minimum
#[test]
fn a_problem_without_constraints_is_bounds_only() {
    let mma = Mma::builder(Sphere::new(10).representation())
        .method(Method::Gcmma)
        .minimize()
        .seed(3)
        .build()
        .unwrap();
    let mut engine = Engine::new(mma, Sphere::new(10)).stop_when(Stop::evaluations(500));
    let outcome = engine.run().unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert!(outcome.best_fitness().score().unwrap() < 1e-15);
    assert!(engine.algorithm().multipliers().is_empty());
}

// a problem's constraint values with its gradient and Jacobian by central differences: the CEC
// 2006 problems give values but not their derivatives
struct Differences<P>(P);

impl<P> Differences<P>
where
    P: FitnessFunction<Reals, Output = (f64, f64)>,
{
    // the score and the constraint values
    fn values(&self, x: &Reals) -> (f64, Vec<f64>) {
        let mut g = vec![0.0; self.0.provides().inequalities];
        let (score, _) = self
            .0
            .evaluate_with(x, &mut Extras::new(None, Some(&mut g), None));
        (score, g)
    }
}

impl<P> FitnessFunction<Reals> for Differences<P>
where
    P: FitnessFunction<Reals, Output = (f64, f64)>,
{
    type Output = (f64, f64);

    fn evaluate(&self, x: &Reals) -> (f64, f64) {
        self.0.evaluate(x)
    }

    fn provides(&self) -> Provided {
        self.0.provides().with_gradient().with_constraint_jacobian()
    }

    fn evaluate_with(&self, x: &Reals, extras: &mut Extras<'_>) -> (f64, f64) {
        let (n, m) = (x.len(), self.0.provides().inequalities);
        let (mut gradient, mut jacobian) = (vec![0.0; n], vec![0.0; m * n]);
        let mut point = x.clone();
        for j in 0..n {
            let h = 6e-6 * x[j].abs().max(1.0);
            point[j] = x[j] + h;
            let (above, g_above) = self.values(&point);
            point[j] = x[j] - h;
            let (below, g_below) = self.values(&point);
            point[j] = x[j];
            gradient[j] = (above - below) / (2.0 * h);
            for i in 0..m {
                jacobian[i * n + j] = (g_above[i] - g_below[i]) / (2.0 * h);
            }
        }
        if let Some(out) = extras.gradient() {
            out.copy_from_slice(&gradient);
        }
        if let Some(out) = extras.constraint_jacobian() {
            out.copy_from_slice(&jacobian);
        }
        let mut inner = Extras::new(None, extras.inequalities(), None);
        self.0.evaluate_with(x, &mut inner)
    }
}

// the report's f*, to 1e-6 relative, feasible, from random starts
fn reaches_the_report<P>(problem: P, cost: f64, seeds: std::ops::Range<u64>)
where
    P: Problem<Representation = Real> + FitnessFunction<Reals, Output = (f64, f64)> + Copy,
{
    let optimum = problem.optimum().expect("known").value();
    for method in METHODS {
        for seed in seeds.clone() {
            let mma = Mma::builder(problem.representation())
                .method(method)
                .constraint_cost(cost)
                .minimize()
                .seed(seed)
                .build()
                .unwrap();
            let outcome = Engine::new(mma, Differences(problem))
                .stop_when(Stop::evaluations(3_000))
                .run()
                .unwrap();
            let best = outcome.best_fitness();
            let score = best.score().unwrap();
            assert!(
                best.is_feasible(),
                "{}, {method:?}, seed {seed}: {best:?}",
                problem.name()
            );
            let error = (score - optimum).abs() / optimum.abs();
            assert!(
                error < 1e-6,
                "{}, {method:?}, seed {seed}: {score}",
                problem.name()
            );
            assert_eq!(outcome.stop_reason(), StopReason::Converged);
        }
    }
}

#[test]
fn cec2006_problems_with_inequalities_reach_the_reports_optimum() {
    // g06's multipliers at the optimum are about 1097 and 1230, above the default cost
    reaches_the_report(cec2006::G06, 1e6, 1..4);
    reaches_the_report(cec2006::G07, 1e3, 1..4);
    reaches_the_report(cec2006::G09, 1e3, 1..4);
}

// Svanberg (1987, section 6, test problem 1): the cantilever beam from x = 5, to the minimum of
// eq. 22, x = (6.016, 5.309, 4.494, 3.502, 2.153) with weight 1.340
#[test]
fn the_cantilever_beam_of_the_1987_paper() {
    const A: [f64; 5] = [61.0, 37.0, 19.0, 7.0, 1.0];
    let beam = |x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]| {
        let mut deflection = 0.0;
        for j in 0..5 {
            gradient[j] = 0.0624;
            deflection += A[j] / (x[j] * x[j] * x[j]);
            jacobian[j] = -3.0 * A[j] / (x[j] * x[j] * x[j] * x[j]);
        }
        g[0] = deflection - 1.0;
        0.0624 * x.iter().sum::<f64>()
    };
    let problem = engineering::CantileverBeam;
    let optimum = problem.optimum().expect("known");
    // GCMMA in the problem's box, [0.01, 100]; MMA, which isn't globally convergent, in a box
    // scaled as Svanberg's notes advise (section 2), where it converges too
    for (method, real) in [
        (Method::Gcmma, problem.representation()),
        (Method::Mma, Real::uniform(5, 1.0..=10.0).unwrap()),
    ] {
        let mma = mma(real, method)
            .initial_genome(Reals::from(vec![5.0; 5]))
            .build()
            .unwrap();
        let outcome = Engine::new(mma, Constrained::differentiable(1, beam))
            .stop_when(Stop::evaluations(500))
            .run()
            .unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Converged);
        let weight = outcome.best_fitness().score().unwrap();
        assert!(
            (weight - optimum.value()).abs() < 1e-9,
            "{method:?}: {weight}"
        );
        assert_eq!(format!("{weight:.3}"), "1.340");
        // the paper's x, printed to 3 decimals: x₄ = 3.50147 prints as 3.501 (the paper has
        // 3.502), and the minimum of genoxide's problem, derived exactly
        let x = outcome.best_genome();
        for (xj, printed) in x.iter().zip([6.016, 5.309, 4.494, 3.502, 2.153]) {
            assert!((xj - printed).abs() < 1e-3, "{method:?}: {x:?}");
        }
        for (xj, exact) in x.iter().zip(optimum.solutions()[0].iter()) {
            assert!((xj - exact).abs() < 1e-6, "{method:?}: {x:?}");
        }
        assert_eq!(problem.evaluate(x).1, 0.0, "feasible");
    }
}

// GCMMA's iterates (the current points) are feasible once one is, and each is better than the
// last (Svanberg 2002, lemma 7.6)
#[test]
fn gcmma_iterates_improve_and_stay_feasible() {
    let mma = mma(Real::uniform(3, -5.0..=5.0).unwrap(), Method::Gcmma)
        .initial_genome(Reals::from(vec![0.0, 0.9, 0.0]))
        .build()
        .unwrap();
    let mut scores = Vec::new();
    let mut engine = Engine::new(mma, Constrained::differentiable(3, projection))
        .stop_when(Stop::evaluations(1_000))
        .on_generation(|snapshot| {
            let fitness = snapshot.population()[0].fitness().unwrap();
            scores.push((fitness.score().unwrap(), fitness.violation()));
        });
    engine.run().unwrap();
    drop(engine);
    // to rounding: an approximation that is conservative to the last bit can still put a point
    // 1e-16 beyond an active constraint
    assert!(
        scores.iter().all(|&(_, violation)| violation <= 1e-14),
        "{scores:?}"
    );
    assert!(
        scores.windows(2).all(|w| w[1].0 <= w[0].0 + 1e-12),
        "{scores:?}"
    );
    assert!(scores.len() > 5);
}

// maximize Σ wⱼ xⱼ subject to Σ xⱼ² ≤ 1 and x₀ ≤ 0.3: the active constraints approached from
// either side, so a converged point can be infeasible by rounding; the restoration step moves it
// back
#[test]
fn the_restoration_step_ends_feasible() {
    let run = |seed: u64, restoration: bool| {
        let n = 2 + (seed % 5) as usize;
        let w: Vec<f64> = (0..n)
            .map(|j| 1.0 + j as f64 * 0.37 + seed as f64 * 0.01)
            .collect();
        let problem = Constrained::differentiable(
            2,
            move |x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]| {
                let n = x.len();
                let (mut value, mut squares) = (0.0, 0.0);
                for j in 0..n {
                    value += w[j] * x[j];
                    gradient[j] = w[j];
                    squares += x[j] * x[j];
                    jacobian[j] = 2.0 * x[j];
                }
                g[0] = squares - 1.0;
                g[1] = x[0] - 0.3;
                jacobian[n] = 1.0;
                value
            },
        );
        let mma = Mma::builder(Real::uniform(n, -2.0..=2.0).unwrap())
            .restoration(restoration)
            .maximize()
            .seed(seed)
            .build()
            .unwrap();
        let outcome = Engine::new(mma, problem)
            .stop_when(Stop::evaluations(2_000))
            .run()
            .unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Converged);
        outcome
    };
    let (mut infeasible, mut extra) = (0, 0);
    for seed in 0..40 {
        let without = run(seed, false);
        let with = run(seed, true);
        if !without.best_fitness().is_feasible() {
            infeasible += 1;
            extra += with.evaluations() - without.evaluations();
            // at most the change of rounding in the score
            let (a, b) = (
                without.best_fitness().score().unwrap(),
                with.best_fitness().score().unwrap(),
            );
            assert!((a - b).abs() <= 1e-9 * a.abs(), "seed {seed}: {a} and {b}");
        }
        assert!(with.best_fitness().is_feasible(), "seed {seed}");
    }
    assert!(
        infeasible > 0,
        "some runs end infeasible without the restoration"
    );
    assert!(extra <= 8 * infeasible);
}

#[test]
fn seeded_runs_are_reproducible() {
    let run = |seed: u64| {
        let mma = mma(Real::uniform(3, -5.0..=5.0).unwrap(), Method::Mma)
            .seed(seed)
            .build()
            .unwrap();
        let mut engine = Engine::new(mma, Constrained::differentiable(3, projection))
            .stop_when(Stop::generations(4));
        engine.run().unwrap();
        engine.algorithm().population()[0].genome().clone()
    };
    assert_eq!(run(5), run(5));
    assert_ne!(run(5), run(6));
}

// a result fixed to the bit: sums in a fixed order, divisions and square roots only, so every
// platform gives these bits
#[test]
fn a_portable_result() {
    let n = 50;
    let mma = mma(Real::uniform(n, 0.01..=10.0).unwrap(), Method::Gcmma)
        .initial_genome(Reals::from(vec![0.5; n]))
        .build()
        .unwrap();
    let problem = Constrained::differentiable(1, volume(costs(n), n as f64));
    let mut engine = Engine::new(mma, problem).stop_when(Stop::generations(12));
    let outcome = engine.run().unwrap();
    let x = engine.algorithm().population()[0].genome();
    let bits: Vec<u64> = [x[0], x[17], x[49], outcome.best_fitness().score().unwrap()]
        .iter()
        .map(|v| v.to_bits())
        .collect();
    let expected = [
        0x3fdf_1ae5_7031_13a6,
        0x3ff6_7451_fe12_77ed,
        0x3ff1_2760_dad2_0b0d,
        0x406b_84d0_583e_68b4,
    ];
    assert_eq!(bits, expected, "{bits:#x?}");
}

#[test]
fn convergence_by_either_criterion() {
    let run = |kkt: f64, step: f64| {
        let mma = mma(Real::uniform(3, -5.0..=5.0).unwrap(), Method::Mma)
            .initial_genome(Reals::from(vec![-2.0, 3.0, 4.0]))
            .kkt_tolerance(kkt)
            .step_tolerance(step)
            .build()
            .unwrap();
        let mut engine = Engine::new(mma, Constrained::differentiable(3, projection))
            .stop_when(Stop::evaluations(1_000));
        let outcome = engine.run().unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Converged);
        assert!(engine.algorithm().is_finished());
        engine.algorithm().converged()
    };
    assert_eq!(run(1e-6, 0.0), Some(Convergence::Kkt));
    assert_eq!(run(0.0, 1e-6), Some(Convergence::Step));
}

// the score scaled during the run: after a re-evaluation, the best is the current point with its
// new score, and the run goes on to the same optimum
#[test]
fn reevaluation_after_the_function_changes() {
    let scale = AtomicU64::new(1.0f64.to_bits());
    let problem = Constrained::differentiable(
        3,
        |x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]| {
            let factor = f64::from_bits(scale.load(Ordering::Relaxed));
            let value = projection(x, gradient, g, jacobian);
            gradient.iter_mut().for_each(|d| *d *= factor);
            factor * value
        },
    );
    let mma = mma(Real::uniform(3, -5.0..=5.0).unwrap(), Method::Gcmma)
        .initial_genome(Reals::from(vec![-2.0, 3.0, 4.0]))
        .build()
        .unwrap();
    let mut seen = None;
    let mut engine = Engine::new(mma, problem)
        .stop_when(Stop::evaluations(1_000))
        .control(|mma, progress| {
            if progress.generation() == 3 && seen.is_none() {
                scale.store(10.0f64.to_bits(), Ordering::Relaxed);
                let evaluations = mma.evaluations();
                mma.reevaluate()?;
                seen = Some(evaluations);
            }
            Ok(())
        });
    let outcome = engine.run().unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert!((outcome.best_fitness().score().unwrap() - 3.6).abs() < 1e-10);
    let mma = engine.algorithm();
    assert!(
        (mma.multipliers()[0] - 8.0).abs() < 1e-5,
        "{:?}",
        mma.multipliers()
    );

    // by hand: the re-evaluation asks the current point, and its tell isn't a generation
    let mut mma = self::mma(Real::uniform(3, -5.0..=5.0).unwrap(), Method::Mma)
        .initial_genome(Reals::from(vec![-2.0, 3.0, 4.0]))
        .build()
        .unwrap();
    mma.prepare(Constrained::differentiable(3, projection).provides())
        .unwrap();
    tell_projection(&mut mma, 1.0);
    tell_projection(&mut mma, 1.0);
    let (generation, evaluations) = (mma.generation(), mma.evaluations());
    mma.reevaluate().unwrap();
    let current = mma.population()[0].genome().clone();
    assert_eq!(mma.ask().get(0), Some(&current));
    tell_projection(&mut mma, 2.0);
    assert_eq!(mma.generation(), generation);
    assert_eq!(mma.evaluations(), evaluations + 1);
    assert_eq!(mma.best().unwrap().genome(), &current);
    assert_eq!(mma.best_generation(), generation);
}

// asks, evaluates the projection problem with its score scaled by `factor`, and tells
fn tell_projection(mma: &mut Mma, factor: f64) {
    let x = mma.ask().get(0).unwrap().clone();
    let (mut gradient, mut g, mut jacobian) = ([0.0; 3], [0.0; 3], [0.0; 9]);
    let value = factor * projection(&x, &mut gradient, &mut g, &mut jacobian);
    gradient.iter_mut().for_each(|d| *d *= factor);
    let violation: f64 = g.iter().map(|g| g.max(0.0)).sum();
    let fitness = [Fitness::constrained(value, violation)];
    let evaluations =
        Evaluations::with_extras(&fitness, Some(&gradient), Some(&g), Some(&jacobian), 3, 3)
            .unwrap();
    mma.tell_evaluations(&evaluations).unwrap();
}

#[test]
fn the_engine_checks_what_the_fitness_function_provides() {
    let build = || {
        mma(Real::uniform(3, -5.0..=5.0).unwrap(), Method::Mma)
            .seed(1)
            .build()
            .unwrap()
    };
    // no gradient
    let plain = |x: &Reals| x.iter().map(|xi| xi * xi).sum::<f64>();
    let error = Engine::new(build(), plain)
        .stop_when(Stop::evaluations(10))
        .run()
        .unwrap_err();
    assert!(
        matches!(
            error,
            Error::InvalidSetting {
                setting: "fitness",
                ..
            }
        ),
        "{error:?}"
    );
    // constraint values without their Jacobian
    let values = Constrained::new(1, |x: &Reals, g: &mut [f64]| {
        g[0] = x[0];
        x[1]
    });
    let error = Engine::new(build(), values)
        .stop_when(Stop::evaluations(10))
        .run()
        .unwrap_err();
    assert!(
        matches!(
            error,
            Error::InvalidSetting {
                setting: "fitness",
                ..
            }
        ),
        "{error:?}"
    );
    // a violation without constraint values
    let hidden = genoxide::gradient::Differentiable(|x: &Reals, gradient: &mut [f64]| {
        gradient.copy_from_slice(&[1.0, 0.0, 0.0]);
        (x[0], 1.0)
    });
    let error = Engine::new(build(), hidden)
        .stop_when(Stop::evaluations(10))
        .run()
        .unwrap_err();
    assert!(
        matches!(
            error,
            Error::InvalidSetting {
                setting: "fitness",
                ..
            }
        ),
        "{error:?}"
    );
    // an invalid initial point
    let invalid = genoxide::gradient::Differentiable(|_: &Reals, _: &mut [f64]| f64::NAN);
    let error = Engine::new(build(), invalid)
        .stop_when(Stop::evaluations(10))
        .run()
        .unwrap_err();
    assert!(
        matches!(
            error,
            Error::InvalidSetting {
                setting: "initial_genome",
                ..
            }
        ),
        "{error:?}"
    );
}

#[test]
fn the_ask_tell_protocol() {
    let mut mma = mma(Real::uniform(3, -5.0..=5.0).unwrap(), Method::Mma)
        .initial_genome(Reals::from(vec![-2.0, 3.0, 4.0]))
        .build()
        .unwrap();
    assert_eq!(mma.tell(&[]).unwrap_err(), Error::TellWithoutAsk);
    mma.prepare(Constrained::differentiable(3, projection).provides())
        .unwrap();
    assert_eq!(
        mma.wants(),
        Wanted::GRADIENT
            .with_inequalities()
            .with_constraint_jacobian()
    );
    let first = mma.ask().get(0).unwrap().clone();
    assert_eq!(
        mma.ask().get(0),
        Some(&first),
        "asking again gives the same"
    );
    assert_eq!(mma.ask().len(), 1);
    // a plain tell lacks the gradient, and a count other than 1 is an error; nothing changes
    let fitness = [Fitness::new(1.0)];
    assert!(matches!(
        mma.tell(&fitness).unwrap_err(),
        Error::InvalidSetting { .. }
    ));
    assert_eq!(
        mma.tell(&[fitness[0]; 2]).unwrap_err(),
        Error::FitnessCount {
            expected: 1,
            got: 2
        }
    );
    assert_eq!(mma.evaluations(), 0);
    assert_eq!(mma.reevaluate().unwrap_err(), Error::ReevaluationOutOfTurn);
    tell_projection(&mut mma, 1.0);
    assert_eq!((mma.generation(), mma.evaluations()), (0, 1));
    assert!(mma.best().is_some());
    let trial = mma.ask().get(0).unwrap().clone();
    assert_ne!(trial, first);
    tell_projection(&mut mma, 1.0);
    assert_eq!((mma.generation(), mma.evaluations()), (1, 2));
    assert_eq!(mma.iterations(), 1);
    assert_eq!(mma.population()[0].genome(), &trial);
    // another run's fitness function with another number of constraints
    assert!(mma.prepare(Provided::GRADIENT).is_err());
    assert!(
        mma.prepare(
            Provided::GRADIENT
                .with_inequalities(3)
                .with_constraint_jacobian()
        )
        .is_ok()
    );
}

#[test]
fn builder_validation() {
    let real = || Real::uniform(2, 0.0..=1.0).unwrap();
    let setting = |builder: MmaBuilder| match builder.build() {
        Err(Error::InvalidSetting { setting, .. }) => setting,
        other => panic!("{other:?}"),
    };
    assert_eq!(
        setting(Mma::builder(real()).asymptote_initial(0.0)),
        "asymptote_initial"
    );
    assert_eq!(
        setting(Mma::builder(real()).asymptote_initial(f64::NAN)),
        "asymptote_initial"
    );
    assert_eq!(
        setting(Mma::builder(real()).asymptote_decrease(0.0)),
        "asymptote_decrease"
    );
    assert_eq!(
        setting(Mma::builder(real()).asymptote_decrease(1.5)),
        "asymptote_decrease"
    );
    assert_eq!(
        setting(Mma::builder(real()).asymptote_increase(0.9)),
        "asymptote_increase"
    );
    assert_eq!(
        setting(Mma::builder(real()).asymptote_increase(f64::INFINITY)),
        "asymptote_increase"
    );
    assert_eq!(setting(Mma::builder(real()).move_limit(0.0)), "move_limit");
    assert_eq!(
        setting(Mma::builder(real()).constraint_cost(-1.0)),
        "constraint_cost"
    );
    assert_eq!(
        setting(Mma::builder(real()).kkt_tolerance(-1.0)),
        "kkt_tolerance"
    );
    assert_eq!(
        setting(Mma::builder(real()).step_tolerance(f64::NAN)),
        "step_tolerance"
    );
    let fixed = Real::new([1.0..=1.0, 2.0..=2.0]).unwrap();
    assert_eq!(setting(Mma::builder(fixed)), "real");
    assert!(matches!(
        Mma::builder(real())
            .initial_genome(Reals::from(vec![2.0, 0.0]))
            .build(),
        Err(Error::InvalidGenome { .. })
    ));
    let built = Mma::builder(real())
        .asymptote_decrease(1.0)
        .asymptote_increase(1.0)
        .kkt_tolerance(0.0)
        .step_tolerance(0.0)
        .seed(4)
        .build()
        .unwrap();
    assert_eq!(built.seed(), 4);
    assert_eq!(built.method(), Method::Mma);
    assert_eq!(built.objective(), Objective::Maximize);
    assert!(built.kkt_residual().is_nan());
    assert!(built.lower_asymptotes().is_empty());
}

// fixed genes stay as they are, and the others reach the optimum
#[test]
fn fixed_genes_stay() {
    let real = Real::new([-5.0..=5.0, 0.8..=0.8, -5.0..=5.0]).unwrap();
    for method in METHODS {
        let mma = mma(real.clone(), method)
            .initial_genome(Reals::from(vec![0.0, 0.8, 0.0]))
            .build()
            .unwrap();
        let outcome = Engine::new(mma, Constrained::differentiable(3, projection))
            .stop_when(Stop::evaluations(1_000))
            .run()
            .unwrap();
        let x = outcome.best_genome();
        assert_eq!(x[1], 0.8);
        assert!(
            (x[0] - 1.6).abs() < 1e-8 && (x[2] + 1.4).abs() < 1e-8,
            "{x:?}"
        );
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_sums_give_the_same_bits() {
    let n = 10_000;
    let run = |parallel: bool| {
        let mma = mma(Real::uniform(n, 0.01..=10.0).unwrap(), Method::Mma)
            .initial_genome(Reals::from(vec![0.5; n]))
            .parallel_sums(parallel)
            .build()
            .unwrap();
        let problem = Constrained::differentiable(1, volume(costs(n), n as f64));
        let mut engine = Engine::new(mma, problem).stop_when(Stop::generations(8));
        let outcome = engine.run().unwrap();
        (
            outcome.best().clone(),
            outcome.evaluations(),
            engine.algorithm().multipliers().to_vec(),
            engine.algorithm().population()[0].genome().clone(),
        )
    };
    assert_eq!(run(false), run(true));
}

// Svanberg (2002, section 8.1): the matrices S, P and Q of n × n, with αᵢⱼ = (i + j − 2) / (2n − 2)
// for i, j from 1: sᵢⱼ = (2 + sin 4παᵢⱼ) / ((1 + |i − j|) ln n), pᵢⱼ = (1 + 2αᵢⱼ) / (…) and
// qᵢⱼ = (3 − 2αᵢⱼ) / (…)
fn svanberg_matrices(n: usize) -> [Vec<f64>; 3] {
    let ln = (n as f64).ln();
    let mut matrices = [vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]];
    for i in 0..n {
        for j in 0..n {
            let alpha = (i + j) as f64 / (2 * n - 2) as f64;
            let divisor = (1.0 + i.abs_diff(j) as f64) * ln;
            let sine = genoxide::math::sin(4.0 * std::f64::consts::PI * alpha);
            matrices[0][i * n + j] = (2.0 + sine) / divisor;
            matrices[1][i * n + j] = (1.0 + 2.0 * alpha) / divisor;
            matrices[2][i * n + j] = (3.0 - 2.0 * alpha) / divisor;
        }
    }
    matrices
}

// xᵀ A x, with its gradient 2 A x into `gradient`, for a symmetric A
fn quadratic(a: &[f64], x: &[f64], gradient: &mut [f64]) -> f64 {
    let n = x.len();
    let mut value = 0.0;
    for (i, row) in a.chunks_exact(n).enumerate() {
        let product: f64 = row.iter().zip(x).map(|(a, x)| a * x).sum();
        gradient[i] = 2.0 * product;
        value += x[i] * product;
    }
    value
}

// Svanberg (2002, section 8.2, eq. 8.1), problem 1: minimize xᵀSx subject to n/2 − xᵀPx ≤ 0 and
// n/2 − xᵀQx ≤ 0 in [−1, 1]ⁿ from x = 0.5; table 8.1 for n = 1000: the objective 260.85, 184
// variables at a bound, and the multipliers 0.138 and 0.451. The paper's CCSA method is its
// example 5.4, whose approximations and curvature updates differ from the 2007 notes' that
// genoxide follows, so the iterations differ (166 outer and 334 inner here, 177 and 209 there):
// the paper's test stops at the mean square of eq. 8.4, 1e-10
#[test]
fn svanbergs_problem_1() {
    let n = 1_000;
    let [s, p, q] = svanberg_matrices(n);
    let half = n as f64 / 2.0;
    let problem = Constrained::differentiable(
        2,
        |x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]| {
            let value = quadratic(&s, x, gradient);
            let (first, second) = jacobian.split_at_mut(n);
            g[0] = half - quadratic(&p, x, first);
            g[1] = half - quadratic(&q, x, second);
            jacobian.iter_mut().for_each(|d| *d = -*d);
            value
        },
    );
    let mma = mma(Real::uniform(n, -1.0..=1.0).unwrap(), Method::Gcmma)
        .initial_genome(Reals::from(vec![0.5; n]))
        .kkt_tolerance(1e-5)
        .step_tolerance(0.0)
        .build()
        .unwrap();
    let mut engine = Engine::new(mma, problem).stop_when(Stop::evaluations(2_000));
    let outcome = engine.run().unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    let mma = engine.algorithm();
    assert_eq!(
        format!("{:.2}", outcome.best_fitness().score().unwrap()),
        "260.85"
    );
    let at_bounds = outcome
        .best_genome()
        .iter()
        .filter(|x| x.abs() == 1.0)
        .count();
    assert_eq!(at_bounds, 184);
    let multipliers: Vec<String> = mma
        .multipliers()
        .iter()
        .map(|l| format!("{l:.3}"))
        .collect();
    assert_eq!(multipliers, ["0.138", "0.451"]);
    assert_eq!((mma.iterations(), mma.inner_iterations()), (166, 334));
}
