//! First-order methods: the update rules against hand-computed steps from the papers' formulas,
//! convergence on smooth problems, bounds, invalid points, re-evaluation, restarts and
//! reproducibility.

use genoxide::algorithm::first_order::{Convergence, Step};
use genoxide::engine::{Evaluations, FitnessFunction, Provided, Wanted};
use genoxide::gradient::{Differentiable, Gradients};
use genoxide::prelude::*;
use genoxide::problems::{AxisParallelEllipsoid, Problem, Rosenbrock, Sphere};

// a first-order method on `real` from `start`, with supplied gradients, minimizing
fn method(real: Real, step: Step, start: &[f64]) -> FirstOrder {
    FirstOrder::builder(real)
        .step(step)
        .gradients(Gradients::Supplied)
        .initial_genome(Reals::from(start.to_vec()))
        .minimize()
        .seed(1)
        .build()
        .unwrap()
}

// asks for the point, and tells `f`'s value and gradient there; returns the point
fn step_by_hand(algorithm: &mut FirstOrder, f: impl Fn(&[f64], &mut [f64]) -> f64) -> Vec<f64> {
    let x = algorithm.ask().get(0).unwrap().to_vec();
    let mut gradient = vec![0.0; x.len()];
    let value = f(&x, &mut gradient);
    let fitness = [Fitness::new(value)];
    let evaluations = Evaluations::with_gradients(&fitness, &gradient, x.len()).unwrap();
    algorithm.tell_evaluations(&evaluations).unwrap();
    x
}

// the linear function 3 x₀ − 0.5 x₁ + 0.25 x₂: its gradient is the same everywhere
fn linear(x: &[f64], gradient: &mut [f64]) -> f64 {
    gradient.copy_from_slice(&LINEAR);
    LINEAR.iter().zip(x).map(|(c, x)| c * x).sum()
}

const LINEAR: [f64; 3] = [3.0, -0.5, 0.25];

fn wide(genes: usize) -> Real {
    Real::uniform(genes, -100.0..=100.0).unwrap()
}

// Kingma and Ba (2015), Algorithm 1, written out
struct Adam {
    alpha: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    // AdamW's decoupled weight decay, Loshchilov and Hutter (2019), Algorithm 2, line 12
    lambda: f64,
    m: Vec<f64>,
    v: Vec<f64>,
    t: i32,
}

impl Adam {
    fn new(alpha: f64, lambda: f64, genes: usize) -> Self {
        Self {
            alpha,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            lambda,
            m: vec![0.0; genes],
            v: vec![0.0; genes],
            t: 0,
        }
    }

    fn step(&mut self, theta: &mut [f64], g: &[f64]) {
        self.t += 1;
        let (beta1_t, beta2_t) = (self.beta1.powi(self.t), self.beta2.powi(self.t));
        for i in 0..theta.len() {
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g[i];
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g[i] * g[i];
            let m_hat = self.m[i] / (1.0 - beta1_t);
            let v_hat = self.v[i] / (1.0 - beta2_t);
            theta[i] -= self.alpha * m_hat / (v_hat.sqrt() + self.epsilon) + self.lambda * theta[i];
        }
    }
}

#[test]
fn adam_follows_algorithm_1() {
    let start = [1.0, -2.0, 0.5];
    let mut adam = method(wide(3), Step::adam(0.01), &start);
    let mut reference = Adam::new(0.01, 0.0, 3);
    let mut theta = start.to_vec();
    for t in 0..4 {
        let x = step_by_hand(&mut adam, linear);
        // β₁ᵗ and β₂ᵗ as powers here, as products in the method: the same to the last bits
        // for a few steps
        for (a, b) in x.iter().zip(&theta) {
            assert!(
                (a - b).abs() <= 1e-15 * b.abs().max(1.0),
                "step {t}: {x:?} {theta:?}"
            );
        }
        reference.step(&mut theta, &LINEAR);
    }
    // the first step: m̂ = g and v̂ = g² after the corrections, so each gene moves by
    // α g / (|g| + ε), about α against the sign of its gradient
    let mut first = method(wide(3), Step::adam(0.01), &start);
    step_by_hand(&mut first, linear);
    let x1 = step_by_hand(&mut first, linear);
    for ((x1, x0), g) in x1.iter().zip(start).zip(LINEAR) {
        let expected = x0 - 0.01 * g / (g.abs() + 1e-8);
        assert!((x1 - expected).abs() < 1e-15, "{x1} {expected}");
    }
    assert_eq!(first.steps(), 1);
    assert_eq!(first.iterations(), 1);
}

#[test]
fn adam_steps_are_the_papers_to_the_bit() {
    // the second step, with the corrections 1 − β₁² and 1 − β₂² as the method computes them
    let start = [1.0, -2.0, 0.5];
    let mut adam = method(wide(3), Step::adam(0.01), &start);
    step_by_hand(&mut adam, linear);
    step_by_hand(&mut adam, linear);
    let x2 = step_by_hand(&mut adam, linear);
    let (beta1, beta2): (f64, f64) = (0.9, 0.999);
    for (i, &g) in LINEAR.iter().enumerate() {
        let mut x = start[i];
        let (mut m, mut v) = (0.0, 0.0);
        let (mut p1, mut p2) = (1.0, 1.0);
        for _ in 0..2 {
            p1 *= beta1;
            p2 *= beta2;
            m = beta1 * m + (1.0 - beta1) * g;
            v = beta2 * v + (1.0 - beta2) * g * g;
            let m_hat = m / (1.0 - p1);
            let v_hat = v / (1.0 - p2);
            x -= 0.01 * m_hat / (v_hat.sqrt() + 1e-8);
        }
        assert_eq!(x2[i].to_bits(), x.to_bits(), "gene {i}");
    }
}

#[test]
fn adamw_decouples_the_weight_decay() {
    let start = [1.0, -2.0, 0.5];
    let lambda = 0.01;
    let mut adamw = method(wide(3), Step::adamw(0.01, lambda), &start);
    let mut reference = Adam::new(0.01, lambda, 3);
    let mut theta = start.to_vec();
    for t in 0..5 {
        let x = step_by_hand(&mut adamw, linear);
        for (a, b) in x.iter().zip(&theta) {
            assert!(
                (a - b).abs() <= 1e-15 * b.abs().max(1.0),
                "step {t}: {x:?} {theta:?}"
            );
        }
        reference.step(&mut theta, &LINEAR);
    }
    // the first step by hand: x₁ = x₀ − (α g / (|g| + ε) + λ x₀)
    let mut first = method(wide(3), Step::adamw(0.01, lambda), &start);
    step_by_hand(&mut first, linear);
    let x1 = step_by_hand(&mut first, linear);
    for ((x1, x0), g) in x1.iter().zip(start).zip(LINEAR) {
        let expected = x0 - (0.01 * g / (g.abs() + 1e-8) + lambda * x0);
        assert!((x1 - expected).abs() < 1e-15, "{x1} {expected}");
    }

    // Adam with the decay in the gradient (L2 regularization, f + λ/2 ‖x‖²) is another method:
    // its decay is divided by √v̂ with the gradient's, so it barely acts where the gradient is
    // large
    let with_l2 = |x: &[f64], gradient: &mut [f64]| {
        let value = linear(x, gradient);
        for (g, xi) in gradient.iter_mut().zip(x) {
            *g += lambda * xi;
        }
        value + 0.5 * lambda * x.iter().map(|xi| xi * xi).sum::<f64>()
    };
    let mut l2 = method(wide(3), Step::adam(0.01), &start);
    let mut decoupled = method(wide(3), Step::adamw(0.01, lambda), &start);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for _ in 0..3 {
        a = step_by_hand(&mut l2, with_l2);
        b = step_by_hand(&mut decoupled, linear);
    }
    for i in 0..3 {
        assert!((a[i] - b[i]).abs() > 1e-4, "gene {i}: {} {}", a[i], b[i]);
    }
    // Adam with L2: the first step is about α against the sign of g + λ x₀, without λ x₀
    // beside it
    let mut l2 = method(wide(3), Step::adam(0.01), &start);
    step_by_hand(&mut l2, with_l2);
    let x1 = step_by_hand(&mut l2, with_l2);
    for i in 0..3 {
        let g = LINEAR[i] + lambda * start[i];
        let expected = start[i] - 0.01 * g / (g.abs() + 1e-8);
        assert!((x1[i] - expected).abs() < 1e-15);
    }

    // with λ = 0, AdamW is Adam to the bit
    let mut adam = method(wide(3), Step::adam(0.01), &start);
    let mut adamw = method(wide(3), Step::adamw(0.01, 0.0), &start);
    for _ in 0..10 {
        assert_eq!(
            step_by_hand(&mut adam, linear),
            step_by_hand(&mut adamw, linear)
        );
    }
}

// the quadratic ½ Σ cᵢ xᵢ², gradient cᵢ xᵢ
fn quadratic(x: &[f64], gradient: &mut [f64]) -> f64 {
    const C: [f64; 2] = [1.0, 10.0];
    let mut value = 0.0;
    for i in 0..2 {
        gradient[i] = C[i] * x[i];
        value += 0.5 * C[i] * x[i] * x[i];
    }
    value
}

#[test]
fn momentum_and_nesterov_follow_sutskever_et_al() {
    let (alpha, mu) = (0.05, 0.9);
    let start = [1.0, 1.0];
    let mut gradient = [0.0; 2];
    // classical momentum, eqs. 1-2: v ← μ v − ε ∇f(θ), θ ← θ + v; the points asked are θ
    let mut momentum = method(wide(2), Step::momentum(alpha, mu), &start);
    let (mut theta, mut v) = (start.to_vec(), vec![0.0; 2]);
    for t in 0..6 {
        let x = step_by_hand(&mut momentum, quadratic);
        assert_eq!(x, theta, "step {t}");
        quadratic(&theta, &mut gradient);
        for i in 0..2 {
            v[i] = mu * v[i] - alpha * gradient[i];
            theta[i] += v[i];
        }
    }
    // Nesterov's accelerated gradient, eqs. 3-4: v ← μ v − ε ∇f(θ + μ v), θ ← θ + v; the
    // points asked are the look-ahead points θ + μ v, where the gradient is taken
    let mut nesterov = method(wide(2), Step::nesterov(alpha, mu), &start);
    let (mut theta, mut v) = (start.to_vec(), vec![0.0; 2]);
    for t in 0..6 {
        let look_ahead: Vec<f64> = (0..2).map(|i| theta[i] + mu * v[i]).collect();
        let x = step_by_hand(&mut nesterov, quadratic);
        for i in 0..2 {
            assert!(
                (x[i] - look_ahead[i]).abs() < 1e-15,
                "step {t}: {x:?} {look_ahead:?}"
            );
        }
        quadratic(&look_ahead, &mut gradient);
        for i in 0..2 {
            v[i] = mu * v[i] - alpha * gradient[i];
            theta[i] += v[i];
        }
    }
    // gradient descent
    let mut descent = method(wide(2), Step::gradient(alpha), &start);
    let mut theta = start.to_vec();
    for _ in 0..4 {
        assert_eq!(step_by_hand(&mut descent, quadratic), theta);
        quadratic(&theta, &mut gradient);
        for i in 0..2 {
            theta[i] -= alpha * gradient[i];
        }
    }
}

#[test]
fn maximizing_climbs_the_score() {
    // −Σ (xᵢ − 1)², maximized at 1
    let hill = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        for (g, xi) in gradient.iter_mut().zip(x.iter()) {
            *g = -2.0 * (xi - 1.0);
        }
        -x.iter().map(|xi| (xi - 1.0) * (xi - 1.0)).sum::<f64>()
    });
    for step in [
        Step::gradient(0.1),
        Step::momentum(0.05, 0.9),
        Step::nesterov(0.05, 0.9),
        Step::adam(0.05),
    ] {
        let algorithm = FirstOrder::builder(Real::uniform(5, -5.0..=5.0).unwrap())
            .step(step)
            .seed(2)
            .build()
            .unwrap();
        let outcome = Engine::new(algorithm, hill)
            .stop_when(Stop::generations(20_000))
            .run()
            .unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Converged, "{step:?}");
        assert!(outcome.best_fitness().score().unwrap() > -1e-10, "{step:?}");
    }
}

#[test]
fn every_rule_converges_on_the_sphere_and_the_ellipsoid() {
    let steps = [
        Step::gradient(0.04),
        Step::momentum(0.02, 0.9),
        Step::nesterov(0.02, 0.9),
        Step::adam(0.05),
        Step::adamw(0.05, 0.0),
    ];
    for step in steps {
        for n in [2, 10, 100] {
            let problems: [Box<dyn Fn() -> Outcome<Reals>>; 2] = [
                Box::new(move || {
                    let problem = Sphere::new(n);
                    let algorithm = FirstOrder::builder(problem.representation())
                        .step(step)
                        .minimize()
                        .seed(n as u64)
                        .build()
                        .unwrap();
                    Engine::new(algorithm, problem)
                        .stop_when(Stop::generations(50_000))
                        .run()
                        .unwrap()
                }),
                Box::new(move || {
                    // curvatures 2 to 2n: a step of 0.04 is stable for n up to 12
                    let problem = AxisParallelEllipsoid::new(n);
                    let rate = if n > 10 {
                        step.learning_rate() / 10.0
                    } else {
                        step.learning_rate()
                    };
                    let algorithm = FirstOrder::builder(problem.representation())
                        .step(step.with_learning_rate(rate))
                        .minimize()
                        .seed(n as u64)
                        .build()
                        .unwrap();
                    Engine::new(algorithm, problem)
                        .stop_when(Stop::generations(100_000))
                        .run()
                        .unwrap()
                }),
            ];
            for (k, run) in problems.iter().enumerate() {
                let outcome = run();
                assert_eq!(
                    outcome.stop_reason(),
                    StopReason::Converged,
                    "{step:?}, n = {n}, problem {k}"
                );
                assert!(
                    outcome.best_fitness().score().unwrap() < 1e-9,
                    "{step:?}, n = {n}, problem {k}: {:?}",
                    outcome.best_fitness()
                );
            }
        }
    }
}

#[test]
fn adam_with_a_schedule_solves_rosenbrock() {
    // Rosenbrock's valley from the classic start, the learning rate halved every 1000 steps
    let problem = Rosenbrock::new(2);
    let adam = FirstOrder::builder(problem.representation())
        .step(Step::adam(0.05))
        .initial_genome(Reals::from(vec![-1.2, 1.0]))
        .gradient_tolerance(1e-8)
        .minimize()
        .seed(1)
        .build()
        .unwrap();
    let mut engine = Engine::new(adam, problem)
        .stop_when(Stop::generations(50_000))
        .control(|adam: &mut FirstOrder, progress| {
            let halvings = (progress.generation() / 1_000) as i32;
            adam.set_learning_rate(0.05 * 0.5f64.powi(halvings))
        });
    let outcome = engine.run().unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert_eq!(engine.algorithm().converged(), Some(Convergence::Gradient));
    let x = outcome.best_genome();
    assert!(
        (x[0] - 1.0).abs() < 1e-6 && (x[1] - 1.0).abs() < 1e-6,
        "{x:?}"
    );
    assert!(engine.algorithm().step().learning_rate() < 0.05);
}

#[test]
fn points_stay_in_the_bounds_and_converge_on_them() {
    // Σ (xᵢ − 3)², whose minimum is outside [−1, 1]: the minimum in the box is at 1, where the
    // projected gradient is 0
    let outside = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        for (g, xi) in gradient.iter_mut().zip(x.iter()) {
            *g = 2.0 * (xi - 3.0);
        }
        x.iter().map(|xi| (xi - 3.0) * (xi - 3.0)).sum::<f64>()
    });
    let real = Real::new([-1.0..=1.0, -1.0..=1.0, 0.0..=0.0, -1.0..=5.0]).unwrap();
    for step in [
        Step::gradient(0.5),
        Step::momentum(0.5, 0.95),
        Step::nesterov(0.5, 0.95),
        Step::adam(0.5),
    ] {
        let algorithm = FirstOrder::builder(real.clone())
            .step(step)
            .minimize()
            .seed(3)
            .build()
            .unwrap();
        let bounds = real.bounds().to_vec();
        let mut engine = Engine::new(algorithm, outside)
            .stop_when(Stop::generations(20_000))
            .on_generation(move |snapshot| {
                for individual in snapshot.population() {
                    let genes = individual.genome().iter().zip(&bounds);
                    assert!(genes.clone().all(|(x, range)| range.contains(x)));
                }
            });
        let outcome = engine.run().unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Converged, "{step:?}");
        let x = outcome.best_genome();
        assert_eq!(x[0], 1.0, "{step:?}");
        assert_eq!(x[1], 1.0, "{step:?}");
        assert_eq!(x[2], 0.0, "{step:?}");
        assert!((x[3] - 3.0).abs() < 1e-6, "{step:?}: {x:?}");
    }
}

#[test]
fn finite_differences_ride_along_with_the_point() {
    let plain = |x: &Reals| x.iter().map(|xi| xi * xi).sum::<f64>();
    for (gradients, per_step) in [
        (Gradients::Auto, 4),
        (Gradients::Forward { step: None }, 4),
        (Gradients::Central { step: None }, 7),
    ] {
        let algorithm = FirstOrder::builder(Real::uniform(3, -5.0..=5.0).unwrap())
            .step(Step::adam(0.05))
            .gradients(gradients)
            .minimize()
            .seed(4)
            .build()
            .unwrap();
        let mut engine = Engine::new(algorithm, plain).stop_when(Stop::generations(5_000));
        let outcome = engine.run().unwrap();
        assert_eq!(
            engine.algorithm().gradients(),
            match gradients {
                Gradients::Auto => Gradients::Forward { step: None },
                other => other,
            }
        );
        assert_eq!(
            outcome.evaluations(),
            per_step * (outcome.generations() + 1)
        );
        assert_eq!(
            outcome.stop_reason(),
            StopReason::Converged,
            "{gradients:?}"
        );
        assert!(outcome.best_fitness().score().unwrap() < 1e-10);
    }
    // a supplied gradient: one evaluation per step, `Auto` resolved to it
    let mut engine = Engine::new(
        FirstOrder::builder(Sphere::new(3).representation())
            .minimize()
            .build()
            .unwrap(),
        Sphere::new(3),
    )
    .stop_when(Stop::generations(10));
    assert_eq!(engine.run().unwrap().evaluations(), 11);
    assert_eq!(engine.algorithm().gradients(), Gradients::Supplied);
    assert_eq!(engine.algorithm().wants(), Wanted::GRADIENT);
}

#[test]
fn supplied_gradients_need_a_fitness_function_that_provides_them() {
    let plain = |x: &Reals| x.iter().sum::<f64>();
    let algorithm = FirstOrder::builder(Real::uniform(2, -1.0..=1.0).unwrap())
        .gradients(Gradients::Supplied)
        .build()
        .unwrap();
    let result = Engine::new(algorithm, plain)
        .stop_when(Stop::generations(1))
        .run();
    let Err(Error::InvalidSetting { setting, .. }) = result else {
        panic!("{result:?}");
    };
    assert_eq!(setting, "gradients");
    // nor driven by hand without the gradient
    let mut algorithm = FirstOrder::builder(Real::uniform(2, -1.0..=1.0).unwrap())
        .gradients(Gradients::Supplied)
        .build()
        .unwrap();
    algorithm.ask();
    let before = format!("{algorithm:?}");
    assert!(matches!(
        algorithm.tell(&[Fitness::new(0.0)]),
        Err(Error::InvalidFitness { .. })
    ));
    assert_eq!(format!("{algorithm:?}"), before);
}

#[test]
fn invalid_points_step_back() {
    // the sphere, invalid where x₀ > 0.5: a learning rate of 1 overshoots into it
    let guarded = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        for (g, xi) in gradient.iter_mut().zip(x.iter()) {
            *g = 2.0 * xi;
        }
        (x[0] <= 0.5).then(|| x.iter().map(|xi| xi * xi).sum::<f64>())
    });
    let algorithm = FirstOrder::builder(Real::uniform(2, -5.0..=5.0).unwrap())
        .step(Step::gradient(0.9))
        .initial_genome(Reals::from(vec![-2.0, 0.3]))
        .minimize()
        .build()
        .unwrap();
    let mut invalid = 0;
    let outcome = Engine::new(algorithm, guarded)
        .stop_when(Stop::generations(1_000))
        .on_generation(|snapshot| {
            // the population is the last valid point
            assert!(snapshot.population()[0].fitness().unwrap().is_valid());
            invalid += snapshot.discarded().len();
        })
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert!(outcome.best_fitness().score().unwrap() < 1e-10);

    // an invalid start, with nothing to step back to
    let algorithm = FirstOrder::builder(Real::uniform(2, -5.0..=5.0).unwrap())
        .initial_genome(Reals::from(vec![2.0, 0.0]))
        .minimize()
        .build()
        .unwrap();
    let mut engine = Engine::new(algorithm, guarded).stop_when(Stop::generations(100));
    let outcome = engine.run().unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert_eq!(engine.algorithm().converged(), Some(Convergence::Invalid));
    assert_eq!(outcome.best_fitness(), Fitness::invalid());

    // restarts find valid points
    let algorithm = FirstOrder::builder(Real::uniform(2, -5.0..=5.0).unwrap())
        .initial_genome(Reals::from(vec![2.0, 0.0]))
        .step(Step::adam(0.05))
        .restarts(local::Restarts::Random { times: 20 })
        .minimize()
        .seed(5)
        .build()
        .unwrap();
    let outcome = Engine::new(algorithm, guarded)
        .stop_when(Stop::generations(100_000))
        .run()
        .unwrap();
    assert!(outcome.best_fitness().score().unwrap() < 1e-10);
}

#[test]
fn a_step_back_halves_the_distance() {
    // invalid beyond x₀ = 1: from 0 a step to 4 fails, then 2 fails, then 1 is valid
    let f = |x: &[f64], gradient: &mut [f64]| {
        gradient[0] = -1.0;
        -x[0]
    };
    let mut algorithm = method(
        Real::uniform(1, -10.0..=10.0).unwrap(),
        Step::gradient(4.0),
        &[0.0],
    );
    let mut points = Vec::new();
    for _ in 0..4 {
        let x = algorithm.ask().get(0).unwrap()[0];
        let mut gradient = [0.0];
        let value = f(&[x], &mut gradient);
        let fitness = [if x <= 1.0 {
            Fitness::new(value)
        } else {
            Fitness::invalid()
        }];
        let evaluations = Evaluations::with_gradients(&fitness, &gradient, 1).unwrap();
        algorithm.tell_evaluations(&evaluations).unwrap();
        points.push(x);
    }
    assert_eq!(points, [0.0, 4.0, 2.0, 1.0]);
    assert_eq!(algorithm.iterations(), 1);
    assert_eq!(algorithm.population()[0].genome()[0], 1.0);
}

#[test]
fn reevaluation_on_a_changed_function() {
    // the same as running on, but for the scores: a doubled function, rescored at the point
    let doubled = |x: &[f64], gradient: &mut [f64]| {
        let value = linear(x, gradient);
        for g in gradient.iter_mut() {
            *g *= 2.0;
        }
        2.0 * value
    };
    let mut algorithm = method(wide(3), Step::adam(0.01), &[1.0, -2.0, 0.5]);
    // nothing to re-evaluate yet
    algorithm.reevaluate().unwrap();
    for _ in 0..5 {
        step_by_hand(&mut algorithm, linear);
    }
    algorithm.ask();
    assert_eq!(algorithm.reevaluate(), Err(Error::ReevaluationOutOfTurn));
    step_by_hand(&mut algorithm, linear);
    let point = algorithm.population()[0].genome().clone();
    let (generation, evaluations, steps) = (
        algorithm.generation(),
        algorithm.evaluations(),
        algorithm.steps(),
    );
    algorithm.reevaluate().unwrap();
    let asked = step_by_hand(&mut algorithm, doubled);
    assert_eq!(asked[..], point[..]);
    assert_eq!(algorithm.generation(), generation);
    assert_eq!(algorithm.evaluations(), evaluations + 1);
    assert_eq!(algorithm.steps(), steps, "the memory is kept");
    assert_eq!(algorithm.best_generation(), generation);
    let mut gradient = [0.0; 3];
    assert_eq!(
        algorithm.best().unwrap().fitness(),
        Some(Fitness::new(doubled(&point, &mut gradient)))
    );
    assert_eq!(algorithm.gradient(), gradient);
    // and the next step continues with Adam's averages: a doubled gradient changes nothing in
    // Adam's direction for a constant gradient, but the averages of the old one are kept
    let next = step_by_hand(&mut algorithm, doubled);
    assert_eq!(algorithm.steps(), steps + 1);
    assert_ne!(next[..], point[..]);
}

#[test]
fn restarts_reset_the_memory() {
    let problem = Sphere::new(4);
    let algorithm = FirstOrder::builder(problem.representation())
        .step(Step::adam(0.05))
        .restarts(local::Restarts::Random { times: 3 })
        .minimize()
        .seed(6)
        .build()
        .unwrap();
    let mut engine = Engine::new(algorithm, problem)
        .stop_when(Stop::generations(100_000))
        .control(|algorithm: &mut FirstOrder, _| {
            // t counts from the restart
            assert!(algorithm.steps() <= algorithm.iterations());
            Ok(())
        });
    let outcome = engine.run().unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    let algorithm = engine.algorithm();
    assert_eq!(algorithm.restart_count(), 3);
    assert!(algorithm.steps() < algorithm.iterations());
    assert!(algorithm.is_finished());
}

#[test]
fn seeded_runs_repeat_and_parallel_ones_match() {
    let run = |seed: u64, parallel: bool| {
        let algorithm = FirstOrder::builder(Real::uniform(6, -5.0..=5.0).unwrap())
            .step(Step::nesterov(0.01, 0.9))
            .gradients(Gradients::Central { step: None })
            .minimize()
            .seed(seed)
            .build()
            .unwrap();
        let rosenbrock = |x: &Reals| Rosenbrock::new(6).evaluate(x);
        let engine = Engine::new(algorithm, rosenbrock).stop_when(Stop::generations(300));
        #[cfg(feature = "parallel")]
        let engine = engine.parallel(parallel);
        #[cfg(not(feature = "parallel"))]
        let _ = parallel;
        let mut engine = engine;
        engine.run().unwrap().into_best()
    };
    assert_eq!(run(7, false), run(7, false));
    assert_eq!(run(7, false), run(7, true));
    assert_ne!(run(7, false), run(8, false));
}

#[test]
fn settings_are_validated() {
    let real = || Real::uniform(2, -1.0..=1.0).unwrap();
    let invalid = |builder: genoxide::algorithm::FirstOrderBuilder| -> &'static str {
        match builder.build() {
            Err(Error::InvalidSetting { setting, .. }) => setting,
            other => panic!("{other:?}"),
        }
    };
    for step in [
        Step::gradient(0.0),
        Step::gradient(f64::INFINITY),
        Step::gradient(f64::NAN),
        Step::momentum(0.1, 1.0),
        Step::nesterov(0.1, -0.1),
        Step::Adam {
            learning_rate: 0.1,
            beta1: 1.0,
            beta2: 0.999,
            epsilon: 1e-8,
        },
        Step::Adam {
            learning_rate: 0.1,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 0.0,
        },
        Step::adamw(0.1, -1.0),
        Step::adamw(0.1, f64::NAN),
    ] {
        assert_eq!(
            invalid(FirstOrder::builder(real()).step(step)),
            "step",
            "{step:?}"
        );
    }
    assert_eq!(
        invalid(FirstOrder::builder(real()).gradient_tolerance(-1.0)),
        "gradient_tolerance"
    );
    assert_eq!(
        invalid(FirstOrder::builder(real()).step_tolerance(f64::NAN)),
        "step_tolerance"
    );
    assert_eq!(
        invalid(FirstOrder::builder(real()).restarts(local::Restarts::Random { times: 0 })),
        "restarts"
    );
    assert_eq!(
        invalid(FirstOrder::builder(real()).gradients(Gradients::Forward { step: Some(0.0) })),
        "gradients"
    );
    assert!(matches!(
        FirstOrder::builder(real())
            .initial_genome(Reals::from(vec![2.0, 0.0]))
            .build(),
        Err(Error::InvalidGenome { .. })
    ));
    // the setters
    let mut algorithm = FirstOrder::builder(real()).build().unwrap();
    assert!(algorithm.set_learning_rate(-1.0).is_err());
    assert!(algorithm.set_multiplier(0.0).is_err());
    assert_eq!(algorithm.step(), Step::adam(0.001));
    assert_eq!(algorithm.multiplier(), 1.0);
    algorithm.set_learning_rate(0.5).unwrap();
    algorithm.set_multiplier(0.25).unwrap();
    assert_eq!(algorithm.step(), Step::adam(0.5));
    assert_eq!(algorithm.multiplier(), 0.25);
    // tell without ask, and the wrong count
    assert_eq!(algorithm.tell(&[]), Err(Error::TellWithoutAsk));
    algorithm.ask();
    assert!(matches!(
        algorithm.tell(&[Fitness::new(1.0)]),
        Err(Error::FitnessCount {
            expected: 3,
            got: 1
        })
    ));
}

#[test]
fn the_multiplier_scales_every_step() {
    // η = 0.5 with α = 0.02 is α = 0.01, to the bit, for every rule but AdamW's decay
    for (step, half) in [
        (Step::gradient(0.02), Step::gradient(0.01)),
        (Step::momentum(0.02, 0.5), Step::momentum(0.01, 0.5)),
        (Step::nesterov(0.02, 0.5), Step::nesterov(0.01, 0.5)),
    ] {
        let mut scaled = method(wide(2), step, &[1.0, 1.0]);
        scaled.set_multiplier(0.5).unwrap();
        let mut plain = method(wide(2), half, &[1.0, 1.0]);
        for _ in 0..5 {
            let a = step_by_hand(&mut scaled, quadratic);
            let b = step_by_hand(&mut plain, quadratic);
            for (a, b) in a.iter().zip(&b) {
                assert!((a - b).abs() < 1e-15, "{step:?}");
            }
        }
    }
    // AdamW: x ← x − η (α m̂ / (√v̂ + ε) + λ x)
    let mut adamw = method(wide(1), Step::adamw(0.01, 0.1), &[2.0]);
    adamw.set_multiplier(0.5).unwrap();
    let f = |x: &[f64], gradient: &mut [f64]| {
        gradient[0] = 1.0;
        x[0]
    };
    step_by_hand(&mut adamw, f);
    let x1 = step_by_hand(&mut adamw, f)[0];
    assert!((x1 - (2.0 - 0.5 * (0.01 / (1.0 + 1e-8) + 0.1 * 2.0))).abs() < 1e-15);
}

#[test]
fn checks_for_a_supplied_gradient_at_the_engine() {
    // a provided gradient, but asked for differences: the engine never wants it
    let mut algorithm = FirstOrder::builder(Sphere::new(2).representation())
        .gradients(Gradients::Forward { step: None })
        .build()
        .unwrap();
    algorithm.prepare(Provided::GRADIENT).unwrap();
    assert_eq!(algorithm.wants(), Wanted::NOTHING);
    algorithm.prepare(Provided::NOTHING).unwrap();
    assert_eq!(algorithm.gradients(), Gradients::Forward { step: None });
    assert_eq!(algorithm.ask().len(), 3);
}

#[test]
fn many_genes_need_supplied_gradients() {
    let n = genoxide::gradient::AUTO_LIMIT + 1;
    let real = Real::uniform(n, -1.0..=1.0).unwrap();
    // driven by hand, `Auto` above the limit is supplied gradients, not n + 1 genomes per ask
    let mut algorithm = FirstOrder::builder(real.clone()).build().unwrap();
    assert_eq!(algorithm.gradients(), Gradients::Supplied);
    assert_eq!(algorithm.ask().len(), 1);
    // finite differences asked for explicitly, beyond what an ask can hold: an error, not an
    // allocation that fails
    let huge = Real::uniform(20_000, -1.0..=1.0).unwrap();
    let result = FirstOrder::builder(huge)
        .gradients(Gradients::Forward { step: None })
        .build();
    let Err(Error::InvalidSetting { setting, reason }) = result else {
        panic!("{result:?}");
    };
    assert_eq!(setting, "gradients");
    assert!(reason.contains("Differentiable"), "{reason}");
    // and `Auto` with a plain function fails at the start of the run
    let plain = |x: &Reals| x.iter().sum::<f64>();
    let result = Engine::new(FirstOrder::builder(real).build().unwrap(), plain)
        .stop_when(Stop::generations(1))
        .run();
    assert!(matches!(
        result,
        Err(Error::InvalidSetting {
            setting: "gradients",
            ..
        })
    ));
}

#[test]
fn a_run_as_in_python() {
    // python/tests/test_first_order.py has the same run, evaluated in Rust, with the same results
    let problem = Rosenbrock::new(4);
    let adam = FirstOrder::builder(problem.representation())
        .step(Step::adam(0.02))
        .restarts(local::Restarts::Random { times: 2 })
        .minimize()
        .seed(5)
        .build()
        .unwrap();
    let outcome = Engine::new(adam, problem)
        .stop_when(Stop::evaluations(200_000))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert_eq!(outcome.evaluations(), 85_572);
    assert_eq!(
        outcome.best_fitness(),
        Fitness::new(4.017_523_816_634_646e-13)
    );
}
