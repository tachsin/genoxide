//! The engine's path for gradients, through a test-only algorithm that asks for them: projected
//! gradient descent with a fixed step, from supplied gradients or a finite-difference stencil.

use genoxide::algorithm::Candidates;
use genoxide::engine::{Evaluations, FitnessFunction, Provided, Wanted};
use genoxide::gradient::{Differentiable, Gradients, Stencil};
use genoxide::prelude::*;
use genoxide::problems::{AxisParallelEllipsoid, Problem};
use std::sync::atomic::{AtomicU64, Ordering};

// projected gradient descent with a fixed step on `Real` genomes, minimizing
struct Descent {
    real: Real,
    gradients: Gradients,
    // the resolved setting: forward differences when driven by hand with `Auto`
    resolved: Gradients,
    // wants supplied gradients whatever the fitness function provides, to test the engine's check
    insist: bool,
    stencil: Option<Stencil>,
    rate: f64,
    x: Reals,
    // the current point, then the stencil's points
    individuals: Vec<Individual<Reals>>,
    indices: Vec<usize>,
    population: Population<Reals>,
    best: Option<Individual<Reals>>,
    generation: u64,
    evaluations: u64,
    best_generation: u64,
    // the gradient of every step, for comparing runs to the bit
    history: Vec<Vec<f64>>,
    gradient: Vec<f64>,
    values: Vec<f64>,
}

impl Descent {
    fn new(real: Real, gradients: Gradients, rate: f64, x: Vec<f64>) -> Self {
        let n = x.len();
        let mut descent = Self {
            resolved: Gradients::Forward { step: None },
            stencil: None,
            real,
            gradients,
            insist: false,
            rate,
            x: Reals::from(x),
            individuals: Vec::new(),
            indices: Vec::new(),
            population: Population::new(Vec::new()),
            best: None,
            generation: 0,
            evaluations: 0,
            best_generation: 0,
            history: Vec::new(),
            gradient: vec![0.0; n],
            values: Vec::new(),
        };
        descent.set_stencil().unwrap();
        descent
    }

    fn set_stencil(&mut self) -> genoxide::Result<()> {
        self.stencil = if self.resolved.is_supplied() {
            None
        } else {
            Some(Stencil::new(&self.real, self.resolved)?)
        };
        Ok(())
    }
}

impl Algorithm for Descent {
    type Genome = Reals;

    fn objective(&self) -> Objective {
        Objective::Minimize
    }

    fn ask(&mut self) -> Candidates<'_, Reals> {
        let points = self.stencil.as_ref().map_or(0, Stencil::len);
        self.individuals
            .resize_with(1 + points, || Individual::new(self.x.clone()));
        self.individuals[0] = Individual::new(self.x.clone());
        if let Some(stencil) = &mut self.stencil {
            stencil.set_center(&self.x).unwrap();
            for (k, individual) in self.individuals[1..].iter_mut().enumerate() {
                stencil.write_point(k, individual.genome_mut()).unwrap();
            }
        }
        self.indices.clear();
        self.indices.extend(0..self.individuals.len());
        Candidates::new(&self.individuals, &self.indices)
    }

    fn tell(&mut self, fitness: &[Fitness]) -> genoxide::Result<()> {
        self.tell_evaluations(&Evaluations::new(fitness))
    }

    fn tell_evaluations(&mut self, evaluations: &Evaluations<'_>) -> genoxide::Result<()> {
        if evaluations.len() != self.indices.len() {
            return Err(Error::FitnessCount {
                expected: self.indices.len(),
                got: evaluations.len(),
            });
        }
        let fitness = evaluations.fitness();
        let score = |fitness: Fitness| fitness.score().unwrap_or(f64::NAN);
        match &self.stencil {
            None => {
                let gradient = evaluations.gradient(0).ok_or(Error::InvalidFitness {
                    reason: "supplied gradients, and none told".to_string(),
                })?;
                self.gradient.copy_from_slice(gradient);
            }
            Some(stencil) => {
                self.values.clear();
                self.values.extend(fitness[1..].iter().map(|&f| score(f)));
                stencil.gradient(score(fitness[0]), &self.values, &mut self.gradient)?;
            }
        }
        if self.best.is_some() {
            self.generation += 1;
        }
        self.evaluations += fitness.len() as u64;
        for (individual, &fitness) in self.individuals.iter_mut().zip(fitness) {
            individual.set_fitness(fitness);
            let better = self.best.as_ref().is_none_or(|best| {
                Objective::Minimize.is_better(fitness, best.fitness().expect("evaluated"))
            });
            if better {
                self.best = Some(individual.clone());
                self.best_generation = self.generation;
            }
        }
        self.population = Population::new(vec![self.individuals[0].clone()]);
        self.history.push(self.gradient.clone());
        // a step down the gradient, projected into the box
        for ((x, g), range) in self
            .x
            .iter_mut()
            .zip(&self.gradient)
            .zip(self.real.bounds())
        {
            *x = (*x - self.rate * g).clamp(*range.start(), *range.end());
        }
        Ok(())
    }

    fn population(&self) -> &Population<Reals> {
        &self.population
    }

    fn best(&self) -> Option<&Individual<Reals>> {
        self.best.as_ref()
    }

    fn generation(&self) -> u64 {
        self.generation
    }

    fn evaluations(&self) -> u64 {
        self.evaluations
    }

    fn best_generation(&self) -> u64 {
        self.best_generation
    }

    fn prepare(&mut self, provided: Provided) -> genoxide::Result<()> {
        self.resolved = if self.insist {
            Gradients::Supplied
        } else {
            self.gradients.resolve(provided, &self.real)?
        };
        self.set_stencil()
    }

    fn wants(&self) -> Wanted {
        if self.resolved.is_supplied() {
            Wanted::GRADIENT
        } else {
            Wanted::NOTHING
        }
    }
}

// the axis-parallel ellipsoid Σ i xᵢ² in 4 dimensions, from (4, −3, 2, −1): its curvatures are 2
// to 8, so a step of 0.1 shrinks each gene by a factor 0.8 to 0.2 per step
fn descent(gradients: Gradients) -> Descent {
    let real = AxisParallelEllipsoid::new(4).representation();
    Descent::new(real, gradients, 0.1, vec![4.0, -3.0, 2.0, -1.0])
}

const GENERATIONS: u64 = 60;

// `engine`, evaluating in parallel if asked and the `parallel` feature is on
fn parallel<'o, A, F>(engine: Engine<'o, A, F>, parallel: bool) -> Engine<'o, A, F>
where
    A: Algorithm,
    F: FitnessFunction<A::Genome>,
{
    #[cfg(feature = "parallel")]
    let engine = engine.parallel(parallel);
    #[cfg(not(feature = "parallel"))]
    let _ = parallel;
    engine
}

// runs `algorithm` with `fitness`, sequentially or in parallel, and returns its outcome and the
// gradients of its steps
fn run<F: FitnessFunction<Reals>>(
    algorithm: Descent,
    fitness: F,
    in_parallel: bool,
) -> (Outcome<Reals>, Vec<Vec<f64>>) {
    let mut engine = parallel(Engine::new(algorithm, fitness), in_parallel)
        .stop_when(Stop::generations(GENERATIONS));
    let outcome = engine.run().unwrap();
    let algorithm = engine.into_algorithm();
    assert_eq!(algorithm.history.len() as u64, GENERATIONS + 1);
    (outcome, algorithm.history)
}

// the ellipsoid as a batch of differentiable functions, through the problem's own gradient
fn ellipsoid_batch(xs: &[&Reals], gradients: &mut [f64]) -> Vec<f64> {
    let problem = AxisParallelEllipsoid::new(4);
    let n = xs.first().map_or(0, |x| x.len());
    xs.iter()
        .zip(gradients.chunks_mut(n.max(1)))
        .map(|(x, row)| problem.evaluate_with(x, &mut genoxide::engine::Extras::with_gradient(row)))
        .collect()
}

// the same best individual, counts and stop reason; the times differ
fn assert_same(a: &Outcome<Reals>, b: &Outcome<Reals>) {
    assert_eq!(a.best(), b.best());
    assert_eq!(a.generations(), b.generations());
    assert_eq!(a.evaluations(), b.evaluations());
    assert_eq!(a.stop_reason(), b.stop_reason());
}

fn to_bits(history: &[Vec<f64>]) -> Vec<Vec<u64>> {
    history
        .iter()
        .map(|gradient| gradient.iter().map(|g| g.to_bits()).collect())
        .collect()
}

#[test]
fn supplied_gradients_have_the_same_bits_sequentially_in_parallel_and_in_a_batch() {
    let problem = AxisParallelEllipsoid::new(4);
    let (outcome, history) = run(descent(Gradients::Supplied), problem, false);
    // one evaluation per step, at the minimum
    assert_eq!(outcome.evaluations(), GENERATIONS + 1);
    assert!(outcome.best_fitness().score().unwrap() < 1e-6);
    // the first gradient is the problem's at the start
    assert_eq!(history[0], [8.0, -12.0, 12.0, -8.0]);
    let closure = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        problem.evaluate_with(x, &mut genoxide::engine::Extras::with_gradient(gradient))
    });
    for (other_outcome, other_history) in [
        run(descent(Gradients::Supplied), problem, true),
        run(descent(Gradients::Supplied), closure, false),
        run(descent(Gradients::Supplied), closure, true),
        run(
            descent(Gradients::Supplied),
            Batch(Differentiable(ellipsoid_batch)),
            false,
        ),
    ] {
        assert_same(&other_outcome, &outcome);
        assert_eq!(to_bits(&other_history), to_bits(&history));
    }
}

#[test]
fn finite_differences_have_the_same_bits_sequentially_in_parallel_and_in_a_batch() {
    let plain = |x: &Reals| AxisParallelEllipsoid::new(4).evaluate(x);
    let batch = Batch(|xs: &[&Reals]| xs.iter().map(|x| plain(x)).collect::<Vec<_>>());
    for gradients in [
        Gradients::Forward { step: None },
        Gradients::Central { step: None },
    ] {
        let (outcome, history) = run(descent(gradients), plain, false);
        // the point and its stencil each step
        let points = gradients.extra_evaluations(4) as u64 + 1;
        assert_eq!(outcome.evaluations(), points * (GENERATIONS + 1));
        assert!(outcome.best_fitness().score().unwrap() < 1e-6);
        // close to the analytic gradient, central closer
        let tolerance = if points == 5 { 1e-6 } else { 1e-9 };
        for (g, t) in history[0].iter().zip([8.0, -12.0, 12.0, -8.0]) {
            assert!((g - t).abs() <= tolerance * 12.0, "{gradients:?}: {g} {t}");
        }
        for (other_outcome, other_history) in [
            run(descent(gradients), plain, true),
            run(descent(gradients), batch, false),
            // finite differences even when a gradient is supplied
            run(descent(gradients), AxisParallelEllipsoid::new(4), true),
        ] {
            assert_same(&other_outcome, &outcome);
            assert_eq!(to_bits(&other_history), to_bits(&history));
        }
    }
}

#[test]
fn auto_resolves_to_what_the_fitness_function_provides() {
    let mut engine = Engine::new(descent(Gradients::Auto), AxisParallelEllipsoid::new(4))
        .stop_when(Stop::generations(1));
    engine.run().unwrap();
    assert_eq!(engine.algorithm().resolved, Gradients::Supplied);
    let plain = |x: &Reals| AxisParallelEllipsoid::new(4).evaluate(x);
    let mut engine = Engine::new(descent(Gradients::Auto), plain).stop_when(Stop::generations(1));
    let outcome = engine.run().unwrap();
    assert_eq!(
        engine.algorithm().resolved,
        Gradients::Forward { step: None }
    );
    assert_eq!(outcome.evaluations(), 2 * 5);
}

#[test]
fn supplied_gradients_need_a_fitness_function_that_provides_them() {
    let plain = |x: &Reals| x.iter().sum::<f64>();
    let result = Engine::new(descent(Gradients::Supplied), plain)
        .stop_when(Stop::generations(1))
        .run();
    let Err(Error::InvalidSetting { setting, reason }) = result else {
        panic!("{result:?}");
    };
    assert_eq!(setting, "gradients");
    assert!(reason.contains("Differentiable"), "{reason}");
}

#[test]
fn auto_needs_a_supplied_gradient_for_many_genes() {
    let n = genoxide::gradient::AUTO_LIMIT + 1;
    let real = Real::uniform(n, -1.0..=1.0).unwrap();
    let plain = |x: &Reals| x.iter().map(|xi| xi * xi).sum::<f64>();
    let result = Engine::new(
        Descent::new(real.clone(), Gradients::Auto, 0.1, vec![0.5; n]),
        plain,
    )
    .stop_when(Stop::generations(1))
    .run();
    let Err(Error::InvalidSetting { setting, reason }) = result else {
        panic!("forward differences in {n} genes");
    };
    assert_eq!(setting, "gradients");
    assert!(reason.contains("Differentiable"), "{reason}");
    // supplied, one evaluation per step
    let sphere = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        for (g, xi) in gradient.iter_mut().zip(x.iter()) {
            *g = 2.0 * xi;
        }
        plain(x)
    });
    let outcome = Engine::new(
        Descent::new(real, Gradients::Auto, 0.25, vec![0.5; n]),
        sphere,
    )
    .stop_when(Stop::generations(1))
    .run()
    .unwrap();
    assert_eq!(outcome.evaluations(), 2);
    assert_eq!(outcome.best_fitness().score(), Some(0.25 * 0.25 * n as f64));
}

#[test]
fn the_engine_checks_that_wanted_extras_are_provided() {
    let mut algorithm = descent(Gradients::Supplied);
    algorithm.insist = true;
    let plain = |x: &Reals| x.iter().sum::<f64>();
    let result = Engine::new(algorithm, plain)
        .stop_when(Stop::generations(1))
        .run();
    let Err(Error::InvalidSetting { setting, reason }) = result else {
        panic!("{result:?}");
    };
    assert_eq!(setting, "fitness");
    assert!(reason.contains("gradient"), "{reason}");
}

// NaN in a gradient makes the evaluation invalid, or stops the run, as NaN in a score does
#[test]
fn nan_in_a_gradient_follows_the_nan_policy() {
    // the sphere, whose gradient is NaN where x₀ > 3
    let calls = AtomicU64::new(0);
    let sphere = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        calls.fetch_add(1, Ordering::Relaxed);
        for (g, xi) in gradient.iter_mut().zip(x.iter()) {
            *g = if x[0] > 3.0 { f64::NAN } else { 2.0 * xi };
        }
        x.iter().map(|xi| xi * xi).sum::<f64>()
    });
    for in_parallel in [false, true] {
        let mut engine = parallel(
            Engine::new(descent(Gradients::Supplied), sphere),
            in_parallel,
        )
        .stop_when(Stop::generations(0));
        let outcome = engine.run().unwrap();
        assert_eq!(outcome.best_fitness(), Fitness::invalid());
        assert!(engine.algorithm().history[0].iter().all(|g| g.is_nan()));
        let result = parallel(
            Engine::new(descent(Gradients::Supplied), sphere),
            in_parallel,
        )
        .nan_policy(NanPolicy::Error)
        .stop_when(Stop::generations(0))
        .run();
        assert_eq!(result.unwrap_err(), Error::NanFitness);
    }
    // in a batch too
    let batch = Batch(Differentiable(|xs: &[&Reals], gradients: &mut [f64]| {
        gradients.fill(f64::NAN);
        xs.iter().map(|x| x[0]).collect::<Vec<_>>()
    }));
    let outcome = Engine::new(descent(Gradients::Supplied), batch)
        .stop_when(Stop::generations(0))
        .run()
        .unwrap();
    assert_eq!(outcome.best_fitness(), Fitness::invalid());
}

#[test]
fn a_batch_with_the_wrong_count_is_an_error() {
    let batch = Batch(Differentiable(|_: &[&Reals], _: &mut [f64]| {
        Vec::<f64>::new()
    }));
    let result = Engine::new(descent(Gradients::Supplied), batch)
        .stop_when(Stop::generations(0))
        .run();
    assert_eq!(
        result.unwrap_err(),
        Error::FitnessCount {
            expected: 1,
            got: 0
        }
    );
}

#[test]
fn info_is_kept_on_the_path_for_gradients() {
    let problem = AxisParallelEllipsoid::new(4);
    let with_info = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        let value =
            problem.evaluate_with(x, &mut genoxide::engine::Extras::with_gradient(gradient));
        Evaluated::new(value, x[0])
    });
    for in_parallel in [false, true] {
        let outcome = parallel(
            Engine::new(descent(Gradients::Supplied), with_info),
            in_parallel,
        )
        .stop_when(Stop::generations(5))
        .run()
        .unwrap();
        assert_eq!(outcome.best_info::<f64>(), Some(&outcome.best_genome()[0]));
    }
}

// driven by hand, with gradients told alongside the fitness
#[test]
fn ask_and_tell_with_gradients() {
    let mut algorithm = descent(Gradients::Supplied);
    algorithm.prepare(Provided::GRADIENT).unwrap();
    assert_eq!(algorithm.wants(), Wanted::GRADIENT);
    let problem = AxisParallelEllipsoid::new(4);
    for _ in 0..3 {
        let x = algorithm.ask().get(0).unwrap().clone();
        let mut gradient = vec![0.0; 4];
        let value = problem.evaluate_with(
            &x,
            &mut genoxide::engine::Extras::with_gradient(&mut gradient),
        );
        let fitness = [Fitness::new(value)];
        let evaluations = Evaluations::with_gradients(&fitness, &gradient, 4).unwrap();
        algorithm.tell_evaluations(&evaluations).unwrap();
    }
    assert_eq!(algorithm.generation(), 2);
    // without a gradient, this algorithm can't step
    algorithm.ask();
    assert!(algorithm.tell(&[Fitness::new(1.0)]).is_err());
}
