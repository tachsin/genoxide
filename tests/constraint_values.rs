//! Constraint values and their Jacobian through the engine: `Constrained`, the test problems that
//! give their values, and the same bits sequentially, in parallel and in a batch.

use genoxide::algorithm::Candidates;
use genoxide::constraint::Constrained;
use genoxide::engine::{BatchExtras, Evaluations, Extras, NanPolicy, Provided, Wanted};
use genoxide::prelude::*;
use genoxide::problems::{Problem, cec2006};

// per generation: the fitness, gradients, constraint values and Jacobians, flat
type Told = Vec<(Vec<Fitness>, Vec<f64>, Vec<f64>, Vec<f64>)>;

// an algorithm that asks the same genomes every generation and records what it's told
struct Recorder {
    population: Population<Reals>,
    indices: Vec<usize>,
    wanted: Wanted,
    told: Told,
    best: Option<Individual<Reals>>,
    generation: u64,
}

impl Recorder {
    fn new(points: &[[f64; 2]], wanted: Wanted) -> Self {
        let genomes = points.iter().map(|point| Reals::from(point.to_vec()));
        Self {
            population: Population::from_genomes(genomes),
            indices: (0..points.len()).collect(),
            wanted,
            told: Vec::new(),
            best: None,
            generation: 0,
        }
    }
}

impl Algorithm for Recorder {
    type Genome = Reals;

    fn objective(&self) -> Objective {
        Objective::Minimize
    }

    fn ask(&mut self) -> Candidates<'_, Reals> {
        Candidates::new(self.population.as_slice(), &self.indices)
    }

    fn tell(&mut self, _: &[Fitness]) -> genoxide::Result<()> {
        unreachable!("told with extras")
    }

    fn tell_evaluations(&mut self, evaluations: &Evaluations<'_>) -> genoxide::Result<()> {
        let count = evaluations.len();
        self.told.push((
            evaluations.fitness().to_vec(),
            flat(count, |p| evaluations.gradient(p)),
            flat(count, |p| evaluations.inequalities(p)),
            flat(count, |p| evaluations.constraint_jacobian(p)),
        ));
        if self.best.is_none() {
            let mut first = self.population[0].clone();
            first.set_fitness(evaluations.fitness()[0]);
            self.best = Some(first);
        } else {
            self.generation += 1;
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
        self.told.len() as u64 * self.indices.len() as u64
    }

    fn best_generation(&self) -> u64 {
        0
    }

    fn wants(&self) -> Wanted {
        self.wanted
    }
}

// the rows of `count` genomes, one after the other
fn flat<'a>(count: usize, row: impl Fn(usize) -> Option<&'a [f64]>) -> Vec<f64> {
    (0..count)
        .flat_map(|p| row(p).unwrap_or(&[]).to_vec())
        .collect()
}

const POINTS: [[f64; 2]; 4] = [[0.5, 1.0], [2.0, -1.0], [-1.5, 0.25], [3.0, 3.0]];

// x₀² + x₁, subject to 1 − x₀ x₁ ≤ 0 and x₀ − 2 ≤ 0
fn hyperbola(x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]) -> f64 {
    gradient.copy_from_slice(&[2.0 * x[0], 1.0]);
    g[0] = 1.0 - x[0] * x[1];
    g[1] = x[0] - 2.0;
    jacobian.copy_from_slice(&[-x[1], -x[0], 1.0, 0.0]);
    x[0] * x[0] + x[1]
}

fn all() -> Wanted {
    Wanted::GRADIENT
        .with_inequalities()
        .with_constraint_jacobian()
}

fn record<F: FitnessFunction<Reals>>(fitness: F, wanted: Wanted, parallel: bool) -> Told {
    let engine = Engine::new(Recorder::new(&POINTS, wanted), fitness);
    #[cfg(feature = "parallel")]
    let engine = engine.parallel(parallel);
    #[cfg(not(feature = "parallel"))]
    let _ = parallel;
    let mut engine = engine.stop_when(Stop::generations(2));
    engine.run().unwrap();
    engine.into_algorithm().told
}

#[test]
fn constrained_functions_give_their_values_and_violation() {
    let values = Constrained::new(2, |x: &Reals, g: &mut [f64]| {
        g[0] = 1.0 - x[0] * x[1];
        g[1] = x[0] - 2.0;
        x[0] * x[0] + x[1]
    });
    assert_eq!(values.inequalities(), 2);
    assert_eq!(values.provides(), Provided::NOTHING.with_inequalities(2));
    // 1 − 0.5 = 0.5 and −1.5: a violation of 0.5
    let x = Reals::from(vec![0.5, 1.0]);
    assert_eq!(values.evaluate(&x), (1.25, 0.5));
    let mut g = [0.0; 2];
    let with = values.evaluate_with(&x, &mut Extras::new(None, Some(&mut g), None));
    assert_eq!((with, g), ((1.25, 0.5), [0.5, -1.5]));
    let derivatives = Constrained::differentiable(2, hyperbola);
    assert_eq!(
        derivatives.provides(),
        Provided::GRADIENT
            .with_inequalities(2)
            .with_constraint_jacobian()
    );
    assert_eq!(derivatives.evaluate(&x), (1.25, 0.5));
    // only some of the buffers, the others scratch
    let mut jacobian = [0.0; 4];
    let with = derivatives.evaluate_with(&x, &mut Extras::new(None, None, Some(&mut jacobian)));
    assert_eq!((with, jacobian), ((1.25, 0.5), [-1.0, -0.5, 1.0, 0.0]));
    // a constrained function is a plain fitness function for any algorithm, by Deb's rules
    let ga = Ga::builder(Real::uniform(2, -3.0..=3.0).unwrap())
        .population_size(30)
        .select(Tournament::new(3).unwrap())
        .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
        .mutate(PolynomialMutation::per_gene(0.5, 20.0).unwrap())
        .minimize()
        .seed(1)
        .build()
        .unwrap();
    let outcome = Engine::new(ga, derivatives)
        .stop_when(Stop::generations(50))
        .run()
        .unwrap();
    assert!(outcome.best_fitness().is_feasible());
}

#[test]
fn the_engine_hands_over_values_and_jacobians() {
    let told = record(Constrained::differentiable(2, hyperbola), all(), false);
    let (fitness, gradients, values, jacobians) = &told[0];
    assert_eq!(fitness.len(), 4);
    // [3, 3]: 9 + 3, g = (−8, 1): a violation of 1
    assert_eq!(fitness[3], Fitness::constrained(12.0, 1.0));
    assert_eq!(&gradients[6..], [6.0, 1.0]);
    assert_eq!(&values[6..], [-8.0, 1.0]);
    assert_eq!(&jacobians[12..], [-3.0, -3.0, 1.0, 0.0]);
    // only the values: no gradient, and no Jacobian
    let told = record(
        Constrained::differentiable(2, hyperbola),
        Wanted::NOTHING.with_inequalities(),
        false,
    );
    let (_, gradients, values, jacobians) = &told[1];
    assert!(gradients.is_empty() && jacobians.is_empty());
    assert_eq!(values.len(), 8);
}

#[test]
fn the_same_bits_sequentially_in_parallel_and_in_a_batch() {
    let sequential = record(Constrained::differentiable(2, hyperbola), all(), false);
    let parallel = record(Constrained::differentiable(2, hyperbola), all(), true);
    assert_eq!(sequential, parallel);
    // a batch function that fills the rows of all genomes, through the default one-by-one
    struct Whole;
    impl FitnessFunction<Reals> for Whole {
        type Output = (f64, f64);
        fn evaluate(&self, x: &Reals) -> (f64, f64) {
            Constrained::differentiable(2, hyperbola).evaluate(x)
        }
        fn is_batch(&self) -> bool {
            true
        }
        fn provides(&self) -> Provided {
            Constrained::differentiable(2, hyperbola).provides()
        }
        fn evaluate_with(&self, x: &Reals, extras: &mut Extras<'_>) -> (f64, f64) {
            Constrained::differentiable(2, hyperbola).evaluate_with(x, extras)
        }
        fn evaluate_batch_with(
            &self,
            xs: &[&Reals],
            extras: &mut BatchExtras<'_>,
        ) -> Vec<(f64, f64)> {
            assert_eq!((extras.dimensions(), extras.constraints()), (2, 2));
            assert_eq!(extras.inequalities().map(|g| g.len()), Some(8));
            assert_eq!(extras.constraint_jacobians().map(|j| j.len()), Some(16));
            xs.iter()
                .enumerate()
                .map(|(p, x)| self.evaluate_with(x, &mut extras.get(p)))
                .collect()
        }
    }
    assert_eq!(record(Whole, all(), false), sequential);
}

#[test]
fn nan_in_a_jacobian_follows_the_nan_policy() {
    let nan = Constrained::differentiable(
        1,
        |x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]| {
            gradient.fill(0.0);
            g[0] = -1.0;
            jacobian[0] = if x[0] > 2.5 { f64::NAN } else { 0.0 };
            x[0]
        },
    );
    let told = record(nan, all(), false);
    assert!(!told[0].0[3].is_valid(), "the point with a NaN is invalid");
    assert!(told[0].0[0].is_valid());
    let error = Engine::new(Recorder::new(&POINTS, all()), nan)
        .nan_policy(NanPolicy::Error)
        .stop_when(Stop::generations(1))
        .run()
        .unwrap_err();
    assert_eq!(error, Error::NanFitness);
}

#[test]
fn values_that_arent_provided_are_an_error_naming_constrained() {
    let plain = |x: &Reals| x[0];
    let error = Engine::new(
        Recorder::new(&POINTS, Wanted::NOTHING.with_inequalities()),
        plain,
    )
    .stop_when(Stop::generations(1))
    .run()
    .unwrap_err();
    let Error::InvalidSetting { setting, reason } = error else {
        panic!("{error:?}");
    };
    assert_eq!(setting, "fitness");
    assert!(reason.contains("inequality constraint values") && reason.contains("Constrained"));
}

#[test]
fn test_problems_give_their_constraint_values_through_the_engine() {
    let told = record(cec2006::G06, Wanted::NOTHING.with_inequalities(), false);
    let (fitness, _, values, _) = &told[0];
    for (p, point) in POINTS.iter().enumerate() {
        let x = Reals::from(point.to_vec());
        let constraints = cec2006::G06.constraints(&x);
        assert_eq!(&values[2 * p..2 * p + 2], constraints.inequalities());
        let (score, violation) = cec2006::G06.evaluate(&x);
        assert_eq!(fitness[p], Fitness::constrained(score, violation));
    }
}
