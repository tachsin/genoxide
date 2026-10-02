//! The shift and the rotation that the CEC and BBOB suites apply to their functions: wrappers that
//! make an instance of any problem on [`Real`] genomes, generated from a seed.

use super::{Constraints, Optimum, Problem};
use crate::Objective;
use crate::StreamRng;
use crate::engine::{Extras, FitnessFunction, Provided};
use crate::genome::{Real, Reals, Representation};

// the streams of the seed's generator from which the shift and the rotation are drawn, so that a
// problem shifted and rotated with the same seed gets independent ones
const SHIFT_STREAM: u64 = 1;
const ROTATION_STREAM: u64 = 2;

// the point that a transformation keeps in place, or moves: the first solution of the problem's
// optimum, or the center of its box when the optimum isn't known
fn anchor<P: Problem<Representation = Real>>(problem: &P) -> Vec<f64> {
    match problem.optimum() {
        Some(optimum) if !optimum.solutions().is_empty() => optimum.solutions()[0].to_vec(),
        _ => problem
            .representation()
            .bounds()
            .iter()
            .map(|range| (range.start() + range.end()) / 2.0)
            .collect(),
    }
}

// the optimum of a wrapped problem, its solutions moved by `moved`, keeping those in the box
fn moved_optimum(
    optimum: Option<Optimum<Reals>>,
    real: &Real,
    moved: impl Fn(&[f64]) -> Reals,
) -> Option<Optimum<Reals>> {
    let optimum = optimum?;
    let solutions: Vec<Reals> = optimum
        .solutions()
        .iter()
        .map(|solution| moved(solution))
        .filter(|solution| real.validate(solution).is_ok())
        .collect();
    Some(if optimum.is_proven() {
        Optimum::proven(optimum.value(), solutions)
    } else {
        Optimum::best_known(optimum.value(), solutions)
    })
}

/// A problem shifted by a random vector: `f(x − o)`, where `f` is the wrapped problem and `o` a
/// shift generated from a seed, as the CEC and BBOB suites shift their functions, so that the
/// optimum is neither at the center of the box nor on its diagonal.
///
/// The shift moves the first solution of the wrapped problem's optimum (the center of the box if
/// the optimum isn't known) to a point drawn uniformly from the middle 80% of each gene's range:
/// the CEC 2013, 2014 and 2017 suites draw their shifted optima from [−80, 80]ⁿ in
/// [−100, 100]ⁿ, and BBOB from [−4, 4]ⁿ in [−5, 5]ⁿ (CEC 2005's data files spread theirs over
/// about 90% of each range). The bounds are the wrapped problem's.
///
/// The optimum keeps its value, and its solutions are shifted (those that the shift moves out of
/// the box are dropped). That holds when the wrapped problem's minimum is its minimum over all of
/// ℝⁿ, as for the functions that CEC and BBOB shift, such as [`Rastrigin`](super::Rastrigin) and
/// [`HighConditionedElliptic`](super::HighConditionedElliptic), but not for a function whose
/// minimum is only the lowest in its box: shifted, [`Schwefel2_26`](super::Schwefel2_26) brings
/// lower values from outside its box into it. The name and the reference are the wrapped
/// problem's, and so is the constraint violation of a constrained problem, at `x − o`; rounding
/// in `x − o` can put a shifted solution a few ulps outside a constraint that's active there
/// (6·10⁻¹⁴ at [`G06`](super::cec2006::G06)'s minimum with seed 3).
///
/// It [provides](FitnessFunction::provides) what the wrapped problem provides, and
/// [`evaluate_with`](FitnessFunction::evaluate_with) gives it at `x − o`: the gradient, the
/// constraints' values and their Jacobian, unchanged, since a shift moves the function without
/// turning or stretching it. A shifted smooth function keeps its analytic gradient, and a shifted
/// constrained problem gives its constraints' values to [`Bo`](crate::algorithm::Bo) (and, with
/// their Jacobian, to [`Mma`](crate::algorithm::Mma)).
///
/// ```
/// use genoxide::genome::Representation;
/// use genoxide::problems::{Problem, Rastrigin, Shifted};
///
/// let problem = Shifted::new(Rastrigin::new(10), 1);
/// let optimum = problem.optimum().expect("known");
/// // the minimum 0, at the shift
/// assert_eq!(optimum.value(), 0.0);
/// assert_eq!(&optimum.solutions()[0][..], problem.shift());
/// assert!(problem.representation().validate(&optimum.solutions()[0]).is_ok());
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Shifted<P> {
    problem: P,
    seed: u64,
    shift: Vec<f64>,
}

impl<P: Problem<Representation = Real>> Shifted<P> {
    /// `problem` shifted by a vector generated from `seed`: the same seed gives the same shift on
    /// every platform.
    pub fn new(problem: P, seed: u64) -> Self {
        let real = problem.representation();
        let anchor = anchor(&problem);
        let mut rng = StreamRng::seed_from_u64(seed).derive(SHIFT_STREAM);
        let shift = real
            .bounds()
            .iter()
            .zip(anchor)
            .map(|(range, anchor)| {
                let (low, high) = (*range.start(), *range.end());
                let target = low + (high - low) * (0.1 + 0.8 * rng.unit_f64());
                target - anchor
            })
            .collect();
        Self {
            problem,
            seed,
            shift,
        }
    }

    /// The wrapped problem.
    pub fn problem(&self) -> &P {
        &self.problem
    }

    /// The seed of the shift.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The shift `o`, a value per gene: the wrapped problem is evaluated at `x − o`.
    pub fn shift(&self) -> &[f64] {
        &self.shift
    }

    // the point at which the wrapped problem is evaluated
    fn unshifted(&self, x: &Reals) -> Reals {
        x.iter().zip(&self.shift).map(|(xi, oi)| xi - oi).collect()
    }
}

impl<P: Problem<Representation = Real>> FitnessFunction<Reals> for Shifted<P> {
    type Output = P::Output;

    fn evaluate(&self, x: &Reals) -> P::Output {
        self.problem.evaluate(&self.unshifted(x))
    }

    /// What the wrapped problem provides.
    fn provides(&self) -> Provided {
        self.problem.provides()
    }

    /// The wrapped problem's [`evaluate_with`](FitnessFunction::evaluate_with) at `x − o`: its
    /// fitness and extras, the same in `x` as in `x − o`.
    ///
    /// # Panics
    ///
    /// As the wrapped problem's.
    fn evaluate_with(&self, x: &Reals, extras: &mut Extras<'_>) -> P::Output {
        self.problem.evaluate_with(&self.unshifted(x), extras)
    }
}

impl<P: Problem<Representation = Real>> Problem for Shifted<P> {
    type Representation = Real;

    /// The wrapped problem's name: a shift makes an instance of the same function.
    fn name(&self) -> &'static str {
        self.problem.name()
    }

    fn representation(&self) -> Real {
        self.problem.representation()
    }

    fn objective(&self) -> Objective {
        self.problem.objective()
    }

    /// The wrapped problem's optimum, its solutions shifted.
    fn optimum(&self) -> Option<Optimum<Reals>> {
        moved_optimum(self.problem.optimum(), &self.representation(), |solution| {
            solution
                .iter()
                .zip(&self.shift)
                .map(|(s, o)| s + o)
                .collect()
        })
    }

    fn reference(&self) -> &'static str {
        self.problem.reference()
    }

    fn reference_url(&self) -> Option<&'static str> {
        self.problem.reference_url()
    }

    fn constraints(&self, genome: &Reals) -> Constraints {
        self.problem.constraints(&self.unshifted(genome))
    }
}

/// A problem rotated by a random orthogonal matrix: `f(c + M (x − c))`, where `f` is the wrapped
/// problem, `M` an orthogonal matrix generated from a seed and `c` the first solution of the
/// wrapped problem's optimum, so that the genes interact and a search can't optimize them one at
/// a time.
///
/// `M` is generated as BBOB generates its rotations (Hansen, Finck, Ros and Auger 2009, section
/// 0.2): a matrix of standard normal numbers, whose rows are made orthonormal by Gram-Schmidt
/// orthonormalization (here twice over, for orthogonality to the precision of `f64`). Like BBOB's
/// and the CEC suites', it may reflect as well as rotate. It turns about the optimum, as CEC 2005
/// does with `z = (x − o) M` and BBOB with `z = R (x − xᵒᵖᵗ)`, so the optimum stays where it is:
/// its first solution, with its value. Other solutions are rotated about it (those that the
/// rotation moves out of the box are dropped). Without a known optimum, `c` is the center of the
/// box.
///
/// The bounds are the wrapped problem's: points of the box can map to points outside it, where
/// the wrapped function must be defined, and a function whose minimum is only the lowest in its
/// box can have lower values there, as for [`Shifted`]. The name and the reference are the wrapped
/// problem's, and so is the constraint violation of a constrained problem, at `c + M (x − c)`.
///
/// It [provides](FitnessFunction::provides) what the wrapped problem provides, and
/// [`evaluate_with`](FitnessFunction::evaluate_with) gives it by the chain rule: the gradient
/// `Mᵀ ∇f(c + M (x − c))`, the constraints' values at `c + M (x − c)`, and their Jacobian `J M`,
/// each row (a constraint's gradient) turned as the gradient is. A rotated smooth function keeps
/// its analytic gradient, and a rotated constrained problem gives its constraints' values to
/// [`Bo`](crate::algorithm::Bo) (and, with their Jacobian, to [`Mma`](crate::algorithm::Mma)).
///
/// Rotating a shifted problem gives CEC 2005's shifted rotated functions, `f((x − o) M)`, e.g. its
/// F10, the shifted rotated Rastrigin, whose rotation turns about the shifted optimum:
///
/// ```
/// use genoxide::prelude::*;
/// use genoxide::problems::{Problem, Rastrigin, Rotated, Shifted};
///
/// let problem = Rotated::new(Shifted::new(Rastrigin::new(5), 1), 1);
/// let optimum = problem.optimum().expect("known");
/// assert_eq!(&optimum.solutions()[0][..], problem.problem().shift());
/// let cmaes = Cmaes::builder(problem.representation())
///     .restarts(cmaes::Restarts::Ipop)
///     .minimize()
///     .seed(1)
///     .build()?;
/// let outcome = Engine::new(cmaes, problem)
///     .stop_when(Stop::target(1e-8).or(Stop::evaluations(200_000)))
///     .run()?;
/// assert_eq!(outcome.stop_reason(), StopReason::Target);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Rotated<P> {
    problem: P,
    seed: u64,
    center: Vec<f64>,
    // row-major, n × n
    matrix: Vec<f64>,
}

impl<P: Problem<Representation = Real>> Rotated<P> {
    /// `problem` rotated by a matrix generated from `seed`: the same seed gives the same matrix
    /// on every platform.
    pub fn new(problem: P, seed: u64) -> Self {
        let center = anchor(&problem);
        let matrix = orthogonal(center.len(), seed);
        Self {
            problem,
            seed,
            center,
            matrix,
        }
    }

    /// The wrapped problem.
    pub fn problem(&self) -> &P {
        &self.problem
    }

    /// The seed of the rotation.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// The orthogonal matrix `M`, row by row: `n × n` values, row `i` from `i n` on.
    pub fn matrix(&self) -> &[f64] {
        &self.matrix
    }

    /// The point `c` that the rotation turns about: the wrapped problem's first optimal solution,
    /// or the center of the box if its optimum isn't known.
    pub fn center(&self) -> &[f64] {
        &self.center
    }

    // `Mᵀ v` into `out`: a gradient at c + M (x − c), or a row of the Jacobian there, as a
    // gradient in x (∂/∂xⱼ = Σᵢ Mᵢⱼ ∂/∂yᵢ, y = c + M (x − c))
    fn turn_back(&self, v: &[f64], out: &mut [f64]) {
        let n = self.center.len();
        out.fill(0.0);
        for (row, vi) in self.matrix.chunks_exact(n.max(1)).zip(v) {
            for (o, m) in out.iter_mut().zip(row) {
                *o += m * vi;
            }
        }
    }

    // the point at which the wrapped problem is evaluated: c + M (x − c)
    fn rotated(&self, x: &Reals) -> Reals {
        let n = self.center.len();
        let difference: Vec<f64> = x.iter().zip(&self.center).map(|(xi, ci)| xi - ci).collect();
        self.matrix
            .chunks_exact(n.max(1))
            .zip(&self.center)
            .map(|(row, ci)| ci + row.iter().zip(&difference).map(|(m, d)| m * d).sum::<f64>())
            .collect()
    }
}

// an n × n orthogonal matrix, row-major: standard normal numbers from `seed`, whose rows are made
// orthonormal by modified Gram-Schmidt, applied twice
fn orthogonal(n: usize, seed: u64) -> Vec<f64> {
    let mut rng = StreamRng::seed_from_u64(seed).derive(ROTATION_STREAM);
    let mut matrix = vec![0.0; n * n];
    let mut i = 0;
    while i < n {
        let (done, rest) = matrix.split_at_mut(i * n);
        let row = &mut rest[..n];
        for value in row.iter_mut() {
            *value = rng.normal();
        }
        for _ in 0..2 {
            for previous in done.chunks_exact(n) {
                let dot: f64 = row.iter().zip(previous).map(|(a, b)| a * b).sum();
                for (value, p) in row.iter_mut().zip(previous) {
                    *value -= dot * p;
                }
            }
        }
        let norm = row.iter().map(|value| value * value).sum::<f64>().sqrt();
        // a row in the span of the others has probability 0: draw it again
        if norm > 1e-8 {
            for value in row.iter_mut() {
                *value /= norm;
            }
            i += 1;
        }
    }
    matrix
}

impl<P: Problem<Representation = Real>> FitnessFunction<Reals> for Rotated<P> {
    type Output = P::Output;

    fn evaluate(&self, x: &Reals) -> P::Output {
        self.problem.evaluate(&self.rotated(x))
    }

    /// What the wrapped problem provides.
    fn provides(&self) -> Provided {
        self.problem.provides()
    }

    /// The wrapped problem's [`evaluate_with`](FitnessFunction::evaluate_with) at
    /// `c + M (x − c)`: its fitness and the constraints' values there, its gradient turned back
    /// by `Mᵀ`, and the Jacobian times `M`.
    ///
    /// # Panics
    ///
    /// As the wrapped problem's, and if the gradient doesn't have a value per gene or the
    /// Jacobian a whole number of rows of a value per gene.
    fn evaluate_with(&self, x: &Reals, extras: &mut Extras<'_>) -> P::Output {
        let n = self.center.len();
        let (gradient, inequalities, jacobian) = extras.buffers();
        // the wrapped problem's derivatives at c + M (x − c), in its own coordinates
        let mut wrapped_gradient = gradient.as_ref().map(|gradient| {
            assert_eq!(gradient.len(), n, "a gradient has a value per gene");
            vec![0.0; n]
        });
        let mut wrapped_jacobian = jacobian.as_ref().map(|jacobian| {
            assert_eq!(
                jacobian.len() % n.max(1),
                0,
                "a Jacobian has a row of a value per gene for each constraint"
            );
            vec![0.0; jacobian.len()]
        });
        let output = self.problem.evaluate_with(
            &self.rotated(x),
            &mut Extras::new(
                wrapped_gradient.as_deref_mut(),
                inequalities,
                wrapped_jacobian.as_deref_mut(),
            ),
        );
        if let (Some(gradient), Some(wrapped)) = (gradient, &wrapped_gradient) {
            self.turn_back(wrapped, gradient);
        }
        if let (Some(jacobian), Some(wrapped)) = (jacobian, &wrapped_jacobian) {
            let rows = jacobian.chunks_exact_mut(n.max(1));
            for (row, wrapped) in rows.zip(wrapped.chunks_exact(n.max(1))) {
                self.turn_back(wrapped, row);
            }
        }
        output
    }
}

impl<P: Problem<Representation = Real>> Problem for Rotated<P> {
    type Representation = Real;

    /// The wrapped problem's name: a rotation makes an instance of the same function.
    fn name(&self) -> &'static str {
        self.problem.name()
    }

    fn representation(&self) -> Real {
        self.problem.representation()
    }

    fn objective(&self) -> Objective {
        self.problem.objective()
    }

    /// The wrapped problem's optimum, its solutions rotated about the first: `c + Mᵀ (s − c)`.
    fn optimum(&self) -> Option<Optimum<Reals>> {
        let n = self.center.len();
        moved_optimum(self.problem.optimum(), &self.representation(), |solution| {
            let difference: Vec<f64> = solution
                .iter()
                .zip(&self.center)
                .map(|(s, c)| s - c)
                .collect();
            (0..n)
                .map(|j| {
                    let column = (0..n).map(|i| self.matrix[i * n + j] * difference[i]);
                    self.center[j] + column.sum::<f64>()
                })
                .collect()
        })
    }

    fn reference(&self) -> &'static str {
        self.problem.reference()
    }

    fn reference_url(&self) -> Option<&'static str> {
        self.problem.reference_url()
    }

    fn constraints(&self, genome: &Reals) -> Constraints {
        self.problem.constraints(&self.rotated(genome))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gradient::Gradients;
    use crate::prelude::{Engine, Lbfgsb, Stop, StopReason};
    use crate::problems::cec2006::G06;
    use crate::problems::{
        BentCigar, Branin, HighConditionedElliptic, Katsuura, Quartic, Rastrigin, Rosenbrock,
        Sphere, Step, boxed,
    };

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
            "{actual} is not {expected}"
        );
    }

    #[test]
    fn a_shift_moves_the_optimum_into_the_middle_of_the_box() {
        for seed in 0..50 {
            let problem = Shifted::new(Rosenbrock::new(6), seed);
            let optimum = problem.optimum().expect("known");
            assert_eq!(optimum.value(), 0.0);
            assert!(optimum.is_proven());
            let solution = &optimum.solutions()[0];
            // in the middle 80% of [−30, 30]
            assert!(solution.iter().all(|x| x.abs() <= 24.0), "{solution:?}");
            // the wrapped minimum (1, …, 1), shifted
            for (x, o) in solution.iter().zip(problem.shift()) {
                assert_close(*x, 1.0 + o, 1e-15);
            }
            assert_close(problem.evaluate(solution), 0.0, 1e-12);
            assert_eq!(problem.seed(), seed);
        }
        // the same seed, the same shift; another seed, another shift
        let a = Shifted::new(Sphere::new(3), 7);
        assert_eq!(a, Shifted::new(Sphere::new(3), 7));
        assert_ne!(a.shift(), Shifted::new(Sphere::new(3), 8).shift());
        // f(x − o)
        let x = Reals::from(vec![1.0, 2.0, 3.0]);
        let expected: f64 = x
            .iter()
            .zip(a.shift())
            .map(|(x, o)| (x - o) * (x - o))
            .sum();
        assert_eq!(a.evaluate(&x), expected);
        assert_eq!(a.name(), "Sphere");
        assert_eq!(a.representation(), Sphere::new(3).representation());
    }

    // fixed values, so that a change of the generator shows
    #[test]
    fn shifts_and_rotations_are_the_same_everywhere() {
        // in the middle 80% of [−100, 100]
        let shift = Shifted::new(Sphere::new(2), 1).shift().to_vec();
        assert_eq!(shift, [46.23653733062747, 24.878588135984046]);
        // a rotation by about 193°: its determinant is 0.9736² + 0.2285² = 1
        let rotation = Rotated::new(Sphere::new(2), 1).matrix().to_vec();
        let expected = [
            -0.9735519448992744,
            0.22846577551756006,
            -0.2284657755175601,
            -0.9735519448992744,
        ];
        assert_eq!(rotation, expected);
    }

    #[test]
    fn rotations_are_orthogonal() {
        for (n, seed) in [(1, 0), (2, 1), (5, 2), (30, 3), (100, 4)] {
            let problem = Rotated::new(Sphere::new(n), seed);
            let m = problem.matrix();
            for i in 0..n {
                for j in 0..n {
                    let dot: f64 = (0..n).map(|k| m[i * n + k] * m[j * n + k]).sum();
                    let expected = if i == j { 1.0 } else { 0.0 };
                    assert!((dot - expected).abs() < 1e-14, "{n}: {i} {j} {dot}");
                }
            }
            // a sphere about the origin doesn't change
            let x: Reals = (0..n).map(|i| i as f64 - 1.5).collect();
            assert_close(problem.evaluate(&x), Sphere::new(n).evaluate(&x), 1e-12);
        }
        assert_ne!(
            Rotated::new(Sphere::new(4), 1).matrix(),
            Rotated::new(Sphere::new(4), 2).matrix()
        );
    }

    #[test]
    fn a_rotation_keeps_the_optimum_in_place() {
        let problem = Rotated::new(Rosenbrock::new(5), 3);
        let optimum = problem.optimum().expect("known");
        assert_eq!(&optimum.solutions()[0][..], &[1.0; 5]);
        assert_eq!(problem.center(), &[1.0; 5]);
        assert_eq!(problem.evaluate(&optimum.solutions()[0]), 0.0);
        // the other points move: the wrapped function at c + M (x − c)
        let x = Reals::from(vec![0.0; 5]);
        let m = problem.matrix();
        let y: Reals = (0..5)
            .map(|i| 1.0 + (0..5).map(|j| -m[i * 5 + j]).sum::<f64>())
            .collect();
        assert_close(problem.evaluate(&x), Rosenbrock::new(5).evaluate(&y), 1e-12);
    }

    // CEC 2005's F10: Rastrigin, shifted, then rotated about the shifted optimum
    #[test]
    fn shifted_and_rotated_rastrigin() {
        let problem = Rotated::new(Shifted::new(Rastrigin::new(10), 5), 5);
        let shift = problem.problem().shift().to_vec();
        let optimum = problem.optimum().expect("known");
        assert_eq!(optimum.solutions()[0].to_vec(), shift);
        assert_close(problem.evaluate(&optimum.solutions()[0]), 0.0, 1e-12);
        // f(M (x − o)): at x = o + Mᵀ e₁, the wrapped Rastrigin sees e₁, where it's 1
        let m = problem.matrix();
        let x: Reals = (0..10).map(|j| shift[j] + m[j]).collect();
        assert_close(problem.evaluate(&x), 1.0, 1e-9);
    }

    // several solutions: those that leave the box are dropped; the first stays
    #[test]
    fn other_solutions_move_with_the_first() {
        for seed in 0..20 {
            let rotated = Rotated::new(Branin, seed);
            let optimum = rotated.optimum().expect("known");
            assert!(!optimum.solutions().is_empty());
            for solution in optimum.solutions() {
                assert_close(rotated.evaluate(solution), optimum.value(), 1e-9);
            }
            let shifted = Shifted::new(Branin, seed);
            for solution in shifted.optimum().expect("known").solutions() {
                assert_close(shifted.evaluate(solution), optimum.value(), 1e-9);
            }
        }
    }

    #[test]
    fn constraints_are_transformed_too() {
        let problem = Shifted::new(G06, 2);
        let optimum = problem.optimum().expect("known");
        let solution = &optimum.solutions()[0];
        let unshifted: Reals = solution
            .iter()
            .zip(problem.shift())
            .map(|(x, o)| x - o)
            .collect();
        assert_eq!(problem.constraints(solution), G06.constraints(&unshifted));
        let rotated = Rotated::new(G06, 2);
        let center = Reals::from(rotated.center().to_vec());
        assert_eq!(rotated.constraints(&center), G06.constraints(&center));
    }

    // a smooth constrained problem with every extra: Σ (xᵢ − 1)² + x₀ x₁, subject to
    // ‖x‖² − 4 <= 0 and x₀ + 2x₁ − x₂³ <= 0, with its gradient and its constraints' Jacobian
    #[derive(Clone, Debug, PartialEq)]
    struct Smooth;

    impl Smooth {
        fn values(x: &[f64]) -> [f64; 2] {
            let squares: f64 = x.iter().map(|xi| xi * xi).sum();
            [squares - 4.0, x[0] + 2.0 * x[1] - x[2] * x[2] * x[2]]
        }
    }

    impl FitnessFunction<Reals> for Smooth {
        type Output = (f64, f64);

        fn evaluate(&self, x: &Reals) -> (f64, f64) {
            let value = x.iter().map(|xi| (xi - 1.0) * (xi - 1.0)).sum::<f64>() + x[0] * x[1];
            (value, self.constraints(x).violation(0.0))
        }

        fn provides(&self) -> Provided {
            Provided::GRADIENT
                .with_inequalities(2)
                .with_constraint_jacobian()
        }

        fn evaluate_with(&self, x: &Reals, extras: &mut Extras<'_>) -> (f64, f64) {
            let (gradient, inequalities, jacobian) = extras.buffers();
            if let Some(gradient) = gradient {
                for (g, xi) in gradient.iter_mut().zip(x.iter()) {
                    *g = 2.0 * (xi - 1.0);
                }
                gradient[0] += x[1];
                gradient[1] += x[0];
            }
            if let Some(g) = inequalities {
                g.copy_from_slice(&Self::values(x));
            }
            if let Some(jacobian) = jacobian {
                let (first, second) = jacobian.split_at_mut(3);
                for (j, xi) in first.iter_mut().zip(x.iter()) {
                    *j = 2.0 * xi;
                }
                second.copy_from_slice(&[1.0, 2.0, -3.0 * x[2] * x[2]]);
            }
            self.evaluate(x)
        }
    }

    impl Problem for Smooth {
        type Representation = Real;

        fn name(&self) -> &'static str {
            "Smooth"
        }

        fn representation(&self) -> Real {
            Real::uniform(3, -2.0..=2.0).expect("valid bounds")
        }

        fn optimum(&self) -> Option<Optimum<Reals>> {
            None
        }

        fn reference(&self) -> &'static str {
            "a test problem"
        }

        fn constraints(&self, x: &Reals) -> Constraints {
            Constraints::new(Self::values(x).to_vec(), Vec::new())
        }
    }

    // the gradient, the constraints' values and their Jacobian of `problem` at `x`, all wanted
    fn extras_of<P>(problem: &P, x: &Reals) -> (P::Output, Vec<f64>, Vec<f64>, Vec<f64>)
    where
        P: Problem<Representation = Real>,
    {
        let (n, m) = (x.len(), problem.provides().inequalities);
        let (mut gradient, mut g, mut jacobian) = (vec![0.0; n], vec![0.0; m], vec![0.0; m * n]);
        let output = problem.evaluate_with(
            x,
            &mut Extras::new(Some(&mut gradient), Some(&mut g), Some(&mut jacobian)),
        );
        (output, gradient, g, jacobian)
    }

    // the slope of `f` along gene `gene` at `x`, by central differences with the step 2⁻¹⁸
    fn slope(f: impl Fn(&Reals) -> f64, x: &Reals, gene: usize) -> f64 {
        let h = 2f64.powi(-18);
        let (mut above, mut below) = (x.clone(), x.clone());
        above[gene] += h;
        below[gene] -= h;
        (f(&above) - f(&below)) / (above[gene] - below[gene])
    }

    // the wrappers' gradients and Jacobians against central differences of their values, and
    // their values the same to the bit with and without extras
    #[test]
    fn the_wrappers_derivatives_match_central_differences() {
        let mut rng = StreamRng::seed_from_u64(1);
        for seed in 0..10 {
            let shifted = Shifted::new(Smooth, seed);
            let rotated = Rotated::new(Smooth, seed);
            let both = Rotated::new(Shifted::new(Smooth, seed), seed);
            for _ in 0..20 {
                let x = Smooth.representation().random_genome(&mut rng);
                check_derivatives(&shifted, &x);
                check_derivatives(&rotated, &x);
                check_derivatives(&both, &x);
            }
        }
    }

    fn check_derivatives<P>(problem: &P, x: &Reals)
    where
        P: Problem<Representation = Real, Output = (f64, f64)>,
    {
        let (output, gradient, g, jacobian) = extras_of(problem, x);
        let plain = problem.evaluate(x);
        assert_eq!(output.0.to_bits(), plain.0.to_bits());
        assert_eq!(output.1.to_bits(), plain.1.to_bits());
        assert_eq!(g, problem.constraints(x).inequalities());
        let close = |analytic: f64, estimate: f64| {
            let error = (analytic - estimate).abs() / analytic.abs().max(1.0);
            assert!(error < 1e-8, "{analytic} against {estimate}");
        };
        for gene in 0..x.len() {
            close(gradient[gene], slope(|x| problem.evaluate(x).0, x, gene));
            for constraint in 0..2 {
                let estimate = slope(
                    |x| problem.constraints(x).inequalities()[constraint],
                    x,
                    gene,
                );
                close(jacobian[constraint * x.len() + gene], estimate);
            }
        }
    }

    // a shift doesn't change the extras: the wrapped problem's at x − o, to the bit
    #[test]
    fn a_shift_passes_the_extras_through() {
        let problem = Shifted::new(Smooth, 4);
        let x = Reals::from(vec![0.5, -1.0, 1.5]);
        let unshifted = problem.unshifted(&x);
        assert_eq!(extras_of(&problem, &x), extras_of(&Smooth, &unshifted));
        // a test problem's analytic gradient, shifted
        let rosenbrock = Shifted::new(Rosenbrock::new(4), 2);
        let x = Reals::from(vec![0.3, -0.2, 1.1, 2.0]);
        let mut gradient = [0.0; 4];
        let mut expected = [0.0; 4];
        let value = rosenbrock.evaluate_with(&x, &mut Extras::with_gradient(&mut gradient));
        let wrapped = Rosenbrock::new(4).evaluate_with(
            &rosenbrock.unshifted(&x),
            &mut Extras::with_gradient(&mut expected),
        );
        assert_eq!((value, gradient), (wrapped, expected));
    }

    // a rotation turns the gradient back by Mᵀ and the Jacobian's rows likewise
    #[test]
    fn a_rotation_turns_the_derivatives_back() {
        let problem = Rotated::new(Smooth, 6);
        let x = Reals::from(vec![0.5, -1.0, 1.5]);
        let y = problem.rotated(&x);
        let (output, gradient, g, jacobian) = extras_of(&problem, &x);
        let (wrapped, inner_gradient, inner_g, inner_jacobian) = extras_of(&Smooth, &y);
        assert_eq!((output, g), (wrapped, inner_g));
        let m = problem.matrix();
        let turned = |v: &[f64], j: usize| (0..3).map(|i| m[i * 3 + j] * v[i]).sum::<f64>();
        for j in 0..3 {
            assert!((gradient[j] - turned(&inner_gradient, j)).abs() < 1e-14);
            for row in 0..2 {
                let expected = turned(&inner_jacobian[row * 3..row * 3 + 3], j);
                assert!((jacobian[row * 3 + j] - expected).abs() < 1e-14);
            }
        }
        // the gradient of a sphere about its center is turned to the same one: M orthogonal
        let sphere = Rotated::new(Sphere::new(5), 3);
        let x = Reals::from(vec![1.0, -2.0, 0.5, 3.0, -1.5]);
        let mut gradient = [0.0; 5];
        sphere.evaluate_with(&x, &mut Extras::with_gradient(&mut gradient));
        for (g, xi) in gradient.iter().zip(x.iter()) {
            assert!((g - 2.0 * xi).abs() < 1e-13, "{g} {xi}");
        }
    }

    // the wrappers provide what they wrap
    #[test]
    fn the_wrappers_provide_what_they_wrap() {
        let gradient = Provided::GRADIENT;
        assert_eq!(Shifted::new(Sphere::new(3), 1).provides(), gradient);
        assert_eq!(Rotated::new(Rastrigin::new(3), 1).provides(), gradient);
        let both = Rotated::new(Shifted::new(BentCigar::new(3), 1), 1);
        assert_eq!(both.provides(), gradient);
        assert_eq!(
            Rotated::new(Shifted::new(Smooth, 1), 1).provides(),
            Smooth.provides()
        );
        let values = Provided::NOTHING.with_inequalities(2);
        assert_eq!(Shifted::new(G06, 1).provides(), values);
        assert_eq!(Rotated::new(G06, 1).provides(), values);
        assert_eq!(Rotated::new(Shifted::new(G06, 1), 1).provides(), values);
        let nothing = Provided::NOTHING;
        assert_eq!(Shifted::new(Step::new(3), 1).provides(), nothing);
        assert_eq!(Rotated::new(Katsuura::new(3), 1).provides(), nothing);
        assert_eq!(
            Rotated::new(Shifted::new(Quartic::noisy(3), 1), 1).provides(),
            nothing
        );
        assert_eq!(Shifted::new(Quartic::new(3), 1).provides(), gradient);
        // and so do they behind DynProblem
        assert_eq!(boxed(Shifted::new(G06, 1)).provides(), values);
        assert_eq!(boxed(Rotated::new(Sphere::new(2), 1)).provides(), gradient);
    }

    // a shifted or rotated G06 gives its constraints' values: those of G06 at the point it's
    // evaluated at
    #[test]
    fn a_wrapped_g06_keeps_its_constraint_values() {
        let x = Reals::from(vec![20.0, 5.0]);
        let shifted = Shifted::new(G06, 2);
        let rotated = Rotated::new(G06, 2);
        for (problem, at) in [
            (&shifted as &dyn DynFitness, shifted.unshifted(&x)),
            (&rotated as &dyn DynFitness, rotated.rotated(&x)),
        ] {
            let mut g = [0.0; 2];
            let fitness = problem.evaluate_with(&x, &mut Extras::new(None, Some(&mut g), None));
            assert_eq!(fitness, G06.evaluate(&at));
            assert_eq!(&g[..], G06.constraints(&at).inequalities());
        }
    }

    // a fitness function of (score, violation) on Real genomes, as a trait object
    trait DynFitness {
        fn evaluate_with(&self, x: &Reals, extras: &mut Extras<'_>) -> (f64, f64);
    }

    impl<F: FitnessFunction<Reals, Output = (f64, f64)>> DynFitness for F {
        fn evaluate_with(&self, x: &Reals, extras: &mut Extras<'_>) -> (f64, f64) {
            FitnessFunction::evaluate_with(self, x, extras)
        }
    }

    // L-BFGS-B takes the analytic gradient through both wrappers, and needs no differences
    #[test]
    fn lbfgsb_runs_on_the_wrapped_gradient() -> crate::Result<()> {
        let problem = Rotated::new(Shifted::new(HighConditionedElliptic::new(10), 1), 1);
        let optimum = problem.optimum().expect("known");
        let lbfgsb = Lbfgsb::builder(problem.representation())
            .gradients(Gradients::Supplied)
            .gradient_tolerance(1e-9)
            .function_tolerance(0.0)
            .minimize()
            .seed(1)
            .build()?;
        let mut engine = Engine::new(lbfgsb, problem).stop_when(Stop::evaluations(10_000));
        let outcome = engine.run()?;
        assert_eq!(outcome.stop_reason(), StopReason::Converged);
        assert!(outcome.best_fitness().score().expect("valid") < 1e-16);
        for (x, o) in outcome
            .best_genome()
            .iter()
            .zip(optimum.solutions()[0].iter())
        {
            assert!((x - o).abs() < 1e-8, "{x} {o}");
        }
        assert_eq!(engine.algorithm().stencil_evaluations(), 0);
        Ok(())
    }
}
