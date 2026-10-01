//! The method of moving asymptotes (MMA) and its globally convergent form (GCMMA): Svanberg's
//! methods for smooth problems with very many variables and few inequality constraints.

use super::{Algorithm, Candidates, Reevaluate};
use crate::engine::{Evaluations, Provided, Wanted};
use crate::genome::{Real, Reals, Representation};
use crate::linalg;
use crate::{Error, Fitness, Individual, Objective, Population, Result, StreamRng};
use rand::Rng;
use std::ops::{Range, RangeInclusive};

#[cfg(test)]
mod tests;

/// Which of Svanberg's two methods an [`Mma`] runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Method {
    /// The method of moving asymptotes (the default): one subproblem and one evaluation per
    /// iteration. Fast in practice, without a proof of convergence from any start.
    #[default]
    Mma,
    /// The globally convergent form: an iteration's point is accepted only once the objective's
    /// and every constraint's approximations are conservative there (at least the function's
    /// value); otherwise the approximations get more curvature and the subproblem is solved
    /// again, at the cost of another evaluation (an inner iteration). Each iterate is then
    /// feasible once one is, and better than the last: Svanberg (2002) proves convergence to a
    /// KKT point from any start.
    Gcmma,
}

/// Why an [`Mma`] run has converged: [`Mma::converged`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Convergence {
    /// The KKT residual at the last iterate, with the subproblem's multipliers, is within the
    /// [tolerance](MmaBuilder::kkt_tolerance): see [`Mma::kkt_residual`].
    Kkt,
    /// The last step moved no gene by more than the
    /// [tolerance](MmaBuilder::step_tolerance), as a fraction of its range.
    Step,
}

// the constants of Svanberg's (2007) notes that aren't settings: eqs. 3.3-3.4's 1.001, 0.001 and
// 10⁻⁵ (raa0), eqs. 3.6-3.7's 0.1 (albefa), and eq. 3.14's 0.01 and 10
const RAA0: f64 = 1e-5;
const ALBEFA: f64 = 0.1;
const ASYMPTOTE_NEAREST: f64 = 0.01;
const ASYMPTOTE_FARTHEST: f64 = 10.0;
// GCMMA: eq. 4.6's 0.1 and its least value 10⁻⁶, and eq. 4.9's 1.1 and 10
const RHO_FRACTION: f64 = 0.1;
const RHO_MIN: f64 = 1e-6;
const RHO_GROWTH: f64 = 1.1;
const RHO_MOST_GROWTH: f64 = 10.0;
// d of the artificial variables, as in Svanberg's (2002) tests (section 8.4)
const D: f64 = 1.0;
// genoxide's: the most inner iterations of an outer one, after which its point is accepted
// anyway (the paper's lemma 7.2 proves a finite number, which rounding can defeat); the tolerance
// of the conservativeness test, relative to the scales of the approximation's terms and of
// evaluating the function, times √n; the most attempts of the restoration step
const MAX_INNER: u64 = 30;
const CONSERVATIVE_TOLERANCE: f64 = 100.0 * f64::EPSILON;
const RESTORATION_ATTEMPTS: u32 = 8;
// the dual solver's limits: Newton iterations, points of each search along a Newton direction,
// and the tolerance of the dual gradient relative to the scale of its terms
const DUAL_ITERATIONS: usize = 100;
const DUAL_SEARCH: usize = 40;
// the genes of a chunk of the dual's sums
const CHUNK: usize = 4096;
const DUAL_TOLERANCE: f64 = 1e-12;

/// What the next ask of an [`Mma`] evaluates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum Phase {
    // the initial point
    Start,
    // the solution of the next subproblem: the next iterate, or (GCMMA) an inner iteration's
    Trial,
    // a restoration of the converged point onto the feasible side of its near-active constraints
    Restore { attempt: u32 },
    // converged, with nothing more to do
    Finished,
}

/// The asymptotes and the iterates their update rule reads (Svanberg 2007, eqs. 3.11-3.14):
/// the state a continuation keeps between stages, apart from the point itself.
#[derive(Clone, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct MovingAsymptotes {
    // lⱼ⁽ᵏ⁾ and uⱼ⁽ᵏ⁾
    lower: Vec<f64>,
    upper: Vec<f64>,
    // x⁽ᵏ⁻¹⁾ and x⁽ᵏ⁻²⁾
    previous: Vec<f64>,
    before_previous: Vec<f64>,
    // k, the outer iteration whose asymptotes these are: 1 at the initial point
    iteration: u64,
}

/// The settings of an [`Mma`], from its builder.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
struct Settings {
    method: Method,
    asymptote_initial: f64,
    asymptote_decrease: f64,
    asymptote_increase: f64,
    move_limit: f64,
    constraint_cost: f64,
    kkt_tolerance: f64,
    step_tolerance: f64,
    restoration: bool,
    #[cfg_attr(feature = "serde", serde(default))]
    parallel: bool,
}

/// The method of moving asymptotes (MMA) and its globally convergent form (GCMMA) on [`Real`]
/// genomes, as an ask / tell [`Algorithm`]: for smooth problems with very many variables (up to
/// millions) and few inequality constraints `gᵢ(x) ≤ 0` (up to a few hundred), with the gradient
/// of the score and of every constraint.
///
/// Each iteration replaces the objective and every constraint by a convex, separable
/// approximation around the current point x⁽ᵏ⁾, built from their values and gradients there and
/// from two asymptotes per gene, `lⱼ < xⱼ < uⱼ`:
///
/// ```text
/// f̃ᵢ(x) = rᵢ + Σⱼ ( pᵢⱼ / (uⱼ − xⱼ) + qᵢⱼ / (xⱼ − lⱼ) )
/// ```
///
/// with pᵢⱼ from the positive part of `∂fᵢ/∂xⱼ` and qᵢⱼ from the negative part, so that f̃ᵢ has
/// fᵢ's value and gradient at x⁽ᵏ⁾ (Svanberg 2007, eqs. 3.2-3.5). The asymptotes move with the
/// iterates: closer to the point where a gene oscillates, which adds curvature and damps it,
/// farther where it moves steadily, which lets it go (eqs. 3.11-3.14). The approximate problem,
/// with each gene also kept within a move limit (eqs. 3.6-3.7), is solved through its dual in the
/// m constraints' multipliers: for given multipliers, each gene's minimizer has a closed form, so
/// the dual function and its derivatives are sums over the genes. Its solution is the next point.
///
/// - **Problem.** A [`Constrained::differentiable`](crate::constraint::Constrained::differentiable)
///   fitness function gives the score, its gradient, the constraints' values and their Jacobian;
///   a [`Differentiable`](crate::gradient::Differentiable) one, a problem without constraints
///   (bounds only). The [`Engine`](crate::Engine) checks them when a run starts. MMA handles
///   inequality constraints only: an equality `h(x) = 0` isn't supported (two inequalities
///   `h ≤ ε` and `−h ≤ ε` make a relaxed one). Bounds are the [`Real`]'s; fixed genes stay as
///   they are.
/// - **Infeasible subproblems.** Each constraint i gets an artificial variable `yᵢ ≥ 0` that
///   relaxes it, at a cost `c yᵢ + ½ yᵢ²` in the objective (Svanberg 2007, eq. 1.1, with
///   `d = 1`), so every subproblem has a solution. With a large enough
///   [cost](MmaBuilder::constraint_cost) `c`, larger than every multiplier, `yᵢ = 0` at a
///   feasible problem's solution. A constraint's [multiplier](Mma::multipliers) at `c` or above
///   means it couldn't be met there.
/// - **Methods.** [`Method::Mma`] (the default) takes the subproblem's solution as the next point.
///   [`Method::Gcmma`] accepts it only if every approximation is conservative there, and otherwise
///   adds curvature and solves again (Svanberg 2007, section 4): more evaluations, but convergence
///   from any start. MMA's approximation of a function adds curvature in proportion to its
///   gradient, so around a minimum inside the bounds where the objective's gradient vanishes (a
///   problem without active constraints, such as a least-squares fit) it can cycle, the
///   asymptotes at their nearest; GCMMA converges there. Like the notes (section 2), MMA expects
///   genes scaled to ranges of about 0.1 to 100 and a score of about 1 to 100.
/// - **Convergence.** After each iteration, the KKT conditions are measured at the new point with
///   the subproblem's multipliers, as in Svanberg (2002, eqs. 8.3-8.4), and the step against
///   each gene's range: the run has [finished](Algorithm::is_finished) when either is within its
///   tolerance, and [`converged`](Mma::converged) says which.
/// - **Restoration.** A method that converges to active constraints approaches them from either
///   side, so its last point can be infeasible by rounding (`gᵢ(x) = 10⁻¹⁵`), which Deb's rules
///   count. When a run converges to an infeasible point, MMA moves it onto the feasible side of
///   its active constraints, with the smallest step in the genes that aren't at a bound, using the
///   last Jacobian: a margin of the violation's size first, doubled on each of up to 8 attempts
///   until the point is feasible. Each attempt is an evaluation; the best by Deb's rules is
///   kept, as always. [`restoration`](MmaBuilder::restoration) turns it off.
/// - **Best.** [`best`](Algorithm::best) is the best point evaluated by the objective and Deb's
///   rules on the violation, as everywhere in genoxide: the iterates aren't monotone in MMA.
/// - **Generations.** A generation is one evaluation: the initial point, an iterate, an inner
///   iteration's rejected point (GCMMA), or a restoration attempt. Each asks for the gradient and
///   the constraint Jacobian, but a restoration attempt, which asks for the constraints' values
///   only.
/// - **Invalid points.** A point whose fitness is invalid, or whose gradient or constraints aren't
///   finite, isn't accepted: the asymptotes move closer to the current point, and once they're at
///   their nearest, the move limit halves, until a point is valid. The initial point must be
///   valid.
///
/// **Scale.** Memory and work are O(n·m) for n genes and m constraints, with no n × n matrix: the
/// state holds a few vectors of n values and the m × n Jacobian, and an iteration makes a pass
/// over the genes for each evaluation of the dual function, typically 5 to 20; each also builds
/// the dual's m × m Hessian, O(n·m²). With a million genes and one constraint, an iteration takes
/// about 0.17 s on one thread, or 0.05 s with [parallel sums](MmaBuilder::parallel_sums) on 20, and
/// the run 15 values of memory per gene besides the representation. Nothing is allocated after
/// the first iteration, and every sum is in a fixed order, so a run gives the same bits on every
/// platform and thread count.
///
/// The fitness is minimized or maximized as the [objective](MmaBuilder::objective) says; the
/// gradient is of the score as the fitness function returns it.
///
/// ```
/// use genoxide::constraint::Constrained;
/// use genoxide::prelude::*;
///
/// // minimize Σ cⱼ/xⱼ subject to Σ xⱼ <= 10: the minimum is at xⱼ = 10 √cⱼ / Σ √cₖ
/// let c = [1.0, 4.0, 9.0, 16.0];
/// let problem = Constrained::differentiable(1, |x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]| {
///     let mut value = 0.0;
///     for j in 0..4 {
///         value += c[j] / x[j];
///         gradient[j] = -c[j] / (x[j] * x[j]);
///         jacobian[j] = 1.0;
///     }
///     g[0] = x.iter().sum::<f64>() - 10.0;
///     value
/// });
/// let mma = Mma::builder(Real::uniform(4, 0.1..=10.0)?)
///     .initial_genome(Reals::from(vec![1.0; 4]))
///     .minimize()
///     .build()?;
/// let outcome = Engine::new(mma, problem).stop_when(Stop::evaluations(500)).run()?;
/// assert_eq!(outcome.stop_reason(), StopReason::Converged);
/// let x = outcome.best_genome();
/// for (xj, cj) in x.iter().zip(c) {
///     assert!((xj - cj.sqrt()).abs() < 1e-6, "{x:?}"); // Σ √cₖ = 10
/// }
/// # Ok::<(), genoxide::Error>(())
/// ```
///
/// References:
///
/// - Svanberg, K. (1987). The method of moving asymptotes: a new method for structural
///   optimization. *International Journal for Numerical Methods in Engineering* 24(2): 359-373.
///   doi:10.1002/nme.1620240207. The approximations (eqs. 2-5), the subproblem and its dual
///   (section 4, eqs. 14-21), and the artificial variables (section 5).
/// - Svanberg, K. (2002). A class of globally convergent optimization methods based on
///   conservative convex separable approximations. *SIAM Journal on Optimization* 12(2):
///   555-573. doi:10.1137/S1052623499362822. GCMMA's outer and inner iterations (section 3), the
///   problem with artificial variables (eq. 2.3), the convergence test (eqs. 8.3-8.4).
/// - Svanberg, K. (2007). *MMA and GCMMA – two methods for nonlinear optimization.* Notes, KTH,
///   Stockholm. The approximations and move limits (eqs. 3.2-3.7), the asymptotes' update with
///   its default constants (eqs. 3.11-3.14), and GCMMA's curvature parameters (eqs. 4.2-4.9),
///   as implemented here.
///
/// The notes solve the subproblem by a primal-dual interior-point method (their section 5);
/// genoxide solves it through its dual, as the 1987 paper does (section 4) and the 2002 paper
/// suggests (section 5: a Newton-type method on the multipliers, with an active set for those at
/// 0), which costs one pass over the genes per evaluation of the dual: on a million genes, many
/// times fewer than the interior point's. With the artificial variables, the dual is smooth and
/// bounded above. The Newton method's details (its search along each direction, for the secant
/// root of the directional derivative, which concavity brackets), the subproblem's form as
/// differences from x⁽ᵏ⁾ without eq. 3.5's cancellation, the tolerances, the KKT residual's
/// scaling by the ranges, the handling of invalid points and the restoration step are genoxide's
/// own. The tests reproduce the 2002 paper's table 8.1 (its problem 1 with n = 1000: the
/// objective, the variables at a bound and the multipliers to their printed digits) and the 1987
/// paper's cantilever beam (eq. 22).
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Mma {
    real: Real,
    settings: Settings,
    objective: Objective,
    seed: u64,
    // m, once a run's `prepare` (or the first evaluations) said it
    constraints: Option<usize>,
    // the current point x⁽ᵏ⁾, a population of one
    current: Population<Reals>,
    // at x⁽ᵏ⁾, in the direction of minimization: f₀ (the score, negated to maximize) and the
    // constraints' values; the gradient of f₀; the constraints' Jacobian, a row per constraint
    values: Vec<f64>,
    gradient: Vec<f64>,
    jacobian: Vec<f64>,
    asymptotes: MovingAsymptotes,
    // GCMMA's ρᵢ, for f₀ and each constraint; MMA's are raa0
    rho: Vec<f64>,
    // the rounding scale of evaluating f₀ and each constraint at x⁽ᵏ⁾: |fᵢ| + Σⱼ |∂fᵢ/∂xⱼ xⱼ|
    rounding: Vec<f64>,
    // the multipliers of the last subproblem
    multipliers: Vec<f64>,
    // the point asked for, and at it (from the subproblem that gave it) the approximations f̃ᵢ
    // and the scales of their terms, and GCMMA's d(x) (Svanberg 2007, eq. 4.7)
    trial: Individual<Reals>,
    approximations: Vec<f64>,
    scales: Vec<f64>,
    distance: f64,
    // the fraction of the move limit, halved after invalid points with the asymptotes nearest
    move_fraction: f64,
    phase: Phase,
    converged: Option<Convergence>,
    kkt_residual: f64,
    iterations: u64,
    inner: u64,
    inner_iterations: u64,
    reevaluating: bool,
    asked: bool,
    discarded_trial: bool,
    generation: u64,
    evaluations: u64,
    best: Option<Individual<Reals>>,
    best_generation: u64,
    #[cfg_attr(feature = "serde", serde(skip))]
    scratch: Scratch,
}

// the dual function at some multipliers: its value, gradient, Hessian (row-major, m × m) and the
// scales of the gradient's terms
#[derive(Clone, Debug, Default)]
struct DualPoint {
    value: f64,
    gradient: Vec<f64>,
    hessian: Vec<f64>,
    scales: Vec<f64>,
}

impl DualPoint {
    fn resize(&mut self, m: usize) {
        self.gradient.resize(m, 0.0);
        self.hessian.resize(m * m, 0.0);
        self.scales.resize(m, 0.0);
    }
}

// buffers reused from one subproblem to the next, never saved
#[derive(Clone, Debug, Default)]
struct Scratch {
    at: DualPoint,
    next: DualPoint,
    // the farthest point of a search still rising, and its multipliers
    best: DualPoint,
    rising: Vec<f64>,
    candidate: Vec<f64>,
    direction: Vec<f64>,
    free: Vec<usize>,
    matrix: Vec<f64>,
    rhs: Vec<f64>,
    work: Work,
    // each gene's place in the subproblem
    genes: Vec<Gene>,
}

// the buffers of a pass of the dual over the genes: a row of partial sums per chunk, and per gene
// (for more than 4 constraints) sᵖ and s^q of f₀ and each constraint and the constraints' slopes
#[derive(Clone, Debug, Default)]
struct Work {
    partials: Vec<f64>,
    coefficients: Vec<f64>,
    slopes: Vec<f64>,
}

impl Scratch {
    fn resize(&mut self, m: usize) {
        self.at.resize(m);
        self.next.resize(m);
        self.best.resize(m);
        for buffer in [
            &mut self.rising,
            &mut self.candidate,
            &mut self.direction,
            &mut self.rhs,
            &mut self.work.slopes,
        ] {
            buffer.resize(m, 0.0);
        }
        self.matrix.resize(m * m, 0.0);
        self.work.coefficients.resize(2 * (m + 1), 0.0);
        if self.free.capacity() < m {
            self.free.reserve(m);
        }
    }
}

// a gene's place in the subproblem: the distances to its asymptotes, a = u − x and b = x − l;
// the range of its step δ = xⱼ − x⁽ᵏ⁾ⱼ, from the move limits and bounds; and 1 / the width of its
// range (1 for a fixed gene, which can't move)
#[derive(Clone, Copy, Debug)]
struct Gene {
    a: f64,
    b: f64,
    low: f64,
    high: f64,
    // 1 / width
    inverse: f64,
}

// the approximations of an iteration, around x⁽ᵏ⁾: everything the subproblem reads
struct Approximation<'a> {
    x: &'a [f64],
    lower: &'a [f64],
    upper: &'a [f64],
    bounds: &'a [RangeInclusive<f64>],
    gradient: &'a [f64],
    jacobian: &'a [f64],
    values: &'a [f64],
    rho: &'a [f64],
    move_limit: f64,
    cost: f64,
    parallel: bool,
    // each gene's place, from `place`
    genes: &'a [Gene],
}

impl Approximation<'_> {
    fn m(&self) -> usize {
        self.values.len() - 1
    }

    #[inline]
    fn gene(&self, j: usize) -> Gene {
        self.genes[j]
    }

    // gene j's place in the subproblem
    fn place(&self, j: usize) -> Gene {
        let x = self.x[j];
        let (low, high) = (*self.bounds[j].start(), *self.bounds[j].end());
        let range = high - low;
        let width = if range > 0.0 { range } else { 1.0 };
        let (a, b) = (self.upper[j] - x, x - self.lower[j]);
        // eqs. 3.6-3.7 as steps: within the bounds, 0.9 of the way to each asymptote and the move
        // limit; a fixed gene's step is 0
        let move_limit = self.move_limit * range;
        Gene {
            a,
            b,
            low: (low - x).max(-(1.0 - ALBEFA) * b).max(-move_limit),
            high: (high - x).min((1.0 - ALBEFA) * a).min(move_limit),
            inverse: 1.0 / width,
        }
    }

    // sᵖᵢ and s^qᵢ of gene j for f₀ and each constraint: pᵢⱼ = a² sᵖᵢ and qᵢⱼ = b² s^qᵢ (eqs.
    // 3.3-3.4, 4.3-4.4: 1.001 of the derivative's positive part, 0.001 of its negative part, and
    // ρᵢ / (xmax − xmin))
    #[inline]
    fn coefficients(&self, j: usize, gene: &Gene, out: &mut [f64]) {
        let n = self.x.len();
        for (i, pair) in out.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let derivative = if i == 0 {
                self.gradient[j]
            } else {
                self.jacobian[(i - 1) * n + j]
            };
            *pair = self::pair(derivative, self.rho[i] * gene.inverse);
        }
    }

    // the minimizer of the Lagrangian in gene j for the weights sᵖ = Σ λᵢ sᵖᵢ, s^q = Σ λᵢ s^qᵢ
    // (with λ₀ = 1), as a step: eq. 19 of the 1987 paper, x = (√P l + √Q u) / (√P + √Q) with
    // P = a² sᵖ and Q = b² s^q, moved within its range
    #[inline]
    fn step(gene: &Gene, sp: f64, sq: f64) -> f64 {
        let (root_p, root_q) = (gene.a * sp.sqrt(), gene.b * sq.sqrt());
        let step = gene.a * gene.b * (sq.sqrt() - sp.sqrt()) / (root_p + root_q);
        step.max(gene.low).min(gene.high)
    }

    // the dual function at `lambda` into `point`, with its gradient and Hessian. The genes are
    // summed in chunks of `CHUNK`, each into its row of `partials` (the sum over the genes, the
    // gradient, the scales and the Hessian), then the chunks in order: the same bits whether the
    // chunks are summed one after the other or in parallel, on any number of threads
    fn dual(&self, lambda: &[f64], point: &mut DualPoint, work: &mut Work) {
        let (n, m) = (self.x.len(), self.m());
        let width = 1 + 2 * m + m * m;
        let chunks = n.div_ceil(CHUNK);
        work.partials.resize(chunks * width, 0.0);
        let partial =
            |chunk: usize, out: &mut [f64], coefficients: &mut [f64], slopes: &mut [f64]| {
                let genes = chunk * CHUNK..((chunk + 1) * CHUNK).min(n);
                match m {
                    0 => self.dual_fixed::<0>(genes, lambda, out),
                    1 => self.dual_fixed::<1>(genes, lambda, out),
                    2 => self.dual_fixed::<2>(genes, lambda, out),
                    3 => self.dual_fixed::<3>(genes, lambda, out),
                    4 => self.dual_fixed::<4>(genes, lambda, out),
                    _ => self.dual_any(genes, lambda, out, coefficients, slopes),
                }
            };
        if self.parallel && chunks > 1 {
            in_parallel(&mut work.partials, width, m, &partial);
        } else {
            let (coefficients, slopes) = (&mut work.coefficients, &mut work.slopes);
            for (chunk, out) in work.partials.chunks_exact_mut(width).enumerate() {
                partial(chunk, out, coefficients, slopes);
            }
        }
        let mut sum = 0.0;
        point.gradient.fill(0.0);
        point.scales.fill(0.0);
        point.hessian.fill(0.0);
        for out in work.partials.chunks_exact(width) {
            sum += out[0];
            for i in 0..m {
                point.gradient[i] += out[1 + i];
                point.scales[i] += out[1 + m + i];
            }
            for (total, value) in point.hessian.iter_mut().zip(&out[1 + 2 * m..]) {
                *total += value;
            }
        }
        self.close_dual(lambda, point, sum);
    }

    // a chunk of the dual's sums for M constraints, with the accumulators in registers; the same
    // operations in the same order as `dual_any`
    fn dual_fixed<const M: usize>(&self, genes: Range<usize>, lambda: &[f64], out: &mut [f64]) {
        let n = self.x.len();
        let lambda: [f64; M] = std::array::from_fn(|i| lambda[i]);
        let rows: [&[f64]; M] = std::array::from_fn(|i| &self.jacobian[i * n..(i + 1) * n]);
        let rho: [f64; M] = std::array::from_fn(|i| self.rho[i + 1]);
        let (mut gradient, mut scales) = ([0.0; M], [0.0; M]);
        let mut hessian = [[0.0; M]; M];
        let mut sum = 0.0;
        for j in genes {
            let gene = self.gene(j);
            let [sp0, sq0] = pair(self.gradient[j], self.rho[0] * gene.inverse);
            let (mut sp, mut sq) = (sp0, sq0);
            let mut coefficients = [[0.0; 2]; M];
            for i in 0..M {
                coefficients[i] = pair(rows[i][j], rho[i] * gene.inverse);
                sp += lambda[i] * coefficients[i][0];
                sq += lambda[i] * coefficients[i][1];
            }
            let step = Self::step(&gene, sp, sq);
            let (beyond_upper, beyond_lower) = (1.0 / (gene.a - step), 1.0 / (gene.b + step));
            let (tp, tq) = (gene.a * beyond_upper, gene.b * beyond_lower);
            sum += step * (sp * tp - sq * tq);
            for i in 0..M {
                let [spi, sqi] = coefficients[i];
                gradient[i] += step * (spi * tp - sqi * tq);
                scales[i] += step.abs() * (spi * tp + sqi * tq);
            }
            if M > 0 && gene.low < step && step < gene.high {
                let slopes: [f64; M] = std::array::from_fn(|i| {
                    let [spi, sqi] = coefficients[i];
                    spi * tp * tp - sqi * tq * tq
                });
                let curvature = 2.0 * (sp * tp * tp * beyond_upper + sq * tq * tq * beyond_lower);
                for i in 0..M {
                    let weighted = slopes[i] / curvature;
                    for k in 0..M {
                        hessian[i][k] -= weighted * slopes[k];
                    }
                }
            }
        }
        out[0] = sum;
        out[1..1 + M].copy_from_slice(&gradient);
        out[1 + M..1 + 2 * M].copy_from_slice(&scales);
        for (i, row) in hessian.iter().enumerate() {
            out[1 + 2 * M + i * M..1 + 2 * M + (i + 1) * M].copy_from_slice(row);
        }
    }

    // a chunk of the dual's sums for any number of constraints
    fn dual_any(
        &self,
        genes: Range<usize>,
        lambda: &[f64],
        out: &mut [f64],
        coefficients: &mut [f64],
        slopes: &mut [f64],
    ) {
        let m = self.m();
        out.fill(0.0);
        let (sum, rest) = out.split_first_mut().expect("a sum");
        let (gradient, rest) = rest.split_at_mut(m);
        let (scales, hessian) = rest.split_at_mut(m);
        for j in genes {
            let gene = self.gene(j);
            self.coefficients(j, &gene, coefficients);
            let (mut sp, mut sq) = (coefficients[0], coefficients[1]);
            for (i, &weight) in lambda.iter().enumerate() {
                sp += weight * coefficients[2 * i + 2];
                sq += weight * coefficients[2 * i + 3];
            }
            let step = Self::step(&gene, sp, sq);
            // 1 / (u − x) and 1 / (x − l) at the minimizer
            let (beyond_upper, beyond_lower) = (1.0 / (gene.a - step), 1.0 / (gene.b + step));
            let (tp, tq) = (gene.a * beyond_upper, gene.b * beyond_lower);
            // the Lagrangian's terms, as differences from x⁽ᵏ⁾: step · (sᵖ a / (u − x) − s^q b /
            // (x − l)), without the cancellation of the eq. 3.5 form
            *sum += step * (sp * tp - sq * tq);
            for i in 0..m {
                let (spi, sqi) = (coefficients[2 * i + 2], coefficients[2 * i + 3]);
                gradient[i] += step * (spi * tp - sqi * tq);
                scales[i] += step.abs() * (spi * tp + sqi * tq);
            }
            if gene.low < step && step < gene.high {
                for i in 0..m {
                    let (spi, sqi) = (coefficients[2 * i + 2], coefficients[2 * i + 3]);
                    slopes[i] = spi * tp * tp - sqi * tq * tq;
                }
                let curvature = 2.0 * (sp * tp * tp * beyond_upper + sq * tq * tq * beyond_lower);
                for i in 0..m {
                    let weighted = slopes[i] / curvature;
                    for k in 0..m {
                        hessian[i * m + k] -= weighted * slopes[k];
                    }
                }
            }
        }
    }

    // the dual's terms outside the sums over the genes: f₀ and λᵢ fᵢ at x⁽ᵏ⁾, and the artificial
    // variables
    fn close_dual(&self, lambda: &[f64], point: &mut DualPoint, sum: f64) {
        let m = lambda.len();
        let mut value = self.values[0] + sum;
        for (i, &weight) in lambda.iter().enumerate() {
            let f = self.values[i + 1];
            value += weight * f;
            point.gradient[i] += f;
            point.scales[i] += f.abs();
            // the artificial variable yᵢ = max(0, (λᵢ − c) / d)
            let excess = weight - self.cost;
            if excess > 0.0 {
                value -= excess * excess / (2.0 * D);
                point.gradient[i] -= excess / D;
                point.scales[i] += excess / D;
                point.hessian[i * m + i] -= 1.0 / D;
            }
        }
        point.value = value;
    }

    // the subproblem's solution for `lambda` into `x`, within the bounds; the approximations f̃ᵢ
    // there and the scales of their terms; returns d(x) (eq. 4.7)
    fn primal(
        &self,
        lambda: &[f64],
        x: &mut [f64],
        coefficients: &mut [f64],
        approximations: &mut [f64],
        scales: &mut [f64],
    ) -> f64 {
        approximations.fill(0.0);
        scales.fill(0.0);
        let mut distance = 0.0;
        for j in 0..self.x.len() {
            let gene = self.gene(j);
            self.coefficients(j, &gene, coefficients);
            let (mut sp, mut sq) = (coefficients[0], coefficients[1]);
            for (i, &weight) in lambda.iter().enumerate() {
                sp += weight * coefficients[2 * i + 2];
                sq += weight * coefficients[2 * i + 3];
            }
            let step = Self::step(&gene, sp, sq);
            let (low, high) = (*self.bounds[j].start(), *self.bounds[j].end());
            let value = (self.x[j] + step).clamp(low, high);
            x[j] = value;
            // the step taken, after rounding
            let step = value - self.x[j];
            let (to_upper, to_lower) = (gene.a - step, gene.b + step);
            let (tp, tq) = (gene.a / to_upper, gene.b / to_lower);
            for (i, &[sp, sq]) in coefficients.as_chunks::<2>().0.iter().enumerate() {
                approximations[i] += step * (sp * tp - sq * tq);
                scales[i] += step.abs() * (sp * tp + sq * tq);
            }
            distance += (gene.a + gene.b) * step * step / (to_upper * to_lower) * gene.inverse;
        }
        for (i, value) in self.values.iter().enumerate() {
            approximations[i] += value;
            scales[i] += value.abs();
        }
        distance
    }
}

// each chunk's partial sums of the dual into its row of `partials`, on rayon's threads: each
// writes its own row, so the same bits as one after the other
#[cfg(feature = "parallel")]
fn in_parallel<F>(partials: &mut [f64], width: usize, m: usize, partial: &F)
where
    F: Fn(usize, &mut [f64], &mut [f64], &mut [f64]) + Sync,
{
    use rayon::prelude::*;
    partials
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(chunk, out)| {
            if m <= 4 {
                partial(chunk, out, &mut [], &mut []);
            } else {
                let (mut coefficients, mut slopes) = (vec![0.0; 2 * (m + 1)], vec![0.0; m]);
                partial(chunk, out, &mut coefficients, &mut slopes);
            }
        });
}

// without the `parallel` feature (a checkpoint of a run in parallel), one after the other
#[cfg(not(feature = "parallel"))]
fn in_parallel<F>(partials: &mut [f64], width: usize, m: usize, partial: &F)
where
    F: Fn(usize, &mut [f64], &mut [f64], &mut [f64]),
{
    let (mut coefficients, mut slopes) = (vec![0.0; 2 * (m + 1)], vec![0.0; m]);
    for (chunk, out) in partials.chunks_exact_mut(width).enumerate() {
        partial(chunk, out, &mut coefficients, &mut slopes);
    }
}

// sᵖ and s^q of a derivative with the curvature ρ / (xmax − xmin) (eqs. 3.3-3.4, 4.3-4.4): 1.001
// of the derivative's positive part and 0.001 of its negative part, and the other way round
#[inline]
fn pair(derivative: f64, curvature: f64) -> [f64; 2] {
    let (plus, minus) = (derivative.max(0.0), (-derivative).max(0.0));
    [
        1.001 * plus + 0.001 * minus + curvature,
        0.001 * plus + 1.001 * minus + curvature,
    ]
}

// solves `matrix · z = rhs` (k × k, symmetric positive definite, row-major) in place into `rhs` by
// Cholesky's method, the matrix becoming its factor; false if it isn't positive definite
fn cholesky_solve(matrix: &mut [f64], rhs: &mut [f64], k: usize) -> bool {
    if linalg::cholesky::cholesky(matrix, k).is_err() {
        return false;
    }
    linalg::triangular::cholesky_solve(matrix, k, rhs);
    true
}

// maximizes the dual function over λ ≥ 0, from `lambda`, into `lambda`: a Newton method on the
// multipliers that aren't held at 0 (those at 0 whose derivative is negative), each step searched
// along its direction for the maximum, which the dual's concavity brackets
fn maximize_dual(approximation: &Approximation<'_>, lambda: &mut [f64], scratch: &mut Scratch) {
    let m = lambda.len();
    if m == 0 {
        return;
    }
    let Scratch {
        at,
        next,
        best,
        rising,
        candidate,
        direction,
        free,
        matrix,
        rhs,
        work,
        ..
    } = scratch;
    for value in lambda.iter_mut() {
        *value = value.max(0.0);
    }
    approximation.dual(lambda, at, work);
    for _ in 0..DUAL_ITERATIONS {
        // optimal: the derivative is 0 where λᵢ > 0, at most 0 where λᵢ = 0
        free.clear();
        let mut optimal = true;
        for (i, &weight) in lambda.iter().enumerate() {
            let derivative = at.gradient[i];
            if weight > 0.0 || derivative > 0.0 {
                free.push(i);
                optimal &= derivative.abs() <= DUAL_TOLERANCE * at.scales[i];
            }
        }
        if optimal || free.is_empty() {
            return;
        }
        // the Newton direction in the free multipliers, (−H + μI) d = ∇W; a multiplier without
        // curvature (every gene it acts on at a bound) gets a little, for the search to scale. A
        // multiplier at 0 that the direction would take below 0 is held there, and the direction
        // found again without it
        loop {
            let k = free.len();
            let largest = free
                .iter()
                .fold(0.0f64, |largest, &i| largest.max(-at.hessian[i * m + i]));
            let mut shift = if largest > 0.0 { 1e-12 * largest } else { 1.0 };
            loop {
                for (row, &i) in free.iter().enumerate() {
                    for (column, &p) in free.iter().enumerate() {
                        matrix[row * k + column] = -at.hessian[i * m + p];
                    }
                    matrix[row * k + row] += shift;
                    rhs[row] = at.gradient[i];
                }
                if cholesky_solve(&mut matrix[..k * k], &mut rhs[..k], k) {
                    break;
                }
                shift *= 10.0;
                if !shift.is_finite() {
                    return;
                }
            }
            direction.fill(0.0);
            for (row, &i) in free.iter().enumerate() {
                direction[i] = rhs[row];
            }
            let held = free.len();
            free.retain(|&i| lambda[i] > 0.0 || direction[i] >= 0.0);
            if free.len() == held || free.is_empty() {
                break;
            }
        }
        if free.is_empty() {
            return;
        }
        let slope = dot(&at.gradient, direction);
        if slope.is_nan() || slope <= 0.0 {
            return;
        }
        // the longest step before a multiplier reaches 0
        let mut longest = f64::INFINITY;
        for i in 0..m {
            if direction[i] < 0.0 {
                longest = longest.min(lambda[i] / -direction[i]);
            }
        }
        // the step s along the direction where the slope φ'(s) = ∇W(λ + s d) · d is near 0 (at
        // most a tenth of φ'(0)): from s = 1, longer while the dual still rises steeply, and
        // within the bracket [low, high] by secant steps once it falls
        let (mut low, mut low_slope) = (0.0, slope);
        let (mut high, mut high_slope) = (f64::INFINITY, f64::NAN);
        // which end moved last, for Illinois's halving of the other end's slope
        let mut last_high = None;
        let mut step = longest.min(1.0);
        let mut taken = false;
        for _ in 0..DUAL_SEARCH {
            for i in 0..m {
                let reaches = direction[i] < 0.0 && lambda[i] / -direction[i] <= step;
                candidate[i] = if step >= longest && reaches {
                    0.0
                } else {
                    (lambda[i] + step * direction[i]).max(0.0)
                };
            }
            approximation.dual(candidate, next, work);
            let trial_slope = dot(&next.gradient, direction);
            if trial_slope.abs() <= 0.1 * slope || (trial_slope >= 0.0 && step >= longest) {
                taken = true;
                break;
            }
            if trial_slope > 0.0 {
                (low, low_slope) = (step, trial_slope);
                // the farthest point still rising, kept
                rising.copy_from_slice(candidate);
                std::mem::swap(next, best);
                if last_high == Some(false) {
                    high_slope *= 0.5;
                }
                last_high = Some(false);
            } else {
                (high, high_slope) = (step, trial_slope);
                if last_high == Some(true) {
                    low_slope *= 0.5;
                }
                last_high = Some(true);
            }
            step = if high.is_infinite() {
                (4.0 * step).min(longest)
            } else {
                // the secant's root of the slope, inside the bracket
                let secant = low + (high - low) * low_slope / (low_slope - high_slope);
                secant.clamp(low + 1e-12 * (high - low), high - 1e-12 * (high - low))
            };
            if !(step > low && step < high) || high - low <= 1e-9 * high {
                break;
            }
        }
        if !taken {
            if low == 0.0 {
                return;
            }
            candidate.copy_from_slice(rising);
            std::mem::swap(next, best);
        }
        if candidate == lambda {
            return;
        }
        lambda.copy_from_slice(candidate);
        std::mem::swap(at, next);
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

impl Mma {
    /// A builder for an MMA run on `real`.
    pub fn builder(real: Real) -> MmaBuilder {
        MmaBuilder {
            real,
            settings: Settings {
                method: Method::Mma,
                asymptote_initial: 0.5,
                asymptote_decrease: 0.7,
                asymptote_increase: 1.2,
                move_limit: 0.5,
                constraint_cost: 1000.0,
                kkt_tolerance: 1e-9,
                step_tolerance: 1e-10,
                restoration: true,
                parallel: false,
            },
            initial_genome: None,
            objective: Objective::default(),
            seed: None,
        }
    }

    /// The representation.
    pub fn real(&self) -> &Real {
        &self.real
    }

    /// The method: MMA or GCMMA.
    pub fn method(&self) -> Method {
        self.settings.method
    }

    /// The seed of the random numbers: the given one, or a random one if none was given.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Why the run has converged, if it has: `None` while it goes on.
    pub fn converged(&self) -> Option<Convergence> {
        self.converged
    }

    /// The number of completed iterations: points accepted after the initial one.
    pub fn iterations(&self) -> u64 {
        self.iterations
    }

    /// The number of GCMMA's inner iterations: points rejected because an approximation wasn't
    /// conservative there. Always 0 with [`Method::Mma`].
    pub fn inner_iterations(&self) -> u64 {
        self.inner_iterations
    }

    /// The multipliers of the constraints from the last subproblem, a value per constraint, at
    /// least 0: at a solution, the Lagrange multipliers of the problem minimized (of the negated
    /// score when maximizing). A multiplier at the [cost](MmaBuilder::constraint_cost) or above
    /// means the constraint is relaxed: it couldn't be met. Empty before the first subproblem.
    pub fn multipliers(&self) -> &[f64] {
        &self.multipliers
    }

    /// The KKT residual at the current point, with the [multipliers](Mma::multipliers) of the
    /// subproblem that gave it (Svanberg 2002, eqs. 8.3-8.4): the root mean square, over the
    /// genes that aren't fixed, of the residuals
    ///
    /// - `(xⱼ − lowⱼ)/widthⱼ · max(0, ∂L/∂xⱼ)` and `(highⱼ − xⱼ)/widthⱼ · max(0, −∂L/∂xⱼ)`, for
    ///   the Lagrangian L = f₀ + Σ λᵢ gᵢ (stationarity, or a bound that holds it);
    /// - `max(0, gᵢ)` and `λᵢ max(0, −gᵢ)` (feasibility and complementarity).
    ///
    /// The paper's box is [−1, 1] and its factors `1 + xⱼ` and `1 − xⱼ`; genoxide divides by the
    /// width, so that the residual has the score's scale whatever the genes'. NaN before the
    /// first iteration.
    pub fn kkt_residual(&self) -> f64 {
        self.kkt_residual
    }

    /// The lower asymptotes lⱼ of the current approximations, a value per gene. Empty before the
    /// initial point is evaluated.
    pub fn lower_asymptotes(&self) -> &[f64] {
        &self.asymptotes.lower
    }

    /// The upper asymptotes uⱼ of the current approximations.
    pub fn upper_asymptotes(&self) -> &[f64] {
        &self.asymptotes.upper
    }

    /// Marks the current point as not evaluated, for a fitness function that changed during the
    /// run. The next [`ask`](Algorithm::ask) gives it, with its gradient and Jacobian, and its
    /// tell sets them, without an iteration:
    ///
    /// - [`generation`](Algorithm::generation) doesn't change; the evaluation counts;
    /// - an inner iteration or a restoration under way is dropped, and the next ask solves a new
    ///   subproblem at the point; the asymptotes are kept;
    /// - [`best`](Algorithm::best) is then the current point: old and new values are never
    ///   compared;
    /// - no random number is drawn.
    ///
    /// Before the initial point is evaluated, it changes nothing.
    ///
    /// # Errors
    ///
    /// [`Error::ReevaluationOutOfTurn`] between an ask and its tell. Nothing changes on errors.
    pub fn reevaluate(&mut self) -> Result<()> {
        if self.asked {
            return Err(Error::ReevaluationOutOfTurn);
        }
        self.reevaluating = self.phase != Phase::Start;
        Ok(())
    }

    fn n(&self) -> usize {
        self.real.bounds().len()
    }

    fn m(&self) -> usize {
        self.constraints.unwrap_or(0)
    }

    fn sign(&self) -> f64 {
        match self.objective {
            Objective::Minimize => 1.0,
            Objective::Maximize => -1.0,
        }
    }

    // the number of genes that aren't fixed
    fn free_genes(&self) -> usize {
        self.real.variable_genes().len()
    }

    // the buffers of a run with m constraints
    fn allocate(&mut self, m: usize) {
        let n = self.n();
        self.values.resize(m + 1, 0.0);
        self.gradient.resize(n, 0.0);
        self.jacobian.resize(m * n, 0.0);
        self.rho.resize(m + 1, RAA0);
        self.rounding.resize(m + 1, 0.0);
        self.multipliers.resize(m, 0.0);
        self.approximations.resize(m + 1, 0.0);
        self.scales.resize(m + 1, 0.0);
        let asymptotes = &mut self.asymptotes;
        for buffer in [
            &mut asymptotes.lower,
            &mut asymptotes.upper,
            &mut asymptotes.previous,
            &mut asymptotes.before_previous,
        ] {
            buffer.resize(n, 0.0);
        }
    }

    // the approximations around the current point
    fn approximation<'a>(&'a self, genes: &'a [Gene]) -> Approximation<'a> {
        Approximation {
            x: self.current[0].genome(),
            lower: &self.asymptotes.lower,
            upper: &self.asymptotes.upper,
            bounds: self.real.bounds(),
            gradient: &self.gradient,
            jacobian: &self.jacobian,
            values: &self.values,
            rho: &self.rho,
            move_limit: self.settings.move_limit * self.move_fraction,
            cost: self.settings.constraint_cost,
            parallel: self.settings.parallel,
            genes,
        }
    }

    // solves the subproblem at the current point into the trial point
    fn solve(&mut self) {
        let m = self.m();
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.resize(m);
        let mut lambda = std::mem::take(&mut self.multipliers);
        lambda.resize(m, 0.0);
        let mut trial = std::mem::replace(&mut self.trial, Individual::new(Reals::default()));
        let mut approximations = std::mem::take(&mut self.approximations);
        let mut scales = std::mem::take(&mut self.scales);
        let mut genes = std::mem::take(&mut scratch.genes);
        {
            let places = self.approximation(&[]);
            genes.clear();
            genes.extend((0..self.n()).map(|j| places.place(j)));
        }
        {
            let approximation = self.approximation(&genes);
            maximize_dual(&approximation, &mut lambda, &mut scratch);
            let x = trial.genome_mut();
            if x.len() != self.n() {
                *x = self.current[0].genome().clone();
            }
            self.distance = approximation.primal(
                &lambda,
                x,
                &mut scratch.work.coefficients,
                &mut approximations,
                &mut scales,
            );
        }
        scratch.genes = genes;
        self.scratch = scratch;
        self.multipliers = lambda;
        self.trial = trial;
        self.approximations = approximations;
        self.scales = scales;
    }

    // the asymptotes of iteration k at the current point (Svanberg 2007, eqs. 3.11-3.14)
    fn update_asymptotes(&mut self) {
        let x = self.current[0].genome();
        let bounds = self.real.bounds();
        let settings = self.settings;
        let asymptotes = &mut self.asymptotes;
        let k = asymptotes.iteration;
        for j in 0..x.len() {
            let range = bounds[j].end() - bounds[j].start();
            let width = if range > 0.0 { range } else { 1.0 };
            let xj = x[j];
            if k <= 2 {
                asymptotes.lower[j] = xj - settings.asymptote_initial * width;
                asymptotes.upper[j] = xj + settings.asymptote_initial * width;
                continue;
            }
            let (previous, before) = (asymptotes.previous[j], asymptotes.before_previous[j]);
            let trend = (xj - previous) * (previous - before);
            let factor = if trend < 0.0 {
                settings.asymptote_decrease
            } else if trend > 0.0 {
                settings.asymptote_increase
            } else {
                1.0
            };
            let lower = xj - factor * (previous - asymptotes.lower[j]);
            let upper = xj + factor * (asymptotes.upper[j] - previous);
            asymptotes.lower[j] = lower
                .min(xj - ASYMPTOTE_NEAREST * width)
                .max(xj - ASYMPTOTE_FARTHEST * width);
            asymptotes.upper[j] = upper
                .max(xj + ASYMPTOTE_NEAREST * width)
                .min(xj + ASYMPTOTE_FARTHEST * width);
        }
    }

    // at the start of an outer iteration: GCMMA's ρᵢ (Svanberg 2007, eq. 4.6), MMA's raa0, and
    // the rounding scales
    fn reset_rho(&mut self) {
        let n = self.n();
        let bounds = self.real.bounds();
        let x = self.current[0].genome();
        let free = self.free_genes() as f64;
        let gcmma = self.settings.method == Method::Gcmma;
        for i in 0..self.rho.len() {
            let row = if i == 0 {
                &self.gradient[..]
            } else {
                &self.jacobian[(i - 1) * n..i * n]
            };
            let (mut sum, mut rounding) = (0.0, self.values[i].abs());
            for ((derivative, range), xj) in row.iter().zip(bounds).zip(x.iter()) {
                sum += derivative.abs() * (range.end() - range.start());
                rounding += (derivative * xj).abs();
            }
            self.rho[i] = if gcmma {
                (RHO_FRACTION * sum / free).max(RHO_MIN)
            } else {
                RAA0
            };
            self.rounding[i] = rounding;
        }
    }

    // the KKT residual at the current point (Svanberg 2002, eqs. 8.3-8.4)
    fn measure_kkt(&self) -> f64 {
        let n = self.n();
        let x = self.current[0].genome();
        let bounds = self.real.bounds();
        let mut sum = 0.0;
        for j in 0..n {
            let (low, high) = (*bounds[j].start(), *bounds[j].end());
            let width = high - low;
            if width <= 0.0 {
                continue;
            }
            let mut derivative = self.gradient[j];
            for (i, &lambda) in self.multipliers.iter().enumerate() {
                derivative += lambda * self.jacobian[i * n + j];
            }
            let above = (x[j] - low) / width * derivative.max(0.0);
            let below = (high - x[j]) / width * (-derivative).max(0.0);
            sum += above * above + below * below;
        }
        for (i, &lambda) in self.multipliers.iter().enumerate() {
            let g = self.values[i + 1];
            let infeasible = g.max(0.0);
            let slack = lambda * (-g).max(0.0);
            sum += infeasible * infeasible + slack * slack;
        }
        (sum / self.free_genes() as f64).sqrt()
    }

    // the largest step of the last iteration, as a fraction of each gene's range
    fn last_step(&self) -> f64 {
        let x = self.current[0].genome();
        let previous = &self.asymptotes.previous;
        let mut largest = 0.0f64;
        for (j, range) in self.real.bounds().iter().enumerate() {
            let width = range.end() - range.start();
            if width > 0.0 {
                largest = largest.max((x[j] - previous[j]).abs() / width);
            }
        }
        largest
    }

    // copies the evaluation of the current point: f₀, the constraints, the gradient and the
    // Jacobian, in the direction of minimization
    fn store(&mut self, score: f64, evaluations: &Evaluations<'_>) {
        let sign = self.sign();
        self.values[0] = sign * score;
        if let Some(g) = evaluations.inequalities(0) {
            self.values[1..].copy_from_slice(g);
        }
        if let Some(gradient) = evaluations.gradient(0) {
            for (stored, &value) in self.gradient.iter_mut().zip(gradient) {
                *stored = sign * value;
            }
        }
        if let Some(jacobian) = evaluations.constraint_jacobian(0) {
            self.jacobian.copy_from_slice(jacobian);
        }
    }

    // the best so far, with `individual` evaluated: the first on ties
    fn update_best(&mut self, from_trial: bool) {
        let individual = if from_trial {
            &self.trial
        } else {
            &self.current[0]
        };
        let fitness = individual.fitness().unwrap_or(Fitness::invalid());
        let better = self.best.as_ref().is_none_or(|best| {
            self.objective
                .is_better(fitness, best.fitness().unwrap_or(Fitness::invalid()))
        });
        if better {
            match &mut self.best {
                Some(best) => best.clone_from(individual),
                None => self.best = Some(individual.clone()),
            }
            self.best_generation = self.generation;
        }
    }

    // takes the trial point as the next iterate
    fn accept(&mut self, evaluations: &Evaluations<'_>, score: f64) {
        let asymptotes = &mut self.asymptotes;
        std::mem::swap(&mut asymptotes.previous, &mut asymptotes.before_previous);
        asymptotes
            .previous
            .copy_from_slice(self.current[0].genome());
        std::mem::swap(&mut self.current.individuals_mut()[0], &mut self.trial);
        self.store(score, evaluations);
        self.iterations += 1;
        self.inner = 0;
        self.move_fraction = 1.0;
        self.asymptotes.iteration += 1;
        self.update_asymptotes();
        self.reset_rho();
        self.kkt_residual = self.measure_kkt();
        self.converged = if self.kkt_residual <= self.settings.kkt_tolerance {
            Some(Convergence::Kkt)
        } else if self.last_step() <= self.settings.step_tolerance {
            Some(Convergence::Step)
        } else {
            None
        };
        if self.converged.is_some() {
            self.finish();
        }
    }

    // after convergence: a restoration if the point is infeasible and the problem seems feasible,
    // else nothing more to do
    fn finish(&mut self) {
        let cost = self.settings.constraint_cost;
        let infeasible = self.values[1..].iter().any(|&g| g > 0.0);
        let relaxed = self.multipliers.iter().any(|&lambda| lambda >= cost);
        self.phase =
            if self.settings.restoration && infeasible && !relaxed && self.restoration_point(0) {
                Phase::Restore { attempt: 0 }
            } else {
                Phase::Finished
            };
    }

    // a point that can't be accepted: the asymptotes move in, and at their nearest the move limit
    // halves
    fn reject_invalid(&mut self) {
        let x = self.current[0].genome();
        let bounds = self.real.bounds();
        let decrease = self.settings.asymptote_decrease;
        let mut moved = false;
        for (j, range) in bounds.iter().enumerate() {
            let width = range.end() - range.start();
            if width <= 0.0 {
                continue;
            }
            let (lower, upper) = (&mut self.asymptotes.lower[j], &mut self.asymptotes.upper[j]);
            let nearest = ASYMPTOTE_NEAREST * width;
            let new_lower = (x[j] - decrease * (x[j] - *lower)).min(x[j] - nearest);
            let new_upper = (x[j] + decrease * (*upper - x[j])).max(x[j] + nearest);
            moved |= new_lower != *lower || new_upper != *upper;
            *lower = new_lower;
            *upper = new_upper;
        }
        if !moved {
            self.move_fraction *= 0.5;
            if self.move_fraction * self.settings.move_limit <= self.settings.step_tolerance {
                self.converged = Some(Convergence::Step);
                self.phase = Phase::Finished;
            }
        }
    }

    // the restoration point of `attempt` into the trial point: the current point moved onto the
    // feasible side of its active constraints by the least step, scaled by the ranges, in the
    // genes that aren't at a bound; false if there's none
    fn restoration_point(&mut self, attempt: u32) -> bool {
        let (n, m) = (self.n(), self.m());
        let mut scratch = std::mem::take(&mut self.scratch);
        scratch.resize(m);
        let x = self.current[0].genome();
        let bounds = self.real.bounds();
        // the active constraints: violated, or with a positive multiplier
        scratch.free.clear();
        for i in 0..m {
            if self.values[i + 1] > 0.0 || self.multipliers.get(i).is_some_and(|&l| l > 0.0) {
                scratch.free.push(i);
            }
        }
        let k = scratch.free.len();
        let movable = |j: usize| {
            let (low, high) = (*bounds[j].start(), *bounds[j].end());
            low < x[j] && x[j] < high
        };
        // J D Jᵀ with D the squared widths, and the targets −(g + t)
        let factor = f64::from(1u32 << attempt);
        for (row, &i) in scratch.free.iter().enumerate() {
            let g = self.values[i + 1];
            let jacobian = &self.jacobian[i * n..(i + 1) * n];
            let rounding: f64 = jacobian
                .iter()
                .zip(x.iter())
                .map(|(d, xj)| (d * xj).abs())
                .sum::<f64>()
                * 16.0
                * f64::EPSILON;
            let margin = factor * g.abs().max(rounding);
            scratch.rhs[row] = g + margin;
            for (column, &p) in scratch.free.iter().enumerate() {
                let other = &self.jacobian[p * n..(p + 1) * n];
                let mut sum = 0.0;
                for j in 0..n {
                    if movable(j) {
                        let width = bounds[j].end() - bounds[j].start();
                        sum += jacobian[j] * width * width * other[j];
                    }
                }
                scratch.matrix[row * k + column] = sum;
            }
        }
        let solved =
            k > 0 && cholesky_solve(&mut scratch.matrix[..k * k], &mut scratch.rhs[..k], k);
        if solved {
            let point = self.trial.genome_mut();
            if point.len() != n {
                *point = x.clone();
            }
            for j in 0..n {
                let (low, high) = (*bounds[j].start(), *bounds[j].end());
                let mut step = 0.0;
                if movable(j) {
                    let width = high - low;
                    for (row, &i) in scratch.free.iter().enumerate() {
                        step -= width * width * self.jacobian[i * n + j] * scratch.rhs[row];
                    }
                }
                point[j] = (x[j] + step).clamp(low, high);
            }
        }
        self.scratch = scratch;
        solved
    }

    fn check_tell(&self, count: usize) -> Result<()> {
        if !self.asked {
            return Err(Error::TellWithoutAsk);
        }
        if count != 1 {
            return Err(Error::FitnessCount {
                expected: 1,
                got: count,
            });
        }
        Ok(())
    }

    // whether an evaluation is usable: a valid fitness with finite extras
    fn usable(fitness: Fitness, evaluations: &Evaluations<'_>) -> bool {
        let finite =
            |values: Option<&[f64]>| values.is_none_or(|v| v.iter().all(|x| x.is_finite()));
        fitness.score().is_some_and(f64::is_finite)
            && finite(evaluations.gradient(0))
            && finite(evaluations.inequalities(0))
            && finite(evaluations.constraint_jacobian(0))
    }

    // the error of an evaluation without the extras the ask wanted
    fn check_extras(&self, evaluations: &Evaluations<'_>) -> Result<()> {
        let (n, m) = (self.n(), self.m());
        let wanted = self.wants();
        let missing = |what: &str| {
            Err(Error::InvalidSetting {
                setting: "fitness",
                reason: format!(
                    "MMA needs the {what} of each evaluated point: tell it with \
                     `tell_evaluations`, from a fitness function that provides it (e.g. \
                     `Constrained::differentiable`)"
                ),
            })
        };
        if wanted.gradient && evaluations.gradient(0).is_none_or(|g| g.len() != n) {
            return missing("gradient");
        }
        if wanted.inequalities && evaluations.inequalities(0).is_none_or(|g| g.len() != m) {
            return missing("values of the constraints");
        }
        if wanted.constraint_jacobian
            && evaluations
                .constraint_jacobian(0)
                .is_none_or(|j| j.len() != m * n)
        {
            return missing("constraint Jacobian");
        }
        Ok(())
    }
}

impl Reevaluate for Mma {
    /// As [`Mma::reevaluate`]: the next ask gives the current point.
    fn reevaluate(&mut self) -> Result<()> {
        Mma::reevaluate(self)
    }
}

impl Algorithm for Mma {
    type Genome = Reals;

    fn objective(&self) -> Objective {
        self.objective
    }

    fn ask(&mut self) -> Candidates<'_, Reals> {
        const FIRST: &[usize] = &[0];
        if !self.asked {
            if !self.reevaluating {
                if self.phase == Phase::Finished {
                    // asked again after converging: go on
                    self.phase = Phase::Trial;
                    self.converged = None;
                }
                if self.phase == Phase::Trial {
                    self.solve();
                }
            }
            self.asked = true;
        }
        let individuals = if self.reevaluating || self.phase == Phase::Start {
            self.current.as_slice()
        } else {
            std::slice::from_ref(&self.trial)
        };
        Candidates::new(individuals, FIRST)
    }

    /// MMA needs the gradient: a plain tell is an error, and changes nothing. The
    /// [`Engine`](crate::Engine) tells it with
    /// [`tell_evaluations`](Algorithm::tell_evaluations).
    ///
    /// # Errors
    ///
    /// [`Error::TellWithoutAsk`] if nothing was asked, [`Error::FitnessCount`] for other than one
    /// value, and [`Error::InvalidSetting`] otherwise: the evaluations have no gradient.
    fn tell(&mut self, fitness: &[Fitness]) -> Result<()> {
        self.check_tell(fitness.len())?;
        self.check_extras(&Evaluations::new(fitness))
    }

    /// # Errors
    ///
    /// [`Error::TellWithoutAsk`] if nothing was asked, [`Error::FitnessCount`] for other than one
    /// evaluation, and [`Error::InvalidSetting`] for evaluations without the gradient, the
    /// constraints' values or their Jacobian that the ask [wanted](Algorithm::wants), for a
    /// fitness with a violation from a function that provides no constraint values (wrap it in
    /// [`Constrained`](crate::constraint::Constrained)), or for an initial point (or a
    /// re-evaluated one) whose fitness is invalid or whose extras aren't finite. Nothing changes
    /// on errors.
    fn tell_evaluations(&mut self, evaluations: &Evaluations<'_>) -> Result<()> {
        self.check_tell(evaluations.len())?;
        if self.constraints.is_none() {
            // driven by hand, without `prepare`: the constraints the evaluations have
            let m = evaluations.constraints();
            self.constraints = Some(m);
            self.allocate(m);
        }
        if let Err(error) = self.check_extras(evaluations) {
            if self.phase == Phase::Start && !self.reevaluating {
                self.constraints = None;
            }
            return Err(error);
        }
        let fitness = evaluations.fitness()[0];
        if self.m() == 0 && fitness.violation() > 0.0 {
            return Err(Error::InvalidSetting {
                setting: "fitness",
                reason: "the fitness function returns a constraint violation, but provides no \
                         constraint values: MMA needs each constraint's value and gradient, e.g. \
                         from `Constrained::differentiable`"
                    .to_string(),
            });
        }
        let usable = Self::usable(fitness, evaluations);
        if (self.reevaluating || self.phase == Phase::Start) && !usable {
            return Err(Error::InvalidSetting {
                setting: "initial_genome",
                reason: format!(
                    "the fitness at the {} point is {fitness:?}: MMA needs a valid score and \
                     finite derivatives there",
                    if self.reevaluating {
                        "re-evaluated"
                    } else {
                        "initial"
                    }
                ),
            });
        }
        self.asked = false;
        self.evaluations += 1;
        self.discarded_trial = false;
        let score = fitness.score().unwrap_or(f64::NAN);
        if self.reevaluating {
            self.current[0].set_fitness(fitness);
            self.store(score, evaluations);
            self.reset_rho();
            let current = self.current[0].clone();
            match &mut self.best {
                Some(best) => best.clone_from(&current),
                None => self.best = Some(current),
            }
            self.best_generation = self.generation;
            self.phase = Phase::Trial;
            self.converged = None;
            self.inner = 0;
            self.reevaluating = false;
            return Ok(());
        }
        if self.phase == Phase::Start {
            self.current[0].set_fitness(fitness);
            self.store(score, evaluations);
            self.asymptotes.iteration = 1;
            self.update_asymptotes();
            self.reset_rho();
            self.update_best(false);
            self.phase = Phase::Trial;
            return Ok(());
        }
        self.generation += 1;
        self.trial.set_fitness(fitness);
        self.update_best(true);
        if let Phase::Restore { attempt } = self.phase {
            self.discarded_trial = true;
            let next = attempt + 1;
            let again = !fitness.is_feasible()
                && next < RESTORATION_ATTEMPTS
                && self.restoration_point(next);
            self.phase = if again {
                Phase::Restore { attempt: next }
            } else {
                Phase::Finished
            };
            return Ok(());
        }
        if !usable {
            self.discarded_trial = true;
            self.reject_invalid();
            return Ok(());
        }
        if self.settings.method == Method::Gcmma && self.inner < MAX_INNER {
            // conservative: fᵢ(x̂) ≤ f̃ᵢ(x̂) for f₀ and every constraint (section 4)
            let sign = self.sign();
            let g = evaluations.inequalities(0).unwrap_or(&[]);
            let mut conservative = true;
            for i in 0..self.values.len() {
                let value = if i == 0 { sign * score } else { g[i - 1] };
                let excess = value - self.approximations[i];
                let tolerance = CONSERVATIVE_TOLERANCE
                    * (self.free_genes() as f64).sqrt()
                    * (self.scales[i] + self.rounding[i]);
                if excess > tolerance {
                    conservative = false;
                    // eqs. 4.8-4.9
                    if self.distance > 0.0 {
                        let delta = excess / self.distance;
                        self.rho[i] =
                            (RHO_GROWTH * (self.rho[i] + delta)).min(RHO_MOST_GROWTH * self.rho[i]);
                    }
                }
            }
            if !conservative && self.distance > 0.0 {
                self.inner += 1;
                self.inner_iterations += 1;
                self.discarded_trial = true;
                return Ok(());
            }
        }
        self.accept(evaluations, score);
        Ok(())
    }

    fn population(&self) -> &Population<Reals> {
        &self.current
    }

    fn best(&self) -> Option<&Individual<Reals>> {
        self.best.as_ref()
    }

    fn discarded(&self) -> &[Individual<Reals>] {
        if self.discarded_trial {
            std::slice::from_ref(&self.trial)
        } else {
            &[]
        }
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

    /// Whether the run has converged, and its restoration, if any, is done.
    fn is_finished(&self) -> bool {
        self.phase == Phase::Finished && !self.reevaluating
    }

    /// Checks that the fitness function provides the gradient, and for constraints their values
    /// and Jacobian.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] for a fitness function without a gradient, with constraint values
    /// but no Jacobian, or with another number of constraints than the run so far.
    fn prepare(&mut self, provided: Provided) -> Result<()> {
        let invalid = |reason: String| {
            Err(Error::InvalidSetting {
                setting: "fitness",
                reason,
            })
        };
        if !provided.gradient {
            return invalid(
                "MMA needs the gradient of the score: supply it with `Differentiable`, or with \
                 `Constrained::differentiable` for a problem with constraints"
                    .to_string(),
            );
        }
        let m = provided.inequalities;
        if m > 0 && !provided.constraint_jacobian {
            return invalid(format!(
                "the fitness function provides {m} constraint values but not their Jacobian, \
                 which MMA needs: supply it with `Constrained::differentiable`"
            ));
        }
        match self.constraints {
            Some(known) if known != m => invalid(format!(
                "the fitness function provides {m} constraints, and the run so far has {known}"
            )),
            Some(_) => Ok(()),
            None => {
                self.constraints = Some(m);
                self.allocate(m);
                Ok(())
            }
        }
    }

    fn wants(&self) -> Wanted {
        let constrained = self.constraints.is_none_or(|m| m > 0);
        if !self.reevaluating && matches!(self.phase, Phase::Restore { .. }) {
            return Wanted::NOTHING.with_inequalities();
        }
        if constrained {
            Wanted::GRADIENT
                .with_inequalities()
                .with_constraint_jacobian()
        } else {
            Wanted::GRADIENT
        }
    }
}

/// A builder for an [`Mma`], from [`Mma::builder`].
///
/// Defaults: maximize, [`Method::Mma`], Svanberg's (2007) asymptote constants (an initial
/// distance of 0.5 of each gene's range, a factor of 0.7 to bring the asymptotes nearer and of 1.2
/// to move them away), a move limit of 0.5 of the range, a constraint cost of 1000, a KKT tolerance
/// of 1e-9 and a step tolerance of 1e-10, the restoration step, a random initial genome and a
/// random seed.
#[derive(Clone, Debug)]
pub struct MmaBuilder {
    real: Real,
    settings: Settings,
    initial_genome: Option<Reals>,
    objective: Objective,
    seed: Option<u64>,
}

impl MmaBuilder {
    /// [`Method::Mma`] (the default) or [`Method::Gcmma`].
    pub fn method(mut self, method: Method) -> Self {
        self.settings.method = method;
        self
    }

    /// The distance of the asymptotes from the point in the first two iterations, as a fraction
    /// of each gene's range, above 0: 0.5 by default (Svanberg 2007, eq. 3.11, `asyinit`). Smaller
    /// is more conservative: more curvature, shorter first steps.
    pub fn asymptote_initial(mut self, fraction: f64) -> Self {
        self.settings.asymptote_initial = fraction;
        self
    }

    /// The factor that brings a gene's asymptotes nearer when the gene oscillates, above 0 and at
    /// most 1: 0.7 by default (eq. 3.13, `asydecr`).
    pub fn asymptote_decrease(mut self, factor: f64) -> Self {
        self.settings.asymptote_decrease = factor;
        self
    }

    /// The factor that moves a gene's asymptotes away when the gene moves the same way twice, at
    /// least 1: 1.2 by default (eq. 3.13, `asyincr`).
    pub fn asymptote_increase(mut self, factor: f64) -> Self {
        self.settings.asymptote_increase = factor;
        self
    }

    /// The largest step of a gene in one iteration, as a fraction of its range, above 0: 0.5 by
    /// default (eqs. 3.6-3.7, `move`). Each step also stays 0.9 of the way to the asymptotes.
    pub fn move_limit(mut self, fraction: f64) -> Self {
        self.settings.move_limit = fraction;
        self
    }

    /// The cost `c` per unit of the artificial variable that relaxes each constraint in the
    /// subproblems, above 0: 1000 by default, as in Svanberg's (2002, section 8.4) tests. It must
    /// exceed the constraints' Lagrange multipliers at the solution, which depend on the scaling
    /// of the score and the constraints; otherwise the solution is infeasible, with a
    /// [multiplier](Mma::multipliers) at `c`. Not too large either: the notes advise "reasonably
    /// large" values, such as 1000 or 10,000 for well-scaled problems.
    pub fn constraint_cost(mut self, cost: f64) -> Self {
        self.settings.constraint_cost = cost;
        self
    }

    /// The [KKT residual](Mma::kkt_residual) at which a run has converged, at least 0: 1e-9 by
    /// default. It's in the units of the score's gradient: scale the score so that it's of the
    /// order of 1 to 100, as Svanberg's notes advise, or set it to match. The paper's tests stop
    /// at 1e-5 (eq. 8.4's mean square of 1e-10).
    pub fn kkt_tolerance(mut self, tolerance: f64) -> Self {
        self.settings.kkt_tolerance = tolerance;
        self
    }

    /// The step at which a run has converged: when no gene moves by more than this fraction of
    /// its range in an iteration, at least 0. 1e-10 by default.
    pub fn step_tolerance(mut self, tolerance: f64) -> Self {
        self.settings.step_tolerance = tolerance;
        self
    }

    /// Whether a run that converges to an infeasible point ends with the restoration step onto
    /// the feasible side of its active constraints (see [`Mma`]). On by default.
    pub fn restoration(mut self, restoration: bool) -> Self {
        self.settings.restoration = restoration;
        self
    }

    /// Sums the dual function's terms over the genes on rayon's threads, in chunks of 4096 genes
    /// whose partial sums are added in order: the same results as without it, on any number of
    /// threads. Off by default; it pays off from about 10⁵ genes, when the subproblems take much
    /// of an iteration.
    #[cfg(feature = "parallel")]
    pub fn parallel_sums(mut self, parallel: bool) -> Self {
        self.settings.parallel = parallel;
        self
    }

    /// The genome to start from, e.g. a known design or the best of a global method. Random by
    /// default.
    pub fn initial_genome(mut self, genome: Reals) -> Self {
        self.initial_genome = Some(genome);
        self
    }

    /// Whether higher or lower fitness is better. Maximize by default.
    pub fn objective(mut self, objective: Objective) -> Self {
        self.objective = objective;
        self
    }

    /// Higher fitness is better (the default).
    pub fn maximize(self) -> Self {
        self.objective(Objective::Maximize)
    }

    /// Lower fitness is better.
    pub fn minimize(self) -> Self {
        self.objective(Objective::Minimize)
    }

    /// The seed of the random numbers, for a reproducible run: the initial genome is random
    /// unless it's given. Random by default.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Validates the settings and creates the run.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidSetting`] for a representation without a gene that has more than one
    ///   value, an initial asymptote distance or a move limit that isn't positive and finite, a
    ///   decrease factor that isn't above 0 and at most 1, an increase factor that isn't finite
    ///   and at least 1, a constraint cost that isn't positive and finite, or a tolerance that
    ///   isn't finite and at least 0.
    /// - [`Error::InvalidGenome`] for an initial genome that doesn't fit the representation.
    pub fn build(self) -> Result<Mma> {
        let invalid =
            |setting: &'static str, reason: String| Err(Error::InvalidSetting { setting, reason });
        let settings = self.settings;
        if self.real.variable_genes().is_empty() {
            return invalid(
                "real",
                "MMA needs a gene with more than one value".to_string(),
            );
        }
        let positive = |value: f64| value > 0.0 && value.is_finite();
        if !positive(settings.asymptote_initial) {
            return invalid(
                "asymptote_initial",
                format!(
                    "must be positive and finite, got {}",
                    settings.asymptote_initial
                ),
            );
        }
        if !(settings.asymptote_decrease > 0.0 && settings.asymptote_decrease <= 1.0) {
            return invalid(
                "asymptote_decrease",
                format!(
                    "must be above 0 and at most 1, got {}",
                    settings.asymptote_decrease
                ),
            );
        }
        if !(settings.asymptote_increase >= 1.0 && settings.asymptote_increase.is_finite()) {
            return invalid(
                "asymptote_increase",
                format!(
                    "must be finite and at least 1, got {}",
                    settings.asymptote_increase
                ),
            );
        }
        if !positive(settings.move_limit) {
            return invalid(
                "move_limit",
                format!("must be positive and finite, got {}", settings.move_limit),
            );
        }
        if !positive(settings.constraint_cost) {
            return invalid(
                "constraint_cost",
                format!(
                    "must be positive and finite, got {}",
                    settings.constraint_cost
                ),
            );
        }
        for (setting, tolerance) in [
            ("kkt_tolerance", settings.kkt_tolerance),
            ("step_tolerance", settings.step_tolerance),
        ] {
            if !(tolerance >= 0.0 && tolerance.is_finite()) {
                return invalid(
                    setting,
                    format!("must be finite and at least 0, got {tolerance}"),
                );
            }
        }
        if let Some(genome) = &self.initial_genome {
            self.real.validate(genome)?;
        }
        let seed = self
            .seed
            .unwrap_or_else(|| StreamRng::from_entropy().next_u64());
        let mut rng = StreamRng::seed_from_u64(seed);
        let start = match self.initial_genome {
            Some(genome) => genome,
            None => self.real.random_genome(&mut rng),
        };
        let trial = Individual::new(start.clone());
        Ok(Mma {
            real: self.real,
            settings,
            objective: self.objective,
            seed,
            constraints: None,
            current: Population::new(vec![Individual::new(start)]),
            values: Vec::new(),
            gradient: Vec::new(),
            jacobian: Vec::new(),
            asymptotes: MovingAsymptotes::default(),
            rho: Vec::new(),
            rounding: Vec::new(),
            multipliers: Vec::new(),
            trial,
            approximations: Vec::new(),
            scales: Vec::new(),
            distance: 0.0,
            move_fraction: 1.0,
            phase: Phase::Start,
            converged: None,
            kkt_residual: f64::NAN,
            iterations: 0,
            inner: 0,
            inner_iterations: 0,
            reevaluating: false,
            asked: false,
            discarded_trial: false,
            generation: 0,
            evaluations: 0,
            best: None,
            best_generation: 0,
            scratch: Scratch::default(),
        })
    }
}
