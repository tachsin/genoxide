//! Multi-objective test problems: fitness functions with their search space, their optimal front
//! where it's known, and the paper that defines them. All objectives are minimized.
//!
//! Each problem implements [`MultiProblem`], and is a fitness function for a
//! [`MultiEngine`](super::MultiEngine) as it is; its optimal front serves the
//! [indicators](super::indicator):
//!
//! ```
//! use genoxide::Objective::Minimize;
//! use genoxide::multi::indicator::igd_plus;
//! use genoxide::multi::problems::{MultiProblem, Zdt1};
//! use genoxide::prelude::*;
//!
//! let problem = Zdt1::new(30);
//! let nsga2 = Nsga2::builder(problem.representation(), [Minimize; 2])
//!     .population_size(100)
//!     .crossover(SimulatedBinaryCrossover::new(15.0)?)
//!     .mutate(PolynomialMutation::per_gene(1.0 / 30.0, 20.0)?)
//!     .seed(1)
//!     .build()?;
//! let outcome = MultiEngine::new(nsga2, problem).stop_when(Stop::generations(200)).run()?;
//! let front = problem.optimal_front(500).expect("known");
//! assert!(igd_plus(&outcome.front_values(), &front, &[Minimize; 2]) < 0.01);
//! # Ok::<(), genoxide::Error>(())
//! ```
//!
//! A constrained problem's fitness is `([f64; M], violation)`, the violation being the sum of
//! its constraint violations, 0 when feasible: the algorithms prefer feasible solutions, then
//! compare objectives or violations (Deb's rules). [`MultiProblem::constraints`] gives each
//! constraint's value. [`all`] lists the problems with `M` objectives at their default sizes,
//! behind the object-safe [`DynMultiProblem`].
//!
//! # The problems
//!
//! | Problem | Variables (default) | Objectives | Constraints | Optimal front |
//! |---|---|---|---|---|
//! | [`Zdt1`], [`Zdt2`], [`Zdt3`] | 2 or more (30) | 2 | | convex; concave; five pieces |
//! | [`Zdt4`], [`Zdt6`] | 2 or more (10) | 2 | | convex, many local fronts; concave, biased |
//! | [`Zdt5`] | 80 bits | 2 | | 31 points, deceptive |
//! | [`Dtlz1`], [`Dtlz2`], [`Dtlz3`], [`Dtlz4`] | M or more (M + 4, M + 9) | any M ≥ 2 | | linear; spherical |
//! | [`Dtlz5`], [`Dtlz6`] | M or more (M + 9) | any M ≥ 2 | | a curve for M ≤ 3; not known for more |
//! | [`Dtlz7`] | M or more (M + 19) | any M ≥ 2 | | 2^(M−1) disconnected regions |
//! | [`ConvexDtlz2`], [`ScaledDtlz2`] | M or more (M + 9) | any M ≥ 2 | | convex; a scaled sphere |
//! | [`ScaledDtlz1`], [`InvertedDtlz1`] | M or more (M + 4) | any M ≥ 2 | | a scaled plane; an inverted simplex |
//! | [`Wfg1`], [`Wfg2`], [`Wfg3`] | k + l (k = 4 or 2(M − 1), l = 20) | any M ≥ 2 | | convex and mixed; convex, disconnected; linear for M = 2 |
//! | [`Wfg4`] to [`Wfg9`] | k + l (k = 4 or 2(M − 1), l = 20) | any M ≥ 2 | | concave |
//! | [`Schaffer1`] | 1 | 2 | | convex |
//! | [`Schaffer2`] | 1 | 2 | | two pieces |
//! | [`FonsecaFleming`] | 1 or more (3) | 2 | | concave |
//! | [`Kursawe`] | 2 or more (3) | 2 | | disconnected, not known |
//! | [`Poloni`] | 2 | 2 | | disconnected, not known |
//! | [`Viennet1`], [`Viennet2`], [`Viennet3`] | 2 | 3 | | VNT1 a surface; not known |
//! | [`Bnh`] | 2 | 2 | 2 | convex |
//! | [`Srn`] | 2 | 2 | 2 | three pieces |
//! | [`Tnk`] | 2 | 2 | 2 | disconnected, sampled |
//! | [`Osy`] | 6 | 2 | 6 | five pieces |
//! | [`Constr`] | 2 | 2 | 2 | convex, two pieces |
//! | [`Ctp1`] | 2 | 2 | 2 | three pieces, two on constraint boundaries |
//! | [`Ctp2`], [`Ctp3`], [`Ctp4`], [`Ctp5`] | 2 | 2 | 1 | 13 pieces; 13 points; 13 points behind tunnels; a piece and 15 points |
//! | [`Ctp6`], [`Ctp7`], [`Ctp8`] | 2 | 2 | 1, 1, 2 | on a boundary, behind infeasible bands; six pieces and a point; three pieces |
//! | [`C1Dtlz1`], [`C1Dtlz3`] | M or more (M + 4, M + 9) | any M ≥ 2 | 1 | DTLZ1's, DTLZ3's, behind infeasible barriers |
//! | [`C2Dtlz2`], [`ConvexC2Dtlz2`] | M or more (M + 9) | any M ≥ 2 | 1 | parts of DTLZ2's, of convex DTLZ2's |
//! | [`C3Dtlz1`], [`C3Dtlz4`] | M or more (M + 4) | any M ≥ 2 | M | on the constraints' boundaries |
//! | [`Mw1`] to [`Mw3`], [`Mw5`] to [`Mw7`], [`Mw9`] to [`Mw13`] | 3 or more (15) | 2 | 1 to 4 | disconnected, points, or on constraint boundaries |
//! | [`Mw4`], [`Mw8`], [`Mw14`] | M + 1 or more (M + 12) | any M ≥ 2 | 1 | linear; spherical in four bands; 2^(M−1) patches |
//! | [`engineering`]: two-bar and four-bar trusses, welded beam, disc brake, speed reducer | 3 to 7 | 2 | 0 to 11 | the trusses' derived; the others not known |
//! | [`engineering`]: car side impact, rocket injector, vehicle crashworthiness; water resource planning | 3 to 7 | 3; 5 | 0 or 10; 7 | not known |
//!
//! ZDT is Zitzler, Deb and Thiele's suite (2000, *Evolutionary Computation* 8(2): 173-195), and
//! DTLZ Deb, Thiele, Laumanns and Zitzler's (2001, TIK-Report 112, ETH Zürich; and 2002,
//! *Proceedings of the 2002 Congress on Evolutionary Computation*: 825-830), in the numbering of
//! their technical report, whose DTLZ6 and DTLZ7 the paper calls DTLZ5 and DTLZ6, and WFG
//! Huband, Hingston, Barone and While's (2006, *IEEE Transactions on Evolutionary Computation*
//! 10(5): 477-506), checked against the authors' C++ toolkit. The convex, scaled and inverted
//! DTLZ problems are Deb and Jain's (2014, *IEEE Transactions on Evolutionary Computation* 18(4):
//! 577-601 and 602-622), CTP Deb, Pratap and Meyarivan's (2001, EMO 2001, LNCS 1993: 284-298),
//! C-DTLZ Jain and Deb's (2014, *IEEE Transactions on Evolutionary Computation* 18(4): 602-622),
//! and MW Ma and Wang's (2019, *IEEE Transactions on Evolutionary Computation* 23(6): 972-986).
//! [`Zdt5`] has
//! [`Binary`](crate::genome::Binary) genomes, and so isn't in [`all`], whose problems have
//! [`Real`] ones. Each other problem's docs give its definition and cite its original authors.
//! Some originals are conference proceedings that aren't online: those definitions are taken from
//! later papers that restate them, named in the docs, and are still to be checked against the
//! originals in [#168](https://github.com/tachsin/genoxide/issues/168). Fronts are derived from
//! the definitions, and the docs say how.
//!
//! The problems compute their trigonometric and exponential functions with [`math`](crate::math),
//! so their values are the same to the bit on every platform, like the rest of genoxide.

mod cdtlz;
mod classic;
mod ctp;
mod dtlz;
mod dtlz_variants;
pub mod engineering;
mod mw;
mod wfg;
mod zdt;

pub use cdtlz::{C1Dtlz1, C1Dtlz3, C2Dtlz2, C3Dtlz1, C3Dtlz4, ConvexC2Dtlz2};
pub use classic::{
    Bnh, Constr, FonsecaFleming, Kursawe, Osy, Poloni, Schaffer1, Schaffer2, Srn, Tnk, Viennet1,
    Viennet2, Viennet3,
};
pub use ctp::{Ctp1, Ctp2, Ctp3, Ctp4, Ctp5, Ctp6, Ctp7, Ctp8};
pub use dtlz::{Dtlz1, Dtlz2, Dtlz3, Dtlz4, Dtlz5, Dtlz6, Dtlz7};
pub use dtlz_variants::{ConvexDtlz2, InvertedDtlz1, ScaledDtlz1, ScaledDtlz2};
pub use mw::{Mw1, Mw2, Mw3, Mw4, Mw5, Mw6, Mw7, Mw8, Mw9, Mw10, Mw11, Mw12, Mw13, Mw14};
pub use wfg::{Wfg1, Wfg2, Wfg3, Wfg4, Wfg5, Wfg6, Wfg7, Wfg8, Wfg9};
pub use zdt::{Zdt1, Zdt2, Zdt3, Zdt4, Zdt5, Zdt6};

use super::{IntoScores, MultiFitnessFunction, Scores};
use crate::genome::{Real, Reals, Representation};
pub use crate::problems::Constraints;

/// A test problem with `M` objectives, all minimized: a fitness function with its search space,
/// its optimal front where it's known, and the paper that defines it.
///
/// Its [`MultiFitnessFunction::Output`] is `[f64; M]` for an unconstrained problem and
/// `([f64; M], f64)`, the objectives and the constraint violation, for a constrained one.
pub trait MultiProblem<const M: usize>:
    MultiFitnessFunction<<<Self as MultiProblem<M>>::Representation as Representation>::Genome, M>
{
    /// The search space, e.g. [`Real`] bounds.
    type Representation: Representation;

    /// The name, e.g. `"ZDT1"`.
    fn name(&self) -> &'static str;

    /// The search space: the number of variables and their bounds.
    fn representation(&self) -> Self::Representation;

    /// The paper, book or report that defines the problem.
    fn reference(&self) -> &'static str;

    /// Its DOI or URL, if any.
    fn reference_url(&self) -> Option<&'static str> {
        None
    }

    /// The number of constraints: 0 for an unconstrained problem.
    fn constraint_count(&self) -> usize {
        0
    }

    /// The values of the constraints at `genome`, in the paper's order, as `g(x) <= 0`; none for
    /// an unconstrained problem.
    fn constraints(
        &self,
        genome: &<Self::Representation as Representation>::Genome,
    ) -> Constraints {
        let _ = genome;
        Constraints::none()
    }

    /// At least `points` points of the optimal front (for 2 objectives, exactly `points`, and
    /// for more, the smallest set of evenly spread points with at least `points`), for
    /// [`igd`](super::indicator::igd) and similar indicators; `None` if the front isn't known.
    fn optimal_front(&self, points: usize) -> Option<Vec<[f64; M]>>;

    /// The best value of each objective on the optimal front, if it's known.
    fn ideal_point(&self) -> Option<[f64; M]> {
        None
    }

    /// The worst value of each objective on the optimal front, if it's known: with the ideal
    /// point, what normalizes the objectives.
    fn nadir_point(&self) -> Option<[f64; M]> {
        None
    }
}

/// A problem with `M` objectives on [`Real`] genomes, as a trait object: for lists of problems
/// of different types, such as [`all`].
///
/// ```
/// use genoxide::multi::problems;
///
/// for problem in problems::all::<2>() {
///     let front = problem.optimal_front(10);
///     assert!(front.is_none_or(|front| front.len() == 10), "{}", problem.name());
/// }
/// ```
pub trait DynMultiProblem<const M: usize>: Send + Sync {
    /// The name, as [`MultiProblem::name`].
    fn name(&self) -> &'static str;

    /// The search space, as [`MultiProblem::representation`].
    fn real(&self) -> Real;

    /// The scores of `genome`: its objective values, and its constraint violation for a
    /// constrained problem.
    fn evaluate(&self, genome: &Reals) -> Scores<M>;

    /// The paper that defines the problem, as [`MultiProblem::reference`].
    fn reference(&self) -> &'static str;

    /// Its DOI or URL, as [`MultiProblem::reference_url`].
    fn reference_url(&self) -> Option<&'static str>;

    /// The number of constraints, as [`MultiProblem::constraint_count`].
    fn constraint_count(&self) -> usize;

    /// The constraint values at `genome`, as [`MultiProblem::constraints`].
    fn constraints(&self, genome: &Reals) -> Constraints;

    /// Points of the optimal front, as [`MultiProblem::optimal_front`].
    fn optimal_front(&self, points: usize) -> Option<Vec<[f64; M]>>;

    /// The ideal point, as [`MultiProblem::ideal_point`].
    fn ideal_point(&self) -> Option<[f64; M]>;

    /// The nadir point, as [`MultiProblem::nadir_point`].
    fn nadir_point(&self) -> Option<[f64; M]>;
}

// a problem with K objectives behind `DynMultiProblem<M>`, built only when M = K; a wrapper, so
// that problems don't have the methods of both traits
struct Boxed<P, const K: usize>(P);

// the values of an array of K values as one of M, when M = K
fn resized<const K: usize, const M: usize>(values: [f64; K]) -> [f64; M] {
    std::array::from_fn(|i| values[i])
}

impl<P, const K: usize, const M: usize> DynMultiProblem<M> for Boxed<P, K>
where
    P: MultiProblem<K, Representation = Real> + Send + Sync,
{
    fn name(&self) -> &'static str {
        self.0.name()
    }

    fn real(&self) -> Real {
        self.0.representation()
    }

    fn evaluate(&self, genome: &Reals) -> Scores<M> {
        let scores: Scores<K> = self
            .0
            .evaluate(genome)
            .into_scores()
            .unwrap_or_else(|_| Scores::invalid());
        match scores.values() {
            Some(values) => Scores::constrained(resized(values), scores.violation()),
            None => Scores::invalid(),
        }
    }

    fn reference(&self) -> &'static str {
        self.0.reference()
    }

    fn reference_url(&self) -> Option<&'static str> {
        self.0.reference_url()
    }

    fn constraint_count(&self) -> usize {
        self.0.constraint_count()
    }

    fn constraints(&self, genome: &Reals) -> Constraints {
        self.0.constraints(genome)
    }

    fn optimal_front(&self, points: usize) -> Option<Vec<[f64; M]>> {
        let front = self.0.optimal_front(points)?;
        Some(front.into_iter().map(resized).collect())
    }

    fn ideal_point(&self) -> Option<[f64; M]> {
        self.0.ideal_point().map(resized)
    }

    fn nadir_point(&self) -> Option<[f64; M]> {
        self.0.nadir_point().map(resized)
    }
}

/// `problem` as a [`DynMultiProblem`].
pub fn boxed<P, const M: usize>(problem: P) -> Box<dyn DynMultiProblem<M>>
where
    P: MultiProblem<M, Representation = Real> + Send + Sync + 'static,
{
    Box::new(Boxed::<P, M>(problem))
}

/// `problem`, with `K` objectives, as a [`DynMultiProblem`] with `M` objectives if `M = K`, and
/// `None` otherwise: for code generic over the number of objectives.
///
/// ```
/// use genoxide::multi::problems::{Viennet1, Zdt1, try_boxed};
///
/// assert!(try_boxed::<_, 2, 2>(Zdt1::default()).is_some());
/// assert!(try_boxed::<_, 3, 2>(Viennet1).is_none());
/// ```
pub fn try_boxed<P, const K: usize, const M: usize>(
    problem: P,
) -> Option<Box<dyn DynMultiProblem<M>>>
where
    P: MultiProblem<K, Representation = Real> + Send + Sync + 'static,
{
    (K == M).then(|| Box::new(Boxed::<P, K>(problem)) as Box<dyn DynMultiProblem<M>>)
}

/// Every problem of this module with `M` objectives and [`Real`] genomes, at its default size:
/// the two-objective problems for `M = 2`, the Viennet problems for `M = 3`, and the
/// [`engineering`] problems with `M` objectives, in the order of the table above, then DTLZ1-7,
/// the convex, scaled and inverted DTLZ problems, WFG1-9, MW4, MW8
/// and MW14 for any `M` from 2 on, then the constrained DTLZ problems (C1-DTLZ3 and convex
/// C2-DTLZ2 only for the numbers of objectives their paper gives a radius for: 3, 5, 8, 10 and
/// 15). [`Zdt5`], on bit strings, isn't in it.
pub fn all<const M: usize>() -> Vec<Box<dyn DynMultiProblem<M>>> {
    let fixed = [
        try_boxed::<_, 2, M>(Zdt1::default()),
        try_boxed::<_, 2, M>(Zdt2::default()),
        try_boxed::<_, 2, M>(Zdt3::default()),
        try_boxed::<_, 2, M>(Zdt4::default()),
        try_boxed::<_, 2, M>(Zdt6::default()),
        try_boxed::<_, 2, M>(Schaffer1),
        try_boxed::<_, 2, M>(Schaffer2),
        try_boxed::<_, 2, M>(FonsecaFleming::default()),
        try_boxed::<_, 2, M>(Kursawe::default()),
        try_boxed::<_, 2, M>(Poloni),
        try_boxed::<_, 3, M>(Viennet1),
        try_boxed::<_, 3, M>(Viennet2),
        try_boxed::<_, 3, M>(Viennet3),
        try_boxed::<_, 2, M>(Bnh),
        try_boxed::<_, 2, M>(Srn),
        try_boxed::<_, 2, M>(Tnk),
        try_boxed::<_, 2, M>(Osy),
        try_boxed::<_, 2, M>(Constr),
        try_boxed::<_, 2, M>(Ctp1),
        try_boxed::<_, 2, M>(Ctp2),
        try_boxed::<_, 2, M>(Ctp3),
        try_boxed::<_, 2, M>(Ctp4),
        try_boxed::<_, 2, M>(Ctp5),
        try_boxed::<_, 2, M>(Ctp6),
        try_boxed::<_, 2, M>(Ctp7),
        try_boxed::<_, 2, M>(Ctp8),
        try_boxed::<_, 2, M>(Mw1::default()),
        try_boxed::<_, 2, M>(Mw2::default()),
        try_boxed::<_, 2, M>(Mw3::default()),
        try_boxed::<_, 2, M>(Mw5::default()),
        try_boxed::<_, 2, M>(Mw6::default()),
        try_boxed::<_, 2, M>(Mw7::default()),
        try_boxed::<_, 2, M>(Mw9::default()),
        try_boxed::<_, 2, M>(Mw10::default()),
        try_boxed::<_, 2, M>(Mw11::default()),
        try_boxed::<_, 2, M>(Mw12::default()),
        try_boxed::<_, 2, M>(Mw13::default()),
        try_boxed::<_, 2, M>(engineering::TwoBarTruss),
        try_boxed::<_, 2, M>(engineering::WeldedBeam),
        try_boxed::<_, 2, M>(engineering::DiscBrake),
        try_boxed::<_, 2, M>(engineering::SpeedReducer),
        try_boxed::<_, 2, M>(engineering::FourBarTruss),
        try_boxed::<_, 3, M>(engineering::CarSideImpact),
        try_boxed::<_, 3, M>(engineering::RocketInjector),
        try_boxed::<_, 3, M>(engineering::VehicleCrashworthiness),
        try_boxed::<_, 5, M>(engineering::WaterResourcePlanning),
    ];
    let mut problems: Vec<_> = fixed.into_iter().flatten().collect();
    if M >= 2 {
        problems.push(boxed(Dtlz1::<M>::default()));
        problems.push(boxed(Dtlz2::<M>::default()));
        problems.push(boxed(Dtlz3::<M>::default()));
        problems.push(boxed(Dtlz4::<M>::default()));
        problems.push(boxed(Dtlz5::<M>::default()));
        problems.push(boxed(Dtlz6::<M>::default()));
        problems.push(boxed(Dtlz7::<M>::default()));
        problems.push(boxed(ConvexDtlz2::<M>::default()));
        problems.push(boxed(ScaledDtlz1::<M>::default()));
        problems.push(boxed(ScaledDtlz2::<M>::default()));
        problems.push(boxed(InvertedDtlz1::<M>::default()));
        problems.push(boxed(Wfg1::<M>::default()));
        problems.push(boxed(Wfg2::<M>::default()));
        problems.push(boxed(Wfg3::<M>::default()));
        problems.push(boxed(Wfg4::<M>::default()));
        problems.push(boxed(Wfg5::<M>::default()));
        problems.push(boxed(Wfg6::<M>::default()));
        problems.push(boxed(Wfg7::<M>::default()));
        problems.push(boxed(Wfg8::<M>::default()));
        problems.push(boxed(Wfg9::<M>::default()));
        problems.push(boxed(Mw4::<M>::default()));
        problems.push(boxed(Mw8::<M>::default()));
        problems.push(boxed(Mw14::<M>::default()));
    }
    if M >= 2 {
        problems.extend(cdtlz::all::<M>());
    }
    problems
}

/// Points evenly spread on the unit simplex (Das and Dennis, 1998): every point with `M`
/// coordinates that are multiples of `1 / divisions` and sum to 1, in lexicographic order.
/// There are `(divisions + M − 1)! / (divisions! (M − 1)!)` of them: 91 for 3 objectives and 12
/// divisions, and none for 0 divisions. They serve as the reference directions of NSGA-III and
/// MOEA/D.
///
/// ```
/// use genoxide::multi::das_dennis;
///
/// let points = das_dennis::<3>(2);
/// assert_eq!(points.len(), 6);
/// assert_eq!(points[0], [0.0, 0.0, 1.0]);
/// assert!(points.iter().all(|p| (p.iter().sum::<f64>() - 1.0).abs() < 1e-12));
/// ```
pub fn das_dennis<const M: usize>(divisions: usize) -> Vec<[f64; M]> {
    let mut points = Vec::new();
    if M == 0 || divisions == 0 {
        return points;
    }
    let mut counts = [0usize; M];
    fill(&mut counts, 0, divisions, divisions, &mut points);
    points
}

// sets the counts from `position` on, with `left` still to share, and records each point
fn fill<const M: usize>(
    counts: &mut [usize; M],
    position: usize,
    left: usize,
    divisions: usize,
    points: &mut Vec<[f64; M]>,
) {
    if position == M - 1 {
        counts[position] = left;
        points.push(counts.map(|count| count as f64 / divisions as f64));
        return;
    }
    for count in 0..=left {
        counts[position] = count;
        fill(counts, position + 1, left - count, divisions, points);
    }
}

// the smallest number of divisions with at least `points` Das-Dennis points
fn divisions_for<const M: usize>(points: usize) -> usize {
    let mut divisions = 1;
    while das_dennis_count(M, divisions) < points {
        divisions += 1;
    }
    divisions
}

fn das_dennis_count(m: usize, divisions: usize) -> usize {
    // C(divisions + m − 1, m − 1), exact in u128 for any practical size
    let (n, k) = ((divisions + m - 1) as u128, (m - 1) as u128);
    let mut count = 1u128;
    for i in 0..k {
        count = count * (n - i) / (i + 1);
    }
    count.min(usize::MAX as u128) as usize
}

// the i-th of `count` values evenly spread over [0, 1], ends included; 0 for a single value
fn evenly(i: usize, count: usize) -> f64 {
    if count > 1 {
        i as f64 / (count - 1) as f64
    } else {
        0.0
    }
}

// a piece of a two-objective front: a curve over t in [0, 1], with or without its end at t = 1
// (a point the next piece dominates)
struct Piece<'a> {
    curve: &'a dyn Fn(f64) -> [f64; 2],
    with_end: bool,
}

// `points` points on `pieces`, in order, shared in proportion to the pieces' lengths in the
// objective space, and evenly spread on each by its parameter
fn pieces_front(pieces: &[Piece<'_>], points: usize) -> Vec<[f64; 2]> {
    const STEPS: usize = 1_000;
    let lengths: Vec<f64> = pieces
        .iter()
        .map(|piece| {
            (1..=STEPS)
                .map(|step| {
                    let [a, b] = (piece.curve)((step - 1) as f64 / STEPS as f64);
                    let [c, d] = (piece.curve)(step as f64 / STEPS as f64);
                    ((c - a) * (c - a) + (d - b) * (d - b)).sqrt()
                })
                .sum()
        })
        .collect();
    let total: f64 = lengths.iter().sum();
    // the points of each piece: its share, rounded down, and the rest to the longest remainders
    let shares: Vec<f64> = lengths
        .iter()
        .map(|length| points as f64 * length / total)
        .collect();
    let mut counts: Vec<usize> = shares.iter().map(|share| share.floor() as usize).collect();
    let mut order: Vec<usize> = (0..pieces.len()).collect();
    order.sort_by(|&a, &b| {
        (shares[b] - shares[b].floor()).total_cmp(&(shares[a] - shares[a].floor()))
    });
    let assigned: usize = counts.iter().sum();
    for &piece in order.iter().take(points - assigned) {
        counts[piece] += 1;
    }
    pieces
        .iter()
        .zip(counts)
        .flat_map(|(piece, count)| {
            (0..count).map(move |i| {
                let t = if piece.with_end {
                    evenly(i, count)
                } else {
                    i as f64 / count as f64
                };
                (piece.curve)(t)
            })
        })
        .collect()
}

// the points that no other point dominates (minimizing both), sorted by the first objective;
// of equal points, one
fn non_dominated(mut points: Vec<[f64; 2]>) -> Vec<[f64; 2]> {
    points.sort_by(|a, b| a[0].total_cmp(&b[0]).then(a[1].total_cmp(&b[1])));
    let mut front: Vec<[f64; 2]> = Vec::new();
    for point in points {
        if front.last().is_none_or(|last| point[1] < last[1]) {
            front.push(point);
        }
    }
    front
}

// `points` points spread over `front`, from the first to the last
fn spread_over(front: &[[f64; 2]], points: usize) -> Vec<[f64; 2]> {
    (0..points)
        .map(|i| {
            let index = (evenly(i, points) * (front.len() - 1) as f64).round() as usize;
            front[index]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Objective::Minimize;
    use crate::StreamRng;
    use crate::multi::non_dominated_sort;
    use std::collections::HashSet;

    #[test]
    fn optimal_fronts_have_the_promised_number_of_points() {
        // exactly `points` with 2 objectives, and at least `points` with more
        for problem in all::<2>() {
            for points in 0..6 {
                if let Some(front) = problem.optimal_front(points) {
                    assert_eq!(front.len(), points, "{} with {points}", problem.name());
                }
            }
        }
        for problem in all::<3>() {
            for points in 0..6 {
                if let Some(front) = problem.optimal_front(points) {
                    assert!(front.len() >= points, "{} with {points}", problem.name());
                }
            }
        }
    }

    #[test]
    fn das_dennis_points() {
        assert_eq!(
            das_dennis::<2>(4),
            [
                [0.0, 1.0],
                [0.25, 0.75],
                [0.5, 0.5],
                [0.75, 0.25],
                [1.0, 0.0]
            ]
        );
        assert_eq!(das_dennis::<3>(12).len(), 91);
        assert_eq!(das_dennis::<5>(6).len(), 210);
        assert!(das_dennis::<3>(0).is_empty());
        assert_eq!(das_dennis::<1>(3), [[1.0]]);
        assert_eq!(das_dennis::<4>(5).len(), das_dennis_count(4, 5));
        assert_eq!(divisions_for::<3>(91), 12);
        assert_eq!(divisions_for::<3>(92), 13);
    }

    #[test]
    fn fronts_of_pieces_share_their_points() {
        let line = |t: f64| [t, 1.0 - t];
        let far = |t: f64| [2.0 + 2.0 * t, -2.0 * t];
        let pieces = [
            Piece {
                curve: &line,
                with_end: false,
            },
            Piece {
                curve: &far,
                with_end: true,
            },
        ];
        // the second piece is twice as long: 3 and 6 points, the first without its end
        let front = pieces_front(&pieces, 9);
        assert_eq!(front.len(), 9);
        assert_eq!(front[0], [0.0, 1.0]);
        assert!((front[2][0] - 2.0 / 3.0).abs() < 1e-12);
        assert_eq!(front[3], [2.0, 0.0]);
        assert_eq!(front[8], [4.0, -2.0]);
        assert!(pieces_front(&pieces, 0).is_empty());
        assert_eq!(pieces_front(&pieces, 1).len(), 1);
        let front = non_dominated(vec![
            [1.0, 1.0],
            [0.0, 2.0],
            [1.0, 0.5],
            [2.0, 0.5],
            [0.0, 2.0],
        ]);
        assert_eq!(front, [[0.0, 2.0], [1.0, 0.5]]);
        assert_eq!(spread_over(&front, 3), [[0.0, 2.0], [1.0, 0.5], [1.0, 0.5]]);
    }

    // what every problem of the registry promises: unique names, references, fronts of mutually
    // non-dominated points of the promised size between the ideal and nadir points, and finite
    // objectives and violations everywhere in the bounds
    fn check_registry<const M: usize>(problems: Vec<Box<dyn DynMultiProblem<M>>>) {
        let names: HashSet<_> = problems.iter().map(|problem| problem.name()).collect();
        assert_eq!(names.len(), problems.len(), "names are unique");
        let mut rng = StreamRng::seed_from_u64(1);
        for problem in &problems {
            let name = problem.name();
            assert!(!problem.reference().is_empty(), "{name}");
            if let Some(url) = problem.reference_url() {
                assert!(url.starts_with("https://"), "{url}");
            }
            if let Some(front) = problem.optimal_front(50) {
                if M == 2 {
                    assert_eq!(front.len(), 50, "{name}");
                } else {
                    assert!(front.len() >= 50, "{name}");
                }
                let scores: Vec<Scores<M>> = front.iter().map(|p| Scores::new(*p)).collect();
                assert_eq!(
                    non_dominated_sort(&scores, &[Minimize; M]).len(),
                    1,
                    "{name}"
                );
                let (ideal, nadir) = (problem.ideal_point(), problem.nadir_point());
                let (ideal, nadir) = (ideal.expect("known"), nadir.expect("known"));
                let slack = |value: f64| 1e-9 * value.abs().max(1.0);
                for point in &front {
                    for j in 0..M {
                        assert!(point[j] >= ideal[j] - slack(ideal[j]), "{name}: {point:?}");
                        assert!(point[j] <= nadir[j] + slack(nadir[j]), "{name}: {point:?}");
                    }
                }
                // the ideal and nadir points are reached
                for j in 0..M {
                    let low = front.iter().map(|p| p[j]).fold(f64::INFINITY, f64::min);
                    let high = front.iter().map(|p| p[j]).fold(f64::NEG_INFINITY, f64::max);
                    assert!(
                        (low - ideal[j]).abs() <= 1e-6 * ideal[j].abs().max(1.0),
                        "{name}"
                    );
                    assert!(
                        (high - nadir[j]).abs() <= 1e-6 * nadir[j].abs().max(1.0),
                        "{name}"
                    );
                }
            }
            let real = problem.real();
            for _ in 0..1_000 {
                let genome = real.random_genome(&mut rng);
                let scores = problem.evaluate(&genome);
                let values = scores.values().expect("valid");
                assert!(values.iter().all(|v| v.is_finite()), "{name}: {genome:?}");
                assert!(scores.violation().is_finite());
                assert_eq!(problem.evaluate(&genome), scores);
                let constraints = problem.constraints(&genome);
                assert_eq!(constraints.len(), problem.constraint_count(), "{name}");
                assert_eq!(constraints.violation(0.0), scores.violation(), "{name}");
            }
        }
    }

    #[test]
    fn the_registries_describe_every_problem() {
        let two = all::<2>();
        assert_eq!(two.len(), 66);
        check_registry(two);
        let three = all::<3>();
        assert_eq!(
            three.iter().map(|p| p.name()).collect::<Vec<_>>(),
            [
                "VNT1",
                "VNT2",
                "VNT3",
                "CarSideImpact",
                "RocketInjector",
                "VehicleCrashworthiness",
                "DTLZ1",
                "DTLZ2",
                "DTLZ3",
                "DTLZ4",
                "DTLZ5",
                "DTLZ6",
                "DTLZ7",
                "Convex DTLZ2",
                "Scaled DTLZ1",
                "Scaled DTLZ2",
                "Inverted DTLZ1",
                "WFG1",
                "WFG2",
                "WFG3",
                "WFG4",
                "WFG5",
                "WFG6",
                "WFG7",
                "WFG8",
                "WFG9",
                "MW4",
                "MW8",
                "MW14",
                "C1-DTLZ1",
                "C1-DTLZ3",
                "C2-DTLZ2",
                "convex C2-DTLZ2",
                "C3-DTLZ1",
                "C3-DTLZ4",
            ]
        );
        check_registry(three);
        let five = all::<5>();
        assert_eq!(five[0].name(), "WaterResourcePlanning");
        check_registry(five);
    }
}
