//! CTP1-CTP8: Deb, Pratap and Meyarivan's constrained test problems, two objectives and one or
//! two constraints each, whose constraints make the optimal front disconnected, a set of points,
//! or hidden behind infeasible bands.
//!
//! The definitions were checked in the authors' KanGAL report 200005 (October 2000), the preprint
//! of the EMO 2001 paper (the number in KanGAL's list of reports; the file's title page misprints
//! 200002, the number of Deb, Pratap and Moitra's report of the same year): CTP1 is its eq. 4
//! (p. 6) with the table of a and b on p. 6, and CTP2 to CTP7 its eq. 5 (p. 7) with the
//! parameters on pp. 7-11. The report leaves the function g, the number of variables and their
//! bounds open (its experiments use "Rastrigin's function as the g functional" and five
//! variables, without the formula); genoxide takes them from the authors'
//! NSGA-II code (version 1.1.6, KanGAL), which defines every CTP problem with two variables,
//! `g = 1 + x₂`, x₁ in [0, 1] and x₂ in [0, 1] (CTP1-CTP5) or [0, 10] (CTP6-CTP8). The report's
//! eq. 5 prints `f₂ = g (1 − f₁/g)`; its figures 6-11 draw the unconstrained front as the curve
//! `f₂ = 1 − √f₁`, and the authors' code computes `f₂ = g (1 − √(f₁/g))`, which genoxide uses.
//! CTP8 isn't in the report: it's credited to Deb's 2001 book (not read), and its definition
//! here is the authors' code's, CTP6's constraint with a second one like CTP7's (b = 2).

use super::{MultiProblem, Piece, evenly, non_dominated, pieces_front};
use crate::constraint::at_most;
use crate::genome::{Real, Reals};
use crate::math;
use crate::multi::MultiFitnessFunction;
use crate::problems::Constraints;
use std::f64::consts::PI;
use std::sync::OnceLock;

const REFERENCE: &str = "Deb, K., Pratap, A. and Meyarivan, T. (2001). Constrained test problems \
                         for multi-objective evolutionary optimization. Evolutionary \
                         Multi-Criterion Optimization (EMO 2001), LNCS 1993: 284-298.";
const REFERENCE_URL: &str = "https://doi.org/10.1007/3-540-44719-9_20";

const BOOK: &str = "Deb, K. (2001). Multi-Objective Optimization Using Evolutionary Algorithms. \
                    Wiley, Chichester.";

// bounds that are valid by construction: x₁ in [0, 1], x₂ in [0, `x2`]
fn bounds(x2: f64) -> Real {
    Real::new([0.0..=1.0, 0.0..=x2]).expect("valid bounds")
}

// the total violation of constraints g(x) <= 0
fn violation(values: &[f64]) -> f64 {
    values.iter().map(|&g| at_most(g, 0.0)).sum()
}

// ---- the constraint of CTP2-CTP8 -----------------------------------------------------------------

// the constraint of the report's eq. 5, `cos θ (f₂ − e) − sin θ f₁ ≥ a |sin(bπ (sin θ (f₂ − e) +
// cos θ f₁)^c)|^d`, with sin θ and cos θ computed once
#[derive(Clone, Copy, Debug)]
struct Wave {
    sin: f64,
    cos: f64,
    a: f64,
    b: f64,
    c: i32,
    d: Power,
    e: f64,
}

// the exponent d: a whole number takes a few multiplications, not the general `powf`
#[derive(Clone, Copy, Debug)]
enum Power {
    Integer(i32),
    Real(f64),
}

impl Wave {
    // sin θ and cos θ
    fn angle(&self) -> (f64, f64) {
        (self.sin, self.cos)
    }

    // the right-hand side at v, the coordinate along the line (f₂ − e) cos θ = f₁ sin θ
    fn height(&self, v: f64) -> f64 {
        let wave = math::sin(self.b * PI * math::powi(v, self.c)).abs();
        self.a
            * match self.d {
                Power::Integer(d) => math::powi(wave, d),
                Power::Real(d) => math::powf(wave, d),
            }
    }

    // the constraint as g <= 0: the right-hand side minus the left-hand side, u, the coordinate
    // across that line
    fn value(&self, f1: f64, f2: f64) -> f64 {
        let (sin, cos) = self.angle();
        let u = cos * (f2 - self.e) - sin * f1;
        let v = sin * (f2 - self.e) + cos * f1;
        self.height(v) - u
    }

    // the point of the constraint's boundary at v: (f₁, f₂) from (u, v), turned back by θ
    fn boundary(&self, v: f64) -> [f64; 2] {
        let (sin, cos) = self.angle();
        let u = self.height(v);
        [-sin * u + cos * v, self.e + cos * u + sin * v]
    }

    // the point on the line at v, where u = 0
    fn on_line(&self, v: f64) -> [f64; 2] {
        let (sin, cos) = self.angle();
        [cos * v, self.e + sin * v]
    }

    // the least and greatest v over the box f₁ in [0, 1], f₂ in [0, `top`]: v is linear in f₁
    // and f₂, so at its corners
    fn span(&self, top: f64) -> (f64, f64) {
        let (sin, cos) = self.angle();
        let corners = [[0.0, 0.0], [1.0, 0.0], [0.0, top], [1.0, top]];
        let vs = corners.map(|[f1, f2]| sin * (f2 - self.e) + cos * f1);
        let low = vs.iter().copied().fold(f64::INFINITY, f64::min);
        let high = vs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        (low, high)
    }

    // the points of the line where the right-hand side is 0, (bπ v^c) a multiple of π, with v
    // in [low, high]; c is 1 or 2
    fn zeros(&self, low: f64, high: f64) -> Vec<f64> {
        let mut zeros = Vec::new();
        if self.c == 1 {
            let (first, last) = ((low * self.b).ceil() as i64, (high * self.b).floor() as i64);
            zeros.extend((first..=last).map(|k| k as f64 / self.b));
        } else {
            let most = low.abs().max(high.abs());
            for k in 0..=(most * most * self.b).floor() as i64 {
                let v = (k as f64 / self.b).sqrt();
                zeros.push(v);
                if k > 0 {
                    zeros.push(-v);
                }
            }
            zeros.retain(|v| (low..=high).contains(v));
        }
        zeros
    }
}

// f₂ of CTP2-CTP8, g (1 − √(f₁/g)), as the authors' code computes it
fn square_root_front(f1: f64, g: f64) -> f64 {
    g * (1.0 - (f1 / g).sqrt())
}

// how far a sampled point of a boundary may be from feasible, in the units of the constraints:
// the rounding of the point's coordinates, and no more
const TOLERANCE: f64 = 1e-10;

// the samples of each curve that bounds the feasible region
const SAMPLES: usize = 200_000;

// fronts are split into pieces where two neighbours are farther apart than this: the gaps of the
// CTP fronts are 0.03 or more, and their samples less than 0.002 apart
const GAP: f64 = 0.01;

// the optimal front of a CTP2-CTP8 problem, dense and sorted by f₁: the non-dominated feasible
// points of the curves that bound the feasible region in the objective space (the boundaries of
// the constraints, where their right-hand side is 0, the unconstrained front g = 1, the largest g,
// and the sides f₁ = 0 and f₁ = 1), each sampled densely, with the ends of their feasible stretches
// found by bisection. Every non-dominated point of the region lies on one of them.
fn sampled_front(waves: &[Wave], top: f64) -> Vec<[f64; 2]> {
    // in the box of the objective space, and feasible for every constraint but `skip`
    let inside = |[f1, f2]: [f64; 2], skip: Option<usize>| {
        (0.0..=1.0).contains(&f1)
            && f2 >= square_root_front(f1, 1.0) - 1e-12
            && f2 <= square_root_front(f1, top) + 1e-12
            && waves
                .iter()
                .enumerate()
                .all(|(j, wave)| Some(j) == skip || wave.value(f1, f2) <= TOLERANCE)
    };
    let feasible = |point: [f64; 2]| inside(point, None);
    let mut curves: Vec<Box<dyn Fn(f64) -> [f64; 2] + '_>> = Vec::new();
    for wave in waves {
        let (low, high) = wave.span(top);
        curves.push(Box::new(move |t: f64| {
            wave.boundary(low + (high - low) * t)
        }));
    }
    curves.push(Box::new(|t: f64| [t, square_root_front(t, 1.0)]));
    curves.push(Box::new(move |t: f64| [t, square_root_front(t, top)]));
    curves.push(Box::new(move |t: f64| [0.0, 1.0 + (top - 1.0) * t]));
    let bottom = square_root_front(1.0, 1.0);
    let far = square_root_front(1.0, top);
    curves.push(Box::new(move |t: f64| [1.0, bottom + (far - bottom) * t]));

    let mut candidates = Vec::new();
    for curve in &curves {
        let mut last: Option<(f64, bool)> = None;
        for i in 0..SAMPLES {
            let t = evenly(i, SAMPLES);
            let point = curve(t);
            let inside = feasible(point);
            if inside {
                candidates.push(point);
            }
            // where the curve enters or leaves the feasible region, its last feasible point
            if let Some((previous, _)) = last.filter(|&(_, was_inside)| was_inside != inside) {
                let (mut good, mut bad) = if inside { (t, previous) } else { (previous, t) };
                for _ in 0..60 {
                    let middle = 0.5 * (good + bad);
                    if feasible(curve(middle)) {
                        good = middle;
                    } else {
                        bad = middle;
                    }
                }
                candidates.push(curve(good));
            }
            last = Some((t, inside));
        }
    }
    // the points where a constraint's right-hand side is 0, on its boundary exactly: in floating
    // point, sin(kπ) is about 1e-15, which the square root of CTP3-CTP5 makes 3e-8
    for (j, wave) in waves.iter().enumerate() {
        let (low, high) = wave.span(top);
        let zeros = wave.zeros(low, high).into_iter().map(|v| wave.on_line(v));
        candidates.extend(zeros.filter(|&point| inside(point, Some(j))));
    }
    non_dominated(candidates)
}

// `points` points of a dense front sorted by f₁, exactly: the front is split into pieces at its
// gaps, each piece gets a point (its start, or the last piece its end, so that both ends of the
// front are kept), and the rest go to the pieces in proportion to their lengths, spread evenly
// along each by length. With fewer points than pieces, pieces evenly chosen get one each.
pub(super) fn spread(front: &[[f64; 2]], points: usize, gap: f64) -> Vec<[f64; 2]> {
    if points == 0 || front.is_empty() {
        return Vec::new();
    }
    let step = |i: usize| {
        let (a, b) = (front[i - 1], front[i]);
        ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt()
    };
    // the pieces, as their first and last indices, and their lengths
    let mut pieces = vec![(0, 0)];
    let mut lengths = vec![0.0];
    for i in 1..front.len() {
        let length = step(i);
        if length > gap {
            pieces.push((i, i));
            lengths.push(0.0);
        } else {
            let last = pieces.len() - 1;
            pieces[last].1 = i;
            lengths[last] += length;
        }
    }
    let count = pieces.len();
    let mut counts = vec![0usize; count];
    if points < count {
        for i in 0..points {
            counts[(evenly(i, points) * (count - 1) as f64).round() as usize] += 1;
        }
    } else {
        counts.fill(1);
        let rest = points - count;
        let total: f64 = lengths.iter().sum();
        if total > 0.0 {
            let shares: Vec<f64> = lengths.iter().map(|l| rest as f64 * l / total).collect();
            for (count, share) in counts.iter_mut().zip(&shares) {
                *count += share.floor() as usize;
            }
            let mut order: Vec<usize> = (0..count).collect();
            order.sort_by(|&a, &b| {
                (shares[b] - shares[b].floor()).total_cmp(&(shares[a] - shares[a].floor()))
            });
            let assigned: usize = counts.iter().sum();
            for &piece in order.iter().take(points - assigned) {
                counts[piece] += 1;
            }
        } else {
            // points only: the rest to the points evenly chosen
            for i in 0..rest {
                counts[(evenly(i, rest) * (count - 1) as f64).round() as usize] += 1;
            }
        }
    }
    let mut spread = Vec::with_capacity(points);
    for (p, (&(start, end), &k)) in pieces.iter().zip(&counts).enumerate() {
        if k == 0 {
            continue;
        }
        if k == 1 || start == end {
            let index = if p == count - 1 { end } else { start };
            spread.extend(std::iter::repeat_n(front[index], k));
            continue;
        }
        // k points evenly spread by length, from the start to the end of the piece
        let length = lengths[p];
        let mut index = start;
        let mut walked = 0.0;
        for j in 0..k {
            let target = length * evenly(j, k);
            while index < end && walked + step(index + 1) <= target + 1e-15 * length {
                walked += step(index + 1);
                index += 1;
            }
            let next = if index < end {
                walked + step(index + 1)
            } else {
                walked
            };
            // the nearer of the two samples around the target
            let pick = if index < end && next - target < target - walked {
                index + 1
            } else {
                index
            };
            spread.push(front[if j == k - 1 { end } else { pick }]);
        }
    }
    spread
}

// the ideal and nadir points of a front sorted by f₁ (and so with f₂ falling)
fn extremes(front: &[[f64; 2]]) -> ([f64; 2], [f64; 2]) {
    let (first, last) = (front[0], front[front.len() - 1]);
    ([first[0], last[1]], [last[0], first[1]])
}

// ---- CTP1 ----------------------------------------------------------------------------------------

// CTP1's a and b, the table of the report (p. 6), which the authors' code uses too
const CTP1_A: [f64; 2] = [0.858, 0.728];
const CTP1_B: [f64; 2] = [0.541, 0.295];

/// CTP1: `f₁ = x₁`, `f₂ = g exp(−f₁/g)` with `g = 1 + x₂`, subject to
/// `f₂ − aⱼ exp(−bⱼ f₁) ≥ 0` for j = 1, 2, with a = (0.858, 0.728) and b = (0.541, 0.295).
///
/// Bounds [0, 1]². The constraints cut off the part of the unconstrained front
/// `f₂ = exp(−f₁)` beyond f₁ ≈ 0.334: the front, derived from the definition, is the largest of
/// the three curves `exp(−f₁)`, `0.858 exp(−0.541 f₁)` and `0.728 exp(−0.295 f₁)`, at g = 1 up to
/// f₁ = ln(0.858)/(−0.459) ≈ 0.33367, on the first constraint's boundary up to
/// f₁ = ln(0.858/0.728)/0.246 ≈ 0.66789, and on the second's to f₁ = 1: from (0, 1) to
/// (1, 0.728 e^−0.295), a front whose two thirds lie on constraint boundaries, as the report says.
///
/// [`constraints`](MultiProblem::constraints) gives `aⱼ exp(−bⱼ f₁) − f₂` for j = 1, 2.
///
/// Deb, K., Pratap, A. and Meyarivan, T. (2001). Constrained test problems for multi-objective
/// evolutionary optimization. *Evolutionary Multi-Criterion Optimization (EMO 2001)*, LNCS 1993:
/// 284-298, eq. 4 and its table of a and b, checked in the authors' KanGAL report 200005
/// (p. 6). The report builds a and b by a procedure for J constraints, and prints them to three
/// digits, the values the authors' code uses and genoxide too.
///
/// What was checked where, for all the CTP problems: the definitions in the authors' KanGAL
/// report 200005 (October 2000, the EMO paper's preprint, whose title page misprints 200002; the
/// published paper wasn't compared):
/// CTP1 is its eq. 4 with the table of a and b (p. 6), and CTP2-CTP7 its eq. 5 (p. 7) with the
/// parameters on pp. 7-11. The report leaves g, the number of variables and their bounds open
/// (its experiments use "Rastrigin's function as the g functional" and five variables, without
/// the formula); genoxide takes them from the authors' NSGA-II code (version 1.1.6, KanGAL),
/// which defines every CTP problem with two variables, `g = 1 + x₂`, x₁ in [0, 1] and x₂ in
/// [0, 1] (CTP1-CTP5) or [0, 10] (CTP6-CTP8). The report's eq. 5 prints `f₂ = g (1 − f₁/g)`;
/// its figures 6-11 draw the unconstrained front as the curve `f₂ = 1 − √f₁`, and the authors'
/// code computes `f₂ = g (1 − √(f₁/g))`, which CTP2-CTP8 use. The code writes each constraint
/// as a ratio, `left/right − 1 ≥ 0`; genoxide keeps the report's difference, which is feasible
/// at the same points and finite where the right-hand side is 0.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Ctp1;

impl Ctp1 {
    /// The number of constraints.
    pub const CONSTRAINTS: usize = 2;

    fn objectives(&self, x: &Reals) -> [f64; 2] {
        let (f1, g) = (x[0], 1.0 + x[1]);
        [f1, g * math::exp(-f1 / g)]
    }

    fn values(&self, x: &Reals) -> [f64; 2] {
        let [f1, f2] = self.objectives(x);
        [0, 1].map(|j| CTP1_A[j] * math::exp(-CTP1_B[j] * f1) - f2)
    }

    // the least feasible f₂ at f₁, at g = 1: the largest of the three curves
    fn lowest(f1: f64) -> f64 {
        let curve = |j: usize| CTP1_A[j] * math::exp(-CTP1_B[j] * f1);
        math::exp(-f1).max(curve(0)).max(curve(1))
    }

    // where the curves cross: exp(−f₁) and the first boundary, and the two boundaries
    fn corners() -> (f64, f64) {
        let first = math::ln(CTP1_A[0]) / (CTP1_B[0] - 1.0);
        let second = math::ln(CTP1_A[0] / CTP1_A[1]) / (CTP1_B[0] - CTP1_B[1]);
        (first, second)
    }
}

impl MultiFitnessFunction<Reals, 2> for Ctp1 {
    type Output = ([f64; 2], f64);

    /// The objective values of `x` and its constraint violation, 0 when it's feasible.
    ///
    /// # Panics
    ///
    /// If `x` has fewer than 2 genes.
    fn evaluate(&self, x: &Reals) -> ([f64; 2], f64) {
        (self.objectives(x), violation(&self.values(x)))
    }
}

impl MultiProblem<2> for Ctp1 {
    type Representation = Real;

    fn name(&self) -> &'static str {
        "CTP1"
    }

    fn representation(&self) -> Real {
        bounds(1.0)
    }

    fn reference(&self) -> &'static str {
        REFERENCE
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some(REFERENCE_URL)
    }

    fn constraint_count(&self) -> usize {
        Self::CONSTRAINTS
    }

    fn constraints(&self, genome: &Reals) -> Constraints {
        Constraints::new(self.values(genome).to_vec(), Vec::new())
    }

    fn optimal_front(&self, points: usize) -> Option<Vec<[f64; 2]>> {
        let (first, second) = Self::corners();
        let on = |from: f64, to: f64| {
            move |t: f64| {
                let f1 = from + (to - from) * t;
                [f1, Self::lowest(f1)]
            }
        };
        let (a, b, c) = (on(0.0, first), on(first, second), on(second, 1.0));
        let pieces = [
            Piece {
                curve: &a,
                with_end: false,
            },
            Piece {
                curve: &b,
                with_end: false,
            },
            Piece {
                curve: &c,
                with_end: true,
            },
        ];
        Some(pieces_front(&pieces, points))
    }

    fn ideal_point(&self) -> Option<[f64; 2]> {
        Some([0.0, Self::lowest(1.0)])
    }

    fn nadir_point(&self) -> Option<[f64; 2]> {
        Some([1.0, 1.0])
    }
}

// ---- CTP2-CTP8 -----------------------------------------------------------------------------------

macro_rules! ctp {
    (
        $(#[$doc:meta])*
        $name:ident, $label:literal, $reference:expr, x2 in [0, $top:literal],
        [$($wave:expr),+ $(,)?]
    ) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
        pub struct $name;

        impl $name {
            /// The number of constraints.
            pub const CONSTRAINTS: usize = [$(stringify!($wave)),+].len();

            // the constraints, built once
            fn waves() -> &'static [Wave; Self::CONSTRAINTS] {
                static WAVES: OnceLock<[Wave; $name::CONSTRAINTS]> = OnceLock::new();
                WAVES.get_or_init(|| [$($wave),+])
            }

            fn objectives(&self, x: &Reals) -> [f64; 2] {
                let (f1, g) = (x[0], 1.0 + x[1]);
                [f1, square_root_front(f1, g)]
            }

            fn values(&self, x: &Reals) -> [f64; Self::CONSTRAINTS] {
                let [f1, f2] = self.objectives(x);
                Self::waves().map(|wave| wave.value(f1, f2))
            }

            // the optimal front, dense, computed once
            fn front() -> &'static [[f64; 2]] {
                static FRONT: OnceLock<Vec<[f64; 2]>> = OnceLock::new();
                FRONT.get_or_init(|| sampled_front(Self::waves(), 1.0 + $top))
            }
        }

        impl MultiFitnessFunction<Reals, 2> for $name {
            type Output = ([f64; 2], f64);

            /// The objective values of `x` and its constraint violation, 0 when it's feasible.
            ///
            /// # Panics
            ///
            /// If `x` has fewer than 2 genes.
            fn evaluate(&self, x: &Reals) -> ([f64; 2], f64) {
                (self.objectives(x), violation(&self.values(x)))
            }
        }

        impl MultiProblem<2> for $name {
            type Representation = Real;

            fn name(&self) -> &'static str {
                $label
            }

            fn representation(&self) -> Real {
                bounds($top)
            }

            fn reference(&self) -> &'static str {
                $reference
            }

            fn reference_url(&self) -> Option<&'static str> {
                ($reference == REFERENCE).then_some(REFERENCE_URL)
            }

            fn constraint_count(&self) -> usize {
                Self::CONSTRAINTS
            }

            fn constraints(&self, genome: &Reals) -> Constraints {
                Constraints::new(self.values(genome).to_vec(), Vec::new())
            }

            fn optimal_front(&self, points: usize) -> Option<Vec<[f64; 2]>> {
                Some(spread(Self::front(), points, GAP))
            }

            fn ideal_point(&self) -> Option<[f64; 2]> {
                Some(extremes(Self::front()).0)
            }

            fn nadir_point(&self) -> Option<[f64; 2]> {
                Some(extremes(Self::front()).1)
            }
        }
    };
}

// the constraint with θ in multiples of π
fn wave(theta: f64, a: f64, b: f64, c: i32, d: f64, e: f64) -> Wave {
    let d = if d.fract() == 0.0 && d.abs() <= f64::from(i32::MAX) {
        Power::Integer(d as i32)
    } else {
        Power::Real(d)
    };
    Wave {
        sin: math::sin(theta * PI),
        cos: math::cos(theta * PI),
        a,
        b,
        c,
        d,
        e,
    }
}

ctp!(
    /// CTP2: `f₁ = x₁`, `f₂ = g (1 − √(f₁/g))` with `g = 1 + x₂`, subject to
    /// `cos θ (f₂ − e) − sin θ f₁ ≥ a |sin(bπ (sin θ (f₂ − e) + cos θ f₁)^c)|^d` with θ = −0.2π,
    /// a = 0.2, b = 10, c = 1, d = 6 and e = 1.
    ///
    /// Bounds [0, 1]². The constraint's boundary waves above the line `(f₂ − e) cos θ = f₁ sin θ`
    /// (the report's eq. 6), `f₂ = 1 − 0.7265 f₁`, and touches it where the sine is 0: the
    /// unconstrained front below the line is infeasible, and the front is 13 disconnected pieces
    /// of the boundary, each starting on the line, from (0, 1) to about (0.9845, 0.2872).
    /// [`optimal_front`](MultiProblem::optimal_front) samples the boundaries of the feasible
    /// region densely and keeps the feasible non-dominated points.
    ///
    /// [`constraints`](MultiProblem::constraints) gives
    /// `a |sin(bπ (sin θ (f₂ − e) + cos θ f₁)^c)|^d − (cos θ (f₂ − e) − sin θ f₁)`.
    ///
    /// Deb, K., Pratap, A. and Meyarivan, T. (2001). Constrained test problems for
    /// multi-objective evolutionary optimization. *Evolutionary Multi-Criterion Optimization
    /// (EMO 2001)*, LNCS 1993: 284-298, eq. 5 and the parameters that follow it, checked in the
    /// authors' KanGAL report 200005 (p. 7); f₂'s square root, g, the variables and their bounds
    /// as in the authors' code (see [`Ctp1`]'s notes).
    Ctp2, "CTP2", REFERENCE, x2 in [0, 1.0],
    [wave(-0.2, 0.2, 10.0, 1, 6.0, 1.0)]
);

ctp!(
    /// CTP3: [`Ctp2`] with a = 0.1 and d = 0.5, so that each piece of the front shrinks to a
    /// point.
    ///
    /// Bounds [0, 1]². The boundary rises steeply on both sides of each point where it touches
    /// the line `f₂ = 1 − 0.7265 f₁`, and the front is those 13 points: `(cos θ v, 1 + sin θ v)`
    /// for v = 0, 0.1, …, 1.2, from (0, 1) to (0.9708, 0.2947) (derived from the definition). At
    /// the points themselves the constraint is 0; in floating point, sin(kπ) is about 1e-15 and
    /// its square root 3e-8, so the exact points count as barely infeasible, and feasible
    /// solutions are found next to them, as close as they like.
    ///
    /// Deb, K., Pratap, A. and Meyarivan, T. (2001). Constrained test problems for
    /// multi-objective evolutionary optimization. *Evolutionary Multi-Criterion Optimization
    /// (EMO 2001)*, LNCS 1993: 284-298, eq. 5, checked in the authors' KanGAL report 200005
    /// (p. 8: d = 0.5 and a = 0.1, the rest as CTP2); f₂'s square root, g, the variables and
    /// their bounds as in the authors' code (see [`Ctp1`]'s notes).
    Ctp3, "CTP3", REFERENCE, x2 in [0, 1.0],
    [wave(-0.2, 0.1, 10.0, 1, 0.5, 1.0)]
);

ctp!(
    /// CTP4: [`Ctp3`] with a = 0.75, which starts the disconnected feasible regions far from the
    /// front: each of the 13 optimal points lies at the end of a long, narrow feasible tunnel.
    ///
    /// Bounds [0, 1]². The front is CTP3's 13 points (derived from the definition).
    ///
    /// Deb, K., Pratap, A. and Meyarivan, T. (2001). Constrained test problems for
    /// multi-objective evolutionary optimization. *Evolutionary Multi-Criterion Optimization
    /// (EMO 2001)*, LNCS 1993: 284-298, eq. 5, checked in the authors' KanGAL report 200005
    /// (p. 8: a = 0.75, the rest as CTP3); f₂'s square root, g, the variables and their bounds as
    /// in the authors' code (see [`Ctp1`]'s notes).
    Ctp4, "CTP4", REFERENCE, x2 in [0, 1.0],
    [wave(-0.2, 0.75, 10.0, 1, 0.5, 1.0)]
);

ctp!(
    /// CTP5: [`Ctp3`] with c = 2, which spaces the optimal points unevenly, closer together as
    /// f₁ grows.
    ///
    /// Bounds [0, 1]². The points where the boundary touches the line are at v = √(k/10) for
    /// k = 0, …, 15. Near v = 0, sin(bπv²)^0.5 grows only linearly in v, and the boundary leaves
    /// the line at a shallow slope: the first stretch of the boundary, from (0, 1) to
    /// f₁ ≈ 0.2558, is a continuous piece of the front. The front is that piece and the 15 other
    /// points, the last at about (0.9908, 0.2801) (derived from the definition, and by sampling).
    ///
    /// Deb, K., Pratap, A. and Meyarivan, T. (2001). Constrained test problems for
    /// multi-objective evolutionary optimization. *Evolutionary Multi-Criterion Optimization
    /// (EMO 2001)*, LNCS 1993: 284-298, eq. 5, checked in the authors' KanGAL report 200005
    /// (p. 9: c = 2, the rest as CTP3); f₂'s square root, g, the variables and their bounds as in
    /// the authors' code (see [`Ctp1`]'s notes).
    Ctp5, "CTP5", REFERENCE, x2 in [0, 1.0],
    [wave(-0.2, 0.1, 10.0, 2, 0.5, 1.0)]
);

ctp!(
    /// CTP6: the constraint of [`Ctp2`] with θ = 0.1π, a = 40, b = 0.5, c = 1, d = 2 and e = −2,
    /// which makes infeasible bands across the whole objective space, parallel to the front.
    ///
    /// Bounds x₁ in [0, 1], x₂ in [0, 10]. The unconstrained front is infeasible, and the front
    /// is a single piece of the boundary of the feasible band nearest to it, from about
    /// (0, 3.6958) to (1, 0.8813) (derived by sampling the boundaries of the feasible region). A
    /// population has to cross the infeasible bands above it to get there.
    ///
    /// Deb, K., Pratap, A. and Meyarivan, T. (2001). Constrained test problems for
    /// multi-objective evolutionary optimization. *Evolutionary Multi-Criterion Optimization
    /// (EMO 2001)*, LNCS 1993: 284-298, eq. 5, checked in the authors' KanGAL report 200005
    /// (p. 10); f₂'s square root, g, the variables and their bounds as in the authors' code (see
    /// [`Ctp1`]'s notes). The report says the front is where
    /// `1 ≤ (f₂ − e) sin θ + f₁ cos θ ≤ 2`; on the front found here, that coordinate runs from
    /// 1.76 to 1.84.
    Ctp6, "CTP6", REFERENCE, x2 in [0, 10.0],
    [wave(0.1, 40.0, 0.5, 1, 2.0, -2.0)]
);

ctp!(
    /// CTP7: the constraint of [`Ctp2`] with θ = −0.05π, a = 40, b = 5, c = 1, d = 6 and e = 0,
    /// which makes infeasible bands across the objective space, nearly perpendicular to the
    /// front.
    ///
    /// Bounds x₁ in [0, 1], x₂ in [0, 10]. The bands leave parts of the unconstrained front
    /// `f₂ = 1 − √f₁` feasible: the front is six disconnected pieces of it, the first from
    /// f₁ ≈ 0.0792 and the last ending at (1, 0), and the point (0, 1.0446), where the least
    /// feasible f₂ at f₁ = 0 lies on the boundary (derived by sampling the boundaries of the
    /// feasible region).
    ///
    /// Deb, K., Pratap, A. and Meyarivan, T. (2001). Constrained test problems for
    /// multi-objective evolutionary optimization. *Evolutionary Multi-Criterion Optimization
    /// (EMO 2001)*, LNCS 1993: 284-298, eq. 5, checked in the authors' KanGAL report 200005
    /// (pp. 10-11); f₂'s square root, g, the variables and their bounds as in the authors' code
    /// (see [`Ctp1`]'s notes).
    Ctp7, "CTP7", REFERENCE, x2 in [0, 10.0],
    [wave(-0.05, 40.0, 5.0, 1, 6.0, 0.0)]
);

ctp!(
    /// CTP8: the constraints of [`Ctp6`] and of [`Ctp7`] with b = 2, together: bands parallel to
    /// the front and bands across it.
    ///
    /// Bounds x₁ in [0, 1], x₂ in [0, 10]. The front is three disconnected pieces of CTP6's,
    /// from about (0, 3.6958) to (0.1341, 3.3139), (0.3264, 2.7696) to (0.4787, 2.3379) and
    /// (0.6824, 1.7656) to (0.8228, 1.3728) (derived by sampling the boundaries of the feasible
    /// region).
    ///
    /// CTP8 isn't in the EMO 2001 paper, which has CTP1-CTP7; later papers credit it to
    /// Deb, K. (2001). *Multi-Objective Optimization Using Evolutionary Algorithms.* Wiley,
    /// which wasn't read. The definition here is the one in the NSGA-II code of Deb's group
    /// (version 1.1.6, KanGAL), with CTP6's g, variables and bounds; not yet checked against the
    /// book ([#168](https://github.com/tachsin/genoxide/issues/168)).
    Ctp8, "CTP8", BOOK, x2 in [0, 10.0],
    [
        wave(0.1, 40.0, 0.5, 1, 2.0, -2.0),
        wave(-0.05, 40.0, 2.0, 1, 6.0, 0.0),
    ]
);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StreamRng;
    use crate::genome::Representation;
    use crate::multi::IntoScores;
    use crate::multi::problems::{DynMultiProblem, boxed};

    fn at(values: &[f64]) -> Reals {
        Reals::from(values.to_vec())
    }

    fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() <= tolerance * e.abs().max(1.0),
                "{actual:?} is not {expected:?}"
            );
        }
    }

    // the genome (x₁, x₂) with the objectives (f₁, f₂) of CTP2-CTP8: x₁ = f₁, and g, from
    // g − √(f₁ g) = f₂, is ((√f₁ + √(f₁ + 4f₂)) / 2)²
    fn genome_of([f1, f2]: [f64; 2]) -> Reals {
        let root = (f1.sqrt() + (f1 + 4.0 * f2).sqrt()) / 2.0;
        at(&[f1, root * root - 1.0])
    }

    fn problems() -> [Box<dyn DynMultiProblem<2>>; 8] {
        [
            boxed(Ctp1),
            boxed(Ctp2),
            boxed(Ctp3),
            boxed(Ctp4),
            boxed(Ctp5),
            boxed(Ctp6),
            boxed(Ctp7),
            boxed(Ctp8),
        ]
    }

    // the report's procedure for CTP1's a and b (p. 6), for J constraints: its table gives them
    // for J = 2, to three digits
    #[test]
    fn ctp1_parameters_follow_the_reports_procedure() {
        let constraints = 2;
        let delta = 1.0 / (constraints + 1) as f64;
        let (mut a, mut b, mut x) = (vec![1.0], vec![1.0], delta);
        for j in 0..constraints {
            let y = a[j] * math::exp(-b[j] * x);
            a.push((a[j] + y) / 2.0);
            b.push(-math::ln(y / a[j + 1]) / x);
            x += delta;
        }
        for j in 0..2 {
            assert!((a[j + 1] - CTP1_A[j]).abs() < 5e-4, "{a:?}");
            assert!((b[j + 1] - CTP1_B[j]).abs() < 5e-4, "{b:?}");
        }
        // the procedure puts the first boundary through the unconstrained front at f₁ = 1/3:
        // with the table's rounded values, they cross at 0.33367
        let (first, second) = Ctp1::corners();
        assert!((first - 0.333_67).abs() < 1e-5 && (second - 0.667_89).abs() < 1e-5);
    }

    #[test]
    fn ctp1_values_and_front() {
        // at the origin: f = (0, 1), and 0.858 − 1, 0.728 − 1
        assert_eq!(Ctp1.evaluate(&at(&[0.0, 0.0])), ([0.0, 1.0], 0.0));
        assert_close(
            Ctp1.constraints(&at(&[0.0, 0.0])).inequalities(),
            &[0.858 - 1.0, 0.728 - 1.0],
            1e-15,
        );
        // at (1, 0), the unconstrained front's end, both constraints break
        let e = math::exp(-1.0);
        let g = [0.858 * math::exp(-0.541) - e, 0.728 * math::exp(-0.295) - e];
        assert_close(Ctp1.constraints(&at(&[1.0, 0.0])).inequalities(), &g, 1e-15);
        assert_close(&[Ctp1.evaluate(&at(&[1.0, 0.0])).1], &[g[0] + g[1]], 1e-15);
        // at (0.5, 1): g = 2, f₂ = 2 exp(−1/4)
        assert_close(
            &Ctp1.evaluate(&at(&[0.5, 1.0])).0,
            &[0.5, 2.0 * math::exp(-0.25)],
            1e-15,
        );
        // the front's points are feasible, and on a boundary past the first corner
        let front = Ctp1.optimal_front(300).expect("known");
        let (first, _) = Ctp1::corners();
        for [f1, f2] in front {
            // g exp(−f₁/g) = f₂, solved for g by Newton's method from g = f₂ e^{f₁}
            let mut g = f2 * math::exp(f1);
            for _ in 0..50 {
                let value = g * math::exp(-f1 / g) - f2;
                let slope = math::exp(-f1 / g) * (1.0 + f1 / g);
                g -= value / slope;
            }
            assert!(g >= 1.0 - 1e-12, "{f1}");
            let values = Ctp1.values(&at(&[f1, (g - 1.0).max(0.0)]));
            assert!(values.iter().all(|&v| v <= 1e-12), "{f1}: {values:?}");
            if f1 > first + 1e-9 {
                assert!(values.iter().any(|v| v.abs() <= 1e-12), "{f1}");
            }
        }
        assert_eq!(Ctp1.representation().bounds()[1], 0.0..=1.0);
    }

    // the report's eq. 6: the optimal solutions of CTP2-CTP5 lie on the line
    // (f₂ − e) cos θ = f₁ sin θ, where the constraint's right-hand side is 0: v = k/b
    #[test]
    fn the_points_on_the_line_are_on_the_boundary() {
        let wave = Ctp3::waves()[0];
        let slope = math::sin(0.2 * PI) / math::cos(0.2 * PI);
        for k in 0..=12 {
            let point = wave.on_line(k as f64 / 10.0);
            // f₂ = 1 − tan(0.2π) f₁
            assert!((point[1] - (1.0 - slope * point[0])).abs() < 1e-15);
            let x = genome_of(point);
            let (f, _) = Ctp3.evaluate(&x);
            assert_close(&f, &point, 1e-13);
            // sin(kπ) is about k × 1e-15 in floating point, and CTP3's square root makes it
            // about 3e-8
            assert!(Ctp3.values(&x)[0] < 1e-7, "{k}");
            assert!(Ctp2.values(&x)[0] < 1e-12, "{k}");
            // a feasible solution next to the point, 1e-6 away across the line; straight above
            // it, the boundary's square root rises too fast
            let (sin, cos) = wave.angle();
            let [f1, f2] = point;
            let (f, violation) = Ctp3.evaluate(&genome_of([f1 - sin * 1e-6, f2 + cos * 1e-6]));
            assert_eq!(violation, 0.0, "{k}: {f:?}");
            let (_, violation) = Ctp3.evaluate(&genome_of([f1, f2 + 1e-6]));
            assert!(violation > 1e-4, "{k}");
        }
    }

    #[test]
    fn ctp2_to_ctp8_values() {
        // at the origin, f = (0, 1): on the line of CTP2-CTP5 at v = 0, feasible
        for problem in &problems()[1..5] {
            let scores = problem.evaluate(&at(&[0.0, 0.0]));
            assert_eq!(scores.values(), Some([0.0, 1.0]));
            assert_eq!(scores.violation(), 0.0);
        }
        // CTP6 at the origin, f = (0, 1): e = −2, so u = 3 cos 0.1π and v = 3 sin 0.1π
        let (sin, cos) = (math::sin(0.1 * PI), math::cos(0.1 * PI));
        let ctp6 = 40.0 * math::powi(math::sin(0.5 * PI * 3.0 * sin), 2) - 3.0 * cos;
        let values = Ctp6.constraints(&at(&[0.0, 0.0]));
        assert_close(values.inequalities(), &[ctp6], 1e-13);
        // CTP7 at the origin: e = 0, u = cos 0.05π and v = −sin 0.05π
        let (sin, cos) = (math::sin(-0.05 * PI), math::cos(-0.05 * PI));
        let ctp7 = 40.0 * math::powi(math::sin(5.0 * PI * sin), 6) - cos;
        let values = Ctp7.constraints(&at(&[0.0, 0.0]));
        assert_close(values.inequalities(), &[ctp7], 1e-13);
        // CTP8 has CTP6's constraint and CTP7's with b = 2
        let second = 40.0 * math::powi(math::sin(2.0 * PI * sin), 6) - cos;
        let values = Ctp8.constraints(&at(&[0.0, 0.0]));
        assert_close(values.inequalities(), &[ctp6, second], 1e-13);
        // f₂ = g (1 − √(f₁/g)): at (0.25, 3), g = 4 and f₂ = 4 (1 − 1/4)
        assert_eq!(Ctp6.evaluate(&at(&[0.25, 3.0])).0, [0.25, 3.0]);
        assert_eq!(Ctp6.representation().bounds()[1], 0.0..=10.0);
        assert_eq!(Ctp5.representation().bounds()[1], 0.0..=1.0);
        assert_eq!(
            (Ctp1::CONSTRAINTS, Ctp2::CONSTRAINTS, Ctp8::CONSTRAINTS),
            (2, 1, 2)
        );
    }

    // the pieces of a front, split where neighbours are more than GAP apart: how many, and how
    // many of them are single points
    fn pieces(front: &[[f64; 2]]) -> (usize, usize) {
        let (mut pieces, mut points, mut length) = (1, 0, 1);
        for pair in front.windows(2) {
            let step =
                ((pair[1][0] - pair[0][0]).powi(2) + (pair[1][1] - pair[0][1]).powi(2)).sqrt();
            if step > GAP {
                points += usize::from(length == 1);
                pieces += 1;
                length = 1;
            } else {
                length += 1;
            }
        }
        (pieces, points + usize::from(length == 1))
    }

    #[test]
    fn the_fronts_have_their_shapes() {
        // (pieces, of which single points), as the report's figures 6-11 draw them
        assert_eq!(pieces(Ctp2::front()), (13, 0));
        assert_eq!(pieces(Ctp3::front()), (13, 13));
        assert_eq!(pieces(Ctp4::front()), (13, 13));
        assert_eq!(pieces(Ctp5::front()), (16, 15));
        assert_eq!(pieces(Ctp6::front()), (1, 0));
        assert_eq!(pieces(Ctp7::front()), (7, 1));
        assert_eq!(pieces(Ctp8::front()), (3, 0));
        // CTP3's and CTP4's are the 13 points of the line, from (0, 1) to v = 1.2
        assert_eq!(Ctp3::front(), Ctp4::front());
        assert_eq!(Ctp3::front().len(), 13);
        assert_eq!(Ctp3::front()[12], Ctp3::waves()[0].on_line(1.2));
        // CTP6's lies where 1 ≤ (f₂ − e) sin θ + f₁ cos θ ≤ 2, as the report says: 1.76 to 1.84
        let (sin, cos) = Ctp6::waves()[0].angle();
        for [f1, f2] in Ctp6::front() {
            let v = sin * (f2 + 2.0) + cos * f1;
            assert!((1.76..1.845).contains(&v), "{v}");
        }
        // CTP7's pieces, but its point at f₁ = 0, lie on the unconstrained front
        for &[f1, f2] in &Ctp7::front()[1..] {
            assert!((f2 - (1.0 - f1.sqrt())).abs() < 1e-12, "{f1}");
        }
        // CTP8's lie on CTP6's
        let on_ctp6 = |p: &[f64; 2]| Ctp6::waves()[0].value(p[0], p[1]).abs() < 1e-9;
        assert!(Ctp8::front().iter().all(on_ctp6));
        // the ends
        let last = Ctp2::front()[Ctp2::front().len() - 1];
        assert_eq!(Ctp2.ideal_point(), Some([0.0, last[1]]));
        assert_close(&Ctp7.nadir_point().unwrap(), &[1.0, 1.044_620_6], 1e-7);
        assert_eq!(Ctp7.ideal_point(), Some([0.0, 0.0]));
    }

    // every point of each front is a feasible objective vector: its genome gives it back and
    // meets the constraints (the points where CTP3-CTP5's right-hand side is 0 to 1e-7)
    #[test]
    fn the_fronts_are_feasible() {
        for problem in &problems()[1..] {
            let top = *problem.real().bounds()[1].end();
            for point in problem.optimal_front(500).expect("known") {
                let x = genome_of(point);
                assert!((-1e-12..=top + 1e-12).contains(&x[1]), "{point:?}");
                let x = at(&[x[0], x[1].clamp(0.0, top)]);
                assert_close(&problem.evaluate(&x).values().unwrap(), &point, 1e-12);
                let values = problem.constraints(&x);
                let worst = values
                    .inequalities()
                    .iter()
                    .copied()
                    .fold(f64::MIN, f64::max);
                assert!(worst <= 1e-7, "{}: {point:?} {worst}", problem.name());
            }
        }
    }

    // no feasible genome dominates a point of the front, and none is below the front: 100,000
    // random genomes, half of them at g = 1, where the fronts mostly lie
    #[test]
    fn random_genomes_agree_with_the_fronts() {
        let mut rng = StreamRng::seed_from_u64(7);
        for problem in problems() {
            let front = problem.optimal_front(20_000).expect("known");
            let real = problem.real();
            for i in 0..100_000 {
                let mut genome = real.random_genome(&mut rng);
                if i % 2 == 0 {
                    genome = at(&[genome[0], 0.0]);
                }
                let scores = problem.evaluate(&genome).into_scores().expect("valid");
                if !scores.is_feasible() {
                    continue;
                }
                let [f1, f2] = scores.values().expect("valid");
                // the points of the front with f₁ at most f1: the last has the least f₂, and
                // if the next is on the same piece, the front runs between them
                let index = front.partition_point(|p| p[0] <= f1);
                assert!(index > 0, "{}: {f1}", problem.name());
                let below = front[index - 1];
                let least = match front.get(index) {
                    Some(next) if next[0] - below[0] < GAP && below[1] - next[1] < GAP => {
                        let t = (f1 - below[0]) / (next[0] - below[0]);
                        below[1] + t * (next[1] - below[1])
                    }
                    _ => below[1],
                };
                let name = problem.name();
                assert!(least <= f2 + 1e-7, "{name}: ({f1}, {f2}) {below:?} {least}");
            }
        }
    }

    #[test]
    fn spreading_a_front() {
        // a piece of length 1, one of length 2 and a point: one point each, the rest by length
        let front = [
            [0.0, 3.0],
            [0.6, 2.2],
            [1.2, 1.4],
            [2.0, 1.0],
            [3.2, 0.1],
            [4.0, -0.5],
            [6.0, -2.0],
        ];
        let spread_front = spread(&front, 7, 1.1);
        assert_eq!(spread_front.len(), 7);
        assert_eq!(spread_front[0], [0.0, 3.0]);
        assert_eq!(spread_front[6], [6.0, -2.0]);
        assert!(spread(&front, 0, 1.1).is_empty());
        assert_eq!(spread(&front, 2, 1.1), [[0.0, 3.0], [6.0, -2.0]]);
        // points only: repeated, evenly
        let points = [[0.0, 1.0], [1.0, 0.0]];
        assert_eq!(
            spread(&points, 3, 0.1),
            [[0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
        );
    }
}
