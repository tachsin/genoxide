//! DAS-CMOP1-DAS-CMOP9: Fan et al.'s difficulty-adjustable constrained problems, two objectives
//! for DAS-CMOP1-6 and three for DAS-CMOP7-9, each with a [`Difficulty`] triplet.
//!
//! Fan, Z., Li, W., Cai, X., Li, H., Wei, C., Zhang, Q., Deb, K. and Goodman, E. (2020).
//! Difficulty adjustable and scalable constrained multiobjective test problem toolkit.
//! *Evolutionary Computation* 28(3): 339-378, sections 4-6 (eqs. 3-7, tables 2 and 3, figure 6),
//! read in arXiv:1612.07603v3 (the journal's text wasn't compared; the arXiv's first version had
//! other problems).
//!
//! Every problem is `fᵢ = αᵢ(x_I) + g(x_II)`: shape functions α of the position variables (x₁ for
//! two objectives, x₁ and x₂ for three) give the unconstrained front, at g = 0, and a distance
//! function g ≥ 0 of the others moves a solution away from it along the diagonal (1, …, 1). Three
//! kinds of constraints each set one kind of difficulty, and the triplet (η, ζ, γ) in [0, 1]³ its
//! level:
//!
//! - **type I, diversity (η):** `sin(aπx₁) − b ≥ 0` (and `cos(aπx₂) − b ≥ 0` for three
//!   objectives), with a = 20 and `b = 2η − 1`: the front in 10 or more pieces, narrower as η
//!   grows; none at η = 0;
//! - **type II, feasibility (ζ):** `(e − g)(g − d) ≥ 0`, with d = 0.5 and `e = d − ln ζ`: g, the
//!   distance from the unconstrained front, must be between d and e, a band of feasible solutions
//!   that narrows as ζ grows; at ζ = 0 (where the paper sets d = 0 and e is infinite) any g, and
//!   at ζ = 1 (e = d) exactly g = 0.5, an equality;
//! - **type III, convergence (γ):** infeasible regions in the objective space, of size `r = γ/2`,
//!   that block the way to the front: nine ellipses for two objectives, `((f₁ − p_k) cos θ −
//!   (f₂ − q_k) sin θ)²/a_k² + ((f₁ − p_k) sin θ + (f₂ − q_k) cos θ)²/b_k² ≥ r` with
//!   `a_k² = 0.3`, `b_k² = 1.2`, θ = −π/4 and centers (p_k, q_k) along the anti-diagonals,
//!   and four spheres for three objectives, `Σⱼ (fⱼ − c_kⱼ)² ≥ r²` with centers at the unit
//!   vectors and at (1, 1, 1)/√3.
//!
//! [`Difficulty::STANDARD`] has the paper's sixteen triplets (table 3).
//!
//! **What was checked where.** The formulas, constants and triplets are the paper's table 2 and
//! 3, compared with the authors' Java code (the jMetal classes `DASCMOP1` to `DASCMOP9` in their
//! "Java source codes of MOEA/D-CDP, NSGA-II-CDP, C-MOEA/DD and C-NSGA-III", linked from the
//! paper and hosted by the first author's laboratory at
//! <http://imagelab.stu.edu.cn/Content.aspx?type=content&Content_ID=1310>; read only, to compare:
//! it has no license). They agree but for these points, where genoxide follows the code:
//!
//! - table 2 sums DAS-CMOP1-3's g from j = 1, which would add `(x₁ − sin(0.5πx₁))²`, a term of
//!   the position variable; the code sums from j = 2, as the paper's own examples (eqs. 3-5) do;
//! - table 2's `a_k² = 0.3, b_k² = 1.2` are squares, which the code divides by (eq. 5's example
//!   has 0.4 and 1.6): the ellipses' semi-axes are √(0.3 r) and √(1.2 r);
//! - at ζ = 1 the code accepts `|g − 0.5| ≤ 10⁻⁴`, where the paper's product is feasible only at
//!   g = 0.5 exactly; genoxide does the same, and [`constraints`](MultiProblem::constraints)
//!   gives `|g − 0.5| − 10⁻⁴` there. At ζ = 0 it gives `−g`, always feasible, the limit of the
//!   product divided by e as e grows (the code sets e = 10³⁰).
//!
//! [`constraints`](MultiProblem::constraints) gives the constraints in the paper's order as
//! `g(x) <= 0`: `b − sin(aπx₁)` (then `b − cos(aπx₂)`), the type II constraint `−(e − g)(g − d)`,
//! then `r − (the ellipse's form)` for each of the nine ellipses or `r² − Σⱼ (fⱼ − c_kⱼ)²` for
//! each of the four spheres.
//!
//! **The fronts.** The paper samples the unconstrained front and the constraints' boundaries and
//! filters them (its figure 6, and the files the authors published, 1,000 points for two
//! objectives and 10,000 for three); genoxide derives them the same way, from the definition.
//! Each value of the position variables gives a ray in the objective space, `α + g (1, …, 1)`,
//! and going further along a ray only makes a solution worse: the front is made of the first
//! feasible point of each ray where the type I constraints hold, the least g in the type II band
//! outside the ellipses or spheres (found exactly: along a ray each is a quadratic in g), then
//! non-dominated. That's 20,000 rays evenly spread over the feasible x₁, with the ends of its
//! intervals, for two objectives, and the 41,905 rays of Das and Dennis's points with 288
//! divisions, mapped to x₁ and x₂, for three; for three objectives `optimal_front` takes the rays
//! of fewer divisions, a divisor of 288, and the points of the dense front where each objective
//! is least and largest, the ideal and nadir points. The fronts depend on the triplet, not on the
//! number of variables, as long as g reaches the values they need (under 1 for the paper's
//! triplets; the default 30 variables reach far more).
//!
//! Compared with the fronts the authors published (`pf_data.zip`, the 16 triplets of each problem;
//! compared only), the fronts here agree to within their sampling: the mean distance from each
//! of their points to the nearest point here is under 0.01 for 130 of the 144 files, and from
//! each point here to the nearest of theirs under 0.01 for 140. The rest are points of theirs
//! that break the constraints as the paper and the code write them, and points here that they
//! leave out: on DAS-CMOP2 and DAS-CMOP5 with the triplets 12, 15 and 16, their points with g
//! other than 0.5 under ζ = 1, or with x₁ outside the type I intervals; on DAS-CMOP3 and
//! DAS-CMOP6 with η = 0.5 (triplets 5, 8, 14 and 16), the isolated point x₁ = 1, feasible only in
//! exact arithmetic (`sin 20π` is −2.4 × 10⁻¹⁵ in floating point); and on DAS-CMOP8 and
//! DAS-CMOP9 with the triplets 7 and 11 (γ alone), the parts of the front near the corners that
//! the type III spheres push out along the diagonal, which their files leave out (the unit
//! sphere's points there are infeasible, and these, further out, aren't dominated by any
//! feasible point).

use super::mw::spread_by_length;
use super::{MultiProblem, das_dennis, divisions_for, non_dominated};
use crate::constraint::at_most;
use crate::genome::{Real, Reals};
use crate::math;
use crate::multi::MultiFitnessFunction;
use crate::problems::Constraints;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::f64::consts::PI;
use std::sync::{Arc, Mutex, OnceLock};

const REFERENCE: &str = "Fan, Z., Li, W., Cai, X., Li, H., Wei, C., Zhang, Q., Deb, K. and \
                         Goodman, E. (2020). Difficulty adjustable and scalable constrained \
                         multiobjective test problem toolkit. Evolutionary Computation 28(3): \
                         339-378.";
const REFERENCE_URL: &str = "https://doi.org/10.1162/evco_a_00259";

// the paper's number of variables
const VARIABLES: usize = 30;

// the type I constraints' a, and the type II constraints' d
const A: f64 = 20.0;
const D: f64 = 0.5;

// the tolerance of the type II equality at ζ = 1, the authors' code's
const TOLERANCE: f64 = 1e-4;

// the type III ellipses: their centers, squared semi-axes and angle
const P: [f64; 9] = [0.0, 1.0, 0.0, 1.0, 2.0, 0.0, 1.0, 2.0, 3.0];
const Q: [f64; 9] = [1.5, 0.5, 2.5, 1.5, 0.5, 3.5, 2.5, 1.5, 0.5];
const A_SQUARED: f64 = 0.3;
const B_SQUARED: f64 = 1.2;

// the rays of a two-objective front, and the divisions of a three-objective one's
const RAYS: usize = 20_000;
const DIVISIONS: usize = 288;

/// A difficulty triplet (η, ζ, γ) of the DAS-CMOP problems, each in [0, 1]: the levels of
/// diversity, feasibility and convergence hardness that the type I, II and III constraints set
/// (see the [`DasCmop1`] docs).
///
/// ```
/// use genoxide::multi::problems::Difficulty;
///
/// let difficulty = Difficulty::standard(8);
/// assert_eq!(difficulty, Difficulty::new(0.5, 0.5, 0.5));
/// assert_eq!(Difficulty::STANDARD.len(), 16);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Difficulty {
    diversity: f64,
    feasibility: f64,
    convergence: f64,
}

impl Difficulty {
    /// The paper's sixteen triplets (table 3), numbered from 1 there: three of one kind of
    /// difficulty at 0.25, then at 0.5 and 0.75, each followed by the three at once
    /// (1-12), then four with an equality, ζ = 1 (13-16).
    pub const STANDARD: [Difficulty; 16] = [
        Self::of(0.25, 0.0, 0.0),
        Self::of(0.0, 0.25, 0.0),
        Self::of(0.0, 0.0, 0.25),
        Self::of(0.25, 0.25, 0.25),
        Self::of(0.5, 0.0, 0.0),
        Self::of(0.0, 0.5, 0.0),
        Self::of(0.0, 0.0, 0.5),
        Self::of(0.5, 0.5, 0.5),
        Self::of(0.75, 0.0, 0.0),
        Self::of(0.0, 0.75, 0.0),
        Self::of(0.0, 0.0, 0.75),
        Self::of(0.75, 0.75, 0.75),
        Self::of(0.0, 1.0, 0.0),
        Self::of(0.5, 1.0, 0.0),
        Self::of(0.0, 1.0, 0.5),
        Self::of(0.5, 1.0, 0.5),
    ];

    const fn of(diversity: f64, feasibility: f64, convergence: f64) -> Self {
        Self {
            diversity,
            feasibility,
            convergence,
        }
    }

    /// The triplet (η, ζ, γ): `diversity` η, `feasibility` ζ and `convergence` γ.
    ///
    /// # Panics
    ///
    /// If one isn't in [0, 1].
    pub fn new(diversity: f64, feasibility: f64, convergence: f64) -> Self {
        for (name, level) in [
            ("diversity", diversity),
            ("feasibility", feasibility),
            ("convergence", convergence),
        ] {
            assert!(
                (0.0..=1.0).contains(&level),
                "a difficulty's {name} is in [0, 1], not {level}"
            );
        }
        Self::of(diversity, feasibility, convergence)
    }

    /// The paper's triplet `number`, from 1 to 16 (table 3): [`STANDARD`](Self::STANDARD)'s
    /// `number − 1`.
    ///
    /// # Panics
    ///
    /// If `number` isn't from 1 to 16.
    pub fn standard(number: usize) -> Self {
        assert!(
            (1..=16).contains(&number),
            "the paper's difficulty triplets are numbered 1 to 16, not {number}"
        );
        Self::STANDARD[number - 1]
    }

    /// η, the diversity hardness: the type I constraints' `b = 2η − 1`.
    pub fn diversity(&self) -> f64 {
        self.diversity
    }

    /// ζ, the feasibility hardness: the type II constraint's `e = 0.5 − ln ζ`.
    pub fn feasibility(&self) -> f64 {
        self.feasibility
    }

    /// γ, the convergence hardness: the type III constraints' `r = γ/2`.
    pub fn convergence(&self) -> f64 {
        self.convergence
    }

    fn b(&self) -> f64 {
        2.0 * self.diversity - 1.0
    }

    fn r(&self) -> f64 {
        0.5 * self.convergence
    }

    // the type II constraint as g <= 0
    fn band(&self, g: f64) -> f64 {
        let zeta = self.feasibility;
        if zeta == 0.0 {
            -g
        } else if zeta == 1.0 {
            (g - D).abs() - TOLERANCE
        } else {
            let e = D - math::ln(zeta);
            -((e - g) * (g - D))
        }
    }

    // the least and largest g that the type II constraint allows, before rounding
    fn distances(&self) -> (f64, f64) {
        let zeta = self.feasibility;
        if zeta == 0.0 {
            (0.0, f64::INFINITY)
        } else if zeta == 1.0 {
            (D - TOLERANCE, D + TOLERANCE)
        } else {
            (D, D - math::ln(zeta))
        }
    }

    // a key for the cache of fronts
    fn bits(&self) -> [u64; 3] {
        [self.diversity, self.feasibility, self.convergence].map(f64::to_bits)
    }
}

// the total violation of constraints g(x) <= 0
fn violation(values: &[f64]) -> f64 {
    values.iter().map(|&g| at_most(g, 0.0)).sum()
}

fn next_up(x: f64) -> f64 {
    if x == 0.0 {
        f64::from_bits(1)
    } else {
        f64::from_bits(x.to_bits() + 1)
    }
}

// ---- the distance functions ----------------------------------------------------------------------

// DAS-CMOP1-3: Σⱼ₌₂ⁿ (xⱼ − sin(0.5πx₁))²
fn g_sine(x: &[f64]) -> f64 {
    let target = math::sin(0.5 * PI * x[0]);
    x[1..].iter().map(|xj| (xj - target) * (xj - target)).sum()
}

// DAS-CMOP4-8: the variables from `first` (0-based) on, (n − first) + Σ ((xⱼ − 0.5)² −
// cos(20π(xⱼ − 0.5)))
fn g_multimodal(x: &[f64], first: usize) -> f64 {
    let sum: f64 = x[first..]
        .iter()
        .map(|xj| (xj - 0.5) * (xj - 0.5) - math::cos(20.0 * PI * (xj - 0.5)))
        .sum();
    (x.len() - first) as f64 + sum
}

// DAS-CMOP9: Σⱼ₌₃ⁿ (xⱼ − cos(0.25 j π (x₁ + x₂)/n))², j 1-based
fn g_linked(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let sum = x[0] + x[1];
    x.iter()
        .enumerate()
        .skip(2)
        .map(|(i, &xj)| {
            let d = xj - math::cos(0.25 * PI * (i + 1) as f64 * sum / n);
            d * d
        })
        .sum()
}

// ---- the shapes ----------------------------------------------------------------------------------

fn concave(x1: f64) -> [f64; 2] {
    [x1, 1.0 - x1 * x1]
}

fn convex(x1: f64) -> [f64; 2] {
    [x1, 1.0 - x1.sqrt()]
}

fn discontinuous(x1: f64) -> [f64; 2] {
    [x1, 1.0 - x1.sqrt() + 0.5 * math::sin(5.0 * PI * x1).abs()]
}

fn simplex(x1: f64, x2: f64) -> [f64; 3] {
    [x1 * x2, x2 * (1.0 - x1), 1.0 - x2]
}

fn sphere(x1: f64, x2: f64) -> [f64; 3] {
    let (sin1, cos1) = (math::sin(0.5 * PI * x1), math::cos(0.5 * PI * x1));
    [
        cos1 * math::cos(0.5 * PI * x2),
        cos1 * math::sin(0.5 * PI * x2),
        sin1,
    ]
}

// ---- the constraints -----------------------------------------------------------------------------

// the type I constraint on x₁ as g <= 0
fn diversity_sine(difficulty: &Difficulty, x1: f64) -> f64 {
    difficulty.b() - math::sin(A * PI * x1)
}

// the type I constraint on x₂ as g <= 0
fn diversity_cosine(difficulty: &Difficulty, x2: f64) -> f64 {
    difficulty.b() - math::cos(A * PI * x2)
}

// the nine ellipses' forms at f, each feasible at r or more
fn ellipses(f: [f64; 2]) -> [f64; 9] {
    let (sin, cos) = (math::sin(-0.25 * PI), math::cos(-0.25 * PI));
    std::array::from_fn(|k| {
        let (u, v) = (f[0] - P[k], f[1] - Q[k]);
        let first = u * cos - v * sin;
        let second = u * sin + v * cos;
        first * first / A_SQUARED + second * second / B_SQUARED
    })
}

// the four spheres' centers
fn centers() -> [[f64; 3]; 4] {
    let middle = 1.0 / 3.0f64.sqrt();
    [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [middle, middle, middle],
    ]
}

// the constraints of a two-objective problem as g <= 0, from x₁, g and f
fn two_values(difficulty: &Difficulty, x1: f64, g: f64, f: [f64; 2]) -> [f64; 11] {
    let r = difficulty.r();
    let forms = ellipses(f);
    std::array::from_fn(|k| match k {
        0 => diversity_sine(difficulty, x1),
        1 => difficulty.band(g),
        _ => r - forms[k - 2],
    })
}

// the constraints of a three-objective problem as g <= 0, from x₁, x₂, g and f
fn three_values(difficulty: &Difficulty, x1: f64, x2: f64, g: f64, f: [f64; 3]) -> [f64; 7] {
    let r = difficulty.r();
    let centers = centers();
    std::array::from_fn(|k| match k {
        0 => diversity_sine(difficulty, x1),
        1 => diversity_cosine(difficulty, x2),
        2 => difficulty.band(g),
        _ => {
            let c = centers[k - 3];
            let squares: f64 = (0..3).map(|j| (f[j] - c[j]) * (f[j] - c[j])).sum();
            r * r - squares
        }
    })
}

// ---- the fronts ----------------------------------------------------------------------------------

// the intervals of g along the ray α + g (1, …, 1) inside an ellipse (two objectives) or a sphere
// (three), open: the boundary is feasible
fn blocked<const M: usize>(difficulty: &Difficulty, alpha: [f64; M]) -> Vec<(f64, f64)> {
    let r = difficulty.r();
    let mut intervals = Vec::new();
    if r == 0.0 {
        return intervals;
    }
    if M == 2 {
        // along the ray, the first rotated coordinate u grows by √2 g and the second is fixed:
        // inside where (u₀ + √2 g)² < a² (r − v²/b²)
        for k in 0..9 {
            let (du, dv) = (alpha[0] - P[k], alpha[1] - Q[k]);
            let u = (du + dv) / 2f64.sqrt();
            let v = (dv - du) / 2f64.sqrt();
            let squared = A_SQUARED * (r - v * v / B_SQUARED);
            if squared > 0.0 {
                let w = squared.sqrt();
                intervals.push(((-w - u) / 2f64.sqrt(), (w - u) / 2f64.sqrt()));
            }
        }
    } else {
        // |α − c + g 1|² < r²: 3g² + 2Sg + |α − c|² − r² < 0, S = Σ (αⱼ − cⱼ)
        for c in centers() {
            let w: [f64; M] = std::array::from_fn(|j| alpha[j] - c[j]);
            let sum: f64 = w.iter().sum();
            let squares: f64 = w.iter().map(|v| v * v).sum();
            let discriminant = sum * sum - 3.0 * (squares - r * r);
            if discriminant > 0.0 {
                let root = discriminant.sqrt();
                intervals.push(((-sum - root) / 3.0, (-sum + root) / 3.0));
            }
        }
    }
    intervals
}

// the first feasible g on the ray α + g (1, …, 1), if any: the least g of the type II band,
// moved past every ellipse or sphere it's in, then up by a few floating-point steps where
// rounding leaves a constraint just broken; `feasible` checks the constraints at g
fn first_feasible<const M: usize>(
    difficulty: &Difficulty,
    alpha: [f64; M],
    feasible: impl Fn(f64) -> bool,
) -> Option<f64> {
    let (low, high) = difficulty.distances();
    let intervals = blocked(difficulty, alpha);
    let mut g = low;
    loop {
        let inside = intervals.iter().find(|&&(start, end)| g > start && g < end);
        match inside {
            Some(&(_, end)) => g = end,
            None => break,
        }
    }
    if g > high + 1e-12 * high.abs().max(1.0) {
        return None;
    }
    for _ in 0..1_000 {
        if feasible(g) {
            return Some(g);
        }
        g = next_up(g);
    }
    None
}

// the intervals of x in [0, 1] where `allowed(x) <= 0`, closed, their ends the last allowed
// floating-point values (found by bisection between `steps` evenly spread values)
fn intervals(allowed: impl Fn(f64) -> f64, steps: usize) -> Vec<(f64, f64)> {
    let inside = |v: f64| allowed(v) <= 0.0;
    let boundary = |mut a: f64, mut b: f64| {
        let side = inside(a);
        loop {
            let middle = 0.5 * (a + b);
            if middle <= a.min(b) || middle >= a.max(b) {
                return a;
            }
            if inside(middle) == side {
                a = middle;
            } else {
                b = middle;
            }
        }
    };
    let mut found = Vec::new();
    let mut start = inside(0.0).then_some(0.0);
    let mut previous = 0.0;
    for i in 1..=steps {
        let v = i as f64 / steps as f64;
        match (start, inside(v)) {
            (Some(low), false) => {
                found.push((low, boundary(previous, v)));
                start = None;
            }
            (None, true) => start = Some(boundary(v, previous)),
            _ => {}
        }
        previous = v;
    }
    if let Some(low) = start {
        found.push((low, 1.0));
    }
    found
}

// the dense front of a two-objective problem with the shape `alpha`: the first feasible point of
// each ray, non-dominated, sorted by f₁
fn two_front(difficulty: &Difficulty, alpha: fn(f64) -> [f64; 2]) -> Vec<[f64; 2]> {
    let allowed = |x1: f64| diversity_sine(difficulty, x1);
    let mut rays: Vec<f64> = (0..=RAYS)
        .map(|i| i as f64 / RAYS as f64)
        .filter(|&x1| allowed(x1) <= 0.0)
        .collect();
    for (low, high) in intervals(allowed, RAYS) {
        rays.extend([low, high]);
    }
    let points = rays
        .into_iter()
        .filter_map(|x1| {
            let a = alpha(x1);
            let feasible = |g: f64| {
                let f = [a[0] + g, a[1] + g];
                two_values(difficulty, x1, g, f)
                    .iter()
                    .all(|&value| value <= 0.0)
            };
            first_feasible(difficulty, a, feasible).map(|g| [a[0] + g, a[1] + g])
        })
        .collect();
    non_dominated(points)
}

// the first feasible points of the rays of Das and Dennis's points with `divisions`, mapped to
// the position variables by `positions`
fn three_rays(
    difficulty: &Difficulty,
    alpha: fn(f64, f64) -> [f64; 3],
    positions: fn([f64; 3]) -> (f64, f64),
    divisions: usize,
) -> Vec<[f64; 3]> {
    das_dennis::<3>(divisions)
        .into_iter()
        .filter_map(|w| {
            let (x1, x2) = positions(w);
            if diversity_sine(difficulty, x1) > 0.0 || diversity_cosine(difficulty, x2) > 0.0 {
                return None;
            }
            let a = alpha(x1, x2);
            let feasible = |g: f64| {
                let f = a.map(|v| v + g);
                three_values(difficulty, x1, x2, g, f)
                    .iter()
                    .all(|&value| value <= 0.0)
            };
            first_feasible(difficulty, a, feasible).map(|g| a.map(|v| v + g))
        })
        .collect()
}

// DAS-CMOP7's position variables for the simplex point w: x₂ = 1 − w₃, x₁ = w₁/x₂
fn simplex_positions(w: [f64; 3]) -> (f64, f64) {
    let x2 = (1.0 - w[2]).clamp(0.0, 1.0);
    let x1 = if x2 > 0.0 {
        (w[0] / x2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (x1, x2)
}

// DAS-CMOP8's and DAS-CMOP9's position variables for the direction w, on the unit sphere
fn sphere_positions(w: [f64; 3]) -> (f64, f64) {
    let norm = w.iter().map(|v| v * v).sum::<f64>().sqrt();
    let s = w.map(|v| v / norm);
    let x1 = (math::asin(s[2].clamp(0.0, 1.0)) / (0.5 * PI)).clamp(0.0, 1.0);
    let x2 = if s[0] > 0.0 || s[1] > 0.0 {
        (math::atan2(s[1], s[0]) / (0.5 * PI)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (x1, x2)
}

// a total order on f64, for the staircase of `non_dominated_3`
#[derive(Clone, Copy, PartialEq)]
struct Ordered(f64);

impl Eq for Ordered {}

impl PartialOrd for Ordered {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ordered {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

// the points of three objectives that no other point dominates, each once, sorted: in order of
// f₁ (then f₂, f₃), each point is dominated by an earlier one iff the earlier ones' staircase of
// (f₂, f₃) has a point at or below it
fn non_dominated_3(mut points: Vec<[f64; 3]>) -> Vec<[f64; 3]> {
    points.sort_by(|a, b| {
        a[0].total_cmp(&b[0])
            .then(a[1].total_cmp(&b[1]))
            .then(a[2].total_cmp(&b[2]))
    });
    // f₂ → f₃, f₃ falling as f₂ rises
    let mut staircase: BTreeMap<Ordered, f64> = BTreeMap::new();
    let mut front = Vec::new();
    for p in points {
        let below = staircase.range(..=Ordered(p[1])).next_back();
        if below.is_some_and(|(_, &f3)| f3 <= p[2]) {
            continue;
        }
        let covered: Vec<Ordered> = staircase
            .range(Ordered(p[1])..)
            .take_while(|(_, f3)| **f3 >= p[2])
            .map(|(key, _)| *key)
            .collect();
        for key in covered {
            staircase.remove(&key);
        }
        staircase.insert(Ordered(p[1]), p[2]);
        front.push(p);
    }
    front
}

// the dense fronts, computed once per problem and triplet
type Fronts<const M: usize> = Mutex<HashMap<(u8, [u64; 3]), Arc<Vec<[f64; M]>>>>;

fn cached<const M: usize>(
    cache: &'static OnceLock<Fronts<M>>,
    problem: u8,
    difficulty: &Difficulty,
    compute: impl FnOnce() -> Vec<[f64; M]>,
) -> Arc<Vec<[f64; M]>> {
    let cache = cache.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (problem, difficulty.bits());
    if let Some(front) = cache.lock().expect("not poisoned").get(&key) {
        return Arc::clone(front);
    }
    let front = Arc::new(compute());
    let mut cache = cache.lock().expect("not poisoned");
    Arc::clone(cache.entry(key).or_insert(front))
}

// the least and largest of each objective on a front
fn corners<const M: usize>(front: &[[f64; M]]) -> ([f64; M], [f64; M]) {
    let low = std::array::from_fn(|j| front.iter().map(|f| f[j]).fold(f64::INFINITY, f64::min));
    let high =
        std::array::from_fn(|j| front.iter().map(|f| f[j]).fold(f64::NEG_INFINITY, f64::max));
    (low, high)
}

// at least `points` points of a three-objective front: the rays of the fewest divisions that
// divide 288 (or are multiples of it), so that they're among the dense front's, with the dense
// front's points where an objective is least or largest
fn three_front(
    dense: &[[f64; 3]],
    points: usize,
    rays: impl Fn(usize) -> Vec<[f64; 3]>,
) -> Vec<[f64; 3]> {
    if points == 0 || dense.is_empty() {
        return Vec::new();
    }
    let (low, high) = corners(dense);
    let extremes: Vec<[f64; 3]> = (0..3)
        .flat_map(|j| {
            let least = dense.iter().find(|f| f[j] == low[j]);
            let largest = dense.iter().find(|f| f[j] == high[j]);
            [least, largest]
        })
        .flatten()
        .copied()
        .collect();
    let least = divisions_for::<3>(points);
    let candidates = (1..=DIVISIONS)
        .filter(|d| DIVISIONS.is_multiple_of(*d))
        .chain((2..).map(|m| m * DIVISIONS));
    for divisions in candidates.filter(|&d| d >= least) {
        let mut front = rays(divisions);
        front.extend(extremes.iter().copied());
        let front = non_dominated_3(front);
        if front.len() >= points {
            return front;
        }
    }
    unreachable!("the candidates go on")
}

// ---- the problems --------------------------------------------------------------------------------

// the struct, sizes, fitness and metadata of a DAS-CMOP problem
macro_rules! das_cmop {
    (
        $(#[$doc:meta])*
        $name:ident, $label:literal, $number:literal, $m:literal, $count:literal,
        ($diversity:literal, $feasibility:literal, $convergence:literal)
    ) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name {
            variables: usize,
            difficulty: Difficulty,
        }

        impl $name {
            /// The number of constraints.
            pub const CONSTRAINTS: usize = $count;

            /// The problem with `variables` variables and the difficulty triplet `difficulty`.
            ///
            /// # Panics
            ///
            #[doc = concat!("With fewer than ", stringify!($m), " variables.")]
            pub fn new(variables: usize, difficulty: Difficulty) -> Self {
                assert!(
                    variables >= $m,
                    concat!($label, " needs at least ", stringify!($m), " variables")
                );
                Self {
                    variables,
                    difficulty,
                }
            }

            /// The problem with the paper's 30 variables and the difficulty triplet
            /// `difficulty`.
            pub fn with_difficulty(difficulty: Difficulty) -> Self {
                Self::new(VARIABLES, difficulty)
            }

            /// The number of variables.
            pub fn variables(&self) -> usize {
                self.variables
            }

            /// The difficulty triplet.
            pub fn difficulty(&self) -> Difficulty {
                self.difficulty
            }

            /// The triplet that the paper plots this problem's front with (figure 6).
            pub fn figure_difficulty() -> Difficulty {
                Difficulty::of($diversity, $feasibility, $convergence)
            }
        }

        impl Default for $name {
            #[doc = concat!("The problem with the paper's 30 variables and the triplet of its figure 6, (", stringify!($diversity), ", ", stringify!($feasibility), ", ", stringify!($convergence), ").")]
            fn default() -> Self {
                Self::with_difficulty(Self::figure_difficulty())
            }
        }

        impl MultiFitnessFunction<Reals, $m> for $name {
            type Output = ([f64; $m], f64);

            /// The objective values of `x` and its constraint violation, 0 when it's feasible.
            ///
            /// # Panics
            ///
            #[doc = concat!("If `x` has fewer than ", stringify!($m), " genes.")]
            fn evaluate(&self, x: &Reals) -> ([f64; $m], f64) {
                let (f, values) = self.parts(x);
                (f, violation(&values))
            }
        }

        impl MultiProblem<$m> for $name {
            type Representation = Real;

            fn name(&self) -> &'static str {
                $label
            }

            fn representation(&self) -> Real {
                Real::uniform(self.variables, 0.0..=1.0).expect("valid bounds")
            }

            fn reference(&self) -> &'static str {
                REFERENCE
            }

            fn reference_url(&self) -> Option<&'static str> {
                Some(REFERENCE_URL)
            }

            fn constraint_count(&self) -> usize {
                $count
            }

            fn constraints(&self, genome: &Reals) -> Constraints {
                Constraints::new(self.parts(genome).1.to_vec(), Vec::new())
            }

            fn optimal_front(&self, points: usize) -> Option<Vec<[f64; $m]>> {
                Some(self.front(points))
            }

            fn ideal_point(&self) -> Option<[f64; $m]> {
                Some(corners(&self.dense()).0)
            }

            fn nadir_point(&self) -> Option<[f64; $m]> {
                Some(corners(&self.dense()).1)
            }
        }
    };
}

// the objectives and constraints of a two-objective problem with the shape `$alpha` and the
// distance function `$g`, and its front
macro_rules! two_objectives {
    ($name:ident, $number:literal, $alpha:ident, |$x:ident| $g:expr) => {
        impl $name {
            fn parts(&self, $x: &Reals) -> ([f64; 2], [f64; 11]) {
                let g = $g;
                let a = $alpha($x[0]);
                let f = [a[0] + g, a[1] + g];
                (f, two_values(&self.difficulty, $x[0], g, f))
            }

            fn dense(&self) -> Arc<Vec<[f64; 2]>> {
                static CACHE: OnceLock<Fronts<2>> = OnceLock::new();
                cached(&CACHE, $number, &self.difficulty, || {
                    two_front(&self.difficulty, $alpha)
                })
            }

            fn front(&self, points: usize) -> Vec<[f64; 2]> {
                let dense = self.dense();
                if dense.is_empty() {
                    return Vec::new();
                }
                spread_by_length(&dense, points)
            }
        }
    };
}

// the same for three objectives, with the position variables of a direction `$positions`
macro_rules! three_objectives {
    ($name:ident, $number:literal, $alpha:ident, $positions:ident, |$x:ident| $g:expr) => {
        impl $name {
            fn parts(&self, $x: &Reals) -> ([f64; 3], [f64; 7]) {
                let g = $g;
                let a = $alpha($x[0], $x[1]);
                let f = a.map(|v| v + g);
                (f, three_values(&self.difficulty, $x[0], $x[1], g, f))
            }

            fn rays(&self, divisions: usize) -> Vec<[f64; 3]> {
                three_rays(&self.difficulty, $alpha, $positions, divisions)
            }

            fn dense(&self) -> Arc<Vec<[f64; 3]>> {
                static CACHE: OnceLock<Fronts<3>> = OnceLock::new();
                cached(&CACHE, $number, &self.difficulty, || {
                    non_dominated_3(self.rays(DIVISIONS))
                })
            }

            fn front(&self, points: usize) -> Vec<[f64; 3]> {
                three_front(&self.dense(), points, |divisions| self.rays(divisions))
            }
        }
    };
}

das_cmop!(
    /// DAS-CMOP1: `f₁ = x₁ + g`, `f₂ = 1 − x₁² + g`, `g = Σⱼ₌₂ⁿ (xⱼ − sin(0.5πx₁))²`, subject to
    /// 11 constraints: type I on x₁, type II on g and nine type III ellipses.
    ///
    /// n variables in [0, 1], 30 by default; a [`Difficulty`] triplet (η, ζ, γ), by default
    /// (0, 0.5, 0.5), the paper's figure 6. The unconstrained front is concave, `f₂ = 1 − f₁²` at
    /// g = 0. The constraints:
    ///
    /// - type I: `sin(20πx₁) ≥ 2η − 1`, which cuts x₁ into ten intervals and the front into as
    ///   many pieces, narrower as η grows (none at η = 0);
    /// - type II: `(e − g)(g − 0.5) ≥ 0` with `e = 0.5 − ln ζ`, which asks g between 0.5 and e,
    ///   moving the front out along the diagonal by 0.5 (any g at ζ = 0, and g = 0.5 within
    ///   10⁻⁴ at ζ = 1);
    /// - type III: the outside of nine ellipses rotated by 45°, centered at (0, 1.5), (1, 0.5),
    ///   (0, 2.5), (1, 1.5), (2, 0.5), (0, 3.5), (1, 2.5), (2, 1.5) and (3, 0.5), of squared
    ///   semi-axes 0.3 r and 1.2 r, r = γ/2, which block the way to the front, and, where they
    ///   reach it, put pieces of it on their boundaries.
    ///
    /// The front depends on the triplet: [`optimal_front`](MultiProblem::optimal_front) samples
    /// it (see the module docs, and [`DasCmop4`], [`DasCmop7`] for the other distance and
    /// shape functions). The distance function here is smooth and easy; the difficulty is all in
    /// the constraints.
    ///
    /// Fan, Z. et al. (2020). Difficulty adjustable and scalable constrained multiobjective
    /// test problem toolkit. *Evolutionary Computation* 28(3): 339-378, table 2.
    DasCmop1,
    "DAS-CMOP1",
    1,
    2,
    11,
    (0.0, 0.5, 0.5)
);

das_cmop!(
    /// DAS-CMOP2: `f₁ = x₁ + g`, `f₂ = 1 − √x₁ + g`, with DAS-CMOP1's g and constraints.
    ///
    /// n variables in [0, 1], 30 by default; a [`Difficulty`] triplet, by default (0, 0.5, 0.5),
    /// the paper's figure 6. The unconstrained front is convex, `f₂ = 1 − √f₁`; the constraints
    /// are [`DasCmop1`]'s.
    ///
    /// Fan, Z. et al. (2020). Difficulty adjustable and scalable constrained multiobjective
    /// test problem toolkit. *Evolutionary Computation* 28(3): 339-378, table 2.
    DasCmop2,
    "DAS-CMOP2",
    2,
    2,
    11,
    (0.0, 0.5, 0.5)
);

das_cmop!(
    /// DAS-CMOP3: `f₁ = x₁ + g`, `f₂ = 1 − √x₁ + 0.5 |sin(5πx₁)| + g`, with DAS-CMOP1's g and
    /// constraints.
    ///
    /// n variables in [0, 1], 30 by default; a [`Difficulty`] triplet, by default (0, 0.5, 0.5),
    /// the paper's figure 6. The unconstrained front is disconnected: the non-dominated parts of
    /// a convex curve with five bumps. The constraints are [`DasCmop1`]'s.
    ///
    /// Fan, Z. et al. (2020). Difficulty adjustable and scalable constrained multiobjective
    /// test problem toolkit. *Evolutionary Computation* 28(3): 339-378, table 2.
    DasCmop3,
    "DAS-CMOP3",
    3,
    2,
    11,
    (0.0, 0.5, 0.5)
);

das_cmop!(
    /// DAS-CMOP4: [`DasCmop1`]'s shape and constraints, with the multimodal distance function
    /// `g = (n − 1) + Σⱼ₌₂ⁿ ((xⱼ − 0.5)² − cos(20π(xⱼ − 0.5)))`.
    ///
    /// n variables in [0, 1], 30 by default; a [`Difficulty`] triplet, by default
    /// (0.5, 0.5, 0.5), the paper's figure 6. g is 0 with every distance variable at 0.5, and
    /// has 11ⁿ⁻¹ − 1 local minima, the cosine's (as DTLZ1's): an algorithm climbs down them, and
    /// the type II band (g between 0.5 and e) is one more trap among them. The front is
    /// DAS-CMOP1's for the same triplet.
    ///
    /// Fan, Z. et al. (2020). Difficulty adjustable and scalable constrained multiobjective
    /// test problem toolkit. *Evolutionary Computation* 28(3): 339-378, table 2.
    DasCmop4,
    "DAS-CMOP4",
    4,
    2,
    11,
    (0.5, 0.5, 0.5)
);

das_cmop!(
    /// DAS-CMOP5: [`DasCmop2`]'s shape, `f₂ = 1 − √x₁ + g`, with [`DasCmop4`]'s multimodal g and
    /// [`DasCmop1`]'s constraints.
    ///
    /// n variables in [0, 1], 30 by default; a [`Difficulty`] triplet, by default
    /// (0.5, 0.5, 0.5), the paper's figure 6. The front is DAS-CMOP2's for the same triplet.
    ///
    /// Fan, Z. et al. (2020). Difficulty adjustable and scalable constrained multiobjective
    /// test problem toolkit. *Evolutionary Computation* 28(3): 339-378, table 2.
    DasCmop5,
    "DAS-CMOP5",
    5,
    2,
    11,
    (0.5, 0.5, 0.5)
);

das_cmop!(
    /// DAS-CMOP6: [`DasCmop3`]'s shape, `f₂ = 1 − √x₁ + 0.5 |sin(5πx₁)| + g`, with
    /// [`DasCmop4`]'s multimodal g and [`DasCmop1`]'s constraints.
    ///
    /// n variables in [0, 1], 30 by default; a [`Difficulty`] triplet, by default
    /// (0.5, 0.5, 0.5), the paper's figure 6. The front is DAS-CMOP3's for the same triplet.
    ///
    /// Fan, Z. et al. (2020). Difficulty adjustable and scalable constrained multiobjective
    /// test problem toolkit. *Evolutionary Computation* 28(3): 339-378, table 2.
    DasCmop6,
    "DAS-CMOP6",
    6,
    2,
    11,
    (0.5, 0.5, 0.5)
);

das_cmop!(
    /// DAS-CMOP7: three objectives, `f₁ = x₁x₂ + g`, `f₂ = x₂(1 − x₁) + g`, `f₃ = 1 − x₂ + g`,
    /// `g = (n − 2) + Σⱼ₌₃ⁿ ((xⱼ − 0.5)² − cos(20π(xⱼ − 0.5)))`, subject to 7 constraints.
    ///
    /// n variables in [0, 1], 30 by default; a [`Difficulty`] triplet (η, ζ, γ), by default
    /// (0.5, 0.5, 0.5), the paper's figure 6. The unconstrained front is the simplex
    /// `f₁ + f₂ + f₃ = 1`. The constraints:
    ///
    /// - type I: `sin(20πx₁) ≥ 2η − 1` and `cos(20πx₂) ≥ 2η − 1`, which cut the front into a
    ///   grid of patches;
    /// - type II: `(e − g)(g − 0.5) ≥ 0`, as [`DasCmop1`]'s;
    /// - type III: the outside of four spheres of radius r = γ/2, centered at the unit vectors,
    ///   the simplex's corners, and at (1, 1, 1)/√3: `Σⱼ fⱼ² − fₖ² + (fₖ − 1)² ≥ r²` and
    ///   `Σⱼ (fⱼ − 1/√3)² ≥ r²`. Near the corners they push the front out along the diagonal.
    ///
    /// Fan, Z. et al. (2020). Difficulty adjustable and scalable constrained multiobjective
    /// test problem toolkit. *Evolutionary Computation* 28(3): 339-378, table 2.
    DasCmop7,
    "DAS-CMOP7",
    7,
    3,
    7,
    (0.5, 0.5, 0.5)
);

das_cmop!(
    /// DAS-CMOP8: three objectives on the sphere, `f₁ = cos(0.5πx₁) cos(0.5πx₂) + g`,
    /// `f₂ = cos(0.5πx₁) sin(0.5πx₂) + g`, `f₃ = sin(0.5πx₁) + g`, with [`DasCmop7`]'s g and
    /// constraints.
    ///
    /// n variables in [0, 1], 30 by default; a [`Difficulty`] triplet, by default
    /// (0.5, 0.5, 0.5), the paper's figure 6. The unconstrained front is the unit sphere's
    /// octant, whose middle, (1, 1, 1)/√3, is the fourth type III sphere's center: the front has a
    /// hole there.
    ///
    /// Fan, Z. et al. (2020). Difficulty adjustable and scalable constrained multiobjective
    /// test problem toolkit. *Evolutionary Computation* 28(3): 339-378, table 2.
    DasCmop8,
    "DAS-CMOP8",
    8,
    3,
    7,
    (0.5, 0.5, 0.5)
);

das_cmop!(
    /// DAS-CMOP9: [`DasCmop8`]'s shape and [`DasCmop7`]'s constraints, with the distance
    /// function `g = Σⱼ₌₃ⁿ (xⱼ − cos(0.25 j π (x₁ + x₂)/n))²`.
    ///
    /// n variables in [0, 1], 30 by default; a [`Difficulty`] triplet, by default
    /// (0.5, 0.5, 0.5), the paper's figure 6. g is 0 where each distance variable is a cosine of
    /// the position variables, different for each: the variables are linked, and no single value
    /// of xⱼ is right everywhere. The front is DAS-CMOP8's for the same triplet.
    ///
    /// Fan, Z. et al. (2020). Difficulty adjustable and scalable constrained multiobjective
    /// test problem toolkit. *Evolutionary Computation* 28(3): 339-378, table 2.
    DasCmop9,
    "DAS-CMOP9",
    9,
    3,
    7,
    (0.5, 0.5, 0.5)
);

two_objectives!(DasCmop1, 1, concave, |x| g_sine(x));
two_objectives!(DasCmop2, 2, convex, |x| g_sine(x));
two_objectives!(DasCmop3, 3, discontinuous, |x| g_sine(x));
two_objectives!(DasCmop4, 4, concave, |x| g_multimodal(x, 1));
two_objectives!(DasCmop5, 5, convex, |x| g_multimodal(x, 1));
two_objectives!(DasCmop6, 6, discontinuous, |x| g_multimodal(x, 1));
three_objectives!(DasCmop7, 7, simplex, simplex_positions, |x| g_multimodal(
    x, 2
));
three_objectives!(DasCmop8, 8, sphere, sphere_positions, |x| g_multimodal(
    x, 2
));
three_objectives!(DasCmop9, 9, sphere, sphere_positions, |x| g_linked(x));

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Objective::Minimize;
    use crate::StreamRng;
    use crate::genome::Representation;
    use crate::multi::{Scores, non_dominated_sort};

    fn at(values: &[f64]) -> Reals {
        Reals::from(values.to_vec())
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(expected) {
            assert!(
                (a - e).abs() <= 1e-12 * e.abs().max(1.0),
                "{actual:?} is not {expected:?}"
            );
        }
    }

    #[test]
    fn the_papers_triplets() {
        assert_eq!(Difficulty::standard(1), Difficulty::new(0.25, 0.0, 0.0));
        assert_eq!(Difficulty::standard(12), Difficulty::new(0.75, 0.75, 0.75));
        assert_eq!(Difficulty::standard(16), Difficulty::new(0.5, 1.0, 0.5));
        // 1-12: three of one kind at a level, then all three at it
        for level in 0..3 {
            let value = 0.25 * (level + 1) as f64;
            let four = &Difficulty::STANDARD[4 * level..4 * level + 4];
            assert_eq!(four[0], Difficulty::new(value, 0.0, 0.0));
            assert_eq!(four[1], Difficulty::new(0.0, value, 0.0));
            assert_eq!(four[2], Difficulty::new(0.0, 0.0, value));
            assert_eq!(four[3], Difficulty::new(value, value, value));
        }
        // 13-16: ζ = 1
        assert!(
            Difficulty::STANDARD[12..]
                .iter()
                .all(|d| d.feasibility() == 1.0)
        );
        assert_eq!(
            DasCmop1::default().difficulty(),
            Difficulty::new(0.0, 0.5, 0.5)
        );
        assert_eq!(DasCmop4::default().difficulty(), Difficulty::standard(8));
        assert_eq!(DasCmop9::default().variables(), 30);
    }

    #[test]
    #[should_panic(expected = "in [0, 1], not 1.5")]
    fn a_level_above_1_panics() {
        let _ = Difficulty::new(0.0, 1.5, 0.0);
    }

    // the formulas of the paper's table 2 and its code, at points chosen so that the values
    // follow by hand
    #[test]
    fn values_match_the_paper() {
        // DAS-CMOP1, (0, 0.5, 0.5): x₁ = 0.5, the distance variables at sin(π/4) but the last,
        // 0.5 away: g = 0.25, f = (0.75, 1); type I off (b = −1: −1 − sin 10π); the band
        // asks g in [0.5, 0.5 + ln 2], broken by −(e − 0.25)(0.25 − 0.5); the ellipse at
        // (1, 1.5): u = (−0.25 − 0.5)/√2, v = (−0.5 + 0.25)/√2, form 0.75²/(2 · 0.3) + 0.25²/
        // (2 · 1.2), r = 0.25 − that
        let problem = DasCmop1::default();
        let s = math::sin(0.25 * PI);
        let mut genes = vec![s; 30];
        genes[0] = 0.5;
        genes[29] = s - 0.5;
        let x = at(&genes);
        let (f, violation) = problem.evaluate(&x);
        assert_close(&f, &[0.75, 1.0]);
        let values = problem.constraints(&x).inequalities().to_vec();
        assert_eq!(values.len(), 11);
        assert_close(&values[..1], &[-1.0 - math::sin(10.0 * PI)]);
        let e = 0.5 + math::ln(2.0);
        assert_close(&values[1..2], &[(e - 0.25) * 0.25]);
        let form = 0.75 * 0.75 / 0.6 + 0.25 * 0.25 / 2.4;
        assert_close(&values[5..6], &[0.25 - form]);
        let expected: f64 = values.iter().map(|&v| v.max(0.0)).sum();
        assert_close(&[violation], &[expected]);

        // DAS-CMOP2 and DAS-CMOP3 at x₁ = 0.25 on the unconstrained front: f₂ = 0.5, and
        // 0.5 + 0.5 |sin(1.25π)|
        let mut genes = vec![math::sin(0.125 * PI); 30];
        genes[0] = 0.25;
        let x = at(&genes);
        assert_close(&DasCmop2::default().evaluate(&x).0, &[0.25, 0.5]);
        let bump = 0.5 * math::sin(1.25 * PI).abs();
        assert_close(&DasCmop3::default().evaluate(&x).0, &[0.25, 0.5 + bump]);

        // DAS-CMOP4: g = 0 at 0.5; a distance variable at 0.6 adds 0.01 − cos 2π + 1 = 0.01
        let mut genes = vec![0.5; 30];
        genes[0] = 0.25;
        assert_close(
            &DasCmop4::default().evaluate(&at(&genes)).0,
            &[0.25, 0.9375],
        );
        genes[7] = 0.6;
        assert_close(&DasCmop5::default().evaluate(&at(&genes)).0, &[0.26, 0.51]);

        // DAS-CMOP7: x₁ = 0.25, x₂ = 0.5, g = 0: (0.125, 0.375, 0.5), on the simplex; at
        // (0.5, 0.5, 0.5): type I sin 5π = 0 ≥ 0 and cos 10π = 1 ≥ 0, the band broken,
        // the middle sphere's distance² 3 (1/√3 − 0.5)²
        let problem = DasCmop7::default();
        let mut genes = vec![0.5; 30];
        genes[0] = 0.25;
        let x = at(&genes);
        assert_close(&problem.evaluate(&x).0, &[0.125, 0.375, 0.5]);
        let values = problem.constraints(&x).inequalities().to_vec();
        assert_eq!(values.len(), 7);
        assert_close(&values[..2], &[-math::sin(5.0 * PI), -math::cos(10.0 * PI)]);
        assert_close(&values[2..3], &[(0.5 + math::ln(2.0)) * 0.5]);
        let middle = 1.0 / 3.0f64.sqrt();
        let squares = (0.125 - middle).powi(2) + (0.375 - middle).powi(2) + (0.5 - middle).powi(2);
        assert_close(&values[6..], &[0.0625 - squares]);

        // DAS-CMOP8 and DAS-CMOP9 at x₁ = 1/3, x₂ = 1/2: (√3/4 · √2 · …) on the sphere
        let mut genes = vec![0.5; 30];
        genes[0] = 1.0 / 3.0;
        let x = at(&genes);
        let (f, _) = DasCmop8::default().evaluate(&x);
        let c = math::cos(PI / 6.0);
        assert_close(
            &f,
            &[c * math::cos(0.25 * PI), c * math::sin(0.25 * PI), 0.5],
        );
        // DAS-CMOP9's distance variables at their optimal values cos(0.25 j π (x₁ + x₂)/n)
        for (i, gene) in genes.iter_mut().enumerate().skip(2) {
            *gene = math::cos(0.25 * PI * (i + 1) as f64 * (1.0 / 3.0 + 0.5) / 30.0);
        }
        let (f9, _) = DasCmop9::default().evaluate(&at(&genes));
        assert_close(&f9, &f);
    }

    // the type II band at its levels: any g at ζ = 0, [0.5, e] between, 0.5 within 10⁻⁴ at 1
    #[test]
    fn the_band_of_distances() {
        let band = |zeta: f64, g: f64| Difficulty::new(0.0, zeta, 0.0).band(g);
        assert!(band(0.0, 0.0) <= 0.0 && band(0.0, 7.0) <= 0.0);
        let e = 0.5 - math::ln(0.25);
        assert!(band(0.25, 0.49) > 0.0 && band(0.25, 0.5) <= 0.0);
        assert!(band(0.25, e - 1e-9) <= 0.0 && band(0.25, e + 1e-9) > 0.0);
        assert!(band(1.0, 0.5) <= 0.0 && band(1.0, 0.50005) <= 0.0 && band(1.0, 0.49995) <= 0.0);
        assert!(band(1.0, 0.5002) > 0.0 && band(1.0, 0.4998) > 0.0);
    }

    // a genome of DAS-CMOP1 or DAS-CMOP4 at x₁ with the distance g, spread over the distance
    // variables (for DAS-CMOP4, within the cosine's central well)
    fn genome_1(x1: f64, g: f64) -> Reals {
        let target = math::sin(0.5 * PI * x1);
        let step = (g / 29.0).sqrt();
        let sign = if target + step <= 1.0 { 1.0 } else { -1.0 };
        let mut genes = vec![target + sign * step; 30];
        genes[0] = x1;
        Reals::from(genes)
    }

    // the fronts: feasible where they lie (the ray's point at its g), reached by genomes,
    // mutually non-dominated, between the ideal and nadir points that they reach
    fn check_two(problem: &impl MultiProblem<2, Representation = Real>, points: usize) {
        let front = problem.optimal_front(points).expect("known");
        assert_eq!(front.len(), points, "{}", problem.name());
        let scores: Vec<Scores<2>> = front.iter().map(|f| Scores::new(*f)).collect();
        assert_eq!(non_dominated_sort(&scores, &[Minimize; 2]).len(), 1);
        let (ideal, nadir) = (
            problem.ideal_point().unwrap(),
            problem.nadir_point().unwrap(),
        );
        assert_eq!(front[0][0], ideal[0]);
        assert_eq!(front[points - 1][1], ideal[1]);
        assert_eq!(front[points - 1][0], nadir[0]);
        assert_eq!(front[0][1], nadir[1]);
    }

    #[test]
    fn two_objective_fronts() {
        for difficulty in Difficulty::STANDARD {
            check_two(&DasCmop1::with_difficulty(difficulty), 50);
            check_two(&DasCmop2::with_difficulty(difficulty), 50);
            check_two(&DasCmop3::with_difficulty(difficulty), 50);
        }
        // the same fronts for the same shapes
        let difficulty = Difficulty::standard(8);
        assert_eq!(
            DasCmop1::with_difficulty(difficulty).optimal_front(100),
            DasCmop4::with_difficulty(difficulty).optimal_front(100)
        );
        // with no constraint, the unconstrained front: f₂ = 1 − f₁², f₁ in [0, 1]
        let problem = DasCmop1::with_difficulty(Difficulty::new(0.0, 0.0, 0.0));
        let front = problem.optimal_front(101).unwrap();
        assert!(
            front
                .iter()
                .all(|f| (f[1] - (1.0 - f[0] * f[0])).abs() < 1e-12)
        );
        assert_eq!((front[0], front[100]), ([0.0, 1.0], [1.0, 0.0]));
        // the type II band moves it by 0.5
        let problem = DasCmop1::with_difficulty(Difficulty::new(0.0, 0.5, 0.0));
        let front = problem.optimal_front(101).unwrap();
        assert!(front.iter().all(|f| {
            let x1 = f[0] - 0.5;
            (f[1] - 0.5 - (1.0 - x1 * x1)).abs() < 1e-12
        }));
        // DAS-CMOP1's default: the front reached by genomes with their g
        let problem = DasCmop1::default();
        for f in problem.optimal_front(200).unwrap() {
            // f₁ − f₂ = x₁ + x₁² − 1 gives x₁, and g = f₁ − x₁
            let x1 = (-1.0 + (1.0 + 4.0 * (f[0] - f[1] + 1.0)).sqrt()) / 2.0;
            let (found, violation) = problem.evaluate(&genome_1(x1, f[0] - x1));
            assert!((found[0] - f[0]).abs() < 1e-9 && (found[1] - f[1]).abs() < 1e-9);
            assert!(violation < 1e-9, "{f:?}: {violation}");
        }
    }

    // no feasible point is better than the front: random x₁ and g, every feasible point weakly
    // dominated by a point of the front, up to its spacing, and dominating none
    #[test]
    fn random_points_agree_with_the_two_objective_fronts() {
        let mut rng = StreamRng::seed_from_u64(9);
        let unit = Real::uniform(2, 0.0..=1.0).expect("valid bounds");
        let shapes: [(fn(f64) -> [f64; 2], fn(Difficulty) -> Vec<[f64; 2]>); 3] = [
            (concave, |d| {
                DasCmop1::with_difficulty(d).optimal_front(4_000).unwrap()
            }),
            (convex, |d| {
                DasCmop2::with_difficulty(d).optimal_front(4_000).unwrap()
            }),
            (discontinuous, |d| {
                DasCmop3::with_difficulty(d).optimal_front(4_000).unwrap()
            }),
        ];
        for difficulty in [3, 4, 8, 12].map(Difficulty::standard) {
            for (alpha, front) in shapes {
                let front = front(difficulty);
                let mut feasible = 0;
                for _ in 0..4_000 {
                    let genome = unit.random_genome(&mut rng);
                    let (x1, g) = (genome[0], 0.4 + 2.0 * genome[1]);
                    let a = alpha(x1);
                    let f = [a[0] + g, a[1] + g];
                    if two_values(&difficulty, x1, g, f).iter().any(|&v| v > 0.0) {
                        continue;
                    }
                    feasible += 1;
                    assert!(
                        front
                            .iter()
                            .any(|q| q[0] <= f[0] + 0.02 && q[1] <= f[1] + 0.02),
                        "{difficulty:?}: {f:?}"
                    );
                    for q in &front {
                        assert!(!(f[0] < q[0] - 1e-9 && f[1] < q[1] - 1e-9), "{f:?} {q:?}");
                    }
                }
                assert!(feasible > 20, "{difficulty:?}: {feasible}");
            }
        }
    }

    #[test]
    fn three_objective_fronts() {
        for number in [1, 3, 4, 8, 11, 12, 16] {
            let difficulty = Difficulty::standard(number);
            for problem in [
                &DasCmop7::with_difficulty(difficulty) as &dyn Check,
                &DasCmop8::with_difficulty(difficulty),
            ] {
                problem.check(91);
            }
        }
        // with no constraint: the simplex and the sphere
        let none = Difficulty::new(0.0, 0.0, 0.0);
        let front = DasCmop7::with_difficulty(none).optimal_front(91).unwrap();
        assert!(
            front
                .iter()
                .all(|f| (f.iter().sum::<f64>() - 1.0).abs() < 1e-12)
        );
        let front = DasCmop9::with_difficulty(none).optimal_front(91).unwrap();
        assert!(
            front
                .iter()
                .all(|f| (f.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12)
        );
        // DAS-CMOP8's and DAS-CMOP9's are the same
        let difficulty = Difficulty::standard(8);
        assert_eq!(
            DasCmop8::with_difficulty(difficulty).optimal_front(200),
            DasCmop9::with_difficulty(difficulty).optimal_front(200)
        );
        // the type III spheres push the corners out: at γ = 0.5, f₁ reaches past 1 near (1, 0, 0)
        let problem = DasCmop8::with_difficulty(Difficulty::new(0.0, 0.0, 0.5));
        let nadir = problem.nadir_point().unwrap();
        assert!(nadir[0] > 1.1, "{nadir:?}");
    }

    trait Check {
        fn check(&self, points: usize);
    }

    impl<P: MultiProblem<3, Representation = Real>> Check for P {
        fn check(&self, points: usize) {
            let front = self.optimal_front(points).expect("known");
            assert!(front.len() >= points, "{}", self.name());
            let scores: Vec<Scores<3>> = front.iter().map(|f| Scores::new(*f)).collect();
            assert_eq!(non_dominated_sort(&scores, &[Minimize; 3]).len(), 1);
            let (ideal, nadir) = (self.ideal_point().unwrap(), self.nadir_point().unwrap());
            for j in 0..3 {
                assert!(front.iter().any(|f| f[j] == ideal[j]), "{}", self.name());
                assert!(front.iter().any(|f| f[j] == nadir[j]), "{}", self.name());
            }
        }
    }

    // random points for three objectives, as for two: random x₁, x₂ and g
    #[test]
    fn random_points_agree_with_the_three_objective_fronts() {
        let mut rng = StreamRng::seed_from_u64(10);
        let unit = Real::uniform(3, 0.0..=1.0).expect("valid bounds");
        for number in [3, 8, 11] {
            let difficulty = Difficulty::standard(number);
            for (alpha, front) in [
                (
                    simplex as fn(f64, f64) -> [f64; 3],
                    DasCmop7::with_difficulty(difficulty)
                        .optimal_front(3_000)
                        .unwrap(),
                ),
                (
                    sphere,
                    DasCmop8::with_difficulty(difficulty)
                        .optimal_front(3_000)
                        .unwrap(),
                ),
            ] {
                let mut feasible = 0;
                for _ in 0..3_000 {
                    let genome = unit.random_genome(&mut rng);
                    let (x1, x2, g) = (genome[0], genome[1], 1.5 * genome[2]);
                    let f = alpha(x1, x2).map(|v| v + g);
                    if three_values(&difficulty, x1, x2, g, f)
                        .iter()
                        .any(|&v| v > 0.0)
                    {
                        continue;
                    }
                    feasible += 1;
                    assert!(
                        front.iter().any(|q| (0..3).all(|j| q[j] <= f[j] + 0.05)),
                        "{number}: {f:?}"
                    );
                    for q in &front {
                        assert!(!(0..3).all(|j| f[j] < q[j] - 1e-9), "{f:?} {q:?}");
                    }
                }
                assert!(feasible > 20, "{number}: {feasible}");
            }
        }
    }

    #[test]
    fn three_objective_non_domination() {
        let points = vec![
            [1.0, 1.0, 1.0],
            [0.0, 2.0, 2.0],
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 2.0],
            [1.0, 2.0, 1.0],
            [2.0, 2.0, 0.0],
            [0.5, 3.0, 3.0],
        ];
        assert_eq!(
            non_dominated_3(points),
            [
                [0.0, 2.0, 2.0],
                [1.0, 1.0, 1.0],
                [2.0, 0.0, 2.0],
                [2.0, 2.0, 0.0]
            ]
        );
    }

    #[test]
    fn sizes() {
        assert_eq!(DasCmop1::default().constraint_count(), 11);
        assert_eq!(DasCmop9::default().constraint_count(), 7);
        assert_eq!(
            DasCmop7::new(3, Difficulty::standard(1))
                .representation()
                .bounds()
                .len(),
            3
        );
    }
}
