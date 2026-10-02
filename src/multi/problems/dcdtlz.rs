//! DC1-DTLZ1, DC1-DTLZ3, DC2-DTLZ1, DC2-DTLZ3, DC3-DTLZ1 and DC3-DTLZ3: Li, Chen, Fu and Yao's
//! DTLZ problems with constraints on the decision variables, any number of objectives.
//!
//! Li, K., Chen, R., Fu, G. and Yao, X. (2019). Two-archive evolutionary algorithm for
//! constrained multiobjective optimization. *IEEE Transactions on Evolutionary Computation*
//! 23(2): 303-315, which names the problems and runs them, and its supplementary document (section
//! 1.2, eqs. 12-16, figures 4-7 and table 2), which defines them; read in the authors' copies, the
//! paper as arXiv:1711.07907 and the supplement as hosted by the first author
//! (<https://colalab.ai/supplementary/ctaea-supp.pdf>).
//!
//! The objectives are C1-DTLZ1's and C1-DTLZ3's, DTLZ1's and DTLZ3's: genoxide's [`Dtlz1`] and
//! [`Dtlz3`]. The constraints act on the decision variables, where C-DTLZ's act on the
//! objectives: type 1 (DC1) on the first position variable, `cos(aπx₁) > b`, which cuts the front
//! into cones from the origin; type 2 (DC2) on the distance function g, `cos(aπg) > b` and
//! `e^(−g) > b`, which keep the front whole and make nearly all of the rest of the space
//! infeasible, with local optima of the violation along the way; type 3 (DC3) on every position
//! variable and on g, which cuts the front into patches and adds DC2's local optima.
//!
//! **What the paper leaves open, and how genoxide settles it.** The supplement gives a = 3 and
//! b = 0.5 for DC1 and no values for DC2 and DC3, and no number of variables. The authors' lab
//! publishes a C++ platform, EMOC (<https://github.com/COLA-Laboratory/EMOC>, read only, to
//! compare: it has no license), whose `dcdtlz.cpp` uses a = 3 throughout, b = 0.5 but for
//! DC2-DTLZ1's 0.9: genoxide takes those values, which [`with_parameters`](Dc1Dtlz1::with_parameters)
//! changes. The supplement writes DC3's position constraints for j = 1, …, m, which would include
//! the first distance variable, 0.5 on the front, where `cos(1.5π) = 0` breaks the constraint and
//! makes the whole front infeasible, against its text and figure 7; EMOC constrains the m − 1
//! position variables, and so does genoxide, which the supplement's table 2 confirms: DC3's
//! feasible share of random solutions, 3.70%, 0.41% and 0.02% for 3, 5 and 8 objectives, is 3⁻ᵐ,
//! a third for each of the m − 1 position constraints and for g's. EMOC computes DTLZ3's g with
//! the factor 10 in place of DTLZ3's 100 (in its C1-DTLZ3 too); genoxide keeps DTLZ3's, as the
//! supplement says that the objectives are C1-DTLZ3's. The number of variables is DTLZ's usual,
//! `M + 4` for DTLZ1 and `M + 9` for DTLZ3 (as C1-DTLZ1 and C1-DTLZ3 have), an assumption. The
//! supplement's table 2 gives DC1 a feasible share of about 10.1%, which is `arccos(b)/π` for
//! b = 0.95, where its text, its figure 4 and EMOC (and its sampled fronts) have b = 0.5, a share
//! of 1/3: genoxide follows the text. The supplement prints the constraints strict ("> b"); a
//! solution here is feasible on the boundary too.
//!
//! **The fronts**, derived here: g = 0 is feasible under every constraint (`cos 0 = 1` and
//! `e⁰ = 1` are above b < 1), and the position constraints don't involve g, so DTLZ's front
//! points are optimal wherever their position variables are feasible, and no other point is (one
//! with g > 0 is dominated by the same position variables at g = 0). DC2's front is DTLZ's, whole;
//! DC1's and DC3's are the parts of it with feasible position variables. Their
//! [`optimal_front`](MultiProblem::optimal_front) maps Das and Dennis's points to the front (the
//! plane `Σ fᵢ = 1/2` or the unit sphere), finds their position variables, keeps the feasible
//! ones, and adds the points where each variable is at the ends of its feasible values (its least,
//! 0, and its largest), where each objective is least and largest; for 2 objectives, it spreads
//! exactly the requested points by length along a dense sample of feasible x₁. Every point is
//! evaluated from a genome, the position variables with the distance variables at 0.5. They agree
//! with the sampled fronts that EMOC ships for 2 and 3 objectives (`pf_data/dc*dtlz`, compared
//! only): the mean distance from each point of either to the nearest of the other is at most
//! 0.011.

use super::dtlz::{rastrigin_g, simplex_points, spherical_front};
use super::mw::spread_by_length;
use super::{Dtlz1, Dtlz3, DynMultiProblem, MultiProblem, boxed, das_dennis, divisions_for};
use crate::constraint::at_most;
use crate::genome::{Real, Reals};
use crate::math;
use crate::multi::MultiFitnessFunction;
use crate::problems::Constraints;
use std::f64::consts::PI;

const REFERENCE: &str = "Li, K., Chen, R., Fu, G. and Yao, X. (2019). Two-archive evolutionary \
                         algorithm for constrained multiobjective optimization. IEEE Transactions \
                         on Evolutionary Computation 23(2): 303-315.";
const REFERENCE_URL: &str = "https://doi.org/10.1109/TEVC.2018.2855411";

// the samples of x₁ for a two-objective front, and of the feasible values of a variable
const SAMPLES: usize = 100_000;

// the total violation of constraints g(x) <= 0
fn violation(values: &[f64]) -> f64 {
    values.iter().map(|&g| at_most(g, 0.0)).sum()
}

// which of DTLZ1 and DTLZ3 a problem builds on
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    // DTLZ1: the front is the plane Σ f = 1/2
    Plane,
    // DTLZ3: the front is the unit sphere
    Sphere,
}

// which constraints: on x₁ (type 1), on g (type 2), or on every position variable and on g
// (type 3)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    One,
    Two,
    Three,
}

// what the six problems share: the base problem's shape, the constraints' type, the number of
// variables and the parameters a and b
#[derive(Clone, Copy, Debug, PartialEq)]
struct Core<const M: usize> {
    shape: Shape,
    kind: Kind,
    variables: usize,
    a: f64,
    b: f64,
}

impl<const M: usize> Core<M> {
    fn new(label: &str, shape: Shape, kind: Kind, variables: usize, a: f64, b: f64) -> Self {
        assert!(M >= 2, "{label} needs at least 2 objectives");
        assert!(
            variables >= M,
            "{label} needs at least as many variables as objectives"
        );
        assert!(a.is_finite() && a > 0.0, "{label} needs a finite a above 0");
        assert!(
            b.is_finite() && b > -1.0 && b < 1.0,
            "{label} needs a finite b between -1 and 1"
        );
        Self {
            shape,
            kind,
            variables,
            a,
            b,
        }
    }

    fn objectives(&self, x: &Reals) -> [f64; M] {
        match self.shape {
            Shape::Plane => Dtlz1::<M>::new(self.variables).evaluate(x),
            Shape::Sphere => Dtlz3::<M>::new(self.variables).evaluate(x),
        }
    }

    // the constraint on one value (a position variable or g): b − cos(aπv) <= 0
    fn cosine(&self, value: f64) -> f64 {
        self.b - math::cos(self.a * PI * value)
    }

    fn values(&self, x: &Reals) -> Vec<f64> {
        match self.kind {
            Kind::One => vec![self.cosine(x[0])],
            Kind::Two => {
                let g = rastrigin_g(&x[M - 1..]);
                vec![self.cosine(g), self.b - math::exp(-g)]
            }
            Kind::Three => {
                let g = rastrigin_g(&x[M - 1..]);
                let mut values: Vec<f64> = x[..M - 1].iter().map(|&xi| self.cosine(xi)).collect();
                values.push(self.cosine(g));
                values
            }
        }
    }

    fn count(&self) -> usize {
        match self.kind {
            Kind::One => 1,
            Kind::Two => 2,
            Kind::Three => M,
        }
    }

    fn evaluate(&self, x: &Reals) -> ([f64; M], f64) {
        (self.objectives(x), violation(&self.values(x)))
    }

    // whether the position variable at `index` (0-based) may take `value` on the front
    fn allowed(&self, index: usize, value: f64) -> bool {
        match self.kind {
            Kind::One => index > 0 || self.cosine(value) <= 0.0,
            Kind::Two => true,
            Kind::Three => self.cosine(value) <= 0.0,
        }
    }

    // the point of the front at the position variables `positions`: evaluated from the genome
    // with the distance variables at 0.5, where g = 0
    fn at(&self, positions: &[f64]) -> [f64; M] {
        let mut genes = positions.to_vec();
        genes.resize(self.variables, 0.5);
        self.objectives(&Reals::from(genes))
    }

    // the feasible values of a constrained position variable in [0, 1], as closed intervals whose
    // ends are the last feasible floating-point values, found by bisection
    fn intervals(&self) -> Vec<(f64, f64)> {
        let feasible = |v: f64| self.cosine(v) <= 0.0;
        // the boundary between a and b, where feasible(a) != feasible(b): the feasible side's
        // last value
        let boundary = |mut a: f64, mut b: f64| {
            let inside = feasible(a);
            loop {
                let middle = 0.5 * (a + b);
                if middle <= a.min(b) || middle >= a.max(b) {
                    return a;
                }
                if feasible(middle) == inside {
                    a = middle;
                } else {
                    b = middle;
                }
            }
        };
        let mut intervals = Vec::new();
        let mut start = feasible(0.0).then_some(0.0);
        let mut previous = 0.0;
        for i in 1..=SAMPLES {
            let v = i as f64 / SAMPLES as f64;
            match (start, feasible(v)) {
                (Some(low), false) => {
                    intervals.push((low, boundary(previous, v)));
                    start = None;
                }
                (None, true) => start = Some(boundary(v, previous)),
                _ => {}
            }
            previous = v;
        }
        if let Some(low) = start {
            intervals.push((low, 1.0));
        }
        intervals
    }

    // the position variables of the front point f: inverting DTLZ1's or DTLZ3's position
    // functions, with a variable whose value doesn't matter (a product before it is 0) at 0
    fn positions(&self, f: &[f64; M]) -> Vec<f64> {
        let mut positions = Vec::with_capacity(M - 1);
        match self.shape {
            Shape::Plane => {
                // f = w/2 with Σ w = 1: f_M = (1 − x₁)/2, f_{M−k+1} = x₁ … x_{k−1} (1 − x_k)/2
                let mut product = 1.0;
                for k in 0..M - 1 {
                    let share = 2.0 * f[M - 1 - k];
                    let x = if product > 0.0 {
                        (1.0 - share / product).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    positions.push(x);
                    product *= x;
                }
            }
            Shape::Sphere => {
                // f_M = sin θ₁, f_{M−k+1} = cos θ₁ … cos θ_{k−1} sin θ_k, θ = πx/2
                let mut product = 1.0;
                for k in 0..M - 1 {
                    let x = if product > 0.0 {
                        let ratio = (f[M - 1 - k] / product).clamp(0.0, 1.0);
                        (math::asin(ratio) / (PI / 2.0)).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    positions.push(x);
                    product *= math::cos(PI / 2.0 * x);
                }
            }
        }
        positions
    }

    // the least and largest value each position variable may take on the front
    fn ends(&self) -> Vec<[f64; 2]> {
        let constrained = match self.intervals().as_slice() {
            [] => [0.0, 0.0],
            intervals => [intervals[0].0, intervals[intervals.len() - 1].1],
        };
        (0..M - 1)
            .map(|index| match (self.kind, index) {
                (Kind::Two, _) | (Kind::One, 1..) => [0.0, 1.0],
                _ => constrained,
            })
            .collect()
    }

    // the front points with every position variable at one of its ends: the least and largest
    // value of each objective on the front are among them, each objective being a product of
    // factors monotonic in separate variables
    fn corners(&self) -> Vec<[f64; M]> {
        let ends = self.ends();
        (0..1usize << (M - 1))
            .map(|choice| {
                let positions: Vec<f64> = (0..M - 1).map(|i| ends[i][(choice >> i) & 1]).collect();
                self.at(&positions)
            })
            .collect()
    }

    fn ideal(&self) -> [f64; M] {
        if self.kind == Kind::Two {
            return [0.0; M];
        }
        let corners = self.corners();
        std::array::from_fn(|j| corners.iter().map(|f| f[j]).fold(f64::INFINITY, f64::min))
    }

    fn nadir(&self) -> [f64; M] {
        if self.kind == Kind::Two {
            return match self.shape {
                Shape::Plane => [0.5; M],
                Shape::Sphere => [1.0; M],
            };
        }
        let corners = self.corners();
        std::array::from_fn(|j| {
            corners
                .iter()
                .map(|f| f[j])
                .fold(f64::NEG_INFINITY, f64::max)
        })
    }

    fn front(&self, points: usize) -> Vec<[f64; M]> {
        if points == 0 {
            return Vec::new();
        }
        if self.kind == Kind::Two {
            return match self.shape {
                Shape::Plane => simplex_points::<M>(points)
                    .into_iter()
                    .map(|w| w.map(|v| v / 2.0))
                    .collect(),
                Shape::Sphere => spherical_front::<M>(points),
            };
        }
        if M == 2 {
            return self.two_objectives(points);
        }
        let corners = self.corners();
        let mut divisions = divisions_for::<M>(points);
        loop {
            let mut front: Vec<[f64; M]> = das_dennis::<M>(divisions)
                .into_iter()
                .filter_map(|w| {
                    let f = match self.shape {
                        Shape::Plane => w.map(|v| v / 2.0),
                        Shape::Sphere => {
                            let norm = w.iter().map(|v| v * v).sum::<f64>().sqrt();
                            w.map(|v| v / norm)
                        }
                    };
                    let positions = self.positions(&f);
                    let allowed = positions
                        .iter()
                        .enumerate()
                        .all(|(index, &x)| self.allowed(index, x));
                    allowed.then(|| self.at(&positions))
                })
                .collect();
            // the corners first, so that a point that rounding sets apart from one is left out
            front.splice(0..0, corners.iter().copied());
            let front = distinct(front);
            if front.len() >= points {
                return front;
            }
            divisions += 1;
        }
    }

    // exactly `points` points for 2 objectives, spread by length along a dense sample of the
    // feasible x₁, the ends of its intervals included
    fn two_objectives(&self, points: usize) -> Vec<[f64; M]> {
        let mut values: Vec<f64> = (0..=SAMPLES)
            .map(|i| i as f64 / SAMPLES as f64)
            .filter(|&x| self.allowed(0, x))
            .collect();
        for (low, high) in self.intervals() {
            values.extend([low, high]);
        }
        let mut dense: Vec<[f64; 2]> = values
            .into_iter()
            .map(|x| {
                let f = self.at(&[x]);
                [f[0], f[1]]
            })
            .collect();
        dense.sort_by(|p, q| p[0].total_cmp(&q[0]).then(q[1].total_cmp(&p[1])));
        dense.dedup();
        spread_by_length(&dense, points)
            .into_iter()
            .map(|p| std::array::from_fn(|j| p[j]))
            .collect()
    }
}

// the points without those within 1e-12 in every objective of an earlier one, in their order.
// Points of the plane or the sphere dominate each other only when rounding sets apart two that
// are the same, and then by less than this: the rest are mutually non-dominated
fn distinct<const M: usize>(points: Vec<[f64; M]>) -> Vec<[f64; M]> {
    let mut order: Vec<usize> = (0..points.len()).collect();
    order.sort_by(|&i, &k| points[i][0].total_cmp(&points[k][0]));
    let mut removed = vec![false; points.len()];
    for (place, &i) in order.iter().enumerate() {
        for &k in &order[place + 1..] {
            if points[k][0] - points[i][0] > 1e-12 {
                break;
            }
            if (0..M).all(|j| (points[k][j] - points[i][j]).abs() <= 1e-12) {
                removed[i.max(k)] = true;
            }
        }
    }
    let kept = points.into_iter().zip(removed);
    kept.filter_map(|(point, removed)| (!removed).then_some(point))
        .collect()
}

macro_rules! dc_dtlz {
    (
        $(#[$doc:meta])*
        $name:ident, $label:literal, $shape:expr, $kind:expr, $k:literal, $b:literal
    ) => {
        $(#[$doc])*
        #[derive(Clone, Copy, Debug, PartialEq)]
        pub struct $name<const M: usize> {
            core: Core<M>,
        }

        impl<const M: usize> $name<M> {
            /// The parameter a of genoxide's default problem.
            pub const A: f64 = 3.0;

            /// The parameter b of genoxide's default problem.
            pub const B: f64 = $b;

            #[doc = concat!("The problem with `variables` variables, at least `M`, a = 3 and b = ", stringify!($b), ".")]
            ///
            /// # Panics
            ///
            /// With fewer than 2 objectives, or fewer variables than objectives.
            pub fn new(variables: usize) -> Self {
                Self::with_parameters(variables, Self::A, Self::B)
            }

            /// The problem with `variables` variables, at least `M`, and the parameters `a`
            /// (the number of feasible segments or local optima, above 0) and `b` (how narrow
            /// they are, between −1 and 1).
            ///
            /// # Panics
            ///
            /// With fewer than 2 objectives, fewer variables than objectives, an `a` that isn't
            /// finite and above 0, or a `b` that isn't between −1 and 1.
            pub fn with_parameters(variables: usize, a: f64, b: f64) -> Self {
                Self {
                    core: Core::new($label, $shape, $kind, variables, a, b),
                }
            }

            /// The number of variables.
            pub fn variables(&self) -> usize {
                self.core.variables
            }

            /// The parameter a.
            pub fn a(&self) -> f64 {
                self.core.a
            }

            /// The parameter b.
            pub fn b(&self) -> f64 {
                self.core.b
            }
        }

        impl<const M: usize> Default for $name<M> {
            #[doc = concat!("The problem with `M + ", stringify!($k), " − 1` variables, a = 3 and b = ", stringify!($b), ".")]
            ///
            /// # Panics
            ///
            /// With fewer than 2 objectives.
            fn default() -> Self {
                Self::new(M + $k - 1)
            }
        }

        impl<const M: usize> MultiFitnessFunction<Reals, M> for $name<M> {
            type Output = ([f64; M], f64);

            /// The objective values of `x` and its constraint violation, 0 when it's feasible.
            ///
            /// # Panics
            ///
            /// If `x` has fewer than `M` genes.
            fn evaluate(&self, x: &Reals) -> ([f64; M], f64) {
                self.core.evaluate(x)
            }
        }

        impl<const M: usize> MultiProblem<M> for $name<M> {
            type Representation = Real;

            fn name(&self) -> &'static str {
                $label
            }

            fn representation(&self) -> Real {
                Real::uniform(self.core.variables, 0.0..=1.0).expect("valid bounds")
            }

            fn reference(&self) -> &'static str {
                REFERENCE
            }

            fn reference_url(&self) -> Option<&'static str> {
                Some(REFERENCE_URL)
            }

            fn constraint_count(&self) -> usize {
                self.core.count()
            }

            fn constraints(&self, genome: &Reals) -> Constraints {
                Constraints::new(self.core.values(genome), Vec::new())
            }

            fn optimal_front(&self, points: usize) -> Option<Vec<[f64; M]>> {
                Some(self.core.front(points))
            }

            fn ideal_point(&self) -> Option<[f64; M]> {
                Some(self.core.ideal())
            }

            fn nadir_point(&self) -> Option<[f64; M]> {
                Some(self.core.nadir())
            }
        }
    };
}

dc_dtlz!(
    /// DC1-DTLZ1: [`Dtlz1`] subject to `cos(aπx₁) ≥ b`, with a = 3 and b = 0.5 (the supplement's
    /// eq. 12): feasible cones from the origin.
    ///
    /// With `M` objectives and `n` variables in [0, 1], `M + 4` by default. With a = 3 and
    /// b = 0.5, x₁ is feasible in [0, 1/9] and [5/9, 7/9], a third of its range, and the front is
    /// the parts of DTLZ1's (the plane `Σ fᵢ = 1/2`) with `f_M = (1 − x₁)/2` in [1/9, 2/9] and
    /// [4/9, 1/2]: two bands, from the corner (0, …, 0, 1/2), each a cone from the origin through
    /// the plane. The ideal point is (0, …, 0, 1/9) and the nadir point (7/18, …, 7/18, 1/2).
    /// DTLZ1's g has 11ᵏ − 1 local fronts, all feasible where x₁ is.
    ///
    /// [`constraints`](MultiProblem::constraints) gives `b − cos(aπx₁)`.
    ///
    /// Li, K., Chen, R., Fu, G. and Yao, X. (2019). Two-archive evolutionary algorithm for
    /// constrained multiobjective optimization. *IEEE Transactions on Evolutionary Computation*
    /// 23(2): 303-315, supplementary document, section 1.2.1, eq. 12; see the module docs for
    /// what was checked where, and what genoxide settles that the paper doesn't.
    Dc1Dtlz1,
    "DC1-DTLZ1",
    Shape::Plane,
    Kind::One,
    5,
    0.5
);

dc_dtlz!(
    /// DC1-DTLZ3: [`Dtlz3`] subject to `cos(aπx₁) ≥ b`, with a = 3 and b = 0.5 (the supplement's
    /// eq. 12): feasible cones from the origin.
    ///
    /// With `M` objectives and `n` variables in [0, 1], `M + 9` by default. With a = 3 and
    /// b = 0.5, x₁ is feasible in [0, 1/9] and [5/9, 7/9], and the front is the parts of DTLZ3's
    /// (the unit sphere) with `f_M = sin(πx₁/2)` in [0, sin(π/18)] and [sin(5π/18), sin(7π/18)]:
    /// two bands, one along the base of the sphere. The ideal point is the origin and the nadir
    /// point (1, …, 1, sin(7π/18)). DTLZ3's g has 3ᵏ − 1 local fronts (Deb et al.), spheres of
    /// larger radii, all feasible where x₁ is.
    ///
    /// [`constraints`](MultiProblem::constraints) gives `b − cos(aπx₁)`.
    ///
    /// Li, K., Chen, R., Fu, G. and Yao, X. (2019). Two-archive evolutionary algorithm for
    /// constrained multiobjective optimization. *IEEE Transactions on Evolutionary Computation*
    /// 23(2): 303-315, supplementary document, section 1.2.1, eq. 12 (see [`Dc1Dtlz1`]).
    Dc1Dtlz3,
    "DC1-DTLZ3",
    Shape::Sphere,
    Kind::One,
    10,
    0.5
);

dc_dtlz!(
    /// DC2-DTLZ1: [`Dtlz1`] subject to `cos(aπg) ≥ b` and `e^(−g) ≥ b`, with a = 3 and b = 0.9
    /// (the supplement's eqs. 13 and 14; b from the authors' lab's code): DTLZ1's front, and
    /// almost nothing else feasible.
    ///
    /// With `M` objectives and `n` variables in [0, 1], `M + 4` by default. With b = 0.9, the
    /// second constraint asks g ≤ −ln 0.9 ≈ 0.105 and the first `3πg ≤ arccos 0.9`, g ≤ 0.0479,
    /// in the first of its bands: the feasible solutions are those within g ≈ 0.048 of the front,
    /// where DTLZ1's g reaches 125 k. The front is DTLZ1's, whole: `Σ fᵢ = 1/2`, ideal point the
    /// origin and nadir point (1/2, …, 1/2). The violation rises and falls with g (the
    /// supplement's figure 6): its local minima, where `cos(aπg)` returns to 1, at g = 2/3,
    /// 4/3, …, hold a population back.
    ///
    /// [`constraints`](MultiProblem::constraints) gives `b − cos(aπg)` and `b − e^(−g)`.
    ///
    /// Li, K., Chen, R., Fu, G. and Yao, X. (2019). Two-archive evolutionary algorithm for
    /// constrained multiobjective optimization. *IEEE Transactions on Evolutionary Computation*
    /// 23(2): 303-315, supplementary document, section 1.2.2, eqs. 13 and 14 (see [`Dc1Dtlz1`]).
    Dc2Dtlz1,
    "DC2-DTLZ1",
    Shape::Plane,
    Kind::Two,
    5,
    0.9
);

dc_dtlz!(
    /// DC2-DTLZ3: [`Dtlz3`] subject to `cos(aπg) ≥ b` and `e^(−g) ≥ b`, with a = 3 and b = 0.5
    /// (the supplement's eqs. 13 and 14; b from the authors' lab's code): DTLZ3's front, and
    /// almost nothing else feasible.
    ///
    /// With `M` objectives and `n` variables in [0, 1], `M + 9` by default. With b = 0.5, g must
    /// be in [0, 1/9] or [5/9, ln 2]: spheres of radius 1 to 10/9 and 14/9 to 1.69 are
    /// feasible. The front is DTLZ3's, whole: the unit sphere, ideal point the origin and nadir
    /// point (1, …, 1). The violation's local minima, where `cos(aπg)` returns to 1, at
    /// g = 2/3, 4/3, …, hold a population back.
    ///
    /// [`constraints`](MultiProblem::constraints) gives `b − cos(aπg)` and `b − e^(−g)`.
    ///
    /// Li, K., Chen, R., Fu, G. and Yao, X. (2019). Two-archive evolutionary algorithm for
    /// constrained multiobjective optimization. *IEEE Transactions on Evolutionary Computation*
    /// 23(2): 303-315, supplementary document, section 1.2.2, eqs. 13 and 14 (see [`Dc1Dtlz1`]).
    Dc2Dtlz3,
    "DC2-DTLZ3",
    Shape::Sphere,
    Kind::Two,
    10,
    0.5
);

dc_dtlz!(
    /// DC3-DTLZ1: [`Dtlz1`] subject to `cos(aπxⱼ) ≥ b` for each position variable,
    /// j = 1, …, M − 1, and `cos(aπg) ≥ b`, with a = 3 and b = 0.5 (the supplement's eqs. 15 and
    /// 16, read as the module docs say): the front in patches, behind DC2's local optima.
    ///
    /// With `M` objectives and `n` variables in [0, 1], `M + 4` by default: M constraints. With
    /// a = 3 and b = 0.5, every position variable is feasible in [0, 1/9] and [5/9, 7/9], and g in
    /// [0, 1/9], [5/9, 7/9], [11/9, 13/9], …. The front is the parts of DTLZ1's where all the
    /// position variables are: 2^(M−1) patches, some of them thin. The ideal point is
    /// (0, …, 0, 1/9) and the nadir point (7/18 (7/9)^(M−2), …, 7/18, 1/2), the first objective
    /// being the product of all the position variables.
    ///
    /// [`constraints`](MultiProblem::constraints) gives `b − cos(aπxⱼ)` for j < M, then
    /// `b − cos(aπg)`.
    ///
    /// Li, K., Chen, R., Fu, G. and Yao, X. (2019). Two-archive evolutionary algorithm for
    /// constrained multiobjective optimization. *IEEE Transactions on Evolutionary Computation*
    /// 23(2): 303-315, supplementary document, section 1.2.3, eqs. 15 and 16 (see [`Dc1Dtlz1`]).
    Dc3Dtlz1,
    "DC3-DTLZ1",
    Shape::Plane,
    Kind::Three,
    5,
    0.5
);

dc_dtlz!(
    /// DC3-DTLZ3: [`Dtlz3`] subject to `cos(aπxⱼ) ≥ b` for each position variable,
    /// j = 1, …, M − 1, and `cos(aπg) ≥ b`, with a = 3 and b = 0.5 (the supplement's eqs. 15 and
    /// 16, read as the module docs say): the front in patches, behind DC2's local optima.
    ///
    /// With `M` objectives and `n` variables in [0, 1], `M + 9` by default: M constraints. With
    /// a = 3 and b = 0.5, every position variable is feasible in [0, 1/9] and [5/9, 7/9], and the
    /// front is the parts of DTLZ3's, the unit sphere, where all of them are: 2^(M−1) patches.
    /// The ideal point is (cos(7π/18)^(M−1), 0, …, 0) and the nadir point
    /// (1, sin(7π/18), …, sin(7π/18)): f₁, a product of cosines, is 1 with every position variable
    /// at 0 and least with all of them at 7/9, and each other objective has a sine factor, of a
    /// variable at most 7/9.
    ///
    /// [`constraints`](MultiProblem::constraints) gives `b − cos(aπxⱼ)` for j < M, then
    /// `b − cos(aπg)`.
    ///
    /// Li, K., Chen, R., Fu, G. and Yao, X. (2019). Two-archive evolutionary algorithm for
    /// constrained multiobjective optimization. *IEEE Transactions on Evolutionary Computation*
    /// 23(2): 303-315, supplementary document, section 1.2.3, eqs. 15 and 16 (see [`Dc1Dtlz1`]).
    Dc3Dtlz3,
    "DC3-DTLZ3",
    Shape::Sphere,
    Kind::Three,
    10,
    0.5
);

/// The DC-DTLZ problems with `M` objectives at their default sizes, for [`all`](super::all).
pub(super) fn all<const M: usize>() -> Vec<Box<dyn DynMultiProblem<M>>> {
    vec![
        boxed(Dc1Dtlz1::<M>::default()),
        boxed(Dc1Dtlz3::<M>::default()),
        boxed(Dc2Dtlz1::<M>::default()),
        boxed(Dc2Dtlz3::<M>::default()),
        boxed(Dc3Dtlz1::<M>::default()),
        boxed(Dc3Dtlz3::<M>::default()),
    ]
}

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

    // the formulas of the supplement, at points chosen so that the values follow by hand
    #[test]
    fn values_match_the_supplement() {
        // DC1-DTLZ1 (M = 3, n = 7): x₁ = 0 is feasible, cos 0 = 1 ≥ 0.5; x₁ = 1/3 isn't,
        // cos π = −1, by 1.5; the objectives are DTLZ1's
        let problem = Dc1Dtlz1::<3>::default();
        assert_eq!(problem.variables(), 7);
        let x = at(&[0.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        assert_eq!(problem.evaluate(&x), ([0.0, 0.0, 0.5], 0.0));
        assert_close(problem.constraints(&x).inequalities(), &[-0.5]);
        let x = at(&[1.0 / 3.0, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
        let (f, violation) = problem.evaluate(&x);
        assert_close(&f, &[1.0 / 12.0, 1.0 / 12.0, 1.0 / 3.0]);
        assert_close(&[violation], &[1.5]);

        // DC2-DTLZ1 (b = 0.9): g = 0 at the distance variables at 0.5, feasible by 0.1 twice; a
        // distance variable at 0.6 adds 100 (0.01 − cos 2π + 1) = 1 to g, cos 3π = −1 and
        // e^(−1): violations 1.9 and 0.9 − e^(−1)
        let problem = Dc2Dtlz1::<3>::default();
        assert_eq!((problem.a(), problem.b()), (3.0, 0.9));
        let x = at(&[0.5; 7]);
        assert_close(problem.constraints(&x).inequalities(), &[-0.1, -0.1]);
        let x = at(&[0.5, 0.5, 0.6, 0.5, 0.5, 0.5, 0.5]);
        let (f, violation) = problem.evaluate(&x);
        assert_close(&f, &[0.25, 0.25, 0.5]);
        assert_close(&[violation], &[1.9 + 0.9 - math::exp(-1.0)]);

        // DC2-DTLZ3 (b = 0.5, n = 12): g = 2/3 at cos(2π) = 1 is a local minimum of the first
        // violation, where e^(−2/3) ≈ 0.513 ≥ 0.5: feasible. Two distance variables at
        // 0.5 ± d with 100 (d² − cos 20πd + 1) = 1/3 each give it
        let problem = Dc2Dtlz3::<3>::default();
        assert_eq!(problem.variables(), 12);
        // d solves d² − cos(20πd) + 1 = 1/300, by bisection
        let h = |d: f64| d * d - math::cos(20.0 * PI * d) + 1.0 - 1.0 / 300.0;
        let (mut low, mut high) = (0.0, 0.01);
        for _ in 0..100 {
            let middle = 0.5 * (low + high);
            if h(middle) < 0.0 {
                low = middle;
            } else {
                high = middle;
            }
        }
        let mut genes = vec![0.5; 12];
        genes[2] = 0.5 + low;
        genes[3] = 0.5 - low;
        let x = at(&genes);
        let g = rastrigin_g(&genes[2..]);
        assert!((g - 2.0 / 3.0).abs() < 1e-9, "{g}");
        let values = problem.constraints(&x).inequalities().to_vec();
        assert!((values[0] + 0.5).abs() < 1e-6, "{values:?}");
        assert_close(&values[1..], &[0.5 - math::exp(-g)]);
        assert_eq!(problem.evaluate(&x).1, 0.0);

        // DC3-DTLZ1 (M = 3): constraints on x₁, x₂ and g; (0.6, 0.7) is feasible in both,
        // (0.6, 0.3) not in x₂: 0.5 − cos(0.9π)
        let problem = Dc3Dtlz1::<3>::default();
        assert_eq!(problem.constraint_count(), 3);
        let x = at(&[0.6, 0.7, 0.5, 0.5, 0.5, 0.5, 0.5]);
        assert_eq!(problem.evaluate(&x).1, 0.0);
        let x = at(&[0.6, 0.3, 0.5, 0.5, 0.5, 0.5, 0.5]);
        let values = problem.constraints(&x).inequalities().to_vec();
        assert_close(&values[1..], &[0.5 - math::cos(0.9 * PI), -0.5]);
        assert_close(&[problem.evaluate(&x).1], &[0.5 - math::cos(0.9 * PI)]);

        // DC3-DTLZ3 with other parameters: a = 2, b = 0
        let problem = Dc3Dtlz3::<3>::with_parameters(12, 2.0, 0.0);
        assert_eq!((problem.a(), problem.b()), (2.0, 0.0));
        let x = at(&genes);
        assert_close(
            &problem.constraints(&x).inequalities()[..2],
            &[-math::cos(PI), -math::cos(PI)],
        );
    }

    #[test]
    fn feasible_values_of_a_variable() {
        let core = Dc1Dtlz1::<3>::default().core;
        let intervals = core.intervals();
        assert_eq!(intervals.len(), 2);
        let expected = [(0.0, 1.0 / 9.0), (5.0 / 9.0, 7.0 / 9.0)];
        for ((low, high), (a, b)) in intervals.iter().zip(expected) {
            assert!((low - a).abs() < 1e-12 && (high - b).abs() < 1e-12);
            assert!(core.cosine(*low) <= 0.0 && core.cosine(*high) <= 0.0);
        }
        // the ends: the last feasible floating-point values
        let (_, high) = intervals[1];
        assert!(core.cosine(next_up(high)) > 0.0);
        // with a = 2 and b = 0, x = 1 is feasible: [0, 1/4], [3/4, 1]
        let core = Dc1Dtlz1::<3>::with_parameters(7, 2.0, 0.0).core;
        let intervals = core.intervals();
        assert_eq!(intervals.len(), 2);
        assert_eq!(intervals[1].1, 1.0);
        assert!((intervals[0].1 - 0.25).abs() < 1e-12);
    }

    fn next_up(x: f64) -> f64 {
        f64::from_bits(x.to_bits() + 1)
    }

    // the fronts: on DTLZ's, with feasible genomes, mutually non-dominated, between the ideal
    // and nadir points that they reach
    fn check_front<P, const M: usize>(
        problem: &P,
        points: usize,
        on_front: impl Fn(&[f64; M]) -> bool,
    ) where
        P: MultiProblem<M, Representation = Real>
            + MultiFitnessFunction<Reals, M, Output = ([f64; M], f64)>,
    {
        let front = problem.optimal_front(points).expect("known");
        if M == 2 {
            assert_eq!(front.len(), points);
        } else {
            assert!(front.len() >= points);
        }
        let scores: Vec<Scores<M>> = front.iter().map(|f| Scores::new(*f)).collect();
        assert_eq!(non_dominated_sort(&scores, &[Minimize; M]).len(), 1);
        let ideal = problem.ideal_point().expect("known");
        let nadir = problem.nadir_point().expect("known");
        for f in &front {
            assert!(on_front(f), "{}: {f:?}", problem.name());
            for j in 0..M {
                assert!(f[j] >= ideal[j] && f[j] <= nadir[j], "{f:?}");
            }
        }
        for j in 0..M {
            assert!(front.iter().any(|f| f[j] == ideal[j]), "{}", problem.name());
            assert!(front.iter().any(|f| f[j] == nadir[j]), "{}", problem.name());
        }
    }

    #[test]
    fn the_fronts_lie_where_derived() {
        let plane = |f: &[f64; 3]| (f.iter().sum::<f64>() - 0.5).abs() < 1e-12;
        let sphere = |f: &[f64; 3]| (f.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12;
        // DC1: f₃ = (1 − x₁)/2 in [1/9, 2/9] or [4/9, 1/2] (DTLZ1), sin(πx₁/2) in [0, sin(π/18)]
        // or [sin(5π/18), sin(7π/18)] (DTLZ3)
        let near = |v: f64, low: f64, high: f64| v >= low - 1e-12 && v <= high + 1e-12;
        check_front(&Dc1Dtlz1::<3>::default(), 200, |f| {
            plane(f) && (near(f[2], 1.0 / 9.0, 2.0 / 9.0) || near(f[2], 4.0 / 9.0, 0.5))
        });
        let s = |v: f64| math::sin(PI * v / 18.0);
        check_front(&Dc1Dtlz3::<3>::default(), 200, |f| {
            sphere(f) && (near(f[2], 0.0, s(1.0)) || near(f[2], s(5.0), s(7.0)))
        });
        check_front(&Dc2Dtlz1::<3>::default(), 91, plane);
        check_front(&Dc2Dtlz3::<3>::default(), 91, sphere);
        let dc3 = Dc3Dtlz1::<3>::default();
        check_front(&dc3, 200, |f| {
            plane(f)
                && dc3
                    .core
                    .positions(f)
                    .iter()
                    .all(|&x| near(x, 0.0, 1.0 / 9.0) || near(x, 5.0 / 9.0, 7.0 / 9.0))
        });
        check_front(&Dc3Dtlz3::<3>::default(), 200, sphere);
        // the ideal and nadir points of the docs
        assert_close(
            &Dc1Dtlz1::<3>::default().ideal_point().unwrap(),
            &[0.0, 0.0, 1.0 / 9.0],
        );
        assert_close(
            &Dc1Dtlz1::<3>::default().nadir_point().unwrap(),
            &[7.0 / 18.0, 7.0 / 18.0, 0.5],
        );
        assert_close(
            &Dc1Dtlz3::<3>::default().nadir_point().unwrap(),
            &[1.0, 1.0, s(7.0)],
        );
        assert_close(
            &Dc3Dtlz1::<3>::default().nadir_point().unwrap(),
            &[7.0 / 18.0 * 7.0 / 9.0, 7.0 / 18.0, 0.5],
        );
        // two objectives: exactly the points asked for, with both ends of each piece
        let problem = Dc1Dtlz1::<2>::default();
        check_front(&problem, 50, |f| (f[0] + f[1] - 0.5).abs() < 1e-12);
        check_front(&Dc3Dtlz3::<2>::default(), 37, |f| {
            (f[0] * f[0] + f[1] * f[1] - 1.0).abs() < 1e-12
        });
        let front = problem.optimal_front(1_000).unwrap();
        let x1 = |f: &[f64; 2]| 2.0 * f[0];
        assert!(
            front
                .iter()
                .all(|f| near(x1(f), 0.0, 1.0 / 9.0) || near(x1(f), 5.0 / 9.0, 7.0 / 9.0))
        );
        // more objectives
        check_front(&Dc1Dtlz1::<4>::default(), 100, |f| {
            (f.iter().sum::<f64>() - 0.5).abs() < 1e-12
        });
        check_front(&Dc3Dtlz3::<4>::default(), 100, |f| {
            (f.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12
        });
    }

    // no feasible solution is better than the front: random genomes with their distance
    // variables near 0.5, every feasible one weakly dominated by a point of the front, up to its
    // spacing, and dominating none
    fn check_against_random<P>(problem: &P, noise: f64, tolerance: f64)
    where
        P: MultiProblem<3, Representation = Real>
            + MultiFitnessFunction<Reals, 3, Output = ([f64; 3], f64)>,
    {
        let front = problem.optimal_front(5_000).expect("known");
        let mut rng = StreamRng::seed_from_u64(5);
        let real = problem.representation();
        let mut feasible = 0;
        for _ in 0..10_000 {
            let mut genome = real.random_genome(&mut rng).to_vec();
            for gene in &mut genome[2..] {
                *gene = 0.5 + (*gene - 0.5) * 2.0 * noise;
            }
            let (f, violation) = problem.evaluate(&at(&genome));
            if violation > 0.0 {
                continue;
            }
            feasible += 1;
            assert!(
                front
                    .iter()
                    .any(|q| (0..3).all(|j| q[j] <= f[j] + tolerance)),
                "{}: {f:?}",
                problem.name()
            );
            for q in &front {
                assert!(
                    !(0..3).all(|j| f[j] < q[j] - 1e-9),
                    "{}: {f:?} dominates {q:?}",
                    problem.name()
                );
            }
        }
        assert!(feasible > 50, "{}: {feasible}", problem.name());
    }

    #[test]
    fn random_solutions_agree_with_the_fronts() {
        check_against_random(&Dc1Dtlz1::<3>::default(), 0.001, 0.02);
        check_against_random(&Dc1Dtlz3::<3>::default(), 0.001, 0.03);
        check_against_random(&Dc2Dtlz1::<3>::default(), 0.0005, 0.02);
        check_against_random(&Dc2Dtlz3::<3>::default(), 0.001, 0.03);
        check_against_random(&Dc3Dtlz1::<3>::default(), 0.001, 0.02);
        check_against_random(&Dc3Dtlz3::<3>::default(), 0.001, 0.03);
    }

    #[test]
    fn sizes_and_parameters() {
        assert_eq!(Dc1Dtlz3::<3>::default().variables(), 12);
        assert_eq!(Dc2Dtlz1::<5>::default().variables(), 9);
        assert_eq!(Dc3Dtlz3::<5>::default().constraint_count(), 5);
        assert_eq!(Dc2Dtlz3::<3>::default().constraint_count(), 2);
        assert_eq!(Dc1Dtlz1::<3>::B, 0.5);
        assert_eq!(Dc2Dtlz1::<3>::B, 0.9);
        let names: Vec<_> = all::<3>().iter().map(|p| p.name()).collect();
        assert_eq!(
            names,
            [
                "DC1-DTLZ1",
                "DC1-DTLZ3",
                "DC2-DTLZ1",
                "DC2-DTLZ3",
                "DC3-DTLZ1",
                "DC3-DTLZ3"
            ]
        );
    }

    #[test]
    #[should_panic(expected = "a finite b between -1 and 1")]
    fn a_b_of_1_panics() {
        let _ = Dc1Dtlz1::<3>::with_parameters(7, 3.0, 1.0);
    }
}
