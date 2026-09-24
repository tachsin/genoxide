//! Standard multi-objective test problems: ZDT (Zitzler, Deb and Thiele, 2000) and DTLZ (Deb,
//! Thiele, Laumanns and Zitzler, 2002), all minimized.
//!
//! Each problem is a fitness function for a [`MultiEngine`](super::MultiEngine), with its
//! [`Real`] representation and points of its optimal front, for the
//! [indicators](super::indicator):
//!
//! ```
//! use genoxide::Objective::Minimize;
//! use genoxide::multi::indicator::igd_plus;
//! use genoxide::multi::problems::{TestProblem, Zdt1};
//! use genoxide::prelude::*;
//!
//! let problem = Zdt1::new(30);
//! let nsga2 = Nsga2::builder(problem.real(), [Minimize; 2])
//!     .population_size(100)
//!     .crossover(SimulatedBinaryCrossover::new(15.0)?)
//!     .mutate(PolynomialMutation::per_gene(1.0 / 30.0, 20.0)?)
//!     .seed(1)
//!     .build()?;
//! let outcome = MultiEngine::new(nsga2, problem).stop_when(Stop::generations(200)).run()?;
//! let distance = igd_plus(&outcome.front_values(), &problem.optimal_front(500), &[Minimize; 2]);
//! assert!(distance < 0.01);
//! # Ok::<(), genoxide::Error>(())
//! ```
//!
//! The problems use the platform's trigonometric and exponential functions, so their values can
//! differ in the last bit between platforms, unlike the rest of genoxide.

use super::MultiFitnessFunction;
use crate::genome::{Real, Reals};
use std::f64::consts::PI;

/// A test problem with `M` objectives to minimize and a known optimal front.
pub trait TestProblem<const M: usize>: MultiFitnessFunction<Reals, M, Output = [f64; M]> {
    /// The name, e.g. `"ZDT1"`.
    fn name(&self) -> &'static str;

    /// The representation: the number of variables and their bounds.
    fn real(&self) -> Real;

    /// At least `points` points of the optimal front (exactly `points` for 2 objectives), for
    /// [`igd`](super::indicator::igd) and similar indicators.
    fn optimal_front(&self, points: usize) -> Vec<[f64; M]>;
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

// the mean of the variables from the second on, ZDT's usual g - 1 over 9
fn tail_mean(x: &[f64]) -> f64 {
    x[1..].iter().sum::<f64>() / (x.len() - 1) as f64
}

fn zdt_front(points: usize, f2: impl Fn(f64) -> f64, start: f64) -> Vec<[f64; 2]> {
    (0..points)
        .map(|i| {
            let t = if points > 1 {
                i as f64 / (points - 1) as f64
            } else {
                0.0
            };
            let f1 = start + (1.0 - start) * t;
            [f1, f2(f1)]
        })
        .collect()
}

macro_rules! zdt {
    ($name:ident, $label:literal, $doc:literal, $default:literal) => {
        #[doc = $doc]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name {
            variables: usize,
        }

        impl $name {
            #[doc = concat!("The problem with `variables` variables, at least 2; ", stringify!($default), " is standard.")]
            ///
            /// # Panics
            ///
            /// If `variables` is below 2.
            pub fn new(variables: usize) -> Self {
                assert!(variables >= 2, "{} needs at least 2 variables", $label);
                Self { variables }
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new($default)
            }
        }
    };
}

zdt!(Zdt1, "ZDT1", "ZDT1: a convex front, `f₂ = 1 − √f₁`.", 30);
zdt!(Zdt2, "ZDT2", "ZDT2: a concave front, `f₂ = 1 − f₁²`.", 30);
zdt!(
    Zdt3,
    "ZDT3",
    "ZDT3: a front of five disconnected pieces.",
    30
);
zdt!(
    Zdt4,
    "ZDT4",
    "ZDT4: the convex front of ZDT1 behind 21⁹ local fronts (Rastrigin in the other variables).",
    10
);
zdt!(
    Zdt6,
    "ZDT6",
    "ZDT6: a concave front with solutions that are dense near one end, non-uniformly spread.",
    10
);

impl MultiFitnessFunction<Reals, 2> for Zdt1 {
    type Output = [f64; 2];

    fn evaluate(&self, x: &Reals) -> [f64; 2] {
        let g = 1.0 + 9.0 * tail_mean(x);
        [x[0], g * (1.0 - (x[0] / g).sqrt())]
    }
}

impl TestProblem<2> for Zdt1 {
    fn name(&self) -> &'static str {
        "ZDT1"
    }

    fn real(&self) -> Real {
        Real::uniform(self.variables, 0.0..=1.0).expect("valid bounds")
    }

    fn optimal_front(&self, points: usize) -> Vec<[f64; 2]> {
        zdt_front(points, |f1| 1.0 - f1.sqrt(), 0.0)
    }
}

impl MultiFitnessFunction<Reals, 2> for Zdt2 {
    type Output = [f64; 2];

    fn evaluate(&self, x: &Reals) -> [f64; 2] {
        let g = 1.0 + 9.0 * tail_mean(x);
        [x[0], g * (1.0 - (x[0] / g) * (x[0] / g))]
    }
}

impl TestProblem<2> for Zdt2 {
    fn name(&self) -> &'static str {
        "ZDT2"
    }

    fn real(&self) -> Real {
        Real::uniform(self.variables, 0.0..=1.0).expect("valid bounds")
    }

    fn optimal_front(&self, points: usize) -> Vec<[f64; 2]> {
        zdt_front(points, |f1| 1.0 - f1 * f1, 0.0)
    }
}

// the pieces of ZDT3's optimal front, as ranges of f1
const ZDT3_PIECES: [(f64, f64); 5] = [
    (0.0, 0.083_001_534_9),
    (0.182_228_728_0, 0.257_762_363_4),
    (0.409_313_674_8, 0.453_882_104_1),
    (0.618_396_794_4, 0.652_511_703_8),
    (0.823_331_798_3, 0.851_832_865_4),
];

impl MultiFitnessFunction<Reals, 2> for Zdt3 {
    type Output = [f64; 2];

    fn evaluate(&self, x: &Reals) -> [f64; 2] {
        let g = 1.0 + 9.0 * tail_mean(x);
        let ratio = x[0] / g;
        [
            x[0],
            g * (1.0 - ratio.sqrt() - ratio * (10.0 * PI * x[0]).sin()),
        ]
    }
}

impl TestProblem<2> for Zdt3 {
    fn name(&self) -> &'static str {
        "ZDT3"
    }

    fn real(&self) -> Real {
        Real::uniform(self.variables, 0.0..=1.0).expect("valid bounds")
    }

    fn optimal_front(&self, points: usize) -> Vec<[f64; 2]> {
        // spread over the pieces in proportion to their widths
        let total: f64 = ZDT3_PIECES.iter().map(|(low, high)| high - low).sum();
        (0..points)
            .map(|i| {
                let t = if points > 1 {
                    i as f64 / (points - 1) as f64
                } else {
                    0.0
                };
                let mut position = t * total;
                let mut f1 = ZDT3_PIECES[4].1;
                for (low, high) in ZDT3_PIECES {
                    if position <= high - low {
                        f1 = low + position;
                        break;
                    }
                    position -= high - low;
                }
                [f1, 1.0 - f1.sqrt() - f1 * (10.0 * PI * f1).sin()]
            })
            .collect()
    }
}

impl MultiFitnessFunction<Reals, 2> for Zdt4 {
    type Output = [f64; 2];

    fn evaluate(&self, x: &Reals) -> [f64; 2] {
        let g = 1.0
            + 10.0 * (x.len() - 1) as f64
            + x[1..]
                .iter()
                .map(|xi| xi * xi - 10.0 * (4.0 * PI * xi).cos())
                .sum::<f64>();
        [x[0], g * (1.0 - (x[0] / g).sqrt())]
    }
}

impl TestProblem<2> for Zdt4 {
    fn name(&self) -> &'static str {
        "ZDT4"
    }

    fn real(&self) -> Real {
        Real::new(std::iter::once(0.0..=1.0).chain((1..self.variables).map(|_| -5.0..=5.0)))
            .expect("valid bounds")
    }

    fn optimal_front(&self, points: usize) -> Vec<[f64; 2]> {
        zdt_front(points, |f1| 1.0 - f1.sqrt(), 0.0)
    }
}

// the smallest f1 of ZDT6's optimal front, 1 − exp(−4 x₁) sin⁶(6π x₁) at its minimum
const ZDT6_START: f64 = 0.280_775_319_1;

impl MultiFitnessFunction<Reals, 2> for Zdt6 {
    type Output = [f64; 2];

    fn evaluate(&self, x: &Reals) -> [f64; 2] {
        let f1 = 1.0 - (-4.0 * x[0]).exp() * (6.0 * PI * x[0]).sin().powi(6);
        let g = 1.0 + 9.0 * tail_mean(x).powf(0.25);
        [f1, g * (1.0 - (f1 / g) * (f1 / g))]
    }
}

impl TestProblem<2> for Zdt6 {
    fn name(&self) -> &'static str {
        "ZDT6"
    }

    fn real(&self) -> Real {
        Real::uniform(self.variables, 0.0..=1.0).expect("valid bounds")
    }

    fn optimal_front(&self, points: usize) -> Vec<[f64; 2]> {
        zdt_front(points, |f1| 1.0 - f1 * f1, ZDT6_START)
    }
}

macro_rules! dtlz {
    ($name:ident, $label:literal, $doc:literal, $k:literal) => {
        #[doc = $doc]
        ///
        #[doc = concat!("With `M` objectives and `n` variables: the first `M − 1` place a solution on the front, the other `k = n − M + 1` its distance to it; ", stringify!($k), " is the standard `k`.")]
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name<const M: usize> {
            variables: usize,
        }

        impl<const M: usize> $name<M> {
            /// The problem with `variables` variables, at least `M`.
            ///
            /// # Panics
            ///
            /// With fewer than 2 objectives, or fewer variables than objectives.
            pub fn new(variables: usize) -> Self {
                assert!(M >= 2, "{} needs at least 2 objectives", $label);
                assert!(variables >= M, "{} needs at least as many variables as objectives", $label);
                Self { variables }
            }
        }

        impl<const M: usize> Default for $name<M> {
            #[doc = concat!("The standard problem, with `M + ", stringify!($k), " − 1` variables.")]
            fn default() -> Self {
                Self::new(M + $k - 1)
            }
        }
    };
}

dtlz!(
    Dtlz1,
    "DTLZ1",
    "DTLZ1: a linear front (the objectives sum to 1/2) behind 11ᵏ − 1 local fronts.",
    5
);
dtlz!(
    Dtlz2,
    "DTLZ2",
    "DTLZ2: a spherical front (the squared objectives sum to 1).",
    10
);
dtlz!(
    Dtlz3,
    "DTLZ3",
    "DTLZ3: the spherical front of DTLZ2 behind 3ᵏ − 1 local fronts.",
    10
);
dtlz!(
    Dtlz4,
    "DTLZ4",
    "DTLZ4: the spherical front of DTLZ2, with solutions biased towards some objectives.",
    10
);

// DTLZ1 and DTLZ3's multimodal distance function
fn rastrigin_g(tail: &[f64]) -> f64 {
    100.0
        * (tail.len() as f64
            + tail
                .iter()
                .map(|x| (x - 0.5) * (x - 0.5) - (20.0 * PI * (x - 0.5)).cos())
                .sum::<f64>())
}

// DTLZ2 and DTLZ4's distance function
fn sphere_g(tail: &[f64]) -> f64 {
    tail.iter().map(|x| (x - 0.5) * (x - 0.5)).sum()
}

// the objectives on a sphere of radius `radius`, with the angles from `x`
fn spherical<const M: usize>(x: &[f64], radius: f64, alpha: f64) -> [f64; M] {
    std::array::from_fn(|m| {
        let mut f = radius;
        let angle = |xi: f64| if alpha == 1.0 { xi } else { xi.powf(alpha) } * PI / 2.0;
        for &xi in &x[..M - 1 - m] {
            f *= angle(xi).cos();
        }
        if m > 0 {
            f *= angle(x[M - 1 - m]).sin();
        }
        f
    })
}

fn spherical_front<const M: usize>(points: usize) -> Vec<[f64; M]> {
    das_dennis::<M>(divisions_for::<M>(points))
        .into_iter()
        .map(|p| {
            let norm = p.iter().map(|v| v * v).sum::<f64>().sqrt();
            p.map(|v| v / norm)
        })
        .collect()
}

impl<const M: usize> MultiFitnessFunction<Reals, M> for Dtlz1<M> {
    type Output = [f64; M];

    fn evaluate(&self, x: &Reals) -> [f64; M] {
        let g = rastrigin_g(&x[M - 1..]);
        std::array::from_fn(|m| {
            let mut f = 0.5 * (1.0 + g);
            for xi in &x[..M - 1 - m] {
                f *= xi;
            }
            if m > 0 {
                f *= 1.0 - x[M - 1 - m];
            }
            f
        })
    }
}

impl<const M: usize> TestProblem<M> for Dtlz1<M> {
    fn name(&self) -> &'static str {
        "DTLZ1"
    }

    fn real(&self) -> Real {
        Real::uniform(self.variables, 0.0..=1.0).expect("valid bounds")
    }

    fn optimal_front(&self, points: usize) -> Vec<[f64; M]> {
        das_dennis::<M>(divisions_for::<M>(points))
            .into_iter()
            .map(|p| p.map(|v| v / 2.0))
            .collect()
    }
}

impl<const M: usize> MultiFitnessFunction<Reals, M> for Dtlz2<M> {
    type Output = [f64; M];

    fn evaluate(&self, x: &Reals) -> [f64; M] {
        spherical(x, 1.0 + sphere_g(&x[M - 1..]), 1.0)
    }
}

impl<const M: usize> TestProblem<M> for Dtlz2<M> {
    fn name(&self) -> &'static str {
        "DTLZ2"
    }

    fn real(&self) -> Real {
        Real::uniform(self.variables, 0.0..=1.0).expect("valid bounds")
    }

    fn optimal_front(&self, points: usize) -> Vec<[f64; M]> {
        spherical_front(points)
    }
}

impl<const M: usize> MultiFitnessFunction<Reals, M> for Dtlz3<M> {
    type Output = [f64; M];

    fn evaluate(&self, x: &Reals) -> [f64; M] {
        spherical(x, 1.0 + rastrigin_g(&x[M - 1..]), 1.0)
    }
}

impl<const M: usize> TestProblem<M> for Dtlz3<M> {
    fn name(&self) -> &'static str {
        "DTLZ3"
    }

    fn real(&self) -> Real {
        Real::uniform(self.variables, 0.0..=1.0).expect("valid bounds")
    }

    fn optimal_front(&self, points: usize) -> Vec<[f64; M]> {
        spherical_front(points)
    }
}

impl<const M: usize> MultiFitnessFunction<Reals, M> for Dtlz4<M> {
    type Output = [f64; M];

    fn evaluate(&self, x: &Reals) -> [f64; M] {
        spherical(x, 1.0 + sphere_g(&x[M - 1..]), 100.0)
    }
}

impl<const M: usize> TestProblem<M> for Dtlz4<M> {
    fn name(&self) -> &'static str {
        "DTLZ4"
    }

    fn real(&self) -> Real {
        Real::uniform(self.variables, 0.0..=1.0).expect("valid bounds")
    }

    fn optimal_front(&self, points: usize) -> Vec<[f64; M]> {
        spherical_front(points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Objective::Minimize;
    use crate::multi::{Scores, non_dominated_sort};

    fn at(values: Vec<f64>) -> Reals {
        Reals::from(values)
    }

    // values computed by pymoo 0.6.2 at random points
    #[test]
    fn values_match_pymoo() {
        fn check<P: TestProblem<M>, const M: usize>(problem: P, x: &[f64], expected: &[f64]) {
            let values = problem.evaluate(&at(x.to_vec()));
            for (value, expected) in values.iter().zip(expected) {
                let tolerance = 1e-12 * expected.abs().max(1.0);
                assert!(
                    (value - expected).abs() <= tolerance,
                    "{}: {values:?} {expected:?}",
                    problem.name()
                );
            }
        }
        check(
            Zdt1::default(),
            &[
                0.6369616873214543,
                0.2697867137638703,
                0.04097352393619469,
                0.016527635528529094,
                0.8132702392002724,
                0.9127555772777217,
                0.6066357757671799,
                0.7294965609839984,
                0.5436249914654229,
                0.9350724237877682,
                0.8158535541215322,
                0.002738500170148095,
                0.8574042765875693,
                0.033585575305464355,
                0.7296554464299441,
                0.17565562060255901,
                0.8631789223498866,
                0.5414612202490917,
                0.2997118905373848,
                0.42268722119765845,
                0.028319671145462966,
                0.12428327649956394,
                0.6706244146936303,
                0.6471895115742501,
                0.6153851114812539,
                0.38367755426188344,
                0.997209935789211,
                0.9808353387762301,
                0.6855419844806947,
                0.6504592762678163,
            ],
            &[0.6369616873214543, 3.8590091483957454],
        );
        check(
            Zdt1::default(),
            &[
                0.6884467305709401,
                0.3889214239791038,
                0.13509650502241122,
                0.7214883401940817,
                0.5253543224757259,
                0.31024187555895566,
                0.4858353588317891,
                0.8894878343490003,
                0.9340435159562497,
                0.35779519670907023,
                0.5715298307297609,
                0.32186939107594215,
                0.5943000301996968,
                0.33791122550713326,
                0.39161900052816123,
                0.8902743520047923,
                0.22715759353337972,
                0.6231871446860424,
                0.08401534358238483,
                0.8326441476533978,
                0.7870983074886834,
                0.23936944299295215,
                0.8764842308107038,
                0.05856803480519435,
                0.3361170605456604,
                0.15027946689483906,
                0.450339366649287,
                0.7963242702872942,
                0.23064220899374743,
                0.05202130106440961,
            ],
            &[0.6884467305709401, 3.324864980568956],
        );
        check(
            Zdt2::default(),
            &[
                0.4045518398215282,
                0.19851304450925533,
                0.0907530456191219,
                0.5803323859868507,
                0.2986961328189226,
                0.6719948779563594,
                0.1995154439682133,
                0.9421131105064978,
                0.36511016824482856,
                0.10549527957022953,
                0.6291081515397092,
                0.9271545530678674,
                0.440377154715784,
                0.9545904936907372,
                0.499895813687647,
                0.42522862484907553,
                0.6202134520153778,
                0.9950965052353241,
                0.9489436749377653,
                0.4600451393090961,
                0.7577288453082914,
                0.49742269548761897,
                0.5293121601967704,
                0.7857857007138075,
                0.4146558493556708,
                0.7344835717887294,
                0.7111428779897498,
                0.9320596866133782,
                0.1149326332809052,
                0.7290151170763094,
            ],
            &[0.4045518398215282, 6.112563809148959],
        );
        check(
            Zdt2::default(),
            &[
                0.9274239286245599,
                0.9679261899246464,
                0.014706304965369288,
                0.8636400902455758,
                0.9811950400663443,
                0.9572101796109636,
                0.1487640122324979,
                0.972628813822955,
                0.8899355557205206,
                0.8223738275430704,
                0.4799879238078322,
                0.23237291963930384,
                0.8018805787183079,
                0.9235301597834695,
                0.2661302722922926,
                0.5389344076221869,
                0.4427528289745315,
                0.931017315981155,
                0.040510711188434634,
                0.7320061956565608,
                0.6143732469489966,
                0.028365365113521057,
                0.7192197728267403,
                0.015991729523571974,
                0.7579510023564281,
                0.512758723262078,
                0.9291042207970062,
                0.06608249672407474,
                0.8413172796123832,
                0.0666900087671014,
            ],
            &[0.9274239286245599, 5.999006658510496],
        );
        check(
            Zdt3::default(),
            &[
                0.34430997880412517,
                0.4302987319478333,
                0.9660620807840702,
                0.562231842228457,
                0.25886459317093224,
                0.24167571409434496,
                0.8881183206591798,
                0.22586942841732438,
                0.1245547058352835,
                0.2883307570075776,
                0.5861230648127328,
                0.5540905021732678,
                0.8097107759127777,
                0.5604759520061858,
                0.2884212144312105,
                0.4128963426808927,
                0.8181209709709104,
                0.6265064624197535,
                0.9590776426974422,
                0.3694044110916809,
                0.5526115105212872,
                0.5939242016131683,
                0.84829120827506,
                0.14547353818653175,
                0.40651033674812664,
                0.909958961662297,
                0.043066888568204176,
                0.8227062801815019,
                0.41538403737122465,
                0.8298039852781027,
            ],
            &[0.34430997880412517, 4.745269248014247],
        );
        check(
            Zdt3::default(),
            &[
                0.009954560807291957,
                0.36504615775827065,
                0.07863003716563988,
                0.6526145763366384,
                0.2738490985995572,
                0.7026520706597863,
                0.9438014269420908,
                0.12681710226124776,
                0.8647782954007741,
                0.059464151600338466,
                0.38077050831088943,
                0.42977406117857664,
                0.48884954683346427,
                0.9764623219360445,
                0.7756911881018284,
                0.308857362719261,
                0.26983678550080015,
                0.8631202041893178,
                0.8813071727376899,
                0.5107065055436453,
                0.34429573096232524,
                0.9949173481609178,
                0.3159435453677002,
                0.18271237892656245,
                0.8800981213040697,
                0.812335398111254,
                0.6678894055713512,
                0.9584136317779519,
                0.9257145772144187,
                0.7482485033017541,
            ],
            &[0.009954560807291957, 5.957034045312921],
        );
        check(
            Zdt4::default(),
            &[
                0.8607014095476777,
                -2.528532596778925,
                -3.5875344309896837,
                1.70061849314936,
                2.1461853665475275,
                -3.329470712177278,
                -1.0444272689512402,
                4.102557662160548,
                0.6140076755022292,
                0.7833591492627265,
            ],
            &[0.8607014095476777, 135.3028685417021],
        );
        check(
            Zdt4::default(),
            &[
                0.19412977289079358,
                0.2602224861787512,
                0.23434727394919896,
                -4.11064359753728,
                4.819426931267062,
                0.713956004557744,
                -4.9359111733568986,
                2.7264920122538863,
                4.782657138401458,
                0.8987002832095046,
            ],
            &[0.19412977289079358, 222.78613238575537],
        );
        check(
            Zdt6::default(),
            &[
                0.319681636282665,
                0.1875077157277849,
                0.6725266339168693,
                0.19510739845680503,
                0.5776878925178592,
                0.6022391763796258,
                0.962423093124381,
                0.07226526552987678,
                0.4999728236586185,
                0.7440974792826482,
            ],
            &[0.999924358368576, 8.457259052325249],
        );
        check(
            Zdt6::default(),
            &[
                0.1772267404746588,
                0.3880667317845192,
                0.06289549845497133,
                0.7258808637757768,
                0.08776788675948677,
                0.3950917083579676,
                0.8735226311207321,
                0.4723003367500115,
                0.9126219336408856,
                0.7659171177388724,
            ],
            &[0.9999705758661059, 8.528621518369352],
        );
        check(
            Dtlz1::<3>::default(),
            &[
                0.9153239601117659,
                0.12740300904890633,
                0.07356290533063203,
                0.07032625356921807,
                0.8688542943473193,
                0.6340699793474432,
                0.496571693798853,
            ],
            &[34.10642034721662, 233.59856246150298, 24.765218425859047],
        );
        check(
            Dtlz1::<3>::default(),
            &[
                0.16354341619648027,
                0.6737334377272737,
                0.318017387845798,
                0.7108798632659449,
                0.4603553288673248,
                0.5074698605445271,
                0.7896657324598704,
            ],
            &[16.970438386594925, 8.218215517598761, 128.82949301975316],
        );
        check(
            Dtlz2::<3>::default(),
            &[
                0.09274547552338075,
                0.5787585033235025,
                0.19723494729586855,
                0.8081367518135681,
                0.4888460361292599,
                0.9886953333678197,
                0.18294332467571872,
                0.9630191401242673,
                0.800917036608609,
                0.4812604965752686,
            ],
            &[1.113362999732723, 1.4295736153981562, 0.26585993186761053],
        );
        check(
            Dtlz2::<3>::default(),
            &[
                0.8135340641796355,
                0.6028489052411163,
                0.6551210639913803,
                0.9136907627073889,
                0.06527041641129139,
                0.8349882039584006,
                0.3818147799662388,
                0.3255456161007044,
                0.9940267712099843,
                0.7811905020763782,
            ],
            &[0.31438028002718943, 0.4368046733605232, 1.7845579196476211],
        );
        check(
            Dtlz3::<3>::default(),
            &[
                0.48553513877958776,
                0.4226283964247812,
                0.8775289058717961,
                0.08681487221489415,
                0.708418756913866,
                0.789154623705146,
                0.7991963797161148,
                0.3222867247398318,
                0.7966391827460546,
                0.22532844187566514,
            ],
            &[235.46637075045513, 184.20854563323843, 285.6742277468477],
        );
        check(
            Dtlz3::<3>::default(),
            &[
                0.3623079504845691,
                0.41744811220437983,
                0.5414099836301646,
                0.11261366554055718,
                0.40694780063930613,
                0.0003006901069229073,
                0.744380726347399,
                0.851875912234257,
                0.13893167912019755,
                0.7037857692667978,
            ],
            &[584.3926910773158, 449.5610572109128, 471.6666890032929],
        );
        check(
            Dtlz4::<3>::default(),
            &[
                0.8211030883946387,
                0.9818283228717938,
                0.8437905623687267,
                0.42410648544401286,
                0.9796887085096565,
                0.9739844048523552,
                0.503676979200579,
                0.7534465385839052,
                0.9138376676731629,
                0.47614707196875306,
            ],
            &[
                1.7579256347161984,
                0.4507470376270612,
                7.847216854062927e-09,
            ],
        );
        check(
            Dtlz4::<3>::default(),
            &[
                0.8637862410970849,
                0.7015685660618728,
                0.2939242559745576,
                0.7676522699834736,
                0.5706847858594991,
                0.09384515343330624,
                0.3913804263046642,
                0.07374101339780592,
                0.4761669632169956,
                0.4285396081429238,
            ],
            &[
                1.4832325849397667,
                9.426301966661368e-16,
                1.0184913417082071e-06,
            ],
        );
        check(
            Dtlz2::<5>::default(),
            &[
                0.42373744297044735,
                0.5863003535907844,
                0.12269066017607344,
                0.9337689099568427,
                0.684050448075033,
                0.8237813583927717,
                0.8968012322637599,
                0.5833200469234759,
                0.0402182209046007,
                0.711486824117758,
            ],
            &[
                0.07563298466216284,
                0.724366976826931,
                0.14212433969380892,
                0.9764378208981872,
                0.9628785072327997,
            ],
        );
        check(
            Dtlz2::<5>::default(),
            &[
                0.5690258542633582,
                0.8259572221703992,
                0.5321604734743441,
                0.8132440953641924,
                0.9970102930724918,
                0.35055481136788813,
                0.1710214400206741,
                0.3916747994539028,
                0.7530499898656764,
                0.43922893185830647,
            ],
            &[
                0.04778002377248,
                0.15817493862323534,
                0.18283232795023638,
                0.8788463847307775,
                1.1357243566732775,
            ],
        );
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
    fn zdt_optimal_solutions_lie_on_the_front() {
        // the optimal solutions have every variable but the first at 0
        for x1 in [0.0, 0.25, 0.5, 1.0] {
            let mut x = vec![0.0; 30];
            x[0] = x1;
            let [f1, f2] = Zdt1::default().evaluate(&at(x.clone()));
            assert!((f2 - (1.0 - f1.sqrt())).abs() < 1e-12);
            let [f1, f2] = Zdt2::default().evaluate(&at(x.clone()));
            assert!((f2 - (1.0 - f1 * f1)).abs() < 1e-12);
            let [f1, f2] = Zdt4::default().evaluate(&at(x[..10].to_vec()));
            assert!((f2 - (1.0 - f1.sqrt())).abs() < 1e-12);
        }
        let front = Zdt1::default().optimal_front(5);
        assert_eq!(front.len(), 5);
        assert_eq!((front[0], front[4]), ([0.0, 1.0], [1.0, 0.0]));
        // ZDT3 and ZDT6: the front points are mutually non-dominated
        for front in [
            Zdt3::default().optimal_front(200),
            Zdt6::default().optimal_front(200),
        ] {
            let scores: Vec<Scores<2>> = front.iter().map(|p| Scores::new(*p)).collect();
            assert_eq!(non_dominated_sort(&scores, &[Minimize; 2]).len(), 1);
        }
        assert_eq!(Zdt6::default().optimal_front(2)[0][0], ZDT6_START);
        assert_eq!(Zdt4::default().real().bounds()[1], -5.0..=5.0);
    }

    #[test]
    fn dtlz_optimal_solutions_lie_on_the_front() {
        // the distance variables at 0.5 put a solution on the front
        let mut x = vec![0.5; 7];
        x[0] = 0.3;
        x[1] = 0.8;
        let f = Dtlz1::<3>::default().evaluate(&at(x.clone()));
        assert!((f.iter().sum::<f64>() - 0.5).abs() < 1e-12);
        let mut x = vec![0.5; 12];
        x[0] = 0.3;
        x[1] = 0.8;
        for f in [
            Dtlz2::<3>::default().evaluate(&at(x.clone())),
            Dtlz3::<3>::default().evaluate(&at(x.clone())),
            Dtlz4::<3>::default().evaluate(&at(x.clone())),
        ] {
            assert!((f.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12);
        }
        let front = Dtlz2::<3>::default().optimal_front(91);
        assert_eq!(front.len(), 91);
        assert!(
            front
                .iter()
                .all(|p| (p.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12)
        );
        let front = Dtlz1::<4>::default().optimal_front(20);
        assert!(front.len() >= 20);
        assert!(
            front
                .iter()
                .all(|p| (p.iter().sum::<f64>() - 0.5).abs() < 1e-12)
        );
        assert_eq!(Dtlz1::<3>::default().real().bounds().len(), 7);
        assert_eq!(Dtlz2::<5>::default().real().bounds().len(), 14);
    }
}
