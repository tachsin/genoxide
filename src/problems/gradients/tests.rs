use crate::engine::{Extras, FitnessFunction, Provided};
use crate::genome::{Reals, Representation};
use crate::gradient;
use crate::problems::classic::{WEIERSTRASS_B, WEIERSTRASS_TERMS, weierstrass_sum};
use crate::problems::{
    self, BentCigar, BucheRastrigin, DifferentPowers, Discus, DynProblem, HappyCat, HgBat,
    HighConditionedElliptic, Katsuura, NonContinuousRastrigin, Penalized1, Penalized2, Problem,
    Quartic, RotatedHyperEllipsoid, SchafferF7, Step, SumOfDifferentPowers, Weierstrass,
};
use crate::{Fitness, StreamRng};
use std::f64::consts::PI;

// the problems with an analytic gradient: every classic function but those whose derivative is
// undefined or 0 on sets of positive measure, or meaningless: a maximum and absolute values (the
// eggholder, Schwefel 2.21 and 2.22), flat steps (the step function, the non-continuous
// Rastrigin) and kinks every 2⁻³³ (Katsuura)
const DIFFERENTIABLE: [&str; 50] = [
    "Sphere",
    "AxisParallelEllipsoid",
    "Schwefel1_2",
    "Rastrigin",
    "Rosenbrock",
    "Ackley",
    "Griewank",
    "Schwefel2_26",
    "Levy",
    "Zakharov",
    "StyblinskiTang",
    "Michalewicz",
    "Himmelblau",
    "Branin",
    "GoldsteinPrice",
    "SixHumpCamel",
    "Hartmann3",
    "Hartmann6",
    "Shekel5",
    "Shekel7",
    "Shekel10",
    "Easom",
    "SchafferF6",
    "DixonPrice",
    "Trid",
    "Powell",
    "Beale",
    "Booth",
    "Matyas",
    "Bohachevsky1",
    "Bohachevsky2",
    "Bohachevsky3",
    "ThreeHumpCamel",
    "Langermann",
    "ShekelFoxholes",
    "Kowalik",
    "SumOfDifferentPowers",
    "Quartic",
    "Penalized1",
    "Penalized2",
    "HighConditionedElliptic",
    "BentCigar",
    "Discus",
    "DifferentPowers",
    "BucheRastrigin",
    "Weierstrass",
    "HappyCat",
    "HgBat",
    "SchafferF7",
    "RotatedHyperEllipsoid",
];

// the problems whose minimum is on a kink, where central differences straddle it and the
// gradient is 0 by convention: Büche-Rastrigin's (whose T_osz has no derivative at 0), HappyCat's
// and HGBat's (cusps, on a sphere and two) and Schaffer F7's (a cusp of each pair)
const KINKED_AT_THE_OPTIMUM: [&str; 4] = ["BucheRastrigin", "HappyCat", "HgBat", "SchafferF7"];

// the analytic gradient of `problem` at `x` against central differences: the value is the plain
// evaluation's, to the bit, and the gradient's error is within 1e-5 relative (absolute below 1),
// plus the error of the differences themselves: their rounding error, about ε |f| / h, which
// dominates where the value is large against the gradient (Rosenbrock's corners), and their
// truncation error, estimated by the change from the step h to 2h, which dominates near
// Kowalik's poles. Returns the error as a fraction of what's allowed: at most 1
fn error(problem: &dyn DynProblem, x: &Reals) -> f64 {
    let mut gradient = vec![f64::NAN; x.len()];
    let fitness = problem.evaluate_with(x, &mut Extras::with_gradient(&mut gradient));
    assert_eq!(fitness, problem.evaluate(x), "{}", problem.name());
    let value = fitness.score().expect("valid").abs().max(1.0);
    let score = |genome: &Reals| problem.evaluate(genome).score().expect("valid");
    let mut point = x.clone();
    let mut central = |gene: usize, h: f64| {
        let xi = x[gene];
        point[gene] = xi + h;
        let above = score(&point);
        point[gene] = xi - h;
        let below = score(&point);
        point[gene] = xi;
        (above - below) / (2.0 * h)
    };
    let mut largest: f64 = 0.0;
    for (gene, &g) in gradient.iter().enumerate() {
        // a power of 2, so that x ± h and x ± 2h are exact
        let h = exp2_floor(gradient::CENTRAL_STEP * x[gene].abs().max(1.0));
        let (estimate, coarse) = (central(gene, h), central(gene, 2.0 * h));
        let allowed = 1e-5 * g.abs().max(estimate.abs()).max(1.0)
            + 1e2 * f64::EPSILON * value / h
            + (estimate - coarse).abs();
        let error = (g - estimate).abs() / allowed;
        assert!(!error.is_nan(), "{}: {x:?}", problem.name());
        largest = largest.max(error);
    }
    largest
}

// the largest power of 2 at most `h`, which is positive and normal
fn exp2_floor(h: f64) -> f64 {
    f64::from_bits(h.to_bits() & 0xfff0_0000_0000_0000)
}

#[test]
fn the_smooth_problems_have_gradients() {
    let names: Vec<&str> = problems::all()
        .iter()
        .filter(|problem| problem.provides().gradient)
        .map(|problem| problem.name())
        .collect();
    assert_eq!(names, DIFFERENTIABLE);
}

#[test]
fn gradients_match_central_differences_at_random_points() {
    let mut rng = StreamRng::seed_from_u64(3);
    for problem in problems::all() {
        // Weierstrass's highest terms are beyond differences (see its own test below)
        if !problem.provides().gradient || problem.name() == "Weierstrass" {
            continue;
        }
        let real = problem.real();
        let mut largest: f64 = 0.0;
        for _ in 0..200 {
            let x = real.random_genome(&mut rng);
            largest = largest.max(error(problem.as_ref(), &x));
        }
        assert!(largest <= 1.0, "{}: {largest}", problem.name());
    }
}

// at the optimum, where the gradient is 0, and near it, where it's small against the value
#[test]
fn gradients_match_central_differences_near_the_optimum() {
    let mut rng = StreamRng::seed_from_u64(4);
    for problem in problems::all() {
        if !problem.provides().gradient {
            continue;
        }
        let real = problem.real();
        let optimum = problem.optimum().expect("known");
        // Weierstrass's highest terms are beyond differences (see its own test below)
        let differences = problem.name() != "Weierstrass";
        let mut largest: f64 = 0.0;
        for solution in optimum.solutions() {
            if differences && !KINKED_AT_THE_OPTIMUM.contains(&problem.name()) {
                largest = largest.max(error(problem.as_ref(), solution));
            }
            // the optima are interior minima, rounded to the precision of f64: a gradient of
            // about 1e-14, but Michalewicz's 2e-6, whose minimizers are found by golden-section
            // search on terms as steep as sin²⁰
            let mut gradient = vec![0.0; solution.len()];
            problem.evaluate_with(solution, &mut Extras::with_gradient(&mut gradient));
            let norm = gradient.iter().fold(0.0, |norm: f64, g| norm.max(g.abs()));
            assert!(norm <= 1e-5, "{}: {norm}", problem.name());
            for _ in 0..if differences { 20 } else { 0 } {
                // about a thousandth of each range away, inside the bounds
                let x: Reals = solution
                    .iter()
                    .zip(real.bounds())
                    .map(|(&xi, range)| {
                        let width = range.end() - range.start();
                        // from half to all of a thousandth: away from a kink at the optimum, where
                        // central differences straddle it
                        let u = rng.unit_f64() * 2.0 - 1.0;
                        let offset = (0.5 + 0.5 * u.abs()) * u.signum() * 1e-3 * width;
                        (xi + offset).clamp(*range.start(), *range.end())
                    })
                    .collect();
                largest = largest.max(error(problem.as_ref(), &x));
            }
        }
        assert!(largest <= 1.0, "{}: {largest}", problem.name());
    }
}

// a problem behind `DynProblem` as a fitness function, for `gradient::check`
struct Function<'a>(&'a dyn DynProblem);

impl FitnessFunction<Reals> for Function<'_> {
    type Output = Fitness;

    fn evaluate(&self, x: &Reals) -> Fitness {
        self.0.evaluate(x)
    }

    fn provides(&self) -> Provided {
        self.0.provides()
    }

    fn evaluate_with(&self, x: &Reals, extras: &mut Extras<'_>) -> Fitness {
        self.0.evaluate_with(x, extras)
    }
}

// the CEC- and BBOB-style functions with a gradient that `gradient::check`'s differences resolve:
// all but Büche-Rastrigin and Weierstrass, whose ripples need finer steps (below)
fn checked() -> Vec<Box<dyn DynProblem>> {
    vec![
        problems::boxed(SumOfDifferentPowers::default()),
        problems::boxed(Quartic::default()),
        problems::boxed(Penalized1::default()),
        problems::boxed(Penalized2::default()),
        problems::boxed(HighConditionedElliptic::default()),
        problems::boxed(BentCigar::default()),
        problems::boxed(Discus::default()),
        problems::boxed(DifferentPowers::default()),
        problems::boxed(HappyCat::default()),
        problems::boxed(HgBat::default()),
        problems::boxed(SchafferF7::default()),
        problems::boxed(RotatedHyperEllipsoid::default()),
    ]
}

// `gradient::check` at 200 random points of the box, which are away from the kinks (a set of
// measure 0) almost surely: every gene within 5e-8, relative (absolute below 1), of central
// differences, beyond the differences' own rounding error, about ε |f| / h, which is the larger
// where the value dwarfs a gene's slope (the bent cigar's first gene, the penalties' far genes)
#[test]
fn the_cec_and_bbob_style_gradients_pass_gradient_check() {
    let mut rng = StreamRng::seed_from_u64(5);
    for problem in checked() {
        let real = problem.real();
        let mut largest: f64 = 0.0;
        for _ in 0..200 {
            let x = real.random_genome(&mut rng);
            let check = gradient::check(&Function(problem.as_ref()), &x).expect("a gradient");
            let value = problem.evaluate(&x).score().expect("valid").abs().max(1.0);
            for (gene, &error) in check.errors().iter().enumerate() {
                let h = gradient::CENTRAL_STEP * x[gene].abs().max(1.0);
                let scale = check.supplied()[gene]
                    .abs()
                    .max(check.estimated()[gene].abs())
                    .max(1.0);
                let rounding = 1e2 * f64::EPSILON * value / (h * scale);
                largest = largest.max(error - rounding);
            }
        }
        assert!(largest <= 5e-8, "{}: {largest}", problem.name());
    }
}

// the slope of `f` at `x` by Richardson's extrapolation of central differences with the steps h
// and h / 2, whose error is O(h⁴) rather than O(h²)
fn richardson(f: impl Fn(f64) -> f64, x: f64, h: f64) -> f64 {
    let central = |h: f64| {
        let (above, below) = (x + h, x - h);
        (f(above) - f(below)) / (above - below)
    };
    (4.0 * central(h / 2.0) - central(h)) / 3.0
}

// Büche-Rastrigin's rings are up to 2000 radians per unit of a gene, and its oscillation T_osz
// wiggles at 10 / |x| near 0: central differences at gradient::check's steps are off by 1e-6 to
// 1e-4 there. A gene's slope, with the others fixed, by Richardson's extrapolation at steps of
// 2⁻²⁰ is within 1e-8 (relative, absolute below 1), beyond the rounding error
#[test]
fn the_buche_rastrigin_gradient_matches_finer_differences() {
    let problem = BucheRastrigin::new(10);
    let real = problem.representation();
    let mut rng = StreamRng::seed_from_u64(6);
    let mut largest: f64 = 0.0;
    for _ in 0..200 {
        let x = real.random_genome(&mut rng);
        let mut gradient = vec![0.0; x.len()];
        let value = problem.evaluate_with(&x, &mut Extras::with_gradient(&mut gradient));
        assert_eq!(value.to_bits(), problem.evaluate(&x).to_bits());
        for (gene, &g) in gradient.iter().enumerate() {
            let h = exp2_floor(x[gene].abs().max(1.0)) * 2f64.powi(-20);
            let along = |xi: f64| {
                let mut point = x.clone();
                point[gene] = xi;
                problem.evaluate(&point)
            };
            let estimate = richardson(along, x[gene], h);
            let scale = g.abs().max(estimate.abs()).max(1.0);
            let rounding = 1e2 * f64::EPSILON * value.abs().max(1.0) / h;
            largest = largest.max(((g - estimate).abs() - rounding) / scale);
        }
    }
    assert!(largest <= 1e-8, "{largest}");
}

// Weierstrass's sum has terms up to 3²⁰ times the first's frequency, and the computed phase
// 2π 3ᵏ (x + 0.5) rounds by up to about ε 2π 3ᵏ, 2·10⁻⁶ rad for the last term: no differences of
// the computed function resolve the derivative of its highest terms. Its slope is checked on the
// sums of its first 1 to 7 terms (to 3⁶ times the first frequency; the phases' rounding shows
// from 3⁷ on, 3e-8 at 3⁷, 6e-8 at 3⁹, 4e-6 at 3¹⁰), with Richardson's extrapolation at steps of
// a hundredth of a radian of the highest term: within 1e-8 (relative, absolute below 1). Every
// term has the same formula, and the gradient is the slope of the whole sum at each gene
#[test]
fn the_weierstrass_gradient_matches_differences_of_its_partial_sums() {
    let mut rng = StreamRng::seed_from_u64(7);
    let mut largest: f64 = 0.0;
    for terms in 1..=7 {
        let frequency = 2.0 * PI * WEIERSTRASS_B.powi(terms - 1);
        let h = exp2_floor(0.01 / frequency);
        for _ in 0..200 {
            let x = rng.unit_f64() - 0.5;
            let slope = super::weierstrass_slope(x, terms);
            let estimate = richardson(|x| weierstrass_sum(x, terms), x, h);
            let error = (slope - estimate).abs() / slope.abs().max(estimate.abs()).max(1.0);
            largest = largest.max(error);
        }
    }
    assert!(largest <= 1e-8, "{largest}");
    // the whole function's gradient: each gene's slope of all 21 terms
    let problem = Weierstrass::new(5);
    let x = problem.representation().random_genome(&mut rng);
    let mut gradient = vec![0.0; 5];
    let value = problem.evaluate_with(&x, &mut Extras::with_gradient(&mut gradient));
    assert_eq!(value.to_bits(), problem.evaluate(&x).to_bits());
    for (g, &xi) in gradient.iter().zip(x.iter()) {
        assert_eq!(*g, super::weierstrass_slope(xi, WEIERSTRASS_TERMS));
    }
    // 0 at the minimum, where every sin(2π 3ᵏ x) is 0
    problem.evaluate_with(
        &Reals::from(vec![0.0; 5]),
        &mut Extras::with_gradient(&mut gradient),
    );
    assert_eq!(gradient, [0.0; 5]);
}

// the noisy quartic's value jumps between any two genomes: it has no gradient
#[test]
fn the_noisy_quartic_has_no_gradient() {
    assert!(Quartic::new(3).provides().gradient);
    assert!(!Quartic::noisy(3).provides().gradient);
    let x = Reals::from(vec![0.5, -0.25, 1.0]);
    let mut gradient = [7.0; 3];
    let noisy = Quartic::noisy(3);
    let value = noisy.evaluate_with(&x, &mut Extras::with_gradient(&mut gradient));
    assert_eq!(value.to_bits(), noisy.evaluate(&x).to_bits());
    assert_eq!(gradient, [7.0; 3]);
    // without noise, 4 i xᵢ³
    Quartic::new(3).evaluate_with(&x, &mut Extras::with_gradient(&mut gradient));
    assert_eq!(gradient, [0.5, -0.125, 12.0]);
}

// the functions whose derivative is undefined or 0 on sets of positive measure provide none
#[test]
fn the_flat_and_rugged_functions_have_no_gradient() {
    assert!(!Step::default().provides().gradient);
    assert!(!NonContinuousRastrigin::default().provides().gradient);
    assert!(!Katsuura::default().provides().gradient);
}

// at the kinks, each gradient's term with the kink is 0: the cones' and cusps' apexes
#[test]
fn gradients_at_the_kinks() {
    let gradient_at = |problem: Box<dyn DynProblem>, x: &[f64]| {
        let mut gradient = vec![f64::NAN; x.len()];
        problem.evaluate_with(
            &Reals::from(x.to_vec()),
            &mut Extras::with_gradient(&mut gradient),
        );
        gradient
    };
    // the apexes are the minima
    assert_eq!(
        gradient_at(problems::boxed(DifferentPowers::new(3)), &[0.0; 3]),
        [0.0; 3]
    );
    assert_eq!(
        gradient_at(problems::boxed(BucheRastrigin::new(3)), &[0.0; 3]),
        [0.0; 3]
    );
    assert_eq!(
        gradient_at(problems::boxed(HappyCat::new(3)), &[-1.0; 3]),
        [0.0; 3]
    );
    assert_eq!(
        gradient_at(problems::boxed(HgBat::new(3)), &[-1.0; 3]),
        [0.0; 3]
    );
    assert_eq!(
        gradient_at(problems::boxed(SchafferF7::new(3)), &[0.0; 3]),
        [0.0; 3]
    );
    // on HappyCat's sphere ‖x‖² = n elsewhere, only the slope (xᵢ + 1) / n is left
    assert_eq!(
        gradient_at(problems::boxed(HappyCat::new(2)), &[1.0, -1.0]),
        [1.0, 0.0]
    );
    // Schaffer F7's pair (0, 0) adds nothing, the pair (0, 3) its slope along the third gene
    let schaffer = gradient_at(problems::boxed(SchafferF7::new(3)), &[0.0, 0.0, 3.0]);
    assert_eq!(schaffer[0], 0.0);
    assert!(schaffer[1] == 0.0 && schaffer[2] != 0.0);
}
