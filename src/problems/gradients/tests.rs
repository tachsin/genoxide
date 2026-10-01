use crate::StreamRng;
use crate::engine::Extras;
use crate::genome::{Reals, Representation};
use crate::gradient;
use crate::problems::{self, DynProblem};

// the problems with an analytic gradient: every classic function but the three that aren't
// differentiable (a maximum, absolute values)
const DIFFERENTIABLE: [&str; 36] = [
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
];

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
        if !problem.provides().gradient {
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
        let mut largest: f64 = 0.0;
        for solution in optimum.solutions() {
            largest = largest.max(error(problem.as_ref(), solution));
            // the optima are interior minima, rounded to the precision of f64: a gradient of
            // about 1e-14, but Michalewicz's 2e-6, whose minimizers are found by golden-section
            // search on terms as steep as sin²⁰
            let mut gradient = vec![0.0; solution.len()];
            problem.evaluate_with(solution, &mut Extras::with_gradient(&mut gradient));
            let norm = gradient.iter().fold(0.0, |norm: f64, g| norm.max(g.abs()));
            assert!(norm <= 1e-5, "{}: {norm}", problem.name());
            for _ in 0..20 {
                // a thousandth of each range away, inside the bounds
                let x: Reals = solution
                    .iter()
                    .zip(real.bounds())
                    .map(|(&xi, range)| {
                        let width = range.end() - range.start();
                        let offset = (rng.unit_f64() * 2.0 - 1.0) * 1e-3 * width;
                        (xi + offset).clamp(*range.start(), *range.end())
                    })
                    .collect();
                largest = largest.max(error(problem.as_ref(), &x));
            }
        }
        assert!(largest <= 1.0, "{}: {largest}", problem.name());
    }
}
