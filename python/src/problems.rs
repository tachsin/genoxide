//! The test problems of `genoxide::problems` and `genoxide::multi::problems`, as the Python
//! package describes them in JSON: their description, their evaluation, and the problem of a run
//! evaluated in Rust.

use crate::run::{WithObjectives, with_objectives};
use genoxide::engine::{FitnessFunction, IntoFitness};
use genoxide::genome::{Binary, Bits, Integer, Integers, Real, Reals, Representation};
use genoxide::multi::problems::{self as multi, DynMultiProblem, MultiProblem, try_boxed};
use genoxide::multi::{IntoScores, MultiFitnessFunction, Scores};
use genoxide::problems::{
    self, Constraints, DynProblem, Optimum, Problem as _, cec2006, engineering,
};
use genoxide::{Fitness, Objective};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde::Deserialize;

/// A problem, as `_describe()` of a `gx.problems` class gives it.
#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum Config {
    Sphere {
        dimensions: usize,
    },
    AxisParallelEllipsoid {
        dimensions: usize,
    },
    #[serde(rename = "schwefel_1_2")]
    Schwefel12 {
        dimensions: usize,
    },
    Rastrigin {
        dimensions: usize,
    },
    Rosenbrock {
        dimensions: usize,
    },
    Ackley {
        dimensions: usize,
    },
    Griewank {
        dimensions: usize,
    },
    #[serde(rename = "schwefel_2_26")]
    Schwefel226 {
        dimensions: usize,
    },
    Levy {
        dimensions: usize,
    },
    Zakharov {
        dimensions: usize,
    },
    StyblinskiTang {
        dimensions: usize,
    },
    Michalewicz {
        dimensions: usize,
    },
    Himmelblau {},
    Branin {},
    GoldsteinPrice {},
    SixHumpCamel {},
    G01 {},
    G02 {},
    G03 {
        tolerance: Option<f64>,
    },
    G04 {},
    G05 {
        tolerance: Option<f64>,
    },
    G06 {},
    WeldedBeam {},
    WeldedBeamRagsdell {},
    PressureVessel {},
    TensionCompressionSpring {},
    SpeedReducer {},
    GearTrain {},
    ThreeBarTruss {},
    CantileverBeam {},
    CarSideImpact {},
    #[serde(untagged)]
    Multi(MultiConfig),
}

/// A multi-objective problem, as `_describe()` of a `gx.problems` class gives it.
#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum MultiConfig {
    Zdt1 {
        variables: usize,
    },
    Zdt2 {
        variables: usize,
    },
    Zdt3 {
        variables: usize,
    },
    Zdt4 {
        variables: usize,
    },
    Zdt6 {
        variables: usize,
    },
    Zdt5 {
        first_bits: usize,
        substrings: usize,
    },
    Dtlz1 {
        objectives: usize,
        variables: Option<usize>,
    },
    Dtlz2 {
        objectives: usize,
        variables: Option<usize>,
    },
    Dtlz3 {
        objectives: usize,
        variables: Option<usize>,
    },
    Dtlz4 {
        objectives: usize,
        variables: Option<usize>,
    },
    Dtlz5 {
        objectives: usize,
        variables: Option<usize>,
    },
    Dtlz6 {
        objectives: usize,
        variables: Option<usize>,
    },
    Dtlz7 {
        objectives: usize,
        variables: Option<usize>,
    },
    Wfg1 {
        objectives: usize,
        position: Option<usize>,
        distance: usize,
    },
    Wfg2 {
        objectives: usize,
        position: Option<usize>,
        distance: usize,
    },
    Wfg3 {
        objectives: usize,
        position: Option<usize>,
        distance: usize,
    },
    Wfg4 {
        objectives: usize,
        position: Option<usize>,
        distance: usize,
    },
    Wfg5 {
        objectives: usize,
        position: Option<usize>,
        distance: usize,
    },
    Wfg6 {
        objectives: usize,
        position: Option<usize>,
        distance: usize,
    },
    Wfg7 {
        objectives: usize,
        position: Option<usize>,
        distance: usize,
    },
    Wfg8 {
        objectives: usize,
        position: Option<usize>,
        distance: usize,
    },
    Wfg9 {
        objectives: usize,
        position: Option<usize>,
        distance: usize,
    },
    Schaffer1 {},
    Schaffer2 {},
    FonsecaFleming {
        variables: usize,
    },
    Kursawe {
        variables: usize,
    },
    Poloni {},
    Viennet1 {},
    Viennet2 {},
    Viennet3 {},
    Bnh {},
    Srn {},
    Tnk {},
    Osy {},
    Constr {},
}

// a size of at least `minimum`, or an error that names it; the constructors panic instead
fn at_least(value: usize, minimum: usize, name: &str, what: &str) -> Result<usize, String> {
    if value >= minimum {
        Ok(value)
    } else {
        Err(format!(
            "{name} needs at least {minimum} {what}, not {value}"
        ))
    }
}

// WFG's sizes, checked: 2 to 6 objectives, a positive multiple of objectives − 1 position
// parameters, and at least 1 distance parameter, an even number for WFG2 and WFG3
fn check_wfg(
    name: &str,
    objectives: usize,
    position: Option<usize>,
    distance: usize,
    even: bool,
) -> Result<(), String> {
    if !(2..=6).contains(&objectives) {
        return Err(format!(
            "WFG takes 2 to 6 objectives in Python, not {objectives}"
        ));
    }
    if position.is_some_and(|position| position == 0 || position % (objectives - 1) != 0) {
        return Err(format!(
            "{name} needs a positive multiple of {} (objectives − 1) position parameters, not {}",
            objectives - 1,
            position.unwrap_or_default()
        ));
    }
    if distance == 0 {
        return Err(format!("{name} needs at least 1 distance parameter"));
    }
    if even && distance % 2 == 1 {
        return Err(format!(
            "{name} needs an even number of distance parameters, not {distance}"
        ));
    }
    Ok(())
}

// a WFG problem with M objectives, `position` position parameters (the default if None) and
// `distance` distance parameters
macro_rules! wfg {
    ($problem:ident, $position:expr, $distance:expr) => {{
        let position = $position.unwrap_or_else(|| multi::$problem::<M>::default().position());
        Some(multi::boxed(multi::$problem::<M>::new(position, $distance)))
    }};
}

impl MultiConfig {
    /// The number of objectives.
    pub fn objectives(&self) -> usize {
        match *self {
            Self::Dtlz1 { objectives, .. }
            | Self::Dtlz2 { objectives, .. }
            | Self::Dtlz3 { objectives, .. }
            | Self::Dtlz4 { objectives, .. }
            | Self::Dtlz5 { objectives, .. }
            | Self::Dtlz6 { objectives, .. }
            | Self::Dtlz7 { objectives, .. }
            | Self::Wfg1 { objectives, .. }
            | Self::Wfg2 { objectives, .. }
            | Self::Wfg3 { objectives, .. }
            | Self::Wfg4 { objectives, .. }
            | Self::Wfg5 { objectives, .. }
            | Self::Wfg6 { objectives, .. }
            | Self::Wfg7 { objectives, .. }
            | Self::Wfg8 { objectives, .. }
            | Self::Wfg9 { objectives, .. } => objectives,
            Self::Viennet1 {} | Self::Viennet2 {} | Self::Viennet3 {} => 3,
            _ => 2,
        }
    }

    // the sizes, checked: an error, not the panic of the constructor
    fn check(&self) -> Result<(), String> {
        let variables = |variables: usize, minimum: usize, name: &str| {
            at_least(variables, minimum, name, "variables").map(|_| ())
        };
        match *self {
            Self::Zdt1 { variables: n } => variables(n, 2, "ZDT1"),
            Self::Zdt2 { variables: n } => variables(n, 2, "ZDT2"),
            Self::Zdt3 { variables: n } => variables(n, 2, "ZDT3"),
            Self::Zdt4 { variables: n } => variables(n, 2, "ZDT4"),
            Self::Zdt6 { variables: n } => variables(n, 2, "ZDT6"),
            Self::FonsecaFleming { variables: n } => variables(n, 1, "FON"),
            Self::Kursawe { variables: n } => variables(n, 2, "KUR"),
            Self::Dtlz1 {
                objectives,
                variables: n,
            }
            | Self::Dtlz2 {
                objectives,
                variables: n,
            }
            | Self::Dtlz3 {
                objectives,
                variables: n,
            }
            | Self::Dtlz4 {
                objectives,
                variables: n,
            }
            | Self::Dtlz5 {
                objectives,
                variables: n,
            }
            | Self::Dtlz6 {
                objectives,
                variables: n,
            }
            | Self::Dtlz7 {
                objectives,
                variables: n,
            } => {
                if !(2..=6).contains(&objectives) {
                    return Err(format!(
                        "DTLZ takes 2 to 6 objectives in Python, not {objectives}"
                    ));
                }
                match n {
                    Some(n) => variables(n, objectives, "DTLZ with this many objectives"),
                    None => Ok(()),
                }
            }
            Self::Zdt5 {
                first_bits,
                substrings,
            } => {
                at_least(first_bits, 1, "ZDT5", "bits in x₁")?;
                at_least(substrings, 1, "ZDT5", "substrings")?;
                let bits = substrings
                    .checked_mul(5)
                    .and_then(|bits| bits.checked_add(first_bits));
                match bits {
                    Some(bits) if bits <= 1 << 24 => Ok(()),
                    _ => Err("ZDT5's genome has at most 2^24 bits".to_string()),
                }
            }
            Self::Wfg1 {
                objectives,
                position,
                distance,
            } => check_wfg("WFG1", objectives, position, distance, false),
            Self::Wfg2 {
                objectives,
                position,
                distance,
            } => check_wfg("WFG2", objectives, position, distance, true),
            Self::Wfg3 {
                objectives,
                position,
                distance,
            } => check_wfg("WFG3", objectives, position, distance, true),
            Self::Wfg4 {
                objectives,
                position,
                distance,
            } => check_wfg("WFG4", objectives, position, distance, false),
            Self::Wfg5 {
                objectives,
                position,
                distance,
            } => check_wfg("WFG5", objectives, position, distance, false),
            Self::Wfg6 {
                objectives,
                position,
                distance,
            } => check_wfg("WFG6", objectives, position, distance, false),
            Self::Wfg7 {
                objectives,
                position,
                distance,
            } => check_wfg("WFG7", objectives, position, distance, false),
            Self::Wfg8 {
                objectives,
                position,
                distance,
            } => check_wfg("WFG8", objectives, position, distance, false),
            Self::Wfg9 {
                objectives,
                position,
                distance,
            } => check_wfg("WFG9", objectives, position, distance, false),
            _ => Ok(()),
        }
    }

    /// The problem, with `M` objectives: its number of objectives, checked before.
    pub fn build<const M: usize>(&self) -> MultiNative<M> {
        if let Self::Zdt5 {
            first_bits,
            substrings,
        } = *self
        {
            let problem = try_binary::<_, 2, M>(multi::Zdt5::new(first_bits, substrings));
            return MultiNative::Binary(
                problem.expect("the number of objectives is the problem's"),
            );
        }
        let problem = match *self {
            Self::Zdt1 { variables } => try_boxed::<_, 2, M>(multi::Zdt1::new(variables)),
            Self::Zdt2 { variables } => try_boxed::<_, 2, M>(multi::Zdt2::new(variables)),
            Self::Zdt3 { variables } => try_boxed::<_, 2, M>(multi::Zdt3::new(variables)),
            Self::Zdt4 { variables } => try_boxed::<_, 2, M>(multi::Zdt4::new(variables)),
            Self::Zdt6 { variables } => try_boxed::<_, 2, M>(multi::Zdt6::new(variables)),
            Self::Dtlz1 { variables, .. } => Some(multi::boxed(
                variables.map_or_else(multi::Dtlz1::<M>::default, multi::Dtlz1::<M>::new),
            )),
            Self::Dtlz2 { variables, .. } => Some(multi::boxed(
                variables.map_or_else(multi::Dtlz2::<M>::default, multi::Dtlz2::<M>::new),
            )),
            Self::Dtlz3 { variables, .. } => Some(multi::boxed(
                variables.map_or_else(multi::Dtlz3::<M>::default, multi::Dtlz3::<M>::new),
            )),
            Self::Dtlz4 { variables, .. } => Some(multi::boxed(
                variables.map_or_else(multi::Dtlz4::<M>::default, multi::Dtlz4::<M>::new),
            )),
            Self::Dtlz5 { variables, .. } => Some(multi::boxed(
                variables.map_or_else(multi::Dtlz5::<M>::default, multi::Dtlz5::<M>::new),
            )),
            Self::Dtlz6 { variables, .. } => Some(multi::boxed(
                variables.map_or_else(multi::Dtlz6::<M>::default, multi::Dtlz6::<M>::new),
            )),
            Self::Dtlz7 { variables, .. } => Some(multi::boxed(
                variables.map_or_else(multi::Dtlz7::<M>::default, multi::Dtlz7::<M>::new),
            )),
            Self::Wfg1 {
                position, distance, ..
            } => wfg!(Wfg1, position, distance),
            Self::Wfg2 {
                position, distance, ..
            } => wfg!(Wfg2, position, distance),
            Self::Wfg3 {
                position, distance, ..
            } => wfg!(Wfg3, position, distance),
            Self::Wfg4 {
                position, distance, ..
            } => wfg!(Wfg4, position, distance),
            Self::Wfg5 {
                position, distance, ..
            } => wfg!(Wfg5, position, distance),
            Self::Wfg6 {
                position, distance, ..
            } => wfg!(Wfg6, position, distance),
            Self::Wfg7 {
                position, distance, ..
            } => wfg!(Wfg7, position, distance),
            Self::Wfg8 {
                position, distance, ..
            } => wfg!(Wfg8, position, distance),
            Self::Wfg9 {
                position, distance, ..
            } => wfg!(Wfg9, position, distance),
            Self::Schaffer1 {} => try_boxed::<_, 2, M>(multi::Schaffer1),
            Self::Schaffer2 {} => try_boxed::<_, 2, M>(multi::Schaffer2),
            Self::FonsecaFleming { variables } => {
                try_boxed::<_, 2, M>(multi::FonsecaFleming::new(variables))
            }
            Self::Kursawe { variables } => try_boxed::<_, 2, M>(multi::Kursawe::new(variables)),
            Self::Poloni {} => try_boxed::<_, 2, M>(multi::Poloni),
            Self::Viennet1 {} => try_boxed::<_, 3, M>(multi::Viennet1),
            Self::Viennet2 {} => try_boxed::<_, 3, M>(multi::Viennet2),
            Self::Viennet3 {} => try_boxed::<_, 3, M>(multi::Viennet3),
            Self::Bnh {} => try_boxed::<_, 2, M>(multi::Bnh),
            Self::Srn {} => try_boxed::<_, 2, M>(multi::Srn),
            Self::Tnk {} => try_boxed::<_, 2, M>(multi::Tnk),
            Self::Osy {} => try_boxed::<_, 2, M>(multi::Osy),
            Self::Constr {} => try_boxed::<_, 2, M>(multi::Constr),
            // built above
            Self::Zdt5 { .. } => None,
        };
        MultiNative::Real(problem.expect("the number of objectives is the problem's"))
    }
}

/// A multi-objective problem with `M` objectives, evaluated in Rust: on real genomes, or on bit
/// strings.
pub enum MultiNative<const M: usize> {
    Real(Box<dyn DynMultiProblem<M>>),
    Binary(Box<dyn BinaryMultiProblem<M>>),
}

/// A multi-objective problem on [`Binary`] genomes, as a trait object: what [`DynMultiProblem`]
/// is for real genomes.
pub trait BinaryMultiProblem<const M: usize>: Send + Sync {
    fn name(&self) -> &'static str;
    fn binary(&self) -> Binary;
    fn evaluate(&self, genome: &Bits) -> Scores<M>;
    fn reference(&self) -> &'static str;
    fn reference_url(&self) -> Option<&'static str>;
    fn constraint_count(&self) -> usize;
    fn constraints(&self, genome: &Bits) -> Constraints;
    fn optimal_front(&self, points: usize) -> Option<Vec<[f64; M]>>;
    fn ideal_point(&self) -> Option<[f64; M]>;
    fn nadir_point(&self) -> Option<[f64; M]>;
}

// a problem with K objectives behind `BinaryMultiProblem<M>`, built only when M = K
struct BinaryBoxed<P, const K: usize>(P);

// the values of an array of K values as one of M, when M = K
fn resized<const K: usize, const M: usize>(values: [f64; K]) -> [f64; M] {
    std::array::from_fn(|i| values[i])
}

impl<P, const K: usize, const M: usize> BinaryMultiProblem<M> for BinaryBoxed<P, K>
where
    P: MultiProblem<K, Representation = Binary> + Send + Sync,
{
    fn name(&self) -> &'static str {
        self.0.name()
    }

    fn binary(&self) -> Binary {
        self.0.representation()
    }

    fn evaluate(&self, genome: &Bits) -> Scores<M> {
        let scores: Scores<K> = MultiFitnessFunction::evaluate(&self.0, genome)
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

    fn constraints(&self, genome: &Bits) -> Constraints {
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

// `problem`, with K objectives, as a `BinaryMultiProblem<M>` if M = K
fn try_binary<P, const K: usize, const M: usize>(
    problem: P,
) -> Option<Box<dyn BinaryMultiProblem<M>>>
where
    P: MultiProblem<K, Representation = Binary> + Send + Sync + 'static,
{
    (K == M).then(|| Box::new(BinaryBoxed::<P, K>(problem)) as Box<dyn BinaryMultiProblem<M>>)
}

// what both kinds of problems have, from the one or the other
macro_rules! either {
    ($problem:expr, $p:ident => $value:expr) => {
        match $problem {
            MultiNative::Real($p) => $value,
            MultiNative::Binary($p) => $value,
        }
    };
}

impl<const M: usize> MultiNative<M> {
    pub fn name(&self) -> &'static str {
        either!(self, p => p.name())
    }

    /// The number of genes: variables, or bits.
    pub fn genome_len(&self) -> usize {
        match self {
            Self::Real(p) => p.real().genome_len(),
            Self::Binary(p) => p.binary().genome_len(),
        }
    }

    /// Whether its genomes are bit strings.
    pub fn is_binary(&self) -> bool {
        matches!(self, Self::Binary(_))
    }

    fn constraint_count(&self) -> usize {
        either!(self, p => p.constraint_count())
    }

    /// The scores of `genome`, or invalid ones for a genome of the other kind (the run checks
    /// that the genome is the problem's).
    pub fn evaluate<G: crate::genes::Genes>(&self, genome: &G) -> Scores<M> {
        let scores = match self {
            Self::Real(p) => genome.reals().map(|genome| p.evaluate(genome)),
            Self::Binary(p) => genome.bits().map(|genome| p.evaluate(genome)),
        };
        scores.unwrap_or_else(Scores::invalid)
    }

    // a genome per row, as numpy arrays of the problem's number of genes: reals, or bits
    fn rows(&self, genomes: &PyReadonlyArray2<'_, f64>) -> PyResult<Genomes> {
        Ok(match self {
            Self::Real(p) => Genomes::Real(rows(p.name(), &p.real(), genomes)?),
            Self::Binary(p) => Genomes::Binary(bit_rows(p.name(), &p.binary(), genomes)?),
        })
    }
}

// the genomes of a problem's rows
enum Genomes {
    Real(Vec<Reals>),
    Binary(Vec<Bits>),
}

/// The name and the number of genes of the problem that `config` describes, and whether its
/// genomes are bit strings.
pub fn name_and_dimensions(config: MultiConfig) -> (&'static str, usize, bool) {
    struct Describe(MultiConfig);

    impl WithObjectives for Describe {
        type Output = (&'static str, usize, bool);

        fn with<const N: usize>(self) -> (&'static str, usize, bool) {
            let problem = self.0.build::<N>();
            (problem.name(), problem.genome_len(), problem.is_binary())
        }
    }

    with_objectives(config.objectives(), Describe(config)).expect("checked")
}

/// A problem that a run evaluates in Rust.
pub enum Problem {
    Single(Box<dyn DynProblem>),
    /// A single-objective problem on integer genomes.
    Integer(Box<dyn IntegerProblem>),
    Multi(MultiConfig),
}

/// A single-objective problem on [`Integer`] genomes, as a trait object: what [`DynProblem`] is
/// for real genomes.
pub trait IntegerProblem: Send + Sync {
    fn name(&self) -> &'static str;
    fn integer(&self) -> Integer;
    fn objective(&self) -> Objective;
    fn evaluate(&self, genome: &Integers) -> Fitness;
    fn optimum(&self) -> Option<Optimum<Integers>>;
    fn reference(&self) -> &'static str;
    fn reference_url(&self) -> Option<&'static str>;
}

impl<P> IntegerProblem for P
where
    P: problems::Problem<Representation = Integer> + Send + Sync,
{
    fn name(&self) -> &'static str {
        problems::Problem::name(self)
    }

    fn integer(&self) -> Integer {
        self.representation()
    }

    fn objective(&self) -> Objective {
        problems::Problem::objective(self)
    }

    fn evaluate(&self, genome: &Integers) -> Fitness {
        FitnessFunction::evaluate(self, genome)
            .into_fitness()
            .unwrap_or_else(|_| Fitness::invalid())
    }

    fn optimum(&self) -> Option<Optimum<Integers>> {
        problems::Problem::optimum(self)
    }

    fn reference(&self) -> &'static str {
        problems::Problem::reference(self)
    }

    fn reference_url(&self) -> Option<&'static str> {
        problems::Problem::reference_url(self)
    }
}

// the problem that `config` describes; a size below the minimum is an error, not the panic of
// the constructor
fn build(config: Config) -> Result<Problem, String> {
    let at_least = |dimensions: usize, minimum: usize, name: &str| {
        at_least(dimensions, minimum, name, "dimensions")
    };
    Ok(Problem::Single(match config {
        Config::Sphere { dimensions } => {
            problems::boxed(problems::Sphere::new(at_least(dimensions, 1, "Sphere")?))
        }
        Config::AxisParallelEllipsoid { dimensions } => problems::boxed(
            problems::AxisParallelEllipsoid::new(at_least(dimensions, 1, "AxisParallelEllipsoid")?),
        ),
        Config::Schwefel12 { dimensions } => problems::boxed(problems::Schwefel1_2::new(at_least(
            dimensions,
            1,
            "Schwefel1_2",
        )?)),
        Config::Rastrigin { dimensions } => problems::boxed(problems::Rastrigin::new(at_least(
            dimensions,
            1,
            "Rastrigin",
        )?)),
        Config::Rosenbrock { dimensions } => problems::boxed(problems::Rosenbrock::new(at_least(
            dimensions,
            2,
            "Rosenbrock",
        )?)),
        Config::Ackley { dimensions } => {
            problems::boxed(problems::Ackley::new(at_least(dimensions, 1, "Ackley")?))
        }
        Config::Griewank { dimensions } => problems::boxed(problems::Griewank::new(at_least(
            dimensions, 1, "Griewank",
        )?)),
        Config::Schwefel226 { dimensions } => problems::boxed(problems::Schwefel2_26::new(
            at_least(dimensions, 1, "Schwefel2_26")?,
        )),
        Config::Levy { dimensions } => {
            problems::boxed(problems::Levy::new(at_least(dimensions, 1, "Levy")?))
        }
        Config::Zakharov { dimensions } => problems::boxed(problems::Zakharov::new(at_least(
            dimensions, 1, "Zakharov",
        )?)),
        Config::StyblinskiTang { dimensions } => problems::boxed(problems::StyblinskiTang::new(
            at_least(dimensions, 1, "StyblinskiTang")?,
        )),
        Config::Michalewicz { dimensions } => problems::boxed(problems::Michalewicz::new(
            at_least(dimensions, 1, "Michalewicz")?,
        )),
        Config::Himmelblau {} => problems::boxed(problems::Himmelblau),
        Config::Branin {} => problems::boxed(problems::Branin),
        Config::GoldsteinPrice {} => problems::boxed(problems::GoldsteinPrice),
        Config::SixHumpCamel {} => problems::boxed(problems::SixHumpCamel),
        Config::G01 {} => problems::boxed(cec2006::G01),
        Config::G02 {} => problems::boxed(cec2006::G02),
        Config::G03 { tolerance } => {
            problems::boxed(cec2006::G03::with_tolerance(equality_tolerance(tolerance)?))
        }
        Config::G04 {} => problems::boxed(cec2006::G04),
        Config::G05 { tolerance } => {
            problems::boxed(cec2006::G05::with_tolerance(equality_tolerance(tolerance)?))
        }
        Config::G06 {} => problems::boxed(cec2006::G06),
        Config::WeldedBeam {} => problems::boxed(engineering::WeldedBeam),
        Config::WeldedBeamRagsdell {} => problems::boxed(engineering::WeldedBeamRagsdell),
        Config::PressureVessel {} => problems::boxed(engineering::PressureVessel),
        Config::TensionCompressionSpring {} => {
            problems::boxed(engineering::TensionCompressionSpring)
        }
        Config::SpeedReducer {} => problems::boxed(engineering::SpeedReducer),
        Config::GearTrain {} => return Ok(Problem::Integer(Box::new(engineering::GearTrain))),
        Config::ThreeBarTruss {} => problems::boxed(engineering::ThreeBarTruss),
        Config::CantileverBeam {} => problems::boxed(engineering::CantileverBeam),
        Config::CarSideImpact {} => problems::boxed(engineering::CarSideImpact),
        Config::Multi(config) => {
            config.check()?;
            return Ok(Problem::Multi(config));
        }
    }))
}

// an equality tolerance: the report's by default, and finite and at least 0 otherwise
fn equality_tolerance(tolerance: Option<f64>) -> Result<f64, String> {
    match tolerance {
        None => Ok(cec2006::EQUALITY_TOLERANCE),
        Some(tolerance) if tolerance.is_finite() && tolerance >= 0.0 => Ok(tolerance),
        Some(tolerance) => Err(format!(
            "the equality tolerance is a finite number of at least 0, not {tolerance}"
        )),
    }
}

/// The problem that `description` (JSON) describes.
pub fn parse(description: &str) -> PyResult<Problem> {
    let mut json = serde_json::Deserializer::from_str(description);
    let config: Config = serde_path_to_error::deserialize(&mut json).map_err(|error| {
        let (path, error) = (error.path().to_string(), error.into_inner());
        PyValueError::new_err(format!("invalid problem `{path}`: {error}"))
    })?;
    build(config).map_err(PyValueError::new_err)
}

// a genome per row, as numpy arrays of the problem's dimensions
fn rows(name: &str, real: &Real, genomes: &PyReadonlyArray2<'_, f64>) -> PyResult<Vec<Reals>> {
    let genomes = genomes.as_array();
    let dimensions = real.genome_len();
    if genomes.ncols() != dimensions {
        return Err(PyValueError::new_err(format!(
            "{name} takes genomes of {dimensions} genes, not {}",
            genomes.ncols()
        )));
    }
    Ok(genomes
        .rows()
        .into_iter()
        .map(|row| row.iter().copied().collect())
        .collect())
}

fn bounds(real: &Real) -> Vec<(f64, f64)> {
    real.bounds()
        .iter()
        .map(|range| (*range.start(), *range.end()))
        .collect()
}

// the rows of points of M values, as a 2-D numpy array
fn matrix<'py, const M: usize>(py: Python<'py>, points: &[[f64; M]]) -> Bound<'py, PyAny> {
    let values: Vec<f64> = points.iter().flatten().copied().collect();
    Array2::from_shape_vec((points.len(), M), values)
        .expect("a row per point")
        .into_pyarray(py)
        .into_any()
}

/// The description of the problem that `problem` (JSON) describes: its name, genome ("real",
/// "integer" or "binary"), bounds (a pair per gene, (0, 1) for a bit), objectives ("minimize" or
/// "maximize" each), number of constraints, reference and the reference's URL; for a
/// single-objective problem, its optimum (None, or its value, solutions a row each, and whether
/// it's proven); for a multi-objective one, its ideal and nadir points (None if unknown).
#[pyfunction]
pub fn problem_info<'py>(py: Python<'py>, problem: &str) -> PyResult<Bound<'py, PyDict>> {
    let info = PyDict::new(py);
    match parse(problem)? {
        Problem::Single(problem) => {
            info.set_item("name", problem.name())?;
            info.set_item("genome", "real")?;
            info.set_item("bounds", bounds(&problem.real()))?;
            let objective = match problem.objective() {
                Objective::Maximize => "maximize",
                Objective::Minimize => "minimize",
            };
            info.set_item("objectives", vec![objective])?;
            let dimensions = problem.real().genome_len();
            info.set_item(
                "constraints",
                problem.constraints(&zeros(&problem.real())).len(),
            )?;
            match problem.optimum() {
                None => info.set_item("optimum", py.None())?,
                Some(optimum) => {
                    let solutions = optimum.solutions();
                    let genes: Vec<f64> =
                        solutions.iter().flat_map(|x| x.iter().copied()).collect();
                    let solutions = Array2::from_shape_vec((solutions.len(), dimensions), genes)
                        .map_err(|error| PyValueError::new_err(error.to_string()))?;
                    let description = PyDict::new(py);
                    description.set_item("value", optimum.value())?;
                    description.set_item("solutions", solutions.into_pyarray(py))?;
                    description.set_item("proven", optimum.is_proven())?;
                    info.set_item("optimum", description)?;
                }
            }
            info.set_item("reference", problem.reference())?;
            info.set_item("reference_url", problem.reference_url())?;
        }
        Problem::Integer(problem) => {
            let integer = problem.integer();
            info.set_item("name", problem.name())?;
            info.set_item("genome", "integer")?;
            let pairs: Vec<(i64, i64)> = integer
                .bounds()
                .iter()
                .map(|range| (*range.start(), *range.end()))
                .collect();
            info.set_item("bounds", pairs)?;
            let objective = match problem.objective() {
                Objective::Maximize => "maximize",
                Objective::Minimize => "minimize",
            };
            info.set_item("objectives", vec![objective])?;
            info.set_item("constraints", 0)?;
            match problem.optimum() {
                None => info.set_item("optimum", py.None())?,
                Some(optimum) => {
                    let solutions = optimum.solutions();
                    let genes: Vec<i64> =
                        solutions.iter().flat_map(|x| x.iter().copied()).collect();
                    let solutions =
                        Array2::from_shape_vec((solutions.len(), integer.genome_len()), genes)
                            .map_err(|error| PyValueError::new_err(error.to_string()))?;
                    let description = PyDict::new(py);
                    description.set_item("value", optimum.value())?;
                    description.set_item("solutions", solutions.into_pyarray(py))?;
                    description.set_item("proven", optimum.is_proven())?;
                    info.set_item("optimum", description)?;
                }
            }
            info.set_item("reference", problem.reference())?;
            info.set_item("reference_url", problem.reference_url())?;
        }
        Problem::Multi(config) => {
            let task = MultiInfo {
                py,
                config,
                info: &info,
            };
            run_with(config, task)??;
        }
    }
    Ok(info)
}

// runs `task` with the problem's number of objectives
fn run_with<T: WithObjectives>(config: MultiConfig, task: T) -> PyResult<T::Output> {
    with_objectives(config.objectives(), task).map_err(|count| {
        PyValueError::new_err(format!("problems take 2 to 6 objectives, not {count}"))
    })
}

struct MultiInfo<'a, 'py> {
    py: Python<'py>,
    config: MultiConfig,
    info: &'a Bound<'py, PyDict>,
}

impl WithObjectives for MultiInfo<'_, '_> {
    type Output = PyResult<()>;

    fn with<const N: usize>(self) -> PyResult<()> {
        let problem = self.config.build::<N>();
        let info = self.info;
        info.set_item("name", problem.name())?;
        match &problem {
            MultiNative::Real(p) => {
                info.set_item("genome", "real")?;
                info.set_item("bounds", bounds(&p.real()))?;
            }
            MultiNative::Binary(p) => {
                // a bit is 0 or 1
                info.set_item("genome", "binary")?;
                info.set_item("bounds", vec![(0, 1); p.binary().genome_len()])?;
            }
        }
        info.set_item("objectives", vec!["minimize"; N])?;
        info.set_item("constraints", problem.constraint_count())?;
        let point = |point: Option<[f64; N]>| point.map(|point| point.to_vec());
        info.set_item(
            "ideal_point",
            point(either!(&problem, p => p.ideal_point())),
        )?;
        info.set_item(
            "nadir_point",
            point(either!(&problem, p => p.nadir_point())),
        )?;
        info.set_item("optimum", self.py.None())?;
        info.set_item("reference", either!(&problem, p => p.reference()))?;
        info.set_item("reference_url", either!(&problem, p => p.reference_url()))?;
        Ok(())
    }
}

/// The values of `genomes`, a row each, for the problem that `problem` (JSON) describes: for a
/// single-objective problem, an array of scores; for a multi-objective one, a 2-D array with a
/// row of objective values per genome; and with constraints, a tuple of that and an array of
/// constraint violations. NaN for an invalid value.
#[pyfunction]
pub fn evaluate<'py>(
    py: Python<'py>,
    problem: &str,
    genomes: PyReadonlyArray2<'py, f64>,
) -> PyResult<Bound<'py, PyAny>> {
    match parse(problem)? {
        Problem::Single(problem) => {
            let genomes = rows(problem.name(), &problem.real(), &genomes)?;
            let constrained = !problem.constraints(&zeros(&problem.real())).is_empty();
            let (scores, violations): (Vec<f64>, Vec<f64>) = py.detach(|| {
                genomes
                    .iter()
                    .map(|genome| {
                        let fitness = problem.evaluate(genome);
                        match fitness.score() {
                            Some(score) => (score, fitness.violation()),
                            None => (f64::NAN, f64::NAN),
                        }
                    })
                    .unzip()
            });
            let scores = PyArray1::from_vec(py, scores).into_any();
            if constrained {
                (scores, PyArray1::from_vec(py, violations)).into_bound_py_any(py)
            } else {
                Ok(scores)
            }
        }
        Problem::Integer(problem) => {
            let genomes = integer_rows(problem.name(), &problem.integer(), &genomes)?;
            let scores: Vec<f64> = py.detach(|| {
                genomes
                    .iter()
                    .map(|genome| problem.evaluate(genome).score().unwrap_or(f64::NAN))
                    .collect()
            });
            Ok(PyArray1::from_vec(py, scores).into_any())
        }
        Problem::Multi(config) => run_with(
            config,
            MultiEvaluate {
                py,
                config,
                genomes,
            },
        )?,
    }
}

// a genome per row, as numpy arrays of the problem's dimensions whose values are whole numbers
fn integer_rows(
    name: &str,
    integer: &Integer,
    genomes: &PyReadonlyArray2<'_, f64>,
) -> PyResult<Vec<Integers>> {
    let genomes = genomes.as_array();
    let dimensions = integer.genome_len();
    if genomes.ncols() != dimensions {
        return Err(PyValueError::new_err(format!(
            "{name} takes genomes of {dimensions} genes, not {}",
            genomes.ncols()
        )));
    }
    genomes
        .rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .map(|&gene| whole(name, gene))
                .collect::<PyResult<Integers>>()
        })
        .collect()
}

// a gene of an integer genome: a whole number that fits in an i64
fn whole(name: &str, gene: f64) -> PyResult<i64> {
    // −2^63 is the least value that fits, and 2^63 the first that doesn't
    const LIMIT: f64 = 9_223_372_036_854_775_808.0;
    if gene.fract() == 0.0 && (-LIMIT..LIMIT).contains(&gene) {
        Ok(gene as i64)
    } else {
        Err(PyValueError::new_err(format!(
            "{name} takes whole numbers as genes, not {gene}"
        )))
    }
}

// a genome per row, as numpy arrays of the problem's number of bits whose values are 0 or 1
fn bit_rows(
    name: &str,
    binary: &Binary,
    genomes: &PyReadonlyArray2<'_, f64>,
) -> PyResult<Vec<Bits>> {
    let genomes = genomes.as_array();
    let length = binary.genome_len();
    if genomes.ncols() != length {
        return Err(PyValueError::new_err(format!(
            "{name} takes genomes of {length} bits, not {}",
            genomes.ncols()
        )));
    }
    genomes
        .rows()
        .into_iter()
        .map(|row| row.iter().map(|&gene| bit(name, gene)).collect())
        .collect()
}

// a gene of a bit string: 0 or 1 (False or True)
fn bit(name: &str, gene: f64) -> PyResult<bool> {
    match gene {
        0.0 => Ok(false),
        1.0 => Ok(true),
        _ => Err(PyValueError::new_err(format!(
            "{name} takes bits, 0 or 1, as genes, not {gene}"
        ))),
    }
}

// a genome of zeros for the problem's bounds, to count its constraints
fn zeros(real: &Real) -> Reals {
    Reals::from(vec![0.0; real.genome_len()])
}

struct MultiEvaluate<'py> {
    py: Python<'py>,
    config: MultiConfig,
    genomes: PyReadonlyArray2<'py, f64>,
}

impl<'py> WithObjectives for MultiEvaluate<'py> {
    type Output = PyResult<Bound<'py, PyAny>>;

    fn with<const N: usize>(self) -> PyResult<Bound<'py, PyAny>> {
        let py = self.py;
        let problem = self.config.build::<N>();
        let genomes = problem.rows(&self.genomes)?;
        let (values, violations): (Vec<[f64; N]>, Vec<f64>) = py.detach(|| {
            let scores: Vec<Scores<N>> = match &genomes {
                Genomes::Real(genomes) => genomes.iter().map(|x| problem.evaluate(x)).collect(),
                Genomes::Binary(genomes) => genomes.iter().map(|x| problem.evaluate(x)).collect(),
            };
            scores
                .iter()
                .map(|scores| match scores.values() {
                    Some(values) => (values, scores.violation()),
                    None => ([f64::NAN; N], f64::NAN),
                })
                .unzip()
        });
        let values = matrix(py, &values);
        if problem.constraint_count() > 0 {
            (values, PyArray1::from_vec(py, violations)).into_bound_py_any(py)
        } else {
            Ok(values)
        }
    }
}

/// The constraint values of `genome` for the problem that `problem` (JSON) describes: the
/// inequalities `g(x) <= 0`, then the equalities `h(x) = 0`.
#[pyfunction]
pub fn constraints<'py>(
    py: Python<'py>,
    problem: &str,
    genome: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let genome: Reals = genome.as_array().iter().copied().collect();
    let constraints = match parse(problem)? {
        Problem::Single(problem) => {
            check_length(problem.name(), &problem.real(), &genome)?;
            problem.constraints(&genome)
        }
        Problem::Integer(problem) => {
            let dimensions = problem.integer().genome_len();
            if genome.len() != dimensions {
                return Err(PyValueError::new_err(format!(
                    "{} takes genomes of {dimensions} genes, not {}",
                    problem.name(),
                    genome.len()
                )));
            }
            Constraints::none()
        }
        Problem::Multi(config) => run_with(
            config,
            MultiConstraints {
                config,
                genome: &genome,
            },
        )??,
    };
    let mut values = constraints.inequalities().to_vec();
    values.extend_from_slice(constraints.equalities());
    Ok(PyArray1::from_vec(py, values))
}

fn check_length(name: &str, real: &Real, genome: &Reals) -> PyResult<()> {
    let dimensions = real.genome_len();
    if genome.len() == dimensions {
        Ok(())
    } else {
        Err(PyValueError::new_err(format!(
            "{name} takes genomes of {dimensions} genes, not {}",
            genome.len()
        )))
    }
}

struct MultiConstraints<'a> {
    config: MultiConfig,
    genome: &'a Reals,
}

impl WithObjectives for MultiConstraints<'_> {
    type Output = PyResult<Constraints>;

    fn with<const N: usize>(self) -> PyResult<Constraints> {
        match self.config.build::<N>() {
            MultiNative::Real(problem) => {
                check_length(problem.name(), &problem.real(), self.genome)?;
                Ok(problem.constraints(self.genome))
            }
            MultiNative::Binary(problem) => {
                let (name, binary) = (problem.name(), problem.binary());
                let bits = self.genome.iter().map(|&gene| bit(name, gene));
                let genome = bits.collect::<PyResult<Bits>>()?;
                if genome.len() != binary.genome_len() {
                    return Err(PyValueError::new_err(format!(
                        "{name} takes genomes of {} bits, not {}",
                        binary.genome_len(),
                        genome.len()
                    )));
                }
                Ok(problem.constraints(&genome))
            }
        }
    }
}

/// At least `points` points of the optimal front of the problem that `problem` (JSON) describes,
/// a row each, or None if the front isn't known.
#[pyfunction]
pub fn optimal_front<'py>(
    py: Python<'py>,
    problem: &str,
    points: usize,
) -> PyResult<Bound<'py, PyAny>> {
    // checked before allocating: a failed allocation would abort the interpreter
    if points as u128 > crate::run::MAX_POINTS {
        return Err(PyValueError::new_err(format!(
            "optimal_front takes at most 2^24 points, not {points}"
        )));
    }
    match parse(problem)? {
        Problem::Single(problem) => Err(PyValueError::new_err(format!(
            "{} has one objective, and an optimum instead of a front",
            problem.name()
        ))),
        Problem::Integer(problem) => Err(PyValueError::new_err(format!(
            "{} has one objective, and an optimum instead of a front",
            problem.name()
        ))),
        Problem::Multi(config) => run_with(config, MultiFront { py, config, points })?,
    }
}

struct MultiFront<'py> {
    py: Python<'py>,
    config: MultiConfig,
    points: usize,
}

impl<'py> WithObjectives for MultiFront<'py> {
    type Output = PyResult<Bound<'py, PyAny>>;

    fn with<const N: usize>(self) -> PyResult<Bound<'py, PyAny>> {
        let problem = self.config.build::<N>();
        let front = self
            .py
            .detach(|| either!(&problem, p => p.optimal_front(self.points)));
        Ok(match front {
            Some(front) => matrix(self.py, &front),
            None => self.py.None().into_bound(self.py),
        })
    }
}

/// The design variables of `genome` for a problem with discrete variables that its genome rounds:
/// the pressure vessel and the speed reducer.
#[pyfunction]
pub fn design<'py>(
    py: Python<'py>,
    problem: &str,
    genome: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let genome: Reals = genome.as_array().iter().copied().collect();
    let mut json = serde_json::Deserializer::from_str(problem);
    let config: Config = serde_path_to_error::deserialize(&mut json)
        .map_err(|error| PyValueError::new_err(format!("invalid problem: {error}")))?;
    let design = match config {
        Config::PressureVessel {} => {
            let problem = engineering::PressureVessel;
            check_length(problem.name(), &problem.representation(), &genome)?;
            problem.design(&genome).to_vec()
        }
        Config::SpeedReducer {} => {
            let problem = engineering::SpeedReducer;
            check_length(problem.name(), &problem.representation(), &genome)?;
            problem.design(&genome).to_vec()
        }
        _ => {
            return Err(PyValueError::new_err(
                "only the problems with discrete variables have a design",
            ));
        }
    };
    Ok(PyArray1::from_vec(py, design))
}

/// The names of the problems of `genoxide::problems::all()`, in its order.
#[pyfunction]
pub fn problem_names() -> Vec<&'static str> {
    problems::all()
        .iter()
        .map(|problem| problem.name())
        .collect()
}

/// The names of the problems of `genoxide::multi::problems::all()` with `objectives` objectives,
/// 2 to 6, in its order.
#[pyfunction]
pub fn multi_problem_names(objectives: usize) -> PyResult<Vec<&'static str>> {
    struct Names;

    impl WithObjectives for Names {
        type Output = Vec<&'static str>;

        fn with<const N: usize>(self) -> Vec<&'static str> {
            multi::all::<N>()
                .iter()
                .map(|problem| problem.name())
                .collect()
        }
    }

    with_objectives(objectives, Names).map_err(|count| {
        PyValueError::new_err(format!("problems take 2 to 6 objectives, not {count}"))
    })
}
