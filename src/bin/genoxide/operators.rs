//! Operators chosen in the run file: an enum per kind of genome, with only the operators that fit.

use crate::config;
use genoxide::genome::{Binary, Genome, Integer, Permutation, Real, Representation};
use genoxide::operator::{
    ArithmeticCrossover, BitFlip, BlendCrossover, Crossover, CycleCrossover,
    EdgeRecombinationCrossover, GaussianMutation, InsertionMutation, InversionMutation, Mutate,
    NoCrossover, OrderCrossover, PartiallyMappedCrossover, PointCrossover, PolynomialMutation,
    RandomSelection, Rank, Roulette, ScrambleMutation, Select, SimulatedBinaryCrossover,
    StochasticUniversalSampling, SwapMutation, Tournament, Truncation, UniformCrossover,
    UniformMutation,
};
use genoxide::{Objective, Population, StreamRng};

type Result<T> = std::result::Result<T, String>;

fn setting<T>(result: genoxide::Result<T>) -> Result<T> {
    result.map_err(|error| error.to_string())
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum AnySelect {
    Tournament(Tournament),
    Rank(Rank),
    Roulette(Roulette),
    StochasticUniversal(StochasticUniversalSampling),
    Truncation(Truncation),
    Random(RandomSelection),
}

impl AnySelect {
    pub fn new(select: config::Select) -> Result<Self> {
        Ok(match select {
            config::Select::Tournament { size } => {
                Self::Tournament(setting(Tournament::new(size))?)
            }
            config::Select::Rank { pressure } => Self::Rank(setting(Rank::new(pressure))?),
            config::Select::Roulette {} => Self::Roulette(Roulette),
            config::Select::StochasticUniversal {} => {
                Self::StochasticUniversal(StochasticUniversalSampling)
            }
            config::Select::Truncation { fraction } => {
                Self::Truncation(setting(Truncation::new(fraction))?)
            }
            config::Select::Random {} => Self::Random(RandomSelection),
        })
    }
}

impl Select for AnySelect {
    fn select<G: Genome>(
        &self,
        population: &Population<G>,
        objective: Objective,
        count: usize,
        rng: &mut StreamRng,
    ) -> Vec<usize> {
        match self {
            Self::Tournament(select) => select.select(population, objective, count, rng),
            Self::Rank(select) => select.select(population, objective, count, rng),
            Self::Roulette(select) => select.select(population, objective, count, rng),
            Self::StochasticUniversal(select) => select.select(population, objective, count, rng),
            Self::Truncation(select) => select.select(population, objective, count, rng),
            Self::Random(select) => select.select(population, objective, count, rng),
        }
    }
}

// the name of an operator in the run file, for errors
fn crossover_name(crossover: config::Crossover) -> &'static str {
    match crossover {
        config::Crossover::Uniform {} => "uniform",
        config::Crossover::OnePoint {} => "one-point",
        config::Crossover::TwoPoint {} => "two-point",
        config::Crossover::KPoint { .. } => "k-point",
        config::Crossover::None {} => "none",
        config::Crossover::SimulatedBinary { .. } => "simulated-binary",
        config::Crossover::Blend { .. } => "blend",
        config::Crossover::Arithmetic {} => "arithmetic",
        config::Crossover::Order {} => "order",
        config::Crossover::PartiallyMapped {} => "partially-mapped",
        config::Crossover::Cycle {} => "cycle",
        config::Crossover::EdgeRecombination {} => "edge-recombination",
    }
}

fn mutate_name(mutate: config::Mutate) -> &'static str {
    match mutate {
        config::Mutate::BitFlip { .. } => "bit-flip",
        config::Mutate::Uniform { .. } => "uniform",
        config::Mutate::Gaussian { .. } => "gaussian",
        config::Mutate::Polynomial { .. } => "polynomial",
        config::Mutate::Swap { .. } => "swap",
        config::Mutate::Inversion {} => "inversion",
        config::Mutate::Insertion {} => "insertion",
        config::Mutate::Scramble {} => "scramble",
    }
}

fn wrong_crossover(crossover: config::Crossover, genome: &str, fits: &str) -> String {
    format!(
        "crossover `{}` doesn't work with {genome} genomes; use {fits}",
        crossover_name(crossover)
    )
}

fn wrong_mutate(mutate: config::Mutate, genome: &str, fits: &str) -> String {
    format!(
        "mutation `{}` doesn't work with {genome} genomes; use {fits}",
        mutate_name(mutate)
    )
}

// exactly one of a rate per gene and a count of genes
fn rate_or_count(rate: Option<f64>, count: Option<usize>) -> Result<RateOrCount> {
    match (rate, count) {
        (Some(rate), None) => Ok(RateOrCount::Rate(rate)),
        (None, Some(count)) => Ok(RateOrCount::Count(count)),
        _ => Err("a mutation needs either `rate` (per gene) or `count` (genes)".to_string()),
    }
}

enum RateOrCount {
    Rate(f64),
    Count(usize),
}

fn point(crossover: config::Crossover) -> Option<Result<PointCrossover>> {
    match crossover {
        config::Crossover::OnePoint {} => Some(Ok(PointCrossover::one_point())),
        config::Crossover::TwoPoint {} => Some(Ok(PointCrossover::two_point())),
        config::Crossover::KPoint { points } => Some(setting(PointCrossover::k_point(points))),
        _ => None,
    }
}

/// A crossover of binary and integer genomes.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ListCrossover {
    Uniform(UniformCrossover),
    Point(PointCrossover),
    None(NoCrossover),
}

impl ListCrossover {
    pub fn new(crossover: config::Crossover, genome: &str) -> Result<Self> {
        if let Some(point) = point(crossover) {
            return Ok(Self::Point(point?));
        }
        match crossover {
            config::Crossover::Uniform {} => Ok(Self::Uniform(UniformCrossover::new())),
            config::Crossover::None {} => Ok(Self::None(NoCrossover)),
            _ => Err(wrong_crossover(
                crossover,
                genome,
                "uniform, one-point, two-point, k-point or none",
            )),
        }
    }
}

macro_rules! list_crossover {
    ($representation:ty) => {
        impl Crossover<$representation> for ListCrossover {
            fn crossover(
                &self,
                representation: &$representation,
                a: &mut <$representation as Representation>::Genome,
                b: &mut <$representation as Representation>::Genome,
                rng: &mut StreamRng,
            ) {
                match self {
                    Self::Uniform(crossover) => crossover.crossover(representation, a, b, rng),
                    Self::Point(crossover) => crossover.crossover(representation, a, b, rng),
                    Self::None(crossover) => crossover.crossover(representation, a, b, rng),
                }
            }
        }
    };
}

list_crossover!(Binary);
list_crossover!(Integer);

/// A crossover of real genomes.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum RealCrossover {
    Uniform(UniformCrossover),
    Point(PointCrossover),
    None(NoCrossover),
    SimulatedBinary(SimulatedBinaryCrossover),
    Blend(BlendCrossover),
    Arithmetic(ArithmeticCrossover),
}

impl RealCrossover {
    pub fn new(crossover: config::Crossover) -> Result<Self> {
        if let Some(point) = point(crossover) {
            return Ok(Self::Point(point?));
        }
        match crossover {
            config::Crossover::Uniform {} => Ok(Self::Uniform(UniformCrossover::new())),
            config::Crossover::None {} => Ok(Self::None(NoCrossover)),
            config::Crossover::SimulatedBinary { eta } => Ok(Self::SimulatedBinary(setting(
                SimulatedBinaryCrossover::new(eta),
            )?)),
            config::Crossover::Blend { alpha } => {
                Ok(Self::Blend(setting(BlendCrossover::new(alpha))?))
            }
            config::Crossover::Arithmetic {} => Ok(Self::Arithmetic(ArithmeticCrossover::new())),
            _ => Err(wrong_crossover(
                crossover,
                "real",
                "simulated-binary, blend, arithmetic, uniform, one-point, two-point, k-point or none",
            )),
        }
    }
}

impl Crossover<Real> for RealCrossover {
    fn crossover(
        &self,
        representation: &Real,
        a: &mut <Real as Representation>::Genome,
        b: &mut <Real as Representation>::Genome,
        rng: &mut StreamRng,
    ) {
        match self {
            Self::Uniform(crossover) => crossover.crossover(representation, a, b, rng),
            Self::Point(crossover) => crossover.crossover(representation, a, b, rng),
            Self::None(crossover) => crossover.crossover(representation, a, b, rng),
            Self::SimulatedBinary(crossover) => crossover.crossover(representation, a, b, rng),
            Self::Blend(crossover) => crossover.crossover(representation, a, b, rng),
            Self::Arithmetic(crossover) => crossover.crossover(representation, a, b, rng),
        }
    }
}

/// A crossover of permutations.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum OrderCrossovers {
    Order(OrderCrossover),
    PartiallyMapped(PartiallyMappedCrossover),
    Cycle(CycleCrossover),
    EdgeRecombination(EdgeRecombinationCrossover),
    None(NoCrossover),
}

impl OrderCrossovers {
    pub fn new(crossover: config::Crossover) -> Result<Self> {
        match crossover {
            config::Crossover::Order {} => Ok(Self::Order(OrderCrossover)),
            config::Crossover::PartiallyMapped {} => {
                Ok(Self::PartiallyMapped(PartiallyMappedCrossover))
            }
            config::Crossover::Cycle {} => Ok(Self::Cycle(CycleCrossover)),
            config::Crossover::EdgeRecombination {} => {
                Ok(Self::EdgeRecombination(EdgeRecombinationCrossover))
            }
            config::Crossover::None {} => Ok(Self::None(NoCrossover)),
            _ => Err(wrong_crossover(
                crossover,
                "permutation",
                "order, partially-mapped, cycle, edge-recombination or none",
            )),
        }
    }
}

impl Crossover<Permutation> for OrderCrossovers {
    fn crossover(
        &self,
        representation: &Permutation,
        a: &mut <Permutation as Representation>::Genome,
        b: &mut <Permutation as Representation>::Genome,
        rng: &mut StreamRng,
    ) {
        match self {
            Self::Order(crossover) => crossover.crossover(representation, a, b, rng),
            Self::PartiallyMapped(crossover) => crossover.crossover(representation, a, b, rng),
            Self::Cycle(crossover) => crossover.crossover(representation, a, b, rng),
            Self::EdgeRecombination(crossover) => crossover.crossover(representation, a, b, rng),
            Self::None(crossover) => crossover.crossover(representation, a, b, rng),
        }
    }
}

/// The mutation of binary genomes.
pub fn bit_flip(mutate: config::Mutate) -> Result<BitFlip> {
    match mutate {
        config::Mutate::BitFlip { rate, count } => match rate_or_count(rate, count)? {
            RateOrCount::Rate(rate) => setting(BitFlip::per_gene(rate)),
            RateOrCount::Count(count) => setting(BitFlip::count(count)),
        },
        _ => Err(wrong_mutate(mutate, "binary", "bit-flip")),
    }
}

/// The mutation of integer genomes.
pub fn integer_mutation(mutate: config::Mutate) -> Result<UniformMutation> {
    match mutate {
        config::Mutate::Uniform { rate, count } => match rate_or_count(rate, count)? {
            RateOrCount::Rate(rate) => setting(UniformMutation::per_gene(rate)),
            RateOrCount::Count(count) => setting(UniformMutation::count(count)),
        },
        _ => Err(wrong_mutate(mutate, "integer", "uniform")),
    }
}

/// A mutation of real genomes.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum RealMutation {
    Uniform(UniformMutation),
    Gaussian(GaussianMutation),
    Polynomial(PolynomialMutation),
}

impl RealMutation {
    pub fn new(mutate: config::Mutate) -> Result<Self> {
        Ok(match mutate {
            config::Mutate::Uniform { rate, count } => {
                Self::Uniform(setting(match rate_or_count(rate, count)? {
                    RateOrCount::Rate(rate) => UniformMutation::per_gene(rate),
                    RateOrCount::Count(count) => UniformMutation::count(count),
                })?)
            }
            config::Mutate::Gaussian { rate, count, sigma } => {
                Self::Gaussian(setting(match rate_or_count(rate, count)? {
                    RateOrCount::Rate(rate) => GaussianMutation::per_gene(rate, sigma),
                    RateOrCount::Count(count) => GaussianMutation::count(count, sigma),
                })?)
            }
            config::Mutate::Polynomial { rate, count, eta } => {
                Self::Polynomial(setting(match rate_or_count(rate, count)? {
                    RateOrCount::Rate(rate) => PolynomialMutation::per_gene(rate, eta),
                    RateOrCount::Count(count) => PolynomialMutation::count(count, eta),
                })?)
            }
            _ => {
                return Err(wrong_mutate(
                    mutate,
                    "real",
                    "polynomial, gaussian or uniform",
                ));
            }
        })
    }
}

impl Mutate<Real> for RealMutation {
    fn mutate(
        &self,
        representation: &Real,
        genome: &mut <Real as Representation>::Genome,
        rng: &mut StreamRng,
    ) {
        match self {
            Self::Uniform(mutate) => mutate.mutate(representation, genome, rng),
            Self::Gaussian(mutate) => mutate.mutate(representation, genome, rng),
            Self::Polynomial(mutate) => mutate.mutate(representation, genome, rng),
        }
    }
}

/// A mutation of permutations.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum OrderMutation {
    Swap(SwapMutation),
    Inversion(InversionMutation),
    Insertion(InsertionMutation),
    Scramble(ScrambleMutation),
}

impl OrderMutation {
    pub fn new(mutate: config::Mutate) -> Result<Self> {
        Ok(match mutate {
            config::Mutate::Swap { count } => Self::Swap(match count {
                Some(count) => setting(SwapMutation::count(count))?,
                None => SwapMutation::new(),
            }),
            config::Mutate::Inversion {} => Self::Inversion(InversionMutation),
            config::Mutate::Insertion {} => Self::Insertion(InsertionMutation),
            config::Mutate::Scramble {} => Self::Scramble(ScrambleMutation),
            _ => {
                return Err(wrong_mutate(
                    mutate,
                    "permutation",
                    "swap, inversion, insertion or scramble",
                ));
            }
        })
    }
}

impl Mutate<Permutation> for OrderMutation {
    fn mutate(
        &self,
        representation: &Permutation,
        genome: &mut <Permutation as Representation>::Genome,
        rng: &mut StreamRng,
    ) {
        match self {
            Self::Swap(mutate) => mutate.mutate(representation, genome, rng),
            Self::Inversion(mutate) => mutate.mutate(representation, genome, rng),
            Self::Insertion(mutate) => mutate.mutate(representation, genome, rng),
            Self::Scramble(mutate) => mutate.mutate(representation, genome, rng),
        }
    }
}
