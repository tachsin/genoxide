//! The operators of a run: an enum per kind of genome, with only the operators that fit it. They
//! implement serde as the operators in them, for checkpoints.

use crate::config;
use crate::errors::{setting, setting_named};
use genoxide::genome::{AdaptiveReal, Binary, Genome, Integer, Permutation, Real, Representation};
use genoxide::operator::{
    ArithmeticCrossover, BitFlip, BlendCrossover, Crossover, CycleCrossover, DoubleTournament,
    EdgeRecombinationCrossover, GaussianMutation, InsertionMutation, InversionMutation,
    LexicographicTournament, Mutate, NoCrossover, OrderCrossover, PartiallyMappedCrossover,
    PointCrossover, PolynomialMutation, RandomSelection, Rank, Roulette, ScrambleMutation, Select,
    SelfAdaptiveMutation, SimulatedBinaryCrossover, StochasticUniversalSampling, SwapMutation,
    Tarpeian, Tournament, Truncation, UniformCrossover, UniformMutation,
};
use genoxide::{Objective, Population, StreamRng};

type Result<T> = std::result::Result<T, String>;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum AnySelect {
    Tournament(Tournament),
    Rank(Rank),
    Roulette(Roulette),
    StochasticUniversalSampling(StochasticUniversalSampling),
    Truncation(Truncation),
    Random(RandomSelection),
    Lexicographic(LexicographicTournament),
    Double(DoubleTournament),
    Tarpeian(Tarpeian<InnerSelect>),
}

/// The selection that a Tarpeian selection wraps: any but another Tarpeian one.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct InnerSelect(Box<AnySelect>);

impl AnySelect {
    pub fn new(select: config::Select) -> Result<Self> {
        Ok(match select {
            config::Select::Tournament { size } => {
                Self::Tournament(setting(Tournament::new(size))?)
            }
            config::Select::Rank { pressure } => Self::Rank(setting(Rank::new(pressure))?),
            config::Select::Roulette {} => Self::Roulette(Roulette),
            config::Select::StochasticUniversalSampling {} => {
                Self::StochasticUniversalSampling(StochasticUniversalSampling)
            }
            config::Select::Truncation { fraction } => {
                Self::Truncation(setting(Truncation::new(fraction))?)
            }
            config::Select::Random {} => Self::Random(RandomSelection),
            config::Select::LexicographicTournament { size, bucket_ratio } => {
                let names = [
                    ("tournament_size", "LexicographicTournament.size"),
                    ("bucket_ratio", "LexicographicTournament.bucket_ratio"),
                ];
                let select = setting_named(LexicographicTournament::new(size), &names)?;
                Self::Lexicographic(match bucket_ratio {
                    Some(ratio) => setting_named(select.ratio_buckets(ratio), &names)?,
                    None => select,
                })
            }
            config::Select::DoubleTournament {
                fitness_size,
                parsimony,
                size_first,
            } => {
                let names = [
                    ("tournament_size", "DoubleTournament.fitness_size"),
                    ("parsimony", "DoubleTournament.parsimony"),
                ];
                let select = setting_named(DoubleTournament::new(fitness_size, parsimony), &names)?;
                Self::Double(if size_first {
                    select.size_first()
                } else {
                    select
                })
            }
            config::Select::Tarpeian { select, rate } => {
                let inner = Self::new(*select)?;
                if let Self::Tarpeian(_) = inner {
                    return Err(
                        "invalid setting `Tarpeian.select`: a selection other than Tarpeian"
                            .to_string(),
                    );
                }
                let names = [("tarpeian_rate", "Tarpeian.rate")];
                Self::Tarpeian(setting_named(
                    Tarpeian::new(InnerSelect(Box::new(inner)), rate),
                    &names,
                )?)
            }
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
            Self::Tarpeian(select) => select.select(population, objective, count, rng),
            select => select_untarpeian(select, population, objective, count, rng),
        }
    }
}

// a Tarpeian selection's view of the population is a `Population<&G>`: the selection it wraps
// selects without another level of references, as none of them is Tarpeian
impl Select for InnerSelect {
    fn select<G: Genome>(
        &self,
        population: &Population<G>,
        objective: Objective,
        count: usize,
        rng: &mut StreamRng,
    ) -> Vec<usize> {
        select_untarpeian(&self.0, population, objective, count, rng)
    }
}

// the selection of any but a Tarpeian selection, which `AnySelect::new` never wraps in another
fn select_untarpeian<G: Genome>(
    select: &AnySelect,
    population: &Population<G>,
    objective: Objective,
    count: usize,
    rng: &mut StreamRng,
) -> Vec<usize> {
    match select {
        AnySelect::Tournament(select) => select.select(population, objective, count, rng),
        AnySelect::Rank(select) => select.select(population, objective, count, rng),
        AnySelect::Roulette(select) => select.select(population, objective, count, rng),
        AnySelect::StochasticUniversalSampling(select) => {
            select.select(population, objective, count, rng)
        }
        AnySelect::Truncation(select) => select.select(population, objective, count, rng),
        AnySelect::Random(select) => select.select(population, objective, count, rng),
        AnySelect::Lexicographic(select) => select.select(population, objective, count, rng),
        AnySelect::Double(select) => select.select(population, objective, count, rng),
        AnySelect::Tarpeian(_) => unreachable!("a Tarpeian selection in a Tarpeian one"),
    }
}

// the Python class of an operator, for errors
fn crossover_name(crossover: config::Crossover) -> &'static str {
    match crossover {
        config::Crossover::Uniform {} => "UniformCrossover",
        config::Crossover::Point { .. } => "PointCrossover",
        config::Crossover::None {} => "NoCrossover",
        config::Crossover::SimulatedBinary { .. } => "SimulatedBinaryCrossover",
        config::Crossover::Blend { .. } => "BlendCrossover",
        config::Crossover::Arithmetic {} => "ArithmeticCrossover",
        config::Crossover::Order {} => "OrderCrossover",
        config::Crossover::PartiallyMapped {} => "PartiallyMappedCrossover",
        config::Crossover::Cycle {} => "CycleCrossover",
        config::Crossover::EdgeRecombination {} => "EdgeRecombinationCrossover",
        config::Crossover::Subtree { .. } => "SubtreeCrossover",
        config::Crossover::OnePoint {} => "OnePointCrossover",
    }
}

fn mutate_name(mutate: &config::Mutate) -> &'static str {
    match mutate {
        config::Mutate::BitFlip { .. } => "BitFlip",
        config::Mutate::Uniform { .. } => "UniformMutation",
        config::Mutate::Gaussian { .. } => "GaussianMutation",
        config::Mutate::Polynomial { .. } => "PolynomialMutation",
        config::Mutate::Swap { .. } => "SwapMutation",
        config::Mutate::Inversion {} => "InversionMutation",
        config::Mutate::Insertion {} => "InsertionMutation",
        config::Mutate::Scramble {} => "ScrambleMutation",
        config::Mutate::SelfAdaptive { .. } => "SelfAdaptiveMutation",
        config::Mutate::Subtree { .. } => "SubtreeMutation",
        config::Mutate::Point { .. } => "PointMutation",
        config::Mutate::Hoist {} => "HoistMutation",
        config::Mutate::Shrink {} => "ShrinkMutation",
        config::Mutate::Constant { .. } => "ConstantMutation",
        config::Mutate::Mutations { .. } => "Mutations",
    }
}

pub fn wrong_crossover(crossover: config::Crossover, genome: &str, fits: &str) -> String {
    format!(
        "{} doesn't work with {genome} genomes; use {fits}",
        crossover_name(crossover)
    )
}

pub fn wrong_mutate(mutate: &config::Mutate, genome: &str, fits: &str) -> String {
    format!(
        "{} doesn't work with {genome} genomes; use {fits}",
        mutate_name(mutate)
    )
}

// exactly one of a rate per gene and a count of genes
pub fn rate_or_count(rate: Option<f64>, count: Option<usize>) -> Result<RateOrCount> {
    match (rate, count) {
        (Some(rate), None) => Ok(RateOrCount::Rate(rate)),
        (None, Some(count)) => Ok(RateOrCount::Count(count)),
        _ => Err("a mutation needs either `rate` (per gene) or `count` (genes)".to_string()),
    }
}

pub enum RateOrCount {
    Rate(f64),
    Count(usize),
}

fn point(points: usize) -> Result<PointCrossover> {
    setting(PointCrossover::k_point(points))
}

/// A crossover of binary, integer and adaptive real genomes.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ListCrossover {
    Uniform(UniformCrossover),
    Point(PointCrossover),
    None(NoCrossover),
}

impl ListCrossover {
    pub fn new(crossover: config::Crossover, genome: &str) -> Result<Self> {
        match crossover {
            config::Crossover::Uniform {} => Ok(Self::Uniform(UniformCrossover::new())),
            config::Crossover::Point { points } => Ok(Self::Point(point(points)?)),
            config::Crossover::None {} => Ok(Self::None(NoCrossover)),
            _ => Err(wrong_crossover(
                crossover,
                genome,
                "UniformCrossover, PointCrossover or NoCrossover",
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

            fn recombines(&self) -> bool {
                !matches!(self, Self::None(_))
            }
        }
    };
}

list_crossover!(Binary);
list_crossover!(Integer);
list_crossover!(AdaptiveReal);

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
        match crossover {
            config::Crossover::Uniform {} => Ok(Self::Uniform(UniformCrossover::new())),
            config::Crossover::Point { points } => Ok(Self::Point(point(points)?)),
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
                "SimulatedBinaryCrossover, BlendCrossover, ArithmeticCrossover, UniformCrossover, PointCrossover or NoCrossover",
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

    fn recombines(&self) -> bool {
        !matches!(self, Self::None(_))
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
                "OrderCrossover, PartiallyMappedCrossover, CycleCrossover, EdgeRecombinationCrossover or NoCrossover",
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

    fn recombines(&self) -> bool {
        !matches!(self, Self::None(_))
    }
}

/// The mutation of binary genomes.
pub fn bit_flip(mutate: config::Mutate) -> Result<BitFlip> {
    match mutate {
        config::Mutate::BitFlip { rate, count } => match rate_or_count(rate, count)? {
            RateOrCount::Rate(rate) => setting(BitFlip::per_gene(rate)),
            RateOrCount::Count(count) => setting(BitFlip::count(count)),
        },
        _ => Err(wrong_mutate(&mutate, "binary", "BitFlip")),
    }
}

/// The mutation of integer genomes.
pub fn integer_mutation(mutate: config::Mutate) -> Result<UniformMutation> {
    match mutate {
        config::Mutate::Uniform { rate, count } => match rate_or_count(rate, count)? {
            RateOrCount::Rate(rate) => setting(UniformMutation::per_gene(rate)),
            RateOrCount::Count(count) => setting(UniformMutation::count(count)),
        },
        _ => Err(wrong_mutate(&mutate, "integer", "UniformMutation")),
    }
}

/// The mutation of adaptive real genomes, which changes their step size too.
pub fn self_adaptive(mutate: config::Mutate) -> Result<SelfAdaptiveMutation> {
    match mutate {
        config::Mutate::SelfAdaptive {
            learning_rate,
            min_step,
        } => {
            let mutation = match learning_rate {
                Some(tau) => setting(SelfAdaptiveMutation::with_learning_rate(tau))?,
                None => SelfAdaptiveMutation::new(),
            };
            match min_step {
                Some(min_step) => setting(mutation.with_min_step(min_step)),
                None => Ok(mutation),
            }
        }
        _ => Err(wrong_mutate(
            &mutate,
            "adaptive real",
            "SelfAdaptiveMutation",
        )),
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
                    &mutate,
                    "real",
                    "PolynomialMutation, GaussianMutation or UniformMutation",
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
            config::Mutate::Swap { count } => Self::Swap(setting(SwapMutation::count(count))?),
            config::Mutate::Inversion {} => Self::Inversion(InversionMutation),
            config::Mutate::Insertion {} => Self::Insertion(InsertionMutation),
            config::Mutate::Scramble {} => Self::Scramble(ScrambleMutation),
            _ => {
                return Err(wrong_mutate(
                    &mutate,
                    "permutation",
                    "SwapMutation, InversionMutation, InsertionMutation or ScrambleMutation",
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
