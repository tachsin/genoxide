//! Genetic operators: selection, crossover and mutation.
//!
//! Crossover and mutation are parameterized by the [`Representation`], so an operator that
//! doesn't fit a genome (e.g. point crossover on a permutation) doesn't compile. Selection works
//! with every representation: [`Tournament`], [`Rank`], [`Truncation`], [`Roulette`],
//! [`StochasticUniversalSampling`] and [`RandomSelection`].
//!
//! | Representation | Crossover | Mutation |
//! |---|---|---|
//! | [`Binary`](crate::genome::Binary) | [`PointCrossover`], [`UniformCrossover`] | [`BitFlip`] (rate `1 / length`) |
//! | [`Integer`](crate::genome::Integer) | [`PointCrossover`], [`UniformCrossover`] | [`UniformMutation`] (rate `1 / length`) |
//! | [`Real`](crate::genome::Real) | [`SimulatedBinaryCrossover`] (η 15 to 20), [`BlendCrossover`] (α 0.5), [`ArithmeticCrossover`], [`PointCrossover`], [`UniformCrossover`] | [`PolynomialMutation`] (η 20, rate `1 / length`), [`GaussianMutation`] (σ 0.01 to 0.1), [`UniformMutation`] |
//! | [`AdaptiveReal`](crate::genome::AdaptiveReal) | [`PointCrossover`], [`UniformCrossover`] | [`SelfAdaptiveMutation`] (τ `1 / √n`) |
//! | [`Permutation`](crate::genome::Permutation) | [`OrderCrossover`], [`PartiallyMappedCrossover`], [`CycleCrossover`], [`EdgeRecombinationCrossover`] | [`SwapMutation`], [`InversionMutation`], [`InsertionMutation`], [`ScrambleMutation`] |
//!
//! [`NoCrossover`] fits every representation, for algorithms that only mutate. For another
//! representation, or another operator, implement [`Crossover`] and [`Mutate`].

pub mod crossover;
pub mod mutate;
pub mod permutation;
pub mod select;

pub use crossover::{
    ArithmeticCrossover, BlendCrossover, NoCrossover, PointCrossover, SimulatedBinaryCrossover,
    UniformCrossover,
};
pub use mutate::{
    BitFlip, GaussianMutation, PolynomialMutation, SelfAdaptiveMutation, SwapMutation,
    UniformMutation,
};
pub use permutation::{
    CycleCrossover, EdgeRecombinationCrossover, InsertionMutation, InversionMutation,
    OrderCrossover, PartiallyMappedCrossover, ScrambleMutation,
};
pub use select::{
    RandomSelection, Rank, Roulette, StochasticUniversalSampling, Tournament, Truncation,
};

use crate::genome::{Genome, Representation};
use crate::{Error, Objective, Population, Result, StreamRng};
use std::fmt::Debug;

/// Selects parents from an evaluated population.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a selection operator",
    label = "not a selection operator",
    note = "set the selection with `.select(...)`, e.g. `.select(Tournament::new(3)?)`"
)]
pub trait Select: Clone + Debug + Send + Sync {
    /// Selects `count` parents, with replacement: the positions of the selected individuals.
    ///
    /// Every individual is expected to be evaluated, unevaluated individuals are treated as
    /// invalid. Returns no parents for an empty population.
    fn select<G: Genome>(
        &self,
        population: &Population<G>,
        objective: Objective,
        count: usize,
        rng: &mut StreamRng,
    ) -> Vec<usize>;
}

/// Recombines two genomes into two children, in place.
///
/// Implement it for a crossover of your own:
///
/// ```
/// use genoxide::StreamRng;
/// use genoxide::genome::{Real, Reals};
/// use genoxide::operator::Crossover;
///
/// // the children are the gene-wise minimum and maximum of their parents
/// #[derive(Clone, Debug)]
/// struct MinMax;
///
/// impl Crossover<Real> for MinMax {
///     fn crossover(&self, _real: &Real, a: &mut Reals, b: &mut Reals, _rng: &mut StreamRng) {
///         for gene in 0..a.len() {
///             let (x, y) = (a[gene], b[gene]);
///             (a[gene], b[gene]) = (x.min(y), x.max(y));
///         }
///     }
/// }
///
/// let real = Real::uniform(2, 0.0..=1.0)?;
/// let (mut a, mut b) = (Reals::from(vec![0.2, 0.9]), Reals::from(vec![0.5, 0.1]));
/// MinMax.crossover(&real, &mut a, &mut b, &mut StreamRng::seed_from_u64(0));
/// assert_eq!(&a[..], &[0.2, 0.1]);
/// assert_eq!(&b[..], &[0.5, 0.9]);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a crossover for `{R}`",
    label = "not a crossover for `{R}`",
    note = "set the crossover with `.crossover(...)`; point and uniform crossover need genomes that implement `SwapGenes`, SBX, blend and arithmetic crossover are for `Real`, order, partially mapped, cycle and edge recombination crossover for `Permutation`, and `NoCrossover` fits every genome"
)]
pub trait Crossover<R: Representation>: Clone + Debug + Send + Sync {
    /// Recombines `a` and `b`, which become the two children.
    ///
    /// # Panics
    ///
    /// May panic if a genome doesn't fit the representation, e.g. has another length.
    fn crossover(
        &self,
        representation: &R,
        a: &mut R::Genome,
        b: &mut R::Genome,
        rng: &mut StreamRng,
    );

    /// Whether the crossover can change the genomes: `true` by default, `false` for
    /// [`NoCrossover`], whose children are copies of their parents. Builders use it to reject a
    /// mutation rate of 0 with a crossover that doesn't recombine, where every child would be a
    /// copy.
    fn recombines(&self) -> bool {
        true
    }
}

/// Changes a genome randomly, in place.
///
/// Implement it for a mutation of your own:
///
/// ```
/// use genoxide::StreamRng;
/// use genoxide::genome::{Integer, Integers};
/// use genoxide::operator::Mutate;
/// use rand::RngExt;
///
/// // adds 1 to a random gene, back to its lower bound after its upper bound
/// #[derive(Clone, Debug)]
/// struct Increment;
///
/// impl Mutate<Integer> for Increment {
///     fn mutate(&self, integer: &Integer, genome: &mut Integers, rng: &mut StreamRng) {
///         let gene = rng.random_range(0..genome.len());
///         let bounds = &integer.bounds()[gene];
///         genome[gene] = if genome[gene] < *bounds.end() {
///             genome[gene] + 1
///         } else {
///             *bounds.start()
///         };
///     }
/// }
///
/// let integer = Integer::uniform(3, 0..=9)?;
/// let mut genome = Integers::from(vec![0, 5, 9]);
/// Increment.mutate(&integer, &mut genome, &mut StreamRng::seed_from_u64(0));
/// assert_eq!(genome.iter().sum::<i64>() % 10, 5); // one gene changed by +1 or -9
/// # Ok::<(), genoxide::Error>(())
/// ```
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a mutation for `{R}`",
    label = "not a mutation for `{R}`",
    note = "set the mutation with `.mutate(...)` (`.neighbor(...)` for local search): `BitFlip` for `Binary`, `UniformMutation` for `Integer`, `GaussianMutation`, `PolynomialMutation` or `UniformMutation` for `Real`, `SelfAdaptiveMutation` for `AdaptiveReal`, `SwapMutation`, `InversionMutation`, `InsertionMutation` or `ScrambleMutation` for `Permutation`"
)]
pub trait Mutate<R: Representation>: Clone + Debug + Send + Sync {
    /// Mutates `genome`. A mutation with a per-gene rate can leave it unchanged when it picks no
    /// gene; the other mutations always change it (unless its space has a single genome).
    ///
    /// # Panics
    ///
    /// May panic if the genome doesn't fit the representation, e.g. has another length.
    fn mutate(&self, representation: &R, genome: &mut R::Genome, rng: &mut StreamRng);
}

// a neighbor of `genome` by `mutate`, drawn again (up to 100 times) while it equals `genome`,
// which a per-gene mutation can leave unchanged: local search never evaluates the solution itself
pub(crate) fn neighbor<R: Representation, X: Mutate<R>>(
    mutate: &X,
    representation: &R,
    genome: &R::Genome,
    rng: &mut StreamRng,
) -> R::Genome {
    let mut candidate = genome.clone();
    for _ in 0..100 {
        mutate.mutate(representation, &mut candidate, rng);
        if &candidate != genome {
            break;
        }
    }
    candidate
}

// The largest population, number of offspring, tournament, number of neighbors or of restart
// kicks the builders accept, 2^24: far beyond practical runs, and small enough that the memory and
// time they take are bounded instead of overflowing or hanging.
pub(crate) const MAX_SIZE: usize = 1 << 24;

// A count of at most `MAX_SIZE` for `setting`.
pub(crate) fn check_size(setting: &'static str, size: usize) -> Result<usize> {
    if size <= MAX_SIZE {
        Ok(size)
    } else {
        Err(Error::InvalidSetting {
            setting,
            reason: format!("must be at most {MAX_SIZE} (2^24), got {size}"),
        })
    }
}

// The crossover and mutation rates of an algorithm that breeds children, each in [0, 1], unless
// every child would be a copy of a parent: both rates 0, or a mutation rate of 0 with a crossover
// that doesn't recombine.
pub(crate) fn check_rates(
    crossover_rate: f64,
    mutation_rate: f64,
    recombines: bool,
) -> Result<(f64, f64)> {
    let crossover_rate = check_probability("crossover_rate", crossover_rate)?;
    let mutation_rate = check_probability("mutation_rate", mutation_rate)?;
    if mutation_rate == 0.0 && (crossover_rate == 0.0 || !recombines) {
        let reason = if recombines {
            "crossover_rate and mutation_rate are both 0, so every child would be a copy of a parent"
        } else {
            "mutation_rate is 0 and the crossover doesn't recombine (NoCrossover), so every child would be a copy of a parent"
        };
        return Err(Error::InvalidSetting {
            setting: "mutation_rate",
            reason: reason.to_string(),
        });
    }
    Ok((crossover_rate, mutation_rate))
}

// A probability in [0, 1] for `setting`.
pub(crate) fn check_probability(setting: &'static str, probability: f64) -> Result<f64> {
    if (0.0..=1.0).contains(&probability) {
        Ok(probability)
    } else {
        Err(Error::InvalidSetting {
            setting,
            reason: format!("must be between 0 and 1, got {probability}"),
        })
    }
}

// A rate in (0, 1] for `setting`.
pub(crate) fn check_rate(setting: &'static str, rate: f64) -> Result<f64> {
    if rate > 0.0 && rate <= 1.0 {
        Ok(rate)
    } else {
        Err(Error::InvalidSetting {
            setting,
            reason: format!("must be greater than 0 and at most 1, got {rate}"),
        })
    }
}
