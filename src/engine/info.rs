//! What a fitness function computed besides the fitness: [`Evaluated`].

use super::IntoFitness;
use crate::genome::Genome;
use crate::multi::{IntoScores, Scores};
use crate::{Fitness, Result};
use std::any::Any;
use std::collections::HashMap;
use std::fmt;
use std::hash::{BuildHasherDefault, Hasher};
use std::panic::{RefUnwindSafe, UnwindSafe};
use std::sync::Arc;

/// A fitness value with extras the fitness function computed along with it: the terms of a
/// weighted sum or a penalty, a secondary measure, which constraints are violated. The engine keeps
/// them next to the fitness, and never uses them in the search: a run gives the same results with
/// or without them.
///
/// `value` is anything a fitness function can return: `f64`, [`Fitness`], `Option<f64>` or
/// `(f64, f64)` for an [`Engine`](crate::Engine) or an [`AsyncEngine`](crate::engine::AsyncEngine),
/// and `[f64; M]`, `([f64; M], f64)`, `Option<[f64; M]>` or [`Scores<M>`] for a
/// [`MultiEngine`](crate::multi::MultiEngine). `info` is any type that is `Send`, `Sync` and
/// `'static`. A [`Batch`](crate::engine::Batch) returns a `Vec` of them.
///
/// The info of a genome is read by its type:
///
/// - [`Outcome::best_info`](crate::Outcome::best_info) for the best individual;
/// - [`Snapshot::info`](crate::observer::Snapshot::info) for the population, the discarded
///   individuals and the best, after every generation;
/// - [`HallOfFame::info`](crate::observer::HallOfFame::info) for its members;
/// - [`MultiSnapshot::info`](crate::multi::MultiSnapshot::info) and
///   [`MultiOutcome::info`](crate::multi::MultiOutcome::info) in a multi-objective run.
///
/// Each gives `None` for another type than the one returned, and for a genome without info.
///
/// Fitness functions are deterministic, so the engine keeps the info by genome: a copy of a parent
/// and a survivor have the info of their genome, without another evaluation. It keeps the info of
/// the population, of the individuals discarded in the last generation and of the best, and drops
/// the rest. A re-evaluation ([`Reevaluate`](crate::algorithm::Reevaluate)) replaces it. The info
/// isn't part of a checkpoint: after resuming, a genome has info once it's evaluated again.
///
/// ```
/// use genoxide::prelude::*;
///
/// const WEIGHTS: [f64; 8] = [12.0, 7.0, 11.0, 8.0, 9.0, 5.0, 14.0, 6.0];
/// const VALUES: [f64; 8] = [24.0, 13.0, 23.0, 15.0, 16.0, 8.0, 25.0, 11.0];
/// const CAPACITY: f64 = 40.0;
///
/// // what the fitness function computes on the way to the fitness
/// #[derive(Debug)]
/// struct Breakdown {
///     weight: f64,
///     items: usize,
/// }
///
/// let knapsack = |selection: &Bits| {
///     let (mut weight, mut value) = (0.0, 0.0);
///     for (item, chosen) in selection.iter().enumerate() {
///         if chosen {
///             weight += WEIGHTS[item];
///             value += VALUES[item];
///         }
///     }
///     let fitness = (value, constraint::at_most(weight, CAPACITY));
///     let items = selection.count_ones();
///     Evaluated::new(fitness, Breakdown { weight, items })
/// };
///
/// let ga = Ga::builder(Binary::new(8)?)
///     .population_size(30)
///     .select(Tournament::new(3)?)
///     .crossover(UniformCrossover::new())
///     .mutate(BitFlip::count(1)?)
///     .seed(1)
///     .build()?;
/// let mut mean_weights = Vec::new();
/// let outcome = Engine::new(ga, knapsack)
///     .stop_when(Stop::generations(50))
///     // the population's mean weight, e.g. for a chart
///     .on_generation(|snapshot| {
///         let population = snapshot.population();
///         let total: f64 = population
///             .iter()
///             .filter_map(|individual| snapshot.info::<Breakdown>(individual.genome()))
///             .map(|breakdown| breakdown.weight)
///             .sum();
///         mean_weights.push(total / population.len() as f64);
///     })
///     .run()?;
///
/// let breakdown = outcome.best_info::<Breakdown>().expect("the best has its breakdown");
/// println!("{} with {} items weighing {}", outcome.best_fitness(), breakdown.items, breakdown.weight);
/// assert!(breakdown.weight <= CAPACITY);
/// assert_eq!(mean_weights.len() as u64, outcome.generations() + 1);
/// assert!(outcome.best_info::<String>().is_none()); // another type
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Evaluated<V, T> {
    value: V,
    info: T,
}

impl<V, T> Evaluated<V, T> {
    /// A fitness value (or a multi-objective result) with its info.
    pub fn new(value: V, info: T) -> Self {
        Self { value, info }
    }

    /// The fitness value.
    pub fn value(&self) -> &V {
        &self.value
    }

    /// The info.
    pub fn info(&self) -> &T {
        &self.info
    }

    /// The fitness value and the info.
    pub fn into_parts(self) -> (V, T) {
        (self.value, self.info)
    }
}

impl<V: IntoFitness, T: Any + Send + Sync> IntoFitness for Evaluated<V, T> {
    fn into_fitness(self) -> Result<Fitness> {
        self.value.into_fitness()
    }

    #[inline]
    fn into_evaluation(self) -> (Result<Fitness>, Option<Info>) {
        (self.value.into_fitness(), Some(Info::new(self.info)))
    }
}

impl<V: IntoScores<M>, T: Any + Send + Sync, const M: usize> IntoScores<M> for Evaluated<V, T> {
    fn into_scores(self) -> Result<Scores<M>> {
        self.value.into_scores()
    }

    #[inline]
    fn into_evaluation(self) -> (Result<Scores<M>>, Option<Info>) {
        (self.value.into_scores(), Some(Info::new(self.info)))
    }
}

/// The info of an [`Evaluated`] result, of any type: shared, so cheap to clone.
#[doc(hidden)]
#[derive(Clone)]
pub struct Info(Arc<dyn Any + Send + Sync>);

impl Info {
    /// The info `info`.
    #[doc(hidden)]
    pub fn new<T: Any + Send + Sync>(info: T) -> Self {
        Self(Arc::new(info))
    }

    // the info as a `T`, or `None` if it's another type
    pub(crate) fn downcast_ref<T: Any>(&self) -> Option<&T> {
        self.0.downcast_ref()
    }
}

// The info is never changed once made: it's only read, through shared references. Without these,
// `Arc<dyn Any + Send + Sync>` would take the unwind safety of the outcome, the snapshot and the
// hall of fame away.
impl UnwindSafe for Info {}
impl RefUnwindSafe for Info {}

impl fmt::Debug for Info {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Info(..)")
    }
}

// A fast hasher of genomes without random keys, for maps and sets of genomes whose order never
// matters: created without a call to the system's random number generator, and results never
// depend on it, as the genomes are compared. Every word is mixed in with a folded multiply (the
// two halves of a 128-bit product xored), and the result is finalized so that the low bits,
// which pick a bucket, depend on every bit. SipHash, the default, spends several times as long on
// a genome of many words.
#[derive(Clone, Copy)]
pub(crate) struct GenomeHasher(u64);

impl Default for GenomeHasher {
    #[inline]
    fn default() -> Self {
        Self(0x243f_6a88_85a3_08d3)
    }
}

impl Hasher for GenomeHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let (words, rest) = bytes.as_chunks::<8>();
        for &word in words {
            self.write_u64(u64::from_le_bytes(word));
        }
        for &byte in rest {
            self.write_u64(u64::from(byte));
        }
    }

    #[inline]
    fn write_u64(&mut self, word: u64) {
        let product = u128::from(self.0 ^ word) * 0x9e37_79b9_7f4a_7c15;
        self.0 = product as u64 ^ (product >> 64) as u64;
    }

    #[inline]
    fn write_u8(&mut self, n: u8) {
        self.write_u64(n.into());
    }

    #[inline]
    fn write_u32(&mut self, n: u32) {
        self.write_u64(n.into());
    }

    #[inline]
    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }

    #[inline]
    fn write_i64(&mut self, n: i64) {
        self.write_u64(n as u64);
    }

    #[inline]
    fn finish(&self) -> u64 {
        // SplitMix64's finalizer
        let mut x = self.0;
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }
}

// builds `GenomeHasher`s
pub(crate) type GenomeHashing = BuildHasherDefault<GenomeHasher>;

// The info of genomes, by genome. Empty, and neither allocated nor hashed into, unless a fitness
// function returns info.
#[derive(Clone)]
pub(crate) struct InfoStore<G> {
    entries: HashMap<G, Entry, GenomeHashing>,
    // the number of the last pruning
    round: u64,
}

#[derive(Clone)]
struct Entry {
    info: Info,
    // the round of the last pruning that kept it
    round: u64,
}

impl<G> Default for InfoStore<G> {
    fn default() -> Self {
        Self {
            entries: HashMap::default(),
            round: 0,
        }
    }
}

impl<G: Genome> InfoStore<G> {
    pub(crate) fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.entries.len()
    }

    // whether it has allocated memory
    #[cfg(test)]
    pub(crate) fn is_allocated(&self) -> bool {
        self.entries.capacity() > 0
    }

    // the info of `genome`
    pub(crate) fn get(&self, genome: &G) -> Option<&Info> {
        if self.entries.is_empty() {
            return None;
        }
        self.entries.get(genome).map(|entry| &entry.info)
    }

    // the info of `genome` as a `T`
    pub(crate) fn info<T: Any>(&self, genome: &G) -> Option<&T> {
        self.get(genome)?.downcast_ref()
    }

    // sets the info of `genome`, replacing any earlier info
    pub(crate) fn insert(&mut self, genome: G, info: Info) {
        self.entries.insert(genome, Entry { info, round: 0 });
    }

    // forgets the info of `genome`
    pub(crate) fn remove(&mut self, genome: &G) {
        if !self.entries.is_empty() {
            self.entries.remove(genome);
        }
    }

    // adds the infos of the last evaluations, then keeps only those of `keep`; nothing to do when
    // no evaluation returned info
    pub(crate) fn update<'a>(
        &mut self,
        evaluated: &mut Vec<(G, Info)>,
        keep: impl IntoIterator<Item = &'a G>,
    ) where
        G: 'a,
    {
        for (genome, info) in evaluated.drain(..) {
            self.insert(genome, info);
        }
        if self.entries.is_empty() {
            return;
        }
        self.round += 1;
        let round = self.round;
        for genome in keep {
            if let Some(entry) = self.entries.get_mut(genome) {
                entry.round = round;
            }
        }
        self.entries.retain(|_, entry| entry.round == round);
    }
}

impl<G> fmt::Debug for InfoStore<G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InfoStore")
            .field("len", &self.entries.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::Bits;

    fn bits(text: &str) -> Bits {
        text.chars().map(|c| c == '1').collect()
    }

    // the types that hold info keep their auto traits
    #[test]
    fn auto_traits() {
        use crate::multi::{MultiOutcome, MultiSnapshot};
        use crate::observer::{HallOfFame, Snapshot};
        fn auto<T: Send + Sync + Unpin + UnwindSafe + RefUnwindSafe>() {}
        auto::<Info>();
        auto::<InfoStore<Bits>>();
        auto::<crate::Outcome<Bits>>();
        auto::<HallOfFame<Bits>>();
        auto::<Snapshot<'static, Bits>>();
        auto::<MultiOutcome<Bits, 2>>();
        auto::<MultiSnapshot<'static, Bits, 2>>();
    }

    #[test]
    fn unused_store_does_not_allocate() {
        let mut store = InfoStore::<Bits>::default();
        store.update(&mut Vec::new(), [&bits("01"), &bits("10")]);
        assert!(store.get(&bits("01")).is_none());
        store.remove(&bits("01"));
        assert!(!store.is_allocated());
    }

    #[test]
    fn keeps_what_is_kept() {
        let mut store = InfoStore::default();
        let mut evaluated = vec![
            (bits("00"), Info::new(0_u8)),
            (bits("01"), Info::new(1_u8)),
            (bits("10"), Info::new(2_u8)),
        ];
        store.update(&mut evaluated, [&bits("01"), &bits("10"), &bits("11")]);
        assert!(evaluated.is_empty());
        assert_eq!(store.len(), 2);
        assert_eq!(store.info::<u8>(&bits("01")), Some(&1));
        assert_eq!(store.info::<u16>(&bits("01")), None);
        assert_eq!(store.info::<u8>(&bits("00")), None);
        // a new evaluation replaces the info
        store.update(&mut vec![(bits("10"), Info::new(5_u8))], [&bits("10")]);
        assert_eq!(store.len(), 1);
        assert_eq!(store.info::<u8>(&bits("10")), Some(&5));
    }
}
