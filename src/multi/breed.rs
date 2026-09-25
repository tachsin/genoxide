//! Breeding shared by the multi-objective genetic algorithms.

use super::Scores;
use crate::genome::Representation;
use crate::operator::{Crossover, Mutate};
use crate::rng::Chance;
use crate::{Individual, Population, StreamRng};
use std::collections::HashSet;
use std::hash::{BuildHasherDefault, Hash, Hasher};

// with duplicate elimination, the children rejected as copies, per child needed, before copies
// are accepted: a population of few distinct genomes still gets its children
const REJECTIONS_PER_CHILD: usize = 100;

// up to this many children, a child is compared with every genome instead of hashing the
// population: SMS-EMOA breeds one child at a time
const COMPARED_CHILDREN: usize = 8;

// a fast, deterministic 64-bit hash of a genome (FxHash's mixing, and SplitMix64's finalizer so
// that the low bits, which pick the set's bucket, depend on every bit). Two genomes with the same
// fingerprint count as copies: at worst, a rare unique child is bred again.
fn fingerprint<G: Hash>(genome: &G) -> u64 {
    let mut hasher = FxHasher(0);
    genome.hash(&mut hasher);
    let mut x = hasher.0;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

#[derive(Default)]
struct FxHasher(u64);

impl Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for chunk in &mut chunks {
            let word = u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
            self.write_u64(word);
        }
        for &byte in chunks.remainder() {
            self.write_u64(u64::from(byte));
        }
    }

    fn write_u64(&mut self, word: u64) {
        self.0 = (self.0.rotate_left(5) ^ word).wrapping_mul(0x517c_c1b7_2722_0a95);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

// a set of fingerprints, which are hashes already
#[derive(Default)]
struct Identity(u64);

impl Hasher for Identity {
    fn write(&mut self, _bytes: &[u8]) {
        unreachable!("only fingerprints are hashed")
    }

    fn write_u64(&mut self, value: u64) {
        self.0 = value;
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

type Fingerprints = HashSet<u64, BuildHasherDefault<Identity>>;

// the operators and rates of a genetic algorithm
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub(crate) struct Variation<R, C, X> {
    pub(crate) representation: R,
    pub(crate) crossover: C,
    pub(crate) mutate: X,
    pub(crate) crossover_chance: Chance,
    pub(crate) mutation_chance: Chance,
    // whether a child that equals a member of the population or an earlier child is bred again;
    // a checkpoint from before this setting resumes without it, as it was saved
    #[cfg_attr(feature = "serde", serde(default))]
    pub(crate) eliminate_duplicates: bool,
}

impl<R, C, X> Variation<R, C, X>
where
    R: Representation,
    C: Crossover<R>,
    X: Mutate<R>,
{
    // `count` children of pairs of parents chosen by `select`: recombined with the crossover
    // chance, each mutated with the mutation chance. With duplicate elimination, a child equal to
    // a member of the population or to an earlier child is dropped. A child equal to a parent
    // (only when copies are accepted) inherits its scores.
    pub(crate) fn breed<const M: usize>(
        &self,
        population: &Population<R::Genome, Scores<M>>,
        count: usize,
        rng: &mut StreamRng,
        mut select: impl FnMut(&mut StreamRng) -> usize,
        offspring: &mut Vec<Individual<R::Genome, Scores<M>>>,
    ) {
        offspring.clear();
        // the fingerprints of the population and of the children so far, for many children
        let hashed = self.eliminate_duplicates && count > COMPARED_CHILDREN;
        let mut seen = Fingerprints::default();
        if hashed {
            seen.reserve(population.len() + count);
            seen.extend(population.iter().map(|x| fingerprint(x.genome())));
        }
        let mut rejections = count.saturating_mul(REJECTIONS_PER_CHILD);
        while offspring.len() < count {
            let parents = [select(rng), select(rng)];
            let mut a = population[parents[0]].genome().clone();
            let mut b = population[parents[1]].genome().clone();
            if rng.chance(self.crossover_chance) {
                self.crossover
                    .crossover(&self.representation, &mut a, &mut b, rng);
            }
            for mut genome in [a, b] {
                if offspring.len() == count {
                    break;
                }
                if rng.chance(self.mutation_chance) {
                    self.mutate.mutate(&self.representation, &mut genome, rng);
                }
                if self.eliminate_duplicates && rejections > 0 {
                    let copy = if hashed {
                        !seen.insert(fingerprint(&genome))
                    } else {
                        population.iter().any(|x| x.genome() == &genome)
                            || offspring.iter().any(|x| x.genome() == &genome)
                    };
                    if copy {
                        rejections -= 1;
                        continue;
                    }
                }
                let inherited = parents
                    .iter()
                    .map(|&parent| &population[parent])
                    .find(|parent| parent.genome() == &genome)
                    .and_then(Individual::fitness);
                let mut child = Individual::unevaluated(genome);
                if let Some(scores) = inherited {
                    child.set_fitness(scores);
                }
                offspring.push(child);
            }
        }
    }
}

// the scores of individuals, invalid if not evaluated
pub(crate) fn scores_of<G, const M: usize>(
    individuals: &[Individual<G, Scores<M>>],
) -> Vec<Scores<M>>
where
    G: crate::genome::Genome,
{
    individuals
        .iter()
        .map(|individual| individual.fitness().unwrap_or(Scores::invalid()))
        .collect()
}
