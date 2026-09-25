//! Breeding shared by the multi-objective genetic algorithms.

use super::Scores;
use crate::genome::Representation;
use crate::operator::{Crossover, Mutate};
use crate::rng::Chance;
use crate::{Individual, Population, StreamRng};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::{BuildHasherDefault, Hash, Hasher};

// with duplicate elimination, the children rejected as copies, per child needed, before copies
// are accepted: a population of few distinct genomes still gets its children
const REJECTIONS_PER_CHILD: usize = 100;

// up to this many children, a child is compared with every genome instead of hashing the
// population: SMS-EMOA breeds one child at a time
const COMPARED_CHILDREN: usize = 8;

// a fast, deterministic 64-bit hash of a genome. It only finds the genomes a child may equal:
// they are compared, so the results never depend on it, on any platform.
fn fingerprint<G: Hash>(genome: &G) -> u64 {
    let mut hasher = Fingerprinter(0x243f_6a88_85a3_08d3);
    genome.hash(&mut hasher);
    // SplitMix64's finalizer: the low bits, which pick the map's bucket, depend on every bit
    let mut x = hasher.0;
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

// every word is mixed in with a folded multiply (the two halves of a 128-bit product xored), so a
// difference in any bit spreads to most bits before the next word. FxHash's rotate and multiply
// only spread a difference upwards: bit 63 of one word, rotated to bit 4, cancelled bit 4 of the
// next.
struct Fingerprinter(u64);

impl Hasher for Fingerprinter {
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
        let product = u128::from(self.0 ^ word) * 0x9e37_79b9_7f4a_7c15;
        self.0 = product as u64 ^ (product >> 64) as u64;
    }

    // every integer is a word, whatever its size, so that a length (a `usize`) hashes the same on
    // 32 and 64 bits. A slice of `usize`, as in `Order`, still reaches `write` as 4 or 8 bytes each.
    fn write_u8(&mut self, n: u8) {
        self.write_u64(n.into());
    }

    fn write_u16(&mut self, n: u16) {
        self.write_u64(n.into());
    }

    fn write_u32(&mut self, n: u32) {
        self.write_u64(n.into());
    }

    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }

    fn write_i8(&mut self, n: i8) {
        self.write_i64(n.into());
    }

    fn write_i16(&mut self, n: i16) {
        self.write_i64(n.into());
    }

    fn write_i32(&mut self, n: i32) {
        self.write_i64(n.into());
    }

    fn write_i64(&mut self, n: i64) {
        self.write_u64(n as u64);
    }

    fn write_isize(&mut self, n: isize) {
        self.write_i64(n as i64);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

// a map keyed by fingerprints, which are hashes already
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

// the fingerprints of the population and of the children so far, each with the index of a genome
// that has it: the population's members, then the children
type Fingerprints = HashMap<u64, usize, BuildHasherDefault<Identity>>;

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
        // for many children, the genomes a child may equal are found by fingerprint
        let hashed = self.eliminate_duplicates && count > COMPARED_CHILDREN;
        let mut seen = Fingerprints::default();
        if hashed {
            seen.reserve(population.len() + count);
            for (index, member) in population.iter().enumerate() {
                seen.entry(fingerprint(member.genome())).or_insert(index);
            }
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
                    let anywhere = |genome: &R::Genome| {
                        population.iter().any(|x| x.genome() == genome)
                            || offspring.iter().any(|x| x.genome() == genome)
                    };
                    let copy = if hashed {
                        match seen.entry(fingerprint(&genome)) {
                            // the index the child gets, as it's kept
                            Entry::Vacant(entry) => {
                                entry.insert(population.len() + offspring.len());
                                false
                            }
                            // almost always the same genome; if not, two genomes share the
                            // fingerprint, and the child is compared with every genome
                            Entry::Occupied(entry) => {
                                let index = *entry.get();
                                let other = match index.checked_sub(population.len()) {
                                    None => population[index].genome(),
                                    Some(child) => offspring[child].genome(),
                                };
                                other == &genome || anywhere(&genome)
                            }
                        }
                    } else {
                        anywhere(&genome)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::{Bits, Integers, Reals};
    use crate::operator::NoCrossover;
    use crate::{Result, StreamRng};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    #[test]
    fn genomes_two_bits_apart_have_different_fingerprints() {
        // with FxHash's mixing, bit 63 of one word cancelled bit 4 of the next
        let a = Bits::zeros(70);
        let mut b = a.clone();
        b.flip(63);
        b.flip(68);
        assert_ne!(fingerprint(&a), fingerprint(&b));
        // and a sign flipped: bit 63 of one gene, and bit 4 of the next
        let a = Reals::from(vec![1.5, 2.0]);
        let b = Reals::from(vec![-1.5, f64::from_bits(2.0f64.to_bits() ^ 16)]);
        assert_ne!(fingerprint(&a), fingerprint(&b));
    }

    #[test]
    fn fingerprints_are_the_same_on_32_and_64_bits() {
        let mut bits = Bits::zeros(70);
        bits.flip(3);
        bits.flip(68);
        assert_eq!(fingerprint(&bits), 0x9311_3c42_a4e1_30d9);
        assert_eq!(
            fingerprint(&Reals::from(vec![1.5, -2.0])),
            0x03da_8c31_8861_8236
        );
        assert_eq!(
            fingerprint(&Integers::from(vec![7, -1, i64::MIN])),
            0x7a0b_97ad_6020_6cff
        );
    }

    // a genome whose fingerprints are all the same
    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Colliding(u64);

    impl Hash for Colliding {
        fn hash<H: Hasher>(&self, _: &mut H) {}
    }

    impl crate::genome::Genome for Colliding {
        fn len(&self) -> usize {
            1
        }
    }

    #[derive(Clone, Debug)]
    struct AnyU64;

    impl Representation for AnyU64 {
        type Genome = Colliding;

        fn genome_len(&self) -> usize {
            1
        }

        fn random_genome(&self, rng: &mut StreamRng) -> Colliding {
            Colliding(rng.below_u64(u64::MAX))
        }

        fn validate(&self, _: &Colliding) -> Result<()> {
            Ok(())
        }
    }

    // the `n`th mutation makes the genome `n / repeats`, and counts the children bred
    #[derive(Clone, Debug)]
    struct Numbered {
        repeats: u64,
        count: Arc<AtomicU64>,
    }

    impl Mutate<AnyU64> for Numbered {
        fn mutate(&self, _: &AnyU64, genome: &mut Colliding, _: &mut StreamRng) {
            *genome = Colliding(self.count.fetch_add(1, Ordering::Relaxed) / self.repeats);
        }
    }

    // the mutations needed for 20 children of a population of 20 genomes that are all different
    // from the children, all with the same fingerprint
    fn mutations_for_20_children(repeats: u64) -> u64 {
        let count = Arc::new(AtomicU64::new(0));
        let variation = Variation {
            representation: AnyU64,
            crossover: NoCrossover,
            mutate: Numbered {
                repeats,
                count: Arc::clone(&count),
            },
            crossover_chance: Chance::new(0.0),
            mutation_chance: Chance::new(1.0),
            eliminate_duplicates: true,
        };
        let population: Population<Colliding, Scores<1>> = (0..20)
            .map(|k| Individual::unevaluated(Colliding(u64::MAX - k)))
            .collect();
        let mut rng = StreamRng::seed_from_u64(1);
        let mut offspring = Vec::new();
        variation.breed(
            &population,
            20,
            &mut rng,
            |rng| rng.below(20),
            &mut offspring,
        );
        let children: Vec<u64> = offspring.iter().map(|x| x.genome().0).collect();
        assert_eq!(children, (0..20).collect::<Vec<_>>());
        count.load(Ordering::Relaxed)
    }

    #[test]
    fn a_child_is_a_copy_only_if_it_equals_a_genome_with_its_fingerprint() {
        // 20 different children: none is bred again
        assert_eq!(mutations_for_20_children(1), 20);
        // every child twice in a row: every second one is a copy, but the last isn't needed
        assert_eq!(mutations_for_20_children(2), 39);
    }
}
