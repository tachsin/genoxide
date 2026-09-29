//! Breeding shared by the multi-objective genetic algorithms.

use super::Scores;
use crate::engine::GenomeHashing;
use crate::genome::Representation;
use crate::operator::{Crossover, Mutate};
use crate::rng::Chance;
use crate::{Individual, Population, StreamRng};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::fmt;
use std::hash::{BuildHasher, BuildHasherDefault, Hash, Hasher};

// with duplicate elimination, the children rejected as copies, per child needed, before copies
// are accepted: a population of few distinct genomes still gets its children
const REJECTIONS_PER_CHILD: usize = 100;

// up to this many children, a child is compared with every genome instead of hashing the
// population: SMS-EMOA breeds one child at a time
const COMPARED_CHILDREN: usize = 8;

// a fast, deterministic 64-bit hash of a genome, the engine's genome hasher's. It only finds the
// genomes a child may equal: they are compared, so the results never depend on it, on any
// platform.
fn fingerprint<G: Hash>(genome: &G) -> u64 {
    GenomeHashing::default().hash_one(genome)
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

// genomes no longer in use, which breeding copies parents into rather than allocating: the
// parents that didn't survive and the children discarded a generation before. Neither a clone nor
// a checkpoint keeps them.
pub(crate) struct Spares<G>(Vec<G>);

impl<G> Default for Spares<G> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<G> Clone for Spares<G> {
    fn clone(&self) -> Self {
        Self::default()
    }
}

impl<G> fmt::Debug for Spares<G> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Spares({})", self.0.len())
    }
}

impl<G: Clone> Spares<G> {
    // a copy of `genome`, in a spare one if there is any
    fn copy(&mut self, genome: &G) -> G {
        match self.0.pop() {
            Some(mut spare) => {
                spare.clone_from(genome);
                spare
            }
            None => genome.clone(),
        }
    }

    // keeps a genome no longer in use
    pub(crate) fn keep(&mut self, genome: G) {
        self.0.push(genome);
    }

    // keeps the genomes of individuals no longer in use
    pub(crate) fn keep_all<const M: usize>(
        &mut self,
        individuals: impl IntoIterator<Item = Individual<G, Scores<M>>>,
    ) where
        G: crate::genome::Genome,
    {
        self.0
            .extend(individuals.into_iter().map(Individual::into_genome));
    }
}

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
    // (only when copies are accepted) inherits its scores. The children's genomes are copies of
    // their parents' in `spares`, as far as there are any, and unused ones go back there.
    pub(crate) fn breed<const M: usize>(
        &self,
        population: &Population<R::Genome, Scores<M>>,
        count: usize,
        rng: &mut StreamRng,
        mut select: impl FnMut(&mut StreamRng) -> usize,
        offspring: &mut Vec<Individual<R::Genome, Scores<M>>>,
        spares: &mut Spares<R::Genome>,
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
            let mut a = spares.copy(population[parents[0]].genome());
            let mut b = spares.copy(population[parents[1]].genome());
            if rng.chance(self.crossover_chance) {
                self.crossover
                    .crossover(&self.representation, &mut a, &mut b, rng);
            }
            for mut genome in [a, b] {
                if offspring.len() == count {
                    spares.keep(genome);
                    continue;
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
                        spares.keep(genome);
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

// the members of `population` at `indices`, each genome once: the first of its copies. A front
// never holds copies of a genome, which MOEA/D's subproblems and a population bred without
// duplicate elimination can.
#[cfg(test)]
pub(crate) fn distinct<G, const M: usize>(
    population: &Population<G, Scores<M>>,
    indices: impl IntoIterator<Item = usize>,
) -> Vec<Individual<G, Scores<M>>>
where
    G: crate::genome::Genome,
{
    let mut members = Vec::new();
    distinct_into(&mut members, population, indices);
    members
}

// `distinct`, in `members`: copied into the memory of its individuals (the previous front), which
// it then holds
pub(crate) fn distinct_into<G, const M: usize>(
    members: &mut Vec<Individual<G, Scores<M>>>,
    population: &Population<G, Scores<M>>,
    indices: impl IntoIterator<Item = usize>,
) where
    G: crate::genome::Genome,
{
    let indices = indices.into_iter();
    // room for every index: at most the population
    let (least, most) = indices.size_hint();
    let room = most.unwrap_or(least).min(population.len());
    members.reserve(room.saturating_sub(members.len()));
    // the members so far: `members[..len]`; the rest is memory to copy into
    let mut len = 0;
    // the fingerprints of the members so far, each with the position of a member that has it
    let mut seen = Fingerprints::with_capacity_and_hasher(room, Default::default());
    for index in indices {
        let individual = &population[index];
        let genome = individual.genome();
        let copy = match seen.entry(fingerprint(genome)) {
            Entry::Vacant(entry) => {
                entry.insert(len);
                false
            }
            // almost always the same genome; if not, two genomes share the fingerprint, and the
            // genome is compared with every member
            Entry::Occupied(entry) => {
                members[*entry.get()].genome() == genome
                    || members[..len]
                        .iter()
                        .any(|member| member.genome() == genome)
            }
        };
        if !copy {
            match members.get_mut(len) {
                Some(member) => member.clone_from(individual),
                None => members.push(individual.clone()),
            }
            len += 1;
        }
    }
    members.truncate(len);
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
            &mut Spares::default(),
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

    #[test]
    fn distinct_keeps_the_first_of_each_genomes_copies() {
        let scored = |genes: Vec<f64>, value: f64| {
            let mut individual = Individual::unevaluated(Reals::from(genes));
            individual.set_fitness(Scores::new([value, -value]));
            individual
        };
        let population = Population::new(vec![
            scored(vec![1.0, 2.0], 1.0),
            scored(vec![3.0, 4.0], 2.0),
            scored(vec![1.0, 2.0], 1.0),
            // the same objective values, another genome: kept
            scored(vec![5.0, 6.0], 2.0),
            scored(vec![3.0, 4.0], 2.0),
        ]);
        let genes = |members: Vec<Individual<Reals, Scores<2>>>| {
            let genomes = members.iter().map(|member| member.genome().to_vec());
            genomes.collect::<Vec<_>>()
        };
        assert_eq!(
            genes(distinct(&population, 0..5)),
            [vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]
        );
        // in the order of the indices
        assert_eq!(
            genes(distinct(&population, [4, 2, 1, 0])),
            [vec![3.0, 4.0], vec![1.0, 2.0]]
        );
        assert!(distinct(&population, []).is_empty());
    }

    #[test]
    fn distinct_compares_genomes_that_share_a_fingerprint() {
        let population: Population<Colliding, Scores<1>> = [3, 1, 3, 2, 1]
            .into_iter()
            .map(|k| Individual::unevaluated(Colliding(k)))
            .collect();
        let members = distinct(&population, 0..5);
        let genomes: Vec<u64> = members.iter().map(|member| member.genome().0).collect();
        assert_eq!(genomes, [3, 1, 2]);
    }

    #[test]
    fn spare_genomes_change_no_child() {
        use crate::genome::Real;
        use crate::operator::{PolynomialMutation, SimulatedBinaryCrossover};
        use rand::Rng;
        let representation = Real::uniform(4, -1.0..=1.0).unwrap();
        let variation = Variation {
            representation: representation.clone(),
            crossover: SimulatedBinaryCrossover::new(15.0).unwrap(),
            mutate: PolynomialMutation::per_gene(0.5, 20.0).unwrap(),
            crossover_chance: Chance::new(0.9),
            mutation_chance: Chance::new(1.0),
            eliminate_duplicates: true,
        };
        let mut rng = StreamRng::seed_from_u64(3);
        let population: Population<Reals, Scores<1>> = (0..10)
            .map(|_| Individual::unevaluated(representation.random_genome(&mut rng)))
            .collect();
        let breed = |spares: &mut Spares<Reals>| {
            let mut rng = StreamRng::seed_from_u64(4);
            let mut offspring = Vec::new();
            variation.breed(
                &population,
                9,
                &mut rng,
                |rng| rng.below(10),
                &mut offspring,
                spares,
            );
            (offspring, rng.next_u64())
        };
        // spares of other lengths and values, fewer than the children need
        let mut spares = Spares::default();
        for length in [0, 3, 9] {
            spares.keep(Reals::from(vec![7.0; length]));
        }
        let (children, next) = breed(&mut Spares::default());
        assert_eq!(breed(&mut spares), (children, next));
        // an odd number of children: the unused second genome of the last pair is kept
        assert_eq!(spares.0.len(), 1);
    }
}
