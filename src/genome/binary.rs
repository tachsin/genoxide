//! Binary genomes: fixed length bit strings, bit-packed.

use super::{Genome, Representation};
use crate::{Error, Result, StreamRng};
use rand::Rng;
use std::fmt;

const WORD_BITS: usize = u64::BITS as usize;

/// A fixed length string of bits, packed 64 per `u64` word.
///
/// ```
/// use genoxide::genome::Bits;
///
/// let mut bits: Bits = [true, false, true].into_iter().collect();
/// bits.flip(1);
/// assert_eq!(bits.count_ones(), 3);
/// assert_eq!(bits.to_string(), "111");
/// ```
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct Bits {
    // ceil(len / 64) words, the unused high bits of the last word are always zero
    words: Vec<u64>,
    len: usize,
}

impl Bits {
    /// `len` bits, all zero.
    pub fn zeros(len: usize) -> Self {
        Self {
            words: vec![0; len.div_ceil(WORD_BITS)],
            len,
        }
    }

    /// `len` bits, all one.
    pub fn ones(len: usize) -> Self {
        let mut bits = Self {
            words: vec![u64::MAX; len.div_ceil(WORD_BITS)],
            len,
        };
        bits.clear_unused_bits();
        bits
    }

    /// The number of bits.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether there are no bits.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The bit at `index`, or `None` if `index` is out of bounds.
    pub fn get(&self, index: usize) -> Option<bool> {
        (index < self.len).then(|| self.words[index / WORD_BITS] >> (index % WORD_BITS) & 1 == 1)
    }

    /// Sets the bit at `index`.
    ///
    /// # Panics
    ///
    /// If `index` is out of bounds.
    pub fn set(&mut self, index: usize, value: bool) {
        self.assert_in_bounds(index);
        let mask = 1 << (index % WORD_BITS);
        if value {
            self.words[index / WORD_BITS] |= mask;
        } else {
            self.words[index / WORD_BITS] &= !mask;
        }
    }

    /// Flips the bit at `index`.
    ///
    /// # Panics
    ///
    /// If `index` is out of bounds.
    pub fn flip(&mut self, index: usize) {
        self.assert_in_bounds(index);
        self.words[index / WORD_BITS] ^= 1 << (index % WORD_BITS);
    }

    /// The number of bits that are one.
    pub fn count_ones(&self) -> usize {
        self.words
            .iter()
            .map(|word| word.count_ones() as usize)
            .sum()
    }

    /// The number of bits that are zero.
    pub fn count_zeros(&self) -> usize {
        self.len - self.count_ones()
    }

    /// The bits, from index 0.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = bool> + '_ {
        (0..self.len).map(|index| self.words[index / WORD_BITS] >> (index % WORD_BITS) & 1 == 1)
    }

    fn assert_in_bounds(&self, index: usize) {
        assert!(
            index < self.len,
            "bit index {index} out of bounds for length {}",
            self.len
        );
    }

    fn clear_unused_bits(&mut self) {
        let used = self.len % WORD_BITS;
        if used > 0 {
            if let Some(last) = self.words.last_mut() {
                *last &= (1 << used) - 1;
            }
        }
    }
}

impl FromIterator<bool> for Bits {
    fn from_iter<I: IntoIterator<Item = bool>>(iter: I) -> Self {
        let mut words = Vec::new();
        let mut len = 0;
        for bit in iter {
            if len % WORD_BITS == 0 {
                words.push(0);
            }
            if bit {
                *words.last_mut().expect("pushed above") |= 1 << (len % WORD_BITS);
            }
            len += 1;
        }
        Self { words, len }
    }
}

impl fmt::Display for Bits {
    /// The bits as `0` and `1`, from index 0.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.iter()
            .try_for_each(|bit| f.write_str(if bit { "1" } else { "0" }))
    }
}

impl fmt::Debug for Bits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bits({self})")
    }
}

impl Genome for Bits {
    fn len(&self) -> usize {
        self.len
    }
}

/// Binary genomes ([`Bits`]) of a fixed length.
///
/// ```
/// use genoxide::genome::{Binary, Representation};
/// use genoxide::StreamRng;
///
/// let binary = Binary::new(100)?;
/// let genome = binary.random_genome(&mut StreamRng::seed_from_u64(0));
/// assert_eq!(genome.len(), 100);
/// assert!(binary.validate(&genome).is_ok());
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Binary {
    len: usize,
}

impl Binary {
    /// Binary genomes of `len` bits. `len` must be at least 1.
    pub fn new(len: usize) -> Result<Self> {
        if len == 0 {
            return Err(Error::InvalidSetting {
                setting: "len",
                reason: "a binary genome needs at least 1 bit".to_string(),
            });
        }
        Ok(Self { len })
    }
}

impl Representation for Binary {
    type Genome = Bits;

    fn genome_len(&self) -> usize {
        self.len
    }

    fn random_genome(&self, rng: &mut StreamRng) -> Bits {
        let mut bits = Bits {
            words: (0..self.len.div_ceil(WORD_BITS))
                .map(|_| rng.next_u64())
                .collect(),
            len: self.len,
        };
        bits.clear_unused_bits();
        bits
    }

    fn validate(&self, genome: &Bits) -> Result<()> {
        if genome.len() == self.len {
            Ok(())
        } else {
            Err(Error::InvalidGenome {
                reason: format!("expected {} bits, got {}", self.len, genome.len()),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn bools() -> impl Strategy<Value = Vec<bool>> {
        prop::collection::vec(any::<bool>(), 0..300)
    }

    fn tail_is_clear(bits: &Bits) -> bool {
        bits.words.len() == bits.len.div_ceil(WORD_BITS)
            && (bits.len % WORD_BITS == 0
                || bits
                    .words
                    .last()
                    .is_none_or(|last| last >> (bits.len % WORD_BITS) == 0))
    }

    #[test]
    fn zeros_and_ones() {
        for len in [0, 1, 63, 64, 65, 200] {
            assert_eq!(Bits::zeros(len).count_ones(), 0);
            assert_eq!(Bits::ones(len).count_ones(), len);
            assert!(tail_is_clear(&Bits::ones(len)));
        }
    }

    #[test]
    #[should_panic(expected = "bit index 3 out of bounds for length 3")]
    fn flip_out_of_bounds() {
        Bits::zeros(3).flip(3);
    }

    #[test]
    fn binary_requires_a_bit() {
        assert!(matches!(
            Binary::new(0),
            Err(Error::InvalidSetting { setting: "len", .. })
        ));
    }

    #[test]
    fn validate_length() {
        let binary = Binary::new(10).unwrap();
        assert!(binary.validate(&Bits::zeros(10)).is_ok());
        assert_eq!(
            binary.validate(&Bits::zeros(9)),
            Err(Error::InvalidGenome {
                reason: "expected 10 bits, got 9".to_string()
            })
        );
    }

    #[test]
    fn random_genome_is_uniform() {
        // 64000 bits: the fraction of ones is within 1% of 0.5
        let genome = Binary::new(64_000)
            .unwrap()
            .random_genome(&mut StreamRng::seed_from_u64(0));
        let fraction = genome.count_ones() as f64 / 64_000.0;
        assert!((fraction - 0.5).abs() < 0.01, "fraction of ones {fraction}");
    }

    proptest! {
        #[test]
        fn from_bools_roundtrip(values in bools()) {
            let bits: Bits = values.iter().copied().collect();
            prop_assert_eq!(bits.len(), values.len());
            prop_assert_eq!(bits.iter().collect::<Vec<_>>(), values.clone());
            prop_assert_eq!(bits.count_ones(), values.iter().filter(|&&v| v).count());
            prop_assert!(tail_is_clear(&bits));
            let text: String = values.iter().map(|&v| if v { '1' } else { '0' }).collect();
            prop_assert_eq!(bits.to_string(), text);
        }

        #[test]
        fn flip_changes_exactly_one_bit(values in bools(), index in any::<prop::sample::Index>()) {
            prop_assume!(!values.is_empty());
            let index = index.index(values.len());
            let original: Bits = values.iter().copied().collect();
            let mut bits = original.clone();
            bits.flip(index);
            prop_assert_ne!(&bits, &original);
            prop_assert_eq!(bits.get(index), original.get(index).map(|v| !v));
            let differences = bits.iter().zip(original.iter()).filter(|(a, b)| a != b).count();
            prop_assert_eq!(differences, 1);
            bits.flip(index);
            prop_assert_eq!(bits, original);
        }

        #[test]
        fn set_and_get(values in bools(), index in any::<prop::sample::Index>(), value: bool) {
            prop_assume!(!values.is_empty());
            let index = index.index(values.len());
            let mut bits: Bits = values.iter().copied().collect();
            bits.set(index, value);
            prop_assert_eq!(bits.get(index), Some(value));
            prop_assert_eq!(bits.get(values.len()), None);
            prop_assert!(tail_is_clear(&bits));
        }

        #[test]
        fn random_genome_is_valid_and_reproducible(len in 1usize..300, seed: u64) {
            let binary = Binary::new(len).unwrap();
            let genome = binary.random_genome(&mut StreamRng::seed_from_u64(seed));
            prop_assert!(binary.validate(&genome).is_ok());
            prop_assert!(tail_is_clear(&genome));
            prop_assert_eq!(genome, binary.random_genome(&mut StreamRng::seed_from_u64(seed)));
        }
    }
}
