//! Portable, seedable random number generation with independent streams.

use rand::{Rng, SeedableRng, TryRng};
use rand_chacha::ChaCha8Rng;
use std::convert::Infallible;

/// The random number generator used throughout genoxide.
///
/// - **Portable:** the same seed gives the same numbers on every platform and every genoxide
///   version with the same major version (ChaCha8, checked against fixed values in the tests).
/// - **Independent streams:** [`derive`](StreamRng::derive) gives a generator for a run, island
///   or worker that depends only on this generator's seed and an id. It doesn't depend on how many
///   numbers were drawn before, so parallel work stays reproducible.
///
/// It implements rand's [`Rng`], so rand's distributions and [`RngExt`](rand::RngExt) methods
/// can be used with it.
///
/// ```
/// use genoxide::StreamRng;
/// use rand::RngExt;
///
/// let mut rng = StreamRng::seed_from_u64(42);
/// let a: f64 = rng.random();
/// assert_eq!(a, StreamRng::seed_from_u64(42).random::<f64>());
///
/// // independent generators for two parallel runs
/// let mut run_0 = rng.derive(0);
/// let mut run_1 = rng.derive(1);
/// assert_ne!(run_0.random::<u64>(), run_1.random::<u64>());
/// ```
#[derive(Clone, Debug)]
pub struct StreamRng {
    inner: ChaCha8Rng,
}

impl StreamRng {
    /// A generator seeded with `seed`: the same seed always gives the same numbers.
    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            inner: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// A generator seeded from the operating system's randomness, so not reproducible.
    pub fn from_entropy() -> Self {
        Self {
            inner: ChaCha8Rng::from_rng(&mut rand::rng()),
        }
    }

    /// An independent generator derived from this generator's seed and `id`.
    ///
    /// The result depends only on the seed and `id`, not on how many numbers this generator has
    /// produced. Different ids give independent generators, and derived generators can be derived
    /// again (e.g. run → island → worker) without collisions.
    pub fn derive(&self, id: u64) -> Self {
        // The child seed is read from stream `id` of the parent seed, far beyond the part of the
        // key stream the parent itself uses (stream 0, from word 0), so it never overlaps the
        // parent's output. A new seed (instead of only a new stream) keeps grandchildren distinct.
        let mut source = ChaCha8Rng::from_seed(self.inner.get_seed());
        source.set_stream(id);
        source.set_word_pos(DERIVE_WORD_POS);
        let mut seed = [0u8; 32];
        source.fill_bytes(&mut seed);
        Self {
            inner: ChaCha8Rng::from_seed(seed),
        }
    }
}

// 2^64 words (of 32 bits) into a stream: unreachable by normal use of a generator
const DERIVE_WORD_POS: u128 = 1 << 64;

impl TryRng for StreamRng {
    type Error = Infallible;

    #[inline]
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok(self.inner.next_u32())
    }

    #[inline]
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        Ok(self.inner.next_u64())
    }

    #[inline]
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        self.inner.fill_bytes(dst);
        Ok(())
    }
}

impl SeedableRng for StreamRng {
    type Seed = [u8; 32];

    fn from_seed(seed: Self::Seed) -> Self {
        Self {
            inner: ChaCha8Rng::from_seed(seed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    // explicit, as the proptest prelude also exports an `Rng`
    use rand::Rng;

    fn first_u64s(mut rng: StreamRng) -> [u64; 3] {
        [rng.next_u64(), rng.next_u64(), rng.next_u64()]
    }

    /// Fixed values: these must never change for the same major version, on any platform.
    #[test]
    fn portable_values() {
        assert_eq!(
            first_u64s(StreamRng::seed_from_u64(42)),
            PORTABLE_SEED_42,
            "the numbers for a seed changed, which breaks reproducibility"
        );
        assert_eq!(
            first_u64s(StreamRng::seed_from_u64(42).derive(7)),
            PORTABLE_SEED_42_DERIVE_7,
            "the numbers for a derived stream changed, which breaks reproducibility"
        );
    }

    const PORTABLE_SEED_42: [u64; 3] = [
        12578764544318200737,
        17529487244874322312,
        7886285670807131020,
    ];
    const PORTABLE_SEED_42_DERIVE_7: [u64; 3] = [
        1169889660524673768,
        4703768454804564706,
        15961940369642500281,
    ];

    #[test]
    fn derive_is_independent_of_the_parent_position() {
        let parent = StreamRng::seed_from_u64(1);
        let mut used_parent = parent.clone();
        for _ in 0..1000 {
            used_parent.next_u64();
        }
        assert_eq!(
            first_u64s(parent.derive(3)),
            first_u64s(used_parent.derive(3))
        );
    }

    #[test]
    fn derive_does_not_collide() {
        let rng = StreamRng::seed_from_u64(1);
        let parent = first_u64s(rng.clone());
        let child_0 = first_u64s(rng.derive(0));
        let child_1 = first_u64s(rng.derive(1));
        let grandchild = first_u64s(rng.derive(0).derive(1));
        let all = [parent, child_0, child_1, grandchild];
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "streams {i} and {j} collide");
            }
        }
    }

    #[test]
    fn from_entropy_differs() {
        assert_ne!(
            first_u64s(StreamRng::from_entropy()),
            first_u64s(StreamRng::from_entropy())
        );
    }

    proptest! {
        #[test]
        fn same_seed_same_numbers(seed: u64, id: u64) {
            prop_assert_eq!(
                first_u64s(StreamRng::seed_from_u64(seed)),
                first_u64s(StreamRng::seed_from_u64(seed))
            );
            prop_assert_eq!(
                first_u64s(StreamRng::seed_from_u64(seed).derive(id)),
                first_u64s(StreamRng::seed_from_u64(seed).derive(id))
            );
        }

        #[test]
        fn different_ids_give_different_streams(seed: u64, id: u64) {
            let rng = StreamRng::seed_from_u64(seed);
            prop_assert_ne!(
                first_u64s(rng.derive(id)),
                first_u64s(rng.derive(id.wrapping_add(1)))
            );
        }
    }
}
