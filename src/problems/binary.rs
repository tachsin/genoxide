//! Binary and combinatorial test problems: functions of bit strings, on [`Binary`] genomes, all
//! maximized.
//!
//! | Problem | Bits (default) | Maximum |
//! |---|---|---|
//! | [`OneMax`] | any (100) | n, at all ones |
//! | [`LeadingOnes`] | any (100) | n, at all ones |
//! | [`Trap`] | blocks × k (10 × 4) | blocks × b, at all ones; the deceptive attractor is all zeros |
//! | [`RoyalRoad`] | blocks × block size (8 × 8) | 64 for R1, 256 for R2, at all ones |
//! | [`NkLandscape`] | n | instance-specific: exhaustive search or dynamic programming |
//! | [`Knapsack`] | items | instance-specific: dynamic programming |
//!
//! Each is a [`FitnessFunction`] of [`Bits`] for an [`Engine`](crate::Engine) as it is, and a
//! [`Problem`] whose [`objective`](Problem::objective) is
//! [`Maximize`](crate::Objective::Maximize), the default of the algorithms:
//!
//! ```
//! use genoxide::prelude::*;
//! use genoxide::problems::Problem;
//! use genoxide::problems::binary::LeadingOnes;
//!
//! // the (1+1) evolutionary algorithm of Droste, Jansen and Wegener: one string, each bit
//! // flipped with probability 1/n, the child kept if it's no worse
//! let problem = LeadingOnes::new(50);
//! let one_plus_one = LocalSearch::builder(problem.representation())
//!     .neighbor(BitFlip::per_gene(1.0 / 50.0)?)
//!     .seed(1)
//!     .build()?;
//! let optimum = problem.optimum().expect("known").value();
//! let outcome = Engine::new(one_plus_one, problem)
//!     .stop_when(Stop::target(optimum).or(Stop::evaluations(100_000)))
//!     .run()?;
//! assert_eq!(outcome.best_fitness(), Fitness::new(50.0));
//! # Ok::<(), genoxide::Error>(())
//! ```
//!
//! [`NkLandscape`] and [`Knapsack`] are instances generated from a seed with genoxide's portable
//! [`StreamRng`](crate::StreamRng): the same seed gives the same landscape or the same items on
//! every platform. Their optimum isn't known in closed form, and
//! [`optimum`](Problem::optimum) computes it, exactly: by dynamic programming (NK landscapes with
//! adjacent neighborhoods, and the knapsack), or by evaluating every bit string (small NK
//! landscapes).
//!
//! These problems aren't in [`all`](super::all), which lists the problems on real genomes.
//!
//! The definitions are taken from these papers, read for them:
//!
//! - Droste, S., Jansen, T. and Wegener, I. (2002). On the analysis of the (1+1) evolutionary
//!   algorithm. *Theoretical Computer Science* 276(1-2): 51-81.
//!   doi:10.1016/S0304-3975(01)00182-7 (OneMax, Definition 9; LeadingOnes, Definition 16)
//! - Ackley, D. H. (1987). *A Connectionist Machine for Genetic Hillclimbing.* Kluwer Academic
//!   Publishers. doi:10.1007/978-1-4613-1997-9 ("One Max" and "Trap", section 3.3)
//! - Deb, K. and Goldberg, D. E. (1993). Analyzing deception in trap functions. In *Foundations
//!   of Genetic Algorithms 2*, Morgan Kaufmann: 93-108. doi:10.1016/B978-0-08-094832-4.50012-X
//! - Mitchell, M., Forrest, S. and Holland, J. H. (1992). The royal road for genetic algorithms:
//!   fitness landscapes and GA performance. *Proceedings of the First European Conference on
//!   Artificial Life*, MIT Press: 245-254 (R2, Figure 1)
//! - Mitchell, M., Holland, J. H. and Forrest, S. (1994). When will a genetic algorithm
//!   outperform hill climbing? *Advances in Neural Information Processing Systems 6*, Morgan
//!   Kaufmann: 51-58 (R1, Figure 1)
//! - Kauffman, S. A. and Weinberger, E. D. (1989). The NK model of rugged fitness landscapes and
//!   its application to maturation of the immune response. *Journal of Theoretical Biology*
//!   141(2): 211-245. doi:10.1016/S0022-5193(89)80019-0
//! - Pisinger, D. (2005). Where are the hard knapsack problems? *Computers & Operations Research*
//!   32(9): 2271-2284. doi:10.1016/j.cor.2004.03.002

mod knapsack;
mod nk;

pub use knapsack::{Knapsack, KnapsackClass, KnapsackGenerator, SpannerDistribution};
pub use nk::{Neighborhood, NkLandscape};

use super::{Optimum, Problem};
use crate::Objective;
use crate::engine::FitnessFunction;
use crate::genome::{Binary, Bits};

// the most bits of a genome, as `Binary::new` accepts
const MAX_BITS: usize = 1 << 24;

const DROSTE: &str = "Droste, S., Jansen, T. and Wegener, I. (2002). On the analysis of the \
                      (1+1) evolutionary algorithm. Theoretical Computer Science 276(1-2): 51-81.";
const DROSTE_URL: &str = "https://doi.org/10.1016/S0304-3975(01)00182-7";

// a genome length checked by the constructor
fn binary(bits: usize) -> Binary {
    Binary::new(bits).expect("checked by the constructor")
}

// the genome length of a problem, or a panic that names it
fn check_length(name: &str, x: &Bits, bits: usize) {
    assert_eq!(x.len(), bits, "{name} takes genomes of {bits} bits");
}

// a number of bits from 1 to 2^24, or a panic that names the problem
fn check_bits(name: &str, bits: usize) -> usize {
    assert!(
        (1..=MAX_BITS).contains(&bits),
        "{name} takes 1 to 2^24 bits, not {bits}"
    );
    bits
}

// the number of ones among the `len` bits of `x` from `start`
fn ones_in(x: &Bits, start: usize, len: usize) -> usize {
    let words = x.as_words();
    let (mut count, mut at, end) = (0, start, start + len);
    while at < end {
        let (word, offset) = (at / 64, at % 64);
        let take = (64 - offset).min(end - at);
        let mask = if take == 64 {
            u64::MAX
        } else {
            ((1u64 << take) - 1) << offset
        };
        count += (words[word] & mask).count_ones() as usize;
        at += take;
    }
    count
}

// ---- OneMax ----------------------------------------------------------------------------------

/// OneMax: the number of ones, `Σ xᵢ`, the simplest function of bit strings.
///
/// Each bit counts on its own, so every string but the optimum has a neighbor one flip away that
/// is better: a hill climber climbs straight to the top. It's the baseline of the binary problems.
/// The (1+1) evolutionary algorithm (one string, each bit flipped with probability 1/n, the child
/// kept if it's no worse) needs Θ(n log n) evaluations on average (Droste et al., 2002, Lemma 10,
/// for every linear function with nonzero weights).
///
/// Maximum n at all ones; 100 bits by default.
///
/// Droste, S., Jansen, T. and Wegener, I. (2002). On the analysis of the (1+1) evolutionary
/// algorithm. *Theoretical Computer Science* 276(1-2): 51-81, Definition 9, where it's the linear
/// function with all weights 1. The function has no single origin; Ackley (1987, *A Connectionist
/// Machine for Genetic Hillclimbing*, section 3.3.1) tests a "One Max" that is ten times the
/// number of ones. Both read; Mühlenbein's (1992) analysis, which Droste et al. cite, wasn't.
///
/// ```
/// use genoxide::genome::Bits;
/// use genoxide::problems::binary::OneMax;
/// use genoxide::prelude::*;
///
/// let x: Bits = [true, false, true, true].into_iter().collect();
/// assert_eq!(OneMax::new(4).evaluate(&x), 3.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OneMax {
    bits: usize,
}

impl OneMax {
    /// OneMax of `bits` bits.
    ///
    /// # Panics
    ///
    /// If `bits` is 0 or above 2^24.
    pub fn new(bits: usize) -> Self {
        Self {
            bits: check_bits("OneMax", bits),
        }
    }

    /// The number of bits.
    pub fn bits(&self) -> usize {
        self.bits
    }
}

impl Default for OneMax {
    /// OneMax of 100 bits.
    fn default() -> Self {
        Self::new(100)
    }
}

impl FitnessFunction<Bits> for OneMax {
    type Output = f64;

    /// The number of ones of `x`.
    ///
    /// # Panics
    ///
    /// If `x` doesn't have [`bits`](OneMax::bits) bits.
    fn evaluate(&self, x: &Bits) -> f64 {
        check_length("OneMax", x, self.bits);
        x.count_ones() as f64
    }
}

impl Problem for OneMax {
    type Representation = Binary;

    fn name(&self) -> &'static str {
        "OneMax"
    }

    fn representation(&self) -> Binary {
        binary(self.bits)
    }

    fn objective(&self) -> Objective {
        Objective::Maximize
    }

    fn optimum(&self) -> Option<Optimum<Bits>> {
        Some(Optimum::proven(
            self.bits as f64,
            vec![Bits::ones(self.bits)],
        ))
    }

    fn reference(&self) -> &'static str {
        DROSTE
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some(DROSTE_URL)
    }
}

// ---- LeadingOnes -----------------------------------------------------------------------------

/// LeadingOnes: the number of ones before the first zero, `Σᵢ Πⱼ≤ᵢ xⱼ`.
///
/// Unimodal, since appending a one to the leading ones always improves a string, but only one
/// bit at a time can: the first zero. The bits after it don't count until the leading ones reach
/// them. The (1+1) evolutionary algorithm (one string, each bit flipped with probability 1/n, the
/// child kept if it's no worse) needs Θ(n²) evaluations on average, and at most e n² (Droste et
/// al., 2002, Theorem 17), where OneMax needs Θ(n log n): Droste et al. give it to disprove the
/// remark, which they attribute to Mühlenbein, that every unimodal function takes O(n log n).
///
/// Maximum n at all ones; 100 bits by default.
///
/// Droste, S., Jansen, T. and Wegener, I. (2002). On the analysis of the (1+1) evolutionary
/// algorithm. *Theoretical Computer Science* 276(1-2): 51-81, Definition 16 and Theorem 17. They
/// take the function from Rudolph, G. (1997). *Convergence Properties of Evolutionary
/// Algorithms*, Kovač, Hamburg, who proved the O(n²) upper bound; Rudolph's book wasn't
/// available to check.
///
/// ```
/// use genoxide::genome::Bits;
/// use genoxide::problems::binary::LeadingOnes;
/// use genoxide::prelude::*;
///
/// let x: Bits = [true, true, false, true].into_iter().collect();
/// assert_eq!(LeadingOnes::new(4).evaluate(&x), 2.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LeadingOnes {
    bits: usize,
}

impl LeadingOnes {
    /// LeadingOnes of `bits` bits.
    ///
    /// # Panics
    ///
    /// If `bits` is 0 or above 2^24.
    pub fn new(bits: usize) -> Self {
        Self {
            bits: check_bits("LeadingOnes", bits),
        }
    }

    /// The number of bits.
    pub fn bits(&self) -> usize {
        self.bits
    }
}

impl Default for LeadingOnes {
    /// LeadingOnes of 100 bits.
    fn default() -> Self {
        Self::new(100)
    }
}

impl FitnessFunction<Bits> for LeadingOnes {
    type Output = f64;

    /// The number of leading ones of `x`.
    ///
    /// # Panics
    ///
    /// If `x` doesn't have [`bits`](LeadingOnes::bits) bits.
    fn evaluate(&self, x: &Bits) -> f64 {
        check_length("LeadingOnes", x, self.bits);
        let mut count = 0;
        for &word in x.as_words() {
            // bit i is bit i % 64 of word i / 64: the leading ones are a word's trailing ones
            count += word.trailing_ones() as usize;
            if word != u64::MAX {
                break;
            }
        }
        // the unused bits of the last word are zero
        count.min(self.bits) as f64
    }
}

impl Problem for LeadingOnes {
    type Representation = Binary;

    fn name(&self) -> &'static str {
        "LeadingOnes"
    }

    fn representation(&self) -> Binary {
        binary(self.bits)
    }

    fn objective(&self) -> Objective {
        Objective::Maximize
    }

    fn optimum(&self) -> Option<Optimum<Bits>> {
        Some(Optimum::proven(
            self.bits as f64,
            vec![Bits::ones(self.bits)],
        ))
    }

    fn reference(&self) -> &'static str {
        DROSTE
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some(DROSTE_URL)
    }
}

// ---- deceptive trap --------------------------------------------------------------------------

/// The deceptive trap: blocks of k bits, each scored by a trap function of its number of ones,
/// that leads away from the optimum.
///
/// A block with u ones scores
///
/// ```text
/// f(u) = a (z − u) / z          for u ≤ z,
///        b (u − z) / (k − z)    otherwise,
/// ```
///
/// and the string scores the sum of its blocks, which are consecutive: bits 0 to k − 1, k to
/// 2k − 1 and so on. f falls from a at u = 0, the deceptive attractor, to 0 at the slope change
/// z, and rises to b > a at u = k, the optimum. Only a block with more than z ones leads up to
/// it; from fewer than z, every step up leads to all zeros.
///
/// [`Trap::new`] takes the values most used since, a = k − 1, b = k and z = k − 1: a block scores
/// k − 1 − u below k ones, and k with all of them. For k ≥ 3, every schema of order below k
/// within a block then favors all zeros: the block is *fully deceptive* (Deb and Goldberg's
/// Theorem 1 and their inequality 16, r = a/b ≥ (2 − 1/(k − z)) / (2 − 1/z), which here is
/// (k − 1)/k ≥ (k − 1)/(2k − 3), their limiting ratio, eq. 20). [`Trap::with_values`] takes any
/// a, b and z: Ackley's 1987 trap of n bits, for one, is
/// `Trap::with_values(1, n, 8n, 10n, ⌊3n/4⌋)`. Deb and Goldberg find that none of Ackley's (of 8,
/// 12, 16 and 20 bits) is fully deceptive, and write that his traps are "fully deceptive only
/// for ℓ < 7"; by their inequality 16, and by enumerating the schemas, only those of 3 and 4 bits
/// are.
///
/// Maximum blocks × b at all ones; the deceptive attractor, all zeros, scores blocks × a; 10
/// blocks of 4 bits by default.
///
/// Deb, K. and Goldberg, D. E. (1993). Analyzing deception in trap functions. In *Foundations of
/// Genetic Algorithms 2*, Morgan Kaufmann: 93-108, equation 1, the function of a block, after
/// Ackley, D. H. (1987). *A Connectionist Machine for Genetic Hillclimbing.* Kluwer Academic
/// Publishers, section 3.3.3 ("Trap"). The paper analyzes one block of ℓ bits; the sum over
/// consecutive blocks, the form of test suites built from deceptive subfunctions, isn't written
/// out in it.
///
/// ```
/// use genoxide::genome::Bits;
/// use genoxide::problems::binary::Trap;
/// use genoxide::prelude::*;
///
/// let trap = Trap::new(2, 4);
/// // a block of 4 ones scores 4; one of a single one, 2
/// let x: Bits = "11110100".chars().map(|bit| bit == '1').collect();
/// assert_eq!(trap.evaluate(&x), 6.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Trap {
    blocks: usize,
    k: usize,
    a: f64,
    b: f64,
    z: usize,
}

impl Trap {
    /// `blocks` blocks of `k` bits, with a = k − 1, b = k and z = k − 1: fully deceptive for
    /// k ≥ 3.
    ///
    /// # Panics
    ///
    /// If `blocks` is 0, `k` is below 2, or the genome would have more than 2^24 bits.
    pub fn new(blocks: usize, k: usize) -> Self {
        assert!(blocks >= 1, "Trap needs at least 1 block");
        assert!(k >= 2, "Trap needs blocks of at least 2 bits, not {k}");
        let bits = blocks.saturating_mul(k);
        check_bits("Trap", bits);
        Self {
            blocks,
            k,
            a: (k - 1) as f64,
            b: k as f64,
            z: k - 1,
        }
    }

    /// `blocks` blocks of `k` bits, with the values `a` at no ones and `b` at k ones and the
    /// slope change `z`.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`](crate::Error::InvalidSetting) if `blocks` is 0, `k` is below 2,
    /// the genome would have more than 2^24 bits, `z` isn't between 1 and k − 1, or `a` and `b`
    /// aren't finite with 0 ≤ a < b.
    ///
    /// ```
    /// use genoxide::problems::binary::Trap;
    ///
    /// // Ackley's trap of 20 bits: a local maximum 8n = 160 at all zeros, the global 10n = 200
    /// let ackley = Trap::with_values(1, 20, 160.0, 200.0, 15)?;
    /// assert!(Trap::with_values(1, 20, 200.0, 160.0, 15).is_err());
    /// # Ok::<(), genoxide::Error>(())
    /// ```
    pub fn with_values(blocks: usize, k: usize, a: f64, b: f64, z: usize) -> crate::Result<Self> {
        let invalid = |setting: &'static str, reason: String| {
            Err(crate::Error::InvalidSetting { setting, reason })
        };
        if blocks == 0 {
            return invalid("blocks", "a trap needs at least 1 block".to_string());
        }
        if k < 2 {
            return invalid("k", format!("a block has at least 2 bits, not {k}"));
        }
        if blocks.checked_mul(k).is_none_or(|bits| bits > MAX_BITS) {
            return invalid(
                "k",
                format!("{blocks} blocks of {k} bits are more than 2^24 bits"),
            );
        }
        if !(1..k).contains(&z) {
            return invalid(
                "z",
                format!("must be between 1 and k − 1 = {}, got {z}", k - 1),
            );
        }
        if !(a.is_finite() && b.is_finite() && 0.0 <= a && a < b) {
            return invalid(
                "a",
                format!("a and b must be finite with 0 ≤ a < b, got a = {a} and b = {b}"),
            );
        }
        Ok(Self { blocks, k, a, b, z })
    }

    /// The number of blocks.
    pub fn blocks(&self) -> usize {
        self.blocks
    }

    /// The number of bits of a block.
    pub fn k(&self) -> usize {
        self.k
    }

    /// The value of a block with no ones, the deceptive attractor.
    pub fn a(&self) -> f64 {
        self.a
    }

    /// The value of a block of ones, the optimum.
    pub fn b(&self) -> f64 {
        self.b
    }

    /// The slope change: the number of ones at which a block scores 0.
    pub fn z(&self) -> usize {
        self.z
    }

    /// The number of bits: blocks × k.
    pub fn bits(&self) -> usize {
        self.blocks * self.k
    }

    /// The value of a block with `ones` ones, f(u).
    ///
    /// ```
    /// use genoxide::problems::binary::Trap;
    ///
    /// let trap = Trap::new(1, 4);
    /// let values: Vec<f64> = (0..=4).map(|u| trap.block(u)).collect();
    /// assert_eq!(values, [3.0, 2.0, 1.0, 0.0, 4.0]);
    /// ```
    ///
    /// # Panics
    ///
    /// If `ones` is above k.
    pub fn block(&self, ones: usize) -> f64 {
        assert!(ones <= self.k, "a block has at most {} ones", self.k);
        if ones <= self.z {
            self.a * (self.z - ones) as f64 / self.z as f64
        } else {
            self.b * (ones - self.z) as f64 / (self.k - self.z) as f64
        }
    }
}

impl Default for Trap {
    /// 10 blocks of 4 bits, with a = 3, b = 4 and z = 3: 40 bits, maximum 40.
    fn default() -> Self {
        Self::new(10, 4)
    }
}

impl FitnessFunction<Bits> for Trap {
    type Output = f64;

    /// The sum of the values of the blocks of `x`, in their order.
    ///
    /// # Panics
    ///
    /// If `x` doesn't have [`bits`](Trap::bits) bits.
    fn evaluate(&self, x: &Bits) -> f64 {
        check_length("Trap", x, self.bits());
        (0..self.blocks)
            .map(|block| self.block(ones_in(x, block * self.k, self.k)))
            .sum()
    }
}

impl Problem for Trap {
    type Representation = Binary;

    fn name(&self) -> &'static str {
        "Trap"
    }

    fn representation(&self) -> Binary {
        binary(self.bits())
    }

    fn objective(&self) -> Objective {
        Objective::Maximize
    }

    fn optimum(&self) -> Option<Optimum<Bits>> {
        let ones = Bits::ones(self.bits());
        Some(Optimum::proven(self.evaluate(&ones), vec![ones]))
    }

    fn reference(&self) -> &'static str {
        "Deb, K. and Goldberg, D. E. (1993). Analyzing deception in trap functions. In Foundations \
         of Genetic Algorithms 2, Morgan Kaufmann: 93-108."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1016/B978-0-08-094832-4.50012-X")
    }
}

// ---- royal road ------------------------------------------------------------------------------

/// The royal road functions R1 and R2: blocks of ones that score only when complete, and, in R2,
/// pairs, quadruples and so on of complete blocks that score again.
///
/// The function is a sum over schemas s, `Σ c_s σ_s(x)`, with `σ_s(x)` 1 if x is an instance of
/// s (has ones wherever s defines a bit) and `c_s = order(s)`, its number of defined bits.
///
/// - **R1** ([`RoyalRoad::r1`]): 8 schemas, each a block of 8 consecutive ones (bits 0 to 7, 8
///   to 15, …): a string scores 8 per complete block, 64 at most (Mitchell, Holland and Forrest,
///   1994, Figure 1).
/// - **R2** ([`RoyalRoad::r2`]): R1's 8 schemas and the 7 above them: the 4 blocks of 16 ones
///   (c = 16), the 2 of 32 (c = 32) and all 64 ones (c = 64), so a string of ones scores
///   8 · 8 + 4 · 16 + 2 · 32 + 64 = 256 (Mitchell, Forrest and Holland, 1992, Figure 1).
///
/// [`RoyalRoad::new`] and [`RoyalRoad::hierarchical`] make the same functions with any number of
/// blocks of any size. The functions were meant as a "royal road" for a genetic algorithm,
/// whose crossover would combine complete blocks into larger ones; but every string short of a
/// complete block scores alike, so the search drifts on plateaus. Mitchell, Holland and Forrest
/// found that random-mutation hill climbing (one bit flipped at a time, the flip kept if it's no
/// worse) solved R1 in 6,179 evaluations on average where their genetic algorithm took 61,334
/// (Table 1).
///
/// Maximum at all ones: blocks × block size for R1; blocks × block size × (log₂ blocks + 1) for
/// R2. R1 by default.
///
/// Mitchell, M., Forrest, S. and Holland, J. H. (1992). The royal road for genetic algorithms:
/// fitness landscapes and GA performance. *Proceedings of the First European Conference on
/// Artificial Life*, MIT Press: 245-254, Figure 1, the function with 15 schemas, which Forrest
/// and Mitchell (1993) later named R2; and Mitchell, M., Holland, J. H. and Forrest, S. (1994).
/// When will a genetic algorithm outperform hill climbing? *Advances in Neural Information
/// Processing Systems 6*, Morgan Kaufmann: 51-58, Figure 1, R1. Forrest and Mitchell (1993,
/// FOGA 2) wasn't available.
///
/// ```
/// use genoxide::genome::Bits;
/// use genoxide::problems::binary::RoyalRoad;
/// use genoxide::prelude::*;
///
/// // the first two blocks complete: 16 in R1, and 8 + 8 + 16 = 32 in R2
/// let x: Bits = (0..64).map(|i| i < 16).collect();
/// assert_eq!(RoyalRoad::r1().evaluate(&x), 16.0);
/// assert_eq!(RoyalRoad::r2().evaluate(&x), 32.0);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RoyalRoad {
    blocks: usize,
    block_size: usize,
    hierarchical: bool,
}

impl RoyalRoad {
    /// R1's form: `blocks` blocks of `block_size` ones, each scoring `block_size` when
    /// complete.
    ///
    /// # Panics
    ///
    /// If `blocks` or `block_size` is 0, or the genome would have more than 2^24 bits.
    pub fn new(blocks: usize, block_size: usize) -> Self {
        Self::checked(blocks, block_size, false)
    }

    /// R2's form: `blocks` blocks of `block_size` ones, and every level of complete pairs,
    /// quadruples and so on up to all the blocks, each schema scoring its number of bits.
    ///
    /// # Panics
    ///
    /// If `blocks` isn't a power of 2, `block_size` is 0, or the genome would have more than
    /// 2^24 bits.
    pub fn hierarchical(blocks: usize, block_size: usize) -> Self {
        assert!(
            blocks.is_power_of_two(),
            "a hierarchical royal road needs a power of 2 blocks, not {blocks}"
        );
        Self::checked(blocks, block_size, true)
    }

    /// R1: 8 blocks of 8 bits, maximum 64.
    pub fn r1() -> Self {
        Self::new(8, 8)
    }

    /// R2: 8 blocks of 8 bits and the levels of 16, 32 and 64 bits above them, maximum 256.
    pub fn r2() -> Self {
        Self::hierarchical(8, 8)
    }

    fn checked(blocks: usize, block_size: usize, hierarchical: bool) -> Self {
        assert!(blocks >= 1, "a royal road needs at least 1 block");
        assert!(block_size >= 1, "a royal road's blocks have at least 1 bit");
        let bits = blocks.saturating_mul(block_size);
        check_bits("RoyalRoad", bits);
        Self {
            blocks,
            block_size,
            hierarchical,
        }
    }

    /// The number of blocks at the lowest level.
    pub fn blocks(&self) -> usize {
        self.blocks
    }

    /// The number of bits of a block at the lowest level.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Whether the function has R2's levels above the blocks.
    pub fn is_hierarchical(&self) -> bool {
        self.hierarchical
    }

    /// The number of bits: blocks × block size.
    pub fn bits(&self) -> usize {
        self.blocks * self.block_size
    }
}

impl Default for RoyalRoad {
    /// R1.
    fn default() -> Self {
        Self::r1()
    }
}

impl FitnessFunction<Bits> for RoyalRoad {
    type Output = f64;

    /// The sum of the orders of the schemas `x` is an instance of.
    ///
    /// # Panics
    ///
    /// If `x` doesn't have [`bits`](RoyalRoad::bits) bits.
    fn evaluate(&self, x: &Bits) -> f64 {
        check_length("RoyalRoad", x, self.bits());
        let mut total = 0;
        // the blocks, then (in R2) the pairs, quadruples and so on: a schema of `len` bits scores
        // `len` when they're all ones
        let mut len = self.block_size;
        loop {
            total += (0..self.bits() / len)
                .filter(|&schema| ones_in(x, schema * len, len) == len)
                .count()
                * len;
            if !self.hierarchical || len == self.bits() {
                break;
            }
            len *= 2;
        }
        total as f64
    }
}

impl Problem for RoyalRoad {
    type Representation = Binary;

    fn name(&self) -> &'static str {
        if self.hierarchical {
            "RoyalRoadR2"
        } else {
            "RoyalRoadR1"
        }
    }

    fn representation(&self) -> Binary {
        binary(self.bits())
    }

    fn objective(&self) -> Objective {
        Objective::Maximize
    }

    fn optimum(&self) -> Option<Optimum<Bits>> {
        let ones = Bits::ones(self.bits());
        Some(Optimum::proven(self.evaluate(&ones), vec![ones]))
    }

    fn reference(&self) -> &'static str {
        if self.hierarchical {
            "Mitchell, M., Forrest, S. and Holland, J. H. (1992). The royal road for genetic \
             algorithms: fitness landscapes and GA performance. Proceedings of the First European \
             Conference on Artificial Life, MIT Press: 245-254."
        } else {
            "Mitchell, M., Holland, J. H. and Forrest, S. (1994). When will a genetic algorithm \
             outperform hill climbing? Advances in Neural Information Processing Systems 6, \
             Morgan Kaufmann: 51-58."
        }
    }

    fn reference_url(&self) -> Option<&'static str> {
        if self.hierarchical {
            None
        } else {
            Some(
                "https://proceedings.neurips.cc/paper_files/paper/1993/hash/\
                 ab88b15733f543179858600245108dd8-Abstract.html",
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::StreamRng;
    use crate::genome::Representation;

    fn bits(text: &str) -> Bits {
        text.chars().map(|bit| bit == '1').collect()
    }

    #[test]
    fn ones_are_counted_in_any_range() {
        let mut rng = StreamRng::seed_from_u64(1);
        let x = Binary::new(200).unwrap().random_genome(&mut rng);
        for start in [0, 1, 63, 64, 65, 127, 130] {
            for len in [0, 1, 5, 63, 64, 65, 70] {
                let expected = (start..start + len)
                    .filter(|&i| x.get(i) == Some(true))
                    .count();
                assert_eq!(ones_in(&x, start, len), expected, "{start} {len}");
            }
        }
    }

    #[test]
    fn one_max_counts_ones() {
        let problem = OneMax::new(5);
        assert_eq!(problem.evaluate(&bits("10110")), 3.0);
        assert_eq!(problem.evaluate(&bits("00000")), 0.0);
        let optimum = problem.optimum().unwrap();
        assert_eq!(optimum.value(), 5.0);
        assert_eq!(problem.evaluate(&optimum.solutions()[0]), 5.0);
        assert_eq!(problem.objective(), Objective::Maximize);
        assert_eq!(OneMax::default().bits(), 100);
    }

    #[test]
    fn leading_ones_stops_at_the_first_zero() {
        let problem = LeadingOnes::new(6);
        assert_eq!(problem.evaluate(&bits("110111")), 2.0);
        assert_eq!(problem.evaluate(&bits("011111")), 0.0);
        assert_eq!(problem.evaluate(&bits("111111")), 6.0);
        // across words: Definition 16's sum of products, against the fast count
        let mut rng = StreamRng::seed_from_u64(2);
        for len in [1, 63, 64, 65, 128, 130] {
            let problem = LeadingOnes::new(len);
            for ones in [0, 1, len / 2, len - 1, len] {
                let mut x = Binary::new(len).unwrap().random_genome(&mut rng);
                for i in 0..ones {
                    x.set(i, true);
                }
                let definition: usize = (0..len)
                    .map(|i| {
                        (0..=i)
                            .map(|j| usize::from(x.get(j) == Some(true)))
                            .product::<usize>()
                    })
                    .sum();
                assert_eq!(problem.evaluate(&x), definition as f64, "{len} {ones}");
            }
        }
    }

    #[test]
    fn traps_follow_deb_and_goldberg() {
        // k = 5: 4 − u below 5 ones, 5 with all of them
        let trap = Trap::new(3, 5);
        let values: Vec<f64> = (0..=5).map(|u| trap.block(u)).collect();
        assert_eq!(values, [4.0, 3.0, 2.0, 1.0, 0.0, 5.0]);
        assert_eq!(
            trap.evaluate(&bits("11111 00000 00100".replace(' ', "").as_str())),
            12.0
        );
        assert_eq!(trap.optimum().unwrap().value(), 15.0);
        // equation 1 with other values: a = 6, b = 10, z = 2 for k = 4
        let trap = Trap::with_values(1, 4, 6.0, 10.0, 2).unwrap();
        let values: Vec<f64> = (0..=4).map(|u| trap.block(u)).collect();
        assert_eq!(values, [6.0, 3.0, 0.0, 5.0, 10.0]);
        // Ackley's trap of 8 bits: z = 6, 8n (z − c)/z and 10n (c − z)/(n − z)
        let ackley = Trap::with_values(1, 8, 64.0, 80.0, 6).unwrap();
        assert_eq!(ackley.block(0), 64.0);
        assert_eq!(ackley.block(6), 0.0);
        assert_eq!(ackley.block(7), 40.0);
        assert_eq!(ackley.block(8), 80.0);
    }

    // the mean value of a block over the strings that match a schema (its fixed bits), as Deb and
    // Goldberg define a schema's fitness
    fn schema_mean(trap: &Trap, fixed: u32, values: u32) -> f64 {
        let k = trap.k();
        let mut sum = 0.0;
        let mut count = 0.0;
        for string in 0u32..1 << k {
            if string & fixed == values {
                sum += trap.block(string.count_ones() as usize);
                count += 1.0;
            }
        }
        sum / count
    }

    // fully deceptive (Deb and Goldberg's definition): in every partition of order below k, the
    // schema with zeros in its fixed bits is no worse than the others
    fn fully_deceptive(trap: &Trap) -> bool {
        let k = trap.k();
        (1u32..(1 << k) - 1).all(|fixed| {
            let zeros = schema_mean(trap, fixed, 0);
            (0u32..1 << k)
                .filter(|values| values & !fixed == 0)
                .all(|values| schema_mean(trap, fixed, values) <= zeros)
        })
    }

    #[test]
    fn the_default_traps_are_fully_deceptive_from_three_bits() {
        for k in 3..=8 {
            assert!(fully_deceptive(&Trap::new(1, k)), "k = {k}");
            // inequality 16 and its limiting ratio, eq. 20, at z = k − 1
            let (r, z) = ((k - 1) as f64 / k as f64, (k - 1) as f64);
            let bound = (2.0 - 1.0 / (k as f64 - z)) / (2.0 - 1.0 / z);
            assert!(r >= bound);
            let limit = (k - 1) as f64 / (2 * k - 3) as f64;
            assert!((bound - limit).abs() < 1e-15);
        }
        // k = 2 isn't: 1 − u below 2 ones and 2 with both, so a single one averages better
        assert!(!fully_deceptive(&Trap::new(1, 2)));
        // Ackley's traps (r = 0.8, z = ⌊3n/4⌋) are fully deceptive exactly where inequality 16
        // holds: for 3 and 4 bits, not 5 or 6 (the paper says "only for ℓ < 7"), and none of the
        // sizes Ackley used, 8, 12, 16 and 20 (section 4)
        for n in 3..=12 {
            let z = 3 * n / 4;
            let ackley = Trap::with_values(1, n, 8.0 * n as f64, 10.0 * n as f64, z).unwrap();
            let bound = (2.0 - 1.0 / (n - z) as f64) / (2.0 - 1.0 / z as f64);
            assert_eq!(fully_deceptive(&ackley), 0.8 >= bound, "n = {n}");
            assert_eq!(fully_deceptive(&ackley), n <= 4, "n = {n}");
        }
    }

    #[test]
    fn invalid_traps_are_errors() {
        assert!(Trap::with_values(0, 4, 3.0, 4.0, 3).is_err());
        assert!(Trap::with_values(1, 1, 0.0, 1.0, 1).is_err());
        assert!(Trap::with_values(1, 4, 3.0, 4.0, 0).is_err());
        assert!(Trap::with_values(1, 4, 3.0, 4.0, 4).is_err());
        assert!(Trap::with_values(1, 4, 4.0, 4.0, 3).is_err());
        assert!(Trap::with_values(1, 4, -1.0, 4.0, 3).is_err());
        assert!(Trap::with_values(1, 4, 3.0, f64::INFINITY, 3).is_err());
        assert!(Trap::with_values(1 << 23, 4, 3.0, 4.0, 3).is_err());
        assert!(Trap::with_values(1, 4, 0.0, 4.0, 3).is_ok());
    }

    #[test]
    #[should_panic(expected = "Trap takes 1 to 2^24 bits")]
    fn traps_of_too_many_bits_panic() {
        Trap::new(1 << 23, 4);
    }

    #[test]
    fn royal_roads_score_their_schemas() {
        let r1 = RoyalRoad::r1();
        let r2 = RoyalRoad::r2();
        assert_eq!(r1.optimum().unwrap().value(), 64.0);
        assert_eq!(r2.optimum().unwrap().value(), 256.0);
        assert_eq!((r1.name(), r2.name()), ("RoyalRoadR1", "RoyalRoadR2"));
        // blocks 1 and 3 complete, block 2 one bit short
        let mut x = Bits::zeros(64);
        for i in (0..8).chain(16..24).chain(9..16) {
            x.set(i, true);
        }
        assert_eq!(r1.evaluate(&x), 16.0);
        assert_eq!(r2.evaluate(&x), 16.0);
        // the first 32 bits: 4 blocks, 2 pairs, 1 of 32
        let x: Bits = (0..64).map(|i| i < 32).collect();
        assert_eq!(r1.evaluate(&x), 32.0);
        assert_eq!(r2.evaluate(&x), 4.0 * 8.0 + 2.0 * 16.0 + 32.0);
        // the last 48: 6 blocks, 3 pairs, 1 of 32
        let x: Bits = (0..64).map(|i| i >= 16).collect();
        assert_eq!(r2.evaluate(&x), 6.0 * 8.0 + 3.0 * 16.0 + 32.0);
        // other sizes: 16 blocks of 3, levels of 3, 6, 12, 24, 48 bits
        let wide = RoyalRoad::hierarchical(16, 3);
        assert_eq!(wide.optimum().unwrap().value(), 48.0 * 5.0);
        assert_eq!(RoyalRoad::new(5, 7).optimum().unwrap().value(), 35.0);
        assert_eq!(
            RoyalRoad::hierarchical(1, 4).optimum().unwrap().value(),
            4.0
        );
    }

    #[test]
    #[should_panic(expected = "power of 2")]
    fn hierarchical_royal_roads_need_a_power_of_two_blocks() {
        RoyalRoad::hierarchical(6, 8);
    }

    #[test]
    #[should_panic(expected = "OneMax takes genomes of 4 bits")]
    fn genomes_of_another_length_panic() {
        OneMax::new(4).evaluate(&Bits::zeros(5));
    }

    #[test]
    fn no_string_beats_the_optimum() {
        let mut rng = StreamRng::seed_from_u64(3);
        type Score = Box<dyn Fn(&Bits) -> f64>;
        let problems: Vec<Score> = vec![
            Box::new(|x| OneMax::new(64).evaluate(x)),
            Box::new(|x| LeadingOnes::new(64).evaluate(x)),
            Box::new(|x| Trap::new(16, 4).evaluate(x)),
            Box::new(|x| RoyalRoad::r1().evaluate(x)),
            Box::new(|x| RoyalRoad::r2().evaluate(x)),
        ];
        let optima = [64.0, 64.0, 64.0, 64.0, 256.0];
        for (problem, optimum) in problems.iter().zip(optima) {
            for _ in 0..1_000 {
                let x = Binary::new(64).unwrap().random_genome(&mut rng);
                assert!(problem(&x) <= optimum);
            }
            assert_eq!(problem(&Bits::ones(64)), optimum);
        }
    }
}
