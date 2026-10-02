//! The 0/1 knapsack problem, with Pisinger's generated instance classes.

use super::{MAX_BITS, Optimum, Problem, binary, check_length};
use crate::constraint::at_most;
use crate::engine::FitnessFunction;
use crate::genome::{Binary, Bits};
use crate::problems::Constraints;
use crate::{Error, Objective, Result, StreamRng};

// the stream of the generator that draws an instance from its seed
const KNAPSACK_STREAM: u64 = 4;
// the largest total weight, profit or capacity: integers up to 2^53 are exact as f64
const MAX_TOTAL: u64 = 1 << 53;
// the most decisions `optimum` records, items × (capacity + 1) bits: 32 MiB
const MAX_DECISIONS: u128 = 1 << 28;

/// How the items of a spanner instance's spanner set are drawn: as in the instance class of the
/// same name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpannerDistribution {
    /// Weights and profits drawn independently.
    Uncorrelated,
    /// Profits within R/10 of the weights.
    WeaklyCorrelated,
    /// Profits R/10 above the weights.
    StronglyCorrelated,
}

/// A class of generated 0/1 knapsack instances (Pisinger, 2005, section 3 and section 3.3).
///
/// R is the data range, [`KnapsackGenerator::range`]; "in [x, y]" is an integer drawn uniformly
/// from x to y; R/10 and R/500 are rounded down.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KnapsackClass {
    /// Weights and profits in [1, R]: no correlation, generally easy.
    Uncorrelated,
    /// Weights in [1, R], profits in [w − R/10, w + R/10] and at least 1 (here
    /// [max(1, w − R/10), w + R/10]).
    WeaklyCorrelated,
    /// Weights in [1, R], profits w + R/10: hard, with a large gap between the continuous and the
    /// integer optimum.
    StronglyCorrelated,
    /// Profits in [1, R], weights p + R/10.
    InverseStronglyCorrelated,
    /// Weights in [1, R], profits in [w + R/10 − R/500, w + R/10 + R/500].
    AlmostStronglyCorrelated,
    /// Weights in [1, R], profits equal to them: the best knapsack is the fullest.
    SubsetSum,
    /// Weights in [100 000, 100 100] and profits in [1, 1000], whatever R.
    UncorrelatedSimilarWeights,
    /// span(v, m): every item a multiple of one of `v` items, the spanner set, drawn with weights
    /// in [1, R] and profits by `distribution`, then scaled down to ⌈2p/m⌉ and ⌈2w/m⌉; each item
    /// is a spanner item, drawn uniformly, times a multiplier in [1, m]. The paper uses
    /// span(2, 10) ([`KnapsackClass::spanner`]).
    Spanner {
        /// The size of the spanner set, at least 1.
        v: usize,
        /// The largest multiplier, at least 1.
        m: u64,
        /// How the spanner set is drawn.
        distribution: SpannerDistribution,
    },
    /// mstr(k₁, k₂, d): weights in [1, R], profits w + k₁ for the weights divisible by d and
    /// w + k₂ for the others. The paper uses mstr(3R/10, 2R/10, 6)
    /// ([`KnapsackClass::multiple_strongly_correlated`]).
    MultipleStronglyCorrelated {
        /// The profit above the weight of the items whose weight is divisible by d.
        k1: u64,
        /// The profit above the weight of the others.
        k2: u64,
        /// The divisor, at least 1.
        d: u64,
    },
    /// pceil(d): weights in [1, R], profits the weights rounded up to a multiple of d,
    /// d⌈w/d⌉. The paper uses pceil(3) ([`KnapsackClass::profit_ceiling`]).
    ProfitCeiling {
        /// The divisor, at least 1.
        d: u64,
    },
    /// circle(d): weights in [1, R], profits on an arc of an ellipse, d √(4R² − (w − 2R)²),
    /// rounded down (computed exactly, with integers), with d = `numerator` / `denominator`. The
    /// paper uses circle(2/3) ([`KnapsackClass::circle`]).
    Circle {
        /// d's numerator, 1 to 2^16.
        numerator: u64,
        /// d's denominator, 1 to 2^16.
        denominator: u64,
    },
}

impl KnapsackClass {
    /// span(2, 10) with the spanner set drawn by `distribution`, the paper's spanner instances.
    pub fn spanner(distribution: SpannerDistribution) -> Self {
        Self::Spanner {
            v: 2,
            m: 10,
            distribution,
        }
    }

    /// mstr(3R/10, 2R/10, 6) for the data range `range`, the paper's.
    pub fn multiple_strongly_correlated(range: u64) -> Self {
        Self::MultipleStronglyCorrelated {
            k1: 3 * range / 10,
            k2: 2 * range / 10,
            d: 6,
        }
    }

    /// pceil(3), the paper's.
    pub fn profit_ceiling() -> Self {
        Self::ProfitCeiling { d: 3 }
    }

    /// circle(2/3), the paper's.
    pub fn circle() -> Self {
        Self::Circle {
            numerator: 2,
            denominator: 3,
        }
    }

    // the parameters checked
    fn check(&self) -> Result<()> {
        let invalid = |reason: &str| {
            Err(Error::InvalidSetting {
                setting: "class",
                reason: reason.to_string(),
            })
        };
        match *self {
            Self::Spanner { v, m, .. } if v == 0 || m == 0 || v > MAX_BITS || m > 1 << 32 => {
                invalid("a spanner set has 1 to 2^24 items and multipliers 1 to 2^32")
            }
            Self::MultipleStronglyCorrelated { k1, k2, d }
                if d == 0 || k1 > 1 << 32 || k2 > 1 << 32 =>
            {
                invalid("mstr(k1, k2, d) needs d of at least 1, and k1 and k2 at most 2^32")
            }
            Self::ProfitCeiling { d } if d == 0 || d > 1 << 32 => {
                invalid("pceil(d) needs d from 1 to 2^32")
            }
            Self::Circle {
                numerator,
                denominator,
            } if !(1..=1 << 16).contains(&numerator) || !(1..=1 << 16).contains(&denominator) => {
                invalid("circle(d) needs d's numerator and denominator from 1 to 2^16")
            }
            _ => Ok(()),
        }
    }
}

/// The settings of a generated [`Knapsack`] instance: its class, number of items, data range,
/// capacity and seed. [`Knapsack::generator`] makes one.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KnapsackGenerator {
    class: KnapsackClass,
    items: usize,
    range: u64,
    instance: u64,
    instances: u64,
    seed: u64,
}

impl KnapsackGenerator {
    /// The data range R: weights (and, by class, profits) are drawn from 1 to R. 1000 by default,
    /// as in the paper's tables 1 to 5 and 9 to 13.
    pub fn range(mut self, range: u64) -> Self {
        self.range = range;
        self
    }

    /// The instance number h, from 1 to H: the capacity is ⌊h/(H + 1) Σ w⌋ (eq. 5). 50 by
    /// default, about half the total weight.
    pub fn instance(mut self, instance: u64) -> Self {
        self.instance = instance;
        self
    }

    /// The number of instances H of a series, 100 by default, as in the paper.
    pub fn instances(mut self, instances: u64) -> Self {
        self.instances = instances;
        self
    }

    /// The seed of the items: the same seed gives the same items on every platform. 0 by
    /// default.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// The instance.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] if there are no items or more than 2^24, the range isn't
    /// between 1 and 2^32, the instance number isn't between 1 and the number of instances (at
    /// most 2^32), the class's parameters are out of their bounds, or the total weight or profit
    /// is above 2^53.
    pub fn generate(&self) -> Result<Knapsack> {
        let invalid =
            |setting: &'static str, reason: String| Err(Error::InvalidSetting { setting, reason });
        if !(1..=MAX_BITS).contains(&self.items) {
            return invalid(
                "items",
                format!("must be between 1 and 2^24, got {}", self.items),
            );
        }
        if !(1..=1 << 32).contains(&self.range) {
            return invalid(
                "range",
                format!("must be between 1 and 2^32, got {}", self.range),
            );
        }
        if !(1..=1 << 32).contains(&self.instances) {
            return invalid(
                "instances",
                format!("must be between 1 and 2^32, got {}", self.instances),
            );
        }
        if !(1..=self.instances).contains(&self.instance) {
            return invalid(
                "instance",
                format!(
                    "must be between 1 and the instances, {}, got {}",
                    self.instances, self.instance
                ),
            );
        }
        self.class.check()?;
        let (weights, profits) = self.items();
        let total: u128 = weights.iter().map(|&w| u128::from(w)).sum();
        let capacity = total * u128::from(self.instance) / u128::from(self.instances + 1);
        let capacity = u64::try_from(capacity).unwrap_or(u64::MAX);
        Knapsack::new(weights, profits, capacity)
    }

    // the weights and profits, drawn in the order of the items
    fn items(&self) -> (Vec<u64>, Vec<u64>) {
        let mut rng = StreamRng::seed_from_u64(self.seed).derive(KNAPSACK_STREAM);
        let range = self.range;
        let (tenth, fivehundredth) = (range / 10, range / 500);
        let mut draw = |low: u64, high: u64| low + rng.below_u64(high - low + 1);
        let mut weights = Vec::with_capacity(self.items);
        let mut profits = Vec::with_capacity(self.items);
        // an item of a simple class, (weight, profit)
        let simple = |draw: &mut dyn FnMut(u64, u64) -> u64, class: SpannerDistribution| {
            let weight = draw(1, range);
            let profit = match class {
                SpannerDistribution::Uncorrelated => draw(1, range),
                SpannerDistribution::WeaklyCorrelated => {
                    draw(weight.saturating_sub(tenth).max(1), weight + tenth)
                }
                SpannerDistribution::StronglyCorrelated => weight + tenth,
            };
            (weight, profit)
        };
        let spanner = match self.class {
            KnapsackClass::Spanner { v, m, distribution } => (0..v)
                .map(|_| {
                    let (weight, profit) = simple(&mut draw, distribution);
                    (
                        weight.saturating_mul(2).div_ceil(m),
                        profit.saturating_mul(2).div_ceil(m),
                    )
                })
                .collect(),
            _ => Vec::new(),
        };
        for _ in 0..self.items {
            let (weight, profit) = match self.class {
                KnapsackClass::Uncorrelated => simple(&mut draw, SpannerDistribution::Uncorrelated),
                KnapsackClass::WeaklyCorrelated => {
                    simple(&mut draw, SpannerDistribution::WeaklyCorrelated)
                }
                KnapsackClass::StronglyCorrelated => {
                    simple(&mut draw, SpannerDistribution::StronglyCorrelated)
                }
                KnapsackClass::InverseStronglyCorrelated => {
                    let profit = draw(1, range);
                    (profit + tenth, profit)
                }
                KnapsackClass::AlmostStronglyCorrelated => {
                    let weight = draw(1, range);
                    let profit = draw(
                        weight + tenth - fivehundredth,
                        weight + tenth + fivehundredth,
                    );
                    (weight, profit)
                }
                KnapsackClass::SubsetSum => {
                    let weight = draw(1, range);
                    (weight, weight)
                }
                KnapsackClass::UncorrelatedSimilarWeights => {
                    let weight = draw(100_000, 100_100);
                    (weight, draw(1, 1000))
                }
                KnapsackClass::Spanner { v, m, .. } => {
                    let (weight, profit) = spanner[draw(0, v as u64 - 1) as usize];
                    let multiplier = draw(1, m);
                    (weight * multiplier, profit * multiplier)
                }
                KnapsackClass::MultipleStronglyCorrelated { k1, k2, d } => {
                    let weight = draw(1, range);
                    let k = if weight % d == 0 { k1 } else { k2 };
                    (weight, weight + k)
                }
                KnapsackClass::ProfitCeiling { d } => {
                    let weight = draw(1, range);
                    (weight, weight.div_ceil(d) * d)
                }
                KnapsackClass::Circle {
                    numerator,
                    denominator,
                } => {
                    let weight = draw(1, range);
                    // d² (4R² − (w − 2R)²) = d² w (4R − w), and its square root rounded down
                    let square = u128::from(weight) * u128::from(4 * range - weight);
                    let scaled = square * u128::from(numerator * numerator)
                        / u128::from(denominator * denominator);
                    (weight, isqrt(scaled) as u64)
                }
            };
            weights.push(weight);
            profits.push(profit);
        }
        (weights, profits)
    }
}

// the square root of `value`, rounded down
fn isqrt(value: u128) -> u128 {
    if value < 2 {
        return value;
    }
    // Newton's method from above
    let mut root = 1u128 << (128 - value.leading_zeros()).div_ceil(2);
    loop {
        let next = (root + value / root) / 2;
        if next >= root {
            return root;
        }
        root = next;
    }
}

/// The 0/1 knapsack problem: items with integer weights and profits, and a knapsack of integer
/// capacity; the most profitable selection whose weight fits.
///
/// A bit per item, 1 to take it. The fitness is `(profit, violation)`, the total profit of the
/// selected items and how far their weight exceeds the capacity, `constraint::at_most(weight,
/// capacity)`, 0 when they fit: under Deb's rules, a selection that fits beats one that doesn't,
/// and overweight ones compare by their excess.
///
/// [`Knapsack::generator`] draws an instance of one of Pisinger's classes from a seed with
/// genoxide's portable [`StreamRng`], so the same settings give the same items on every
/// platform; [`Knapsack::new`] takes the items of any instance. The weights and profits are
/// drawn item by item, in order (the weight first where both are drawn), each from its interval
/// with Lemire's unbiased method; the paper doesn't give its generator's random numbers, so the
/// classes are Pisinger's and the instances genoxide's.
///
/// The problem is NP-hard, though only weakly: dynamic programming solves it in O(n c) time
/// (Bellman's recursion), which [`optimum`](Problem::optimum) runs, exactly, when the items times
/// the capacity are at most 2^28, and gives `None` above. Pisinger's classes are hard for
/// branch-and-bound algorithms in different ways: strongly correlated instances have a large gap
/// between the continuous and the integer optimum, subset-sum instances bounds that don't help,
/// and the spanner, multiple strongly correlated, profit ceiling and circle instances bounds that
/// stay loose (section 3.3).
///
/// Pisinger, D. (2005). Where are the hard knapsack problems? *Computers & Operations Research*
/// 32(9): 2271-2284, the model (1)-(3), the instance classes of section 3 and section 3.3, and
/// the capacity of eq. 5. Martello, S. and Toth, P. (1990). *Knapsack Problems: Algorithms and
/// Computer Implementations*, Wiley, is the general reference.
///
/// ```
/// use genoxide::prelude::*;
/// use genoxide::problems::Problem;
/// use genoxide::problems::binary::{Knapsack, KnapsackClass};
///
/// let knapsack = Knapsack::generator(KnapsackClass::WeaklyCorrelated, 30).seed(1).generate()?;
/// let optimum = knapsack.optimum().expect("small enough").value();
/// let ga = Ga::builder(knapsack.representation())
///     .population_size(100)
///     .select(Tournament::new(3)?)
///     .crossover(UniformCrossover::new())
///     .mutate(BitFlip::per_gene(1.0 / 30.0)?)
///     .seed(1)
///     .build()?;
/// let outcome = Engine::new(ga, knapsack)
///     .stop_when(Stop::target(optimum).or(Stop::generations(500)))
///     .run()?;
/// assert!(outcome.best_fitness().is_feasible());
/// assert!(outcome.best_fitness().score().unwrap() >= 0.99 * optimum);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Knapsack {
    weights: Vec<u64>,
    profits: Vec<u64>,
    capacity: u64,
}

impl Knapsack {
    /// An instance with these items, a weight and a profit each, and this capacity.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] if there are no items or more than 2^24, the weights and the
    /// profits differ in number, or the total weight, the total profit or the capacity is above
    /// 2^53 (so that the fitness is exact).
    pub fn new(weights: Vec<u64>, profits: Vec<u64>, capacity: u64) -> Result<Self> {
        let invalid =
            |setting: &'static str, reason: String| Err(Error::InvalidSetting { setting, reason });
        if !(1..=MAX_BITS).contains(&weights.len()) {
            return invalid(
                "weights",
                format!("must have 1 to 2^24 items, got {}", weights.len()),
            );
        }
        if profits.len() != weights.len() {
            return invalid(
                "profits",
                format!(
                    "must have a profit per item, {}, got {}",
                    weights.len(),
                    profits.len()
                ),
            );
        }
        let total = |values: &[u64]| values.iter().map(|&v| u128::from(v)).sum::<u128>();
        for (setting, sum) in [
            ("weights", total(&weights)),
            ("profits", total(&profits)),
            ("capacity", u128::from(capacity)),
        ] {
            if sum > u128::from(MAX_TOTAL) {
                return invalid(setting, format!("must add up to at most 2^53, got {sum}"));
            }
        }
        Ok(Self {
            weights,
            profits,
            capacity,
        })
    }

    /// The settings of an instance of `class` with `items` items, to generate it: R = 1000,
    /// instance 50 of 100 and seed 0 by default.
    ///
    /// ```
    /// use genoxide::problems::binary::{Knapsack, KnapsackClass};
    ///
    /// let knapsack = Knapsack::generator(KnapsackClass::StronglyCorrelated, 50)
    ///     .range(10_000)
    ///     .seed(7)
    ///     .generate()?;
    /// // profits 1000 above the weights
    /// for (weight, profit) in knapsack.weights().iter().zip(knapsack.profits()) {
    ///     assert_eq!(profit - weight, 1000);
    /// }
    /// # Ok::<(), genoxide::Error>(())
    /// ```
    pub fn generator(class: KnapsackClass, items: usize) -> KnapsackGenerator {
        KnapsackGenerator {
            class,
            items,
            range: 1000,
            instance: 50,
            instances: 100,
            seed: 0,
        }
    }

    /// The number of items.
    pub fn items(&self) -> usize {
        self.weights.len()
    }

    /// The weight of each item.
    pub fn weights(&self) -> &[u64] {
        &self.weights
    }

    /// The profit of each item.
    pub fn profits(&self) -> &[u64] {
        &self.profits
    }

    /// The capacity.
    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    /// The total weight and profit of the items `x` selects.
    ///
    /// # Panics
    ///
    /// If `x` doesn't have a bit per item.
    pub fn totals(&self, x: &Bits) -> (u64, u64) {
        check_length("Knapsack", x, self.items());
        let (mut weight, mut profit) = (0, 0);
        for (w, &word) in x.as_words().iter().enumerate() {
            let mut word = word;
            while word != 0 {
                let item = w * 64 + word.trailing_zeros() as usize;
                weight += self.weights[item];
                profit += self.profits[item];
                word &= word - 1;
            }
        }
        (weight, profit)
    }

    // the best selection by Bellman's recursion over the capacities, or None if the decisions
    // to record are too many
    fn solve(&self) -> Option<Bits> {
        let total: u64 = self.weights.iter().sum();
        let capacity = self.capacity.min(total);
        let n = self.items();
        if n as u128 * (u128::from(capacity) + 1) > MAX_DECISIONS {
            return None;
        }
        let capacity = capacity as usize;
        let row = (capacity + 1).div_ceil(64);
        let mut best = vec![0u64; capacity + 1];
        let mut taken = vec![0u64; n * row];
        for (item, (&weight, &profit)) in self.weights.iter().zip(&self.profits).enumerate() {
            let Ok(weight) = usize::try_from(weight) else {
                continue;
            };
            if weight > capacity {
                continue;
            }
            for c in (weight..=capacity).rev() {
                let with = best[c - weight] + profit;
                if with > best[c] {
                    best[c] = with;
                    taken[item * row + c / 64] |= 1 << (c % 64);
                }
            }
        }
        let mut x = Bits::zeros(n);
        let mut c = capacity;
        for item in (0..n).rev() {
            if taken[item * row + c / 64] >> (c % 64) & 1 == 1 {
                x.set(item, true);
                c -= self.weights[item] as usize;
            }
        }
        Some(x)
    }
}

impl FitnessFunction<Bits> for Knapsack {
    type Output = (f64, f64);

    /// The total profit of the items `x` selects, and how far their weight exceeds the capacity.
    ///
    /// # Panics
    ///
    /// If `x` doesn't have a bit per item.
    fn evaluate(&self, x: &Bits) -> (f64, f64) {
        let (weight, profit) = self.totals(x);
        (profit as f64, at_most(weight as f64, self.capacity as f64))
    }
}

impl Problem for Knapsack {
    type Representation = Binary;

    fn name(&self) -> &'static str {
        "Knapsack"
    }

    fn representation(&self) -> Binary {
        binary(self.items())
    }

    fn objective(&self) -> Objective {
        Objective::Maximize
    }

    /// The most profitable selection that fits, by dynamic programming; `None` if the items
    /// times the capacity (or the total weight, if less) are above 2^28.
    fn optimum(&self) -> Option<Optimum<Bits>> {
        let solution = self.solve()?;
        let (_, profit) = self.totals(&solution);
        Some(Optimum::proven(profit as f64, vec![solution]))
    }

    fn reference(&self) -> &'static str {
        "Pisinger, D. (2005). Where are the hard knapsack problems? Computers & Operations \
         Research 32(9): 2271-2284."
    }

    fn reference_url(&self) -> Option<&'static str> {
        Some("https://doi.org/10.1016/j.cor.2004.03.002")
    }

    /// The capacity constraint, `weight − capacity ≤ 0`.
    fn constraints(&self, x: &Bits) -> Constraints {
        let (weight, _) = self.totals(x);
        Constraints::new(vec![weight as f64 - self.capacity as f64], Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::genome::Representation;

    fn classes() -> Vec<KnapsackClass> {
        vec![
            KnapsackClass::Uncorrelated,
            KnapsackClass::WeaklyCorrelated,
            KnapsackClass::StronglyCorrelated,
            KnapsackClass::InverseStronglyCorrelated,
            KnapsackClass::AlmostStronglyCorrelated,
            KnapsackClass::SubsetSum,
            KnapsackClass::UncorrelatedSimilarWeights,
            KnapsackClass::spanner(SpannerDistribution::Uncorrelated),
            KnapsackClass::spanner(SpannerDistribution::WeaklyCorrelated),
            KnapsackClass::spanner(SpannerDistribution::StronglyCorrelated),
            KnapsackClass::multiple_strongly_correlated(1000),
            KnapsackClass::profit_ceiling(),
            KnapsackClass::circle(),
        ]
    }

    // the best profit that fits, by trying every selection
    fn brute_force(knapsack: &Knapsack) -> u64 {
        let n = knapsack.items();
        (0u64..1 << n)
            .filter_map(|selection| {
                let x: Bits = (0..n).map(|i| selection >> i & 1 == 1).collect();
                let (weight, profit) = knapsack.totals(&x);
                (weight <= knapsack.capacity()).then_some(profit)
            })
            .max()
            .unwrap()
    }

    #[test]
    fn the_classes_follow_pisinger() {
        let r = 1000;
        for class in classes() {
            let knapsack = Knapsack::generator(class, 500).seed(3).generate().unwrap();
            let pairs = knapsack.weights().iter().zip(knapsack.profits());
            for (&w, &p) in pairs {
                let in_range = |v: u64| (1..=r).contains(&v);
                let holds = match class {
                    KnapsackClass::Uncorrelated => in_range(w) && in_range(p),
                    KnapsackClass::WeaklyCorrelated => {
                        in_range(w) && p >= 1 && p + 100 >= w && p <= w + 100
                    }
                    KnapsackClass::StronglyCorrelated => in_range(w) && p == w + 100,
                    KnapsackClass::InverseStronglyCorrelated => in_range(p) && w == p + 100,
                    KnapsackClass::AlmostStronglyCorrelated => {
                        in_range(w) && p + 2 >= w + 100 && p <= w + 102
                    }
                    KnapsackClass::SubsetSum => in_range(w) && p == w,
                    KnapsackClass::UncorrelatedSimilarWeights => {
                        (100_000..=100_100).contains(&w) && (1..=1000).contains(&p)
                    }
                    KnapsackClass::Spanner { .. } => (1..=2000).contains(&w),
                    KnapsackClass::MultipleStronglyCorrelated { .. } => {
                        in_range(w) && p == w + if w % 6 == 0 { 300 } else { 200 }
                    }
                    KnapsackClass::ProfitCeiling { .. } => {
                        in_range(w) && p % 3 == 0 && p >= w && p < w + 3
                    }
                    KnapsackClass::Circle { .. } => {
                        // (3p/2)² ≤ 4R² − (w − 2R)² < (3(p + 1)/2)²
                        let square = 4 * r * r - (2 * r - w) * (2 * r - w);
                        in_range(w) && 9 * p * p <= 4 * square && 4 * square < 9 * (p + 1) * (p + 1)
                    }
                };
                assert!(holds, "{class:?}: weight {w}, profit {p}");
            }
            // eq. 5: instance 50 of 100
            let total: u64 = knapsack.weights().iter().sum();
            assert_eq!(knapsack.capacity(), total * 50 / 101, "{class:?}");
        }
    }

    #[test]
    fn spanner_items_are_multiples_of_the_spanner_set() {
        fn gcd(a: u64, b: u64) -> u64 {
            if b == 0 { a } else { gcd(b, a % b) }
        }
        for distribution in [
            SpannerDistribution::Uncorrelated,
            SpannerDistribution::WeaklyCorrelated,
            SpannerDistribution::StronglyCorrelated,
        ] {
            let class = KnapsackClass::spanner(distribution);
            let knapsack = Knapsack::generator(class, 200).seed(5).generate().unwrap();
            // each item is a multiple of one of 2 spanner items: their reduced pairs are at most 2
            let mut reduced: Vec<(u64, u64)> = knapsack
                .weights()
                .iter()
                .zip(knapsack.profits())
                .map(|(&w, &p)| (w / gcd(w, p), p / gcd(w, p)))
                .collect();
            reduced.sort_unstable();
            reduced.dedup();
            assert!(reduced.len() <= 2, "{distribution:?}: {reduced:?}");
            // a base weight is at most ⌈2R/m⌉ = 200, times a multiplier of at most 10
            assert!(knapsack.weights().iter().all(|&w| (1..=2000).contains(&w)));
        }
    }

    // instances are the same on every platform: fixed values, to the bit
    #[test]
    fn instances_are_reproducible_from_their_seed() {
        let knapsack = Knapsack::generator(KnapsackClass::Uncorrelated, 5)
            .seed(1)
            .generate()
            .unwrap();
        assert_eq!(knapsack.weights(), [613, 858, 688, 724, 989]);
        assert_eq!(knapsack.profits(), [705, 45, 998, 22, 438]);
        assert_eq!(knapsack.capacity(), 1916);
        // the circle's profits, rounded down: (2/3) √(898 · 3102) = 1112.67
        let circle = Knapsack::generator(KnapsackClass::circle(), 3)
            .seed(2)
            .generate()
            .unwrap();
        assert_eq!(circle.weights(), [898, 956, 782]);
        assert_eq!(circle.profits(), [1112, 1137, 1057]);
        assert_eq!(circle.capacity(), 1304);
    }

    #[test]
    fn the_optimum_is_the_best_selection() {
        for (i, class) in classes().into_iter().enumerate() {
            for seed in 0..3 {
                let knapsack = Knapsack::generator(class, 14)
                    .range(100)
                    .instance(1 + 30 * seed)
                    .seed(i as u64 * 10 + seed)
                    .generate()
                    .unwrap();
                let optimum = knapsack.optimum().unwrap();
                assert_eq!(
                    optimum.value(),
                    brute_force(&knapsack) as f64,
                    "{class:?} {seed}"
                );
                let fitness = knapsack.evaluate(&optimum.solutions()[0]);
                assert_eq!(fitness, (optimum.value(), 0.0));
            }
        }
    }

    #[test]
    fn the_fitness_is_the_profit_and_the_excess_weight() {
        let knapsack = Knapsack::new(vec![5, 4, 3], vec![10, 40, 30], 7).unwrap();
        let x = |text: &str| -> Bits { text.chars().map(|bit| bit == '1').collect() };
        assert_eq!(knapsack.evaluate(&x("011")), (70.0, 0.0));
        assert_eq!(knapsack.evaluate(&x("111")), (80.0, 5.0));
        assert_eq!(knapsack.constraints(&x("111")).inequalities(), [5.0]);
        assert_eq!(knapsack.constraints(&x("100")).inequalities(), [-2.0]);
        assert_eq!(knapsack.optimum().unwrap().value(), 70.0);
        // a capacity above the total weight: everything fits
        let roomy = Knapsack::new(vec![5, 4], vec![1, 2], 100).unwrap();
        assert_eq!(roomy.optimum().unwrap().value(), 3.0);
        // the items of more than one word
        let many = Knapsack::new(vec![1; 130], (0..130).collect(), 2).unwrap();
        let optimum = many.optimum().unwrap();
        assert_eq!(optimum.value(), 129.0 + 128.0);
        assert_eq!(many.representation().genome_len(), 130);
    }

    #[test]
    fn integer_square_roots_round_down() {
        for value in (0u128..10_000).chain([u64::MAX as u128, 1 << 100, (1 << 100) - 1]) {
            let root = isqrt(value);
            assert!(
                root * root <= value && (root + 1) * (root + 1) > value,
                "{value}"
            );
        }
    }

    #[test]
    fn invalid_instances_are_errors() {
        let class = KnapsackClass::Uncorrelated;
        assert!(Knapsack::generator(class, 0).generate().is_err());
        assert!(Knapsack::generator(class, 10).range(0).generate().is_err());
        assert!(
            Knapsack::generator(class, 10)
                .instance(0)
                .generate()
                .is_err()
        );
        assert!(
            Knapsack::generator(class, 10)
                .instance(101)
                .generate()
                .is_err()
        );
        assert!(
            Knapsack::generator(class, 10)
                .instances(0)
                .generate()
                .is_err()
        );
        let spanner = KnapsackClass::Spanner {
            v: 0,
            m: 10,
            distribution: SpannerDistribution::Uncorrelated,
        };
        assert!(Knapsack::generator(spanner, 10).generate().is_err());
        let ceiling = KnapsackClass::ProfitCeiling { d: 0 };
        assert!(Knapsack::generator(ceiling, 10).generate().is_err());
        let circle = KnapsackClass::Circle {
            numerator: 2,
            denominator: 0,
        };
        assert!(Knapsack::generator(circle, 10).generate().is_err());
        assert!(Knapsack::new(vec![], vec![], 1).is_err());
        assert!(Knapsack::new(vec![1, 2], vec![1], 1).is_err());
        assert!(Knapsack::new(vec![1 << 53, 1], vec![1, 1], 1).is_err());
        assert!(Knapsack::new(vec![1], vec![1], (1 << 53) + 1).is_err());
        // too large to solve
        let large = Knapsack::new(vec![1 << 40; 2], vec![1; 2], 1 << 41).unwrap();
        assert!(large.optimum().is_none());
    }
}
