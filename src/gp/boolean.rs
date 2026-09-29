//! Boolean problems: Koza's multiplexer and even-parity functions, learned from their whole
//! truth tables.
//!
//! Each is a fitness function for trees of its paper's primitives, [`Logic`], in the set its
//! [`primitives`](Multiplexer::primitives) gives: the number of cases of the truth table that a
//! tree gets wrong, minimized, 0 at the optimum (Koza's standardized fitness). The trees are
//! evaluated on 64 cases at once, a bit per case in a `u64`, with [`Tree::evaluate`].
//!
//! ```
//! use genoxide::gp::boolean::Multiplexer;
//! use genoxide::gp::{Gp, SubtreeCrossover, SubtreeMutation};
//! use genoxide::prelude::*;
//!
//! // the 6-multiplexer: 2 address bits select one of 4 data bits
//! let problem = Multiplexer::new(2)?;
//! let gp = Gp::builder(problem.primitives().clone()).build()?;
//! let ga = Ga::builder(gp)
//!     .population_size(500)
//!     .select(Tournament::new(7)?)
//!     .crossover(SubtreeCrossover::new())
//!     .mutate(SubtreeMutation::new())
//!     .mutation_rate(0.1)
//!     .minimize()
//!     .seed(1)
//!     .build()?;
//! let outcome = Engine::new(ga, problem)
//!     .stop_when(Stop::target(0.0).or(Stop::generations(50)))
//!     .run()?;
//! assert_eq!(outcome.best_fitness(), Fitness::new(0.0)); // all 64 cases right
//! # Ok::<(), genoxide::Error>(())
//! ```
//!
//! They are among the problems that McDermott et al. (2012, Genetic programming needs better
//! benchmarks, GECCO 2012: 791-798) found overused; they stay GP's classic tests with an exact
//! optimum.

use super::primitives::PrimitiveSet;
use super::tree::Tree;
use crate::engine::FitnessFunction;
use crate::{Error, Result};

/// The primitives of the Boolean problems: Koza's (1992) functions, and the inputs.
///
/// `If` takes three arguments and returns the second if the first is true, else the third;
/// the others are the usual Boolean functions. Each problem's set holds the functions of its
/// paper and its inputs, by name (`a0`, `d0`, ...).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Logic {
    /// a ∧ b.
    And,
    /// a ∨ b.
    Or,
    /// ¬(a ∧ b).
    Nand,
    /// ¬(a ∨ b).
    Nor,
    /// ¬a.
    Not,
    /// b if a, else c.
    If,
    /// The input at this position of the problem's inputs.
    Input(u16),
}

impl Logic {
    /// The value of the primitive for 64 cases at once, a bit per case: `args` are its
    /// arguments' values, `inputs` the values of the inputs.
    pub fn apply(self, args: &[u64], inputs: &[u64]) -> u64 {
        match self {
            Logic::And => args[0] & args[1],
            Logic::Or => args[0] | args[1],
            Logic::Nand => !(args[0] & args[1]),
            Logic::Nor => !(args[0] | args[1]),
            Logic::Not => !args[0],
            Logic::If => (args[0] & args[1]) | (!args[0] & args[2]),
            Logic::Input(input) => inputs[usize::from(input)],
        }
    }
}

// a truth table: the inputs' and the target's bits, 64 cases per word
#[derive(Clone, Debug)]
struct Table {
    primitives: PrimitiveSet<Logic>,
    inputs: usize,
    cases: u64,
    // the inputs' values in word w at [w * inputs + input]
    columns: Vec<u64>,
    targets: Vec<u64>,
}

// the most inputs: 2^20 cases, 16,384 words per input
const MAX_INPUTS: usize = 20;

impl Table {
    // the table of `target` over every case of `inputs` inputs, input i of case c being bit i of c
    fn new(primitives: PrimitiveSet<Logic>, inputs: usize, target: impl Fn(u64) -> bool) -> Self {
        let cases = 1u64 << inputs;
        let words = cases.div_ceil(64) as usize;
        let mut columns = vec![0; words * inputs];
        let mut targets = vec![0; words];
        for case in 0..cases {
            let (word, bit) = ((case / 64) as usize, case % 64);
            for input in 0..inputs {
                columns[word * inputs + input] |= (case >> input & 1) << bit;
            }
            targets[word] |= u64::from(target(case)) << bit;
        }
        Self {
            primitives,
            inputs,
            cases,
            columns,
            targets,
        }
    }

    // the tree's outputs, 64 cases per word (bits beyond the last case are 0)
    fn outputs(&self, tree: &Tree) -> Vec<u64> {
        let mut stack = Vec::new();
        let words = self.targets.len();
        (0..words)
            .map(|word| {
                let inputs = &self.columns[word * self.inputs..(word + 1) * self.inputs];
                let value = tree.evaluate(
                    &self.primitives,
                    &mut stack,
                    |op: Logic, args: &[u64]| op.apply(args, inputs),
                    |_, constant| if constant == 0.0 { 0 } else { !0 },
                );
                value & self.mask(word)
            })
            .collect()
    }

    // the cases of the word
    fn mask(&self, word: usize) -> u64 {
        let left = self.cases - word as u64 * 64;
        if left >= 64 { !0 } else { (1 << left) - 1 }
    }

    // the number of cases the tree gets wrong
    fn errors(&self, tree: &Tree) -> u64 {
        let mut stack = Vec::new();
        let mut errors = 0;
        for (word, &target) in self.targets.iter().enumerate() {
            let inputs = &self.columns[word * self.inputs..(word + 1) * self.inputs];
            let value = tree.evaluate(
                &self.primitives,
                &mut stack,
                |op: Logic, args: &[u64]| op.apply(args, inputs),
                |_, constant| if constant == 0.0 { 0 } else { !0 },
            );
            errors += u64::from(((value ^ target) & self.mask(word)).count_ones());
        }
        errors
    }
}

fn check_inputs(setting: &'static str, inputs: usize, least: usize) -> Result<()> {
    if (least..=MAX_INPUTS).contains(&inputs) {
        Ok(())
    } else {
        Err(Error::InvalidSetting {
            setting,
            reason: format!(
                "the problem must have {least} to {MAX_INPUTS} inputs (2^{MAX_INPUTS} cases), got \
                 {inputs}"
            ),
        })
    }
}

macro_rules! boolean_problem {
    ($name:ident) => {
        impl $name {
            /// The primitive set of the paper, over [`Logic`]: build a
            /// [`Gp`](crate::gp::Gp) from it, so the trees' primitives are the problem's.
            pub fn primitives(&self) -> &PrimitiveSet<Logic> {
                &self.table.primitives
            }

            /// The number of inputs.
            pub fn inputs(&self) -> usize {
                self.table.inputs
            }

            /// The number of cases of the truth table, 2^inputs. In case `c`, input `i` (in the
            /// order of the set's terminals) is bit `i` of `c`.
            pub fn cases(&self) -> u64 {
                self.table.cases
            }

            /// The right outputs, 64 cases per word: case `c` at bit `c % 64` of word `c / 64`.
            pub fn targets(&self) -> &[u64] {
                &self.table.targets
            }

            /// The tree's outputs, as [`targets`](Self::targets) lays them out.
            ///
            /// # Panics
            ///
            /// If the tree has primitives that aren't the problem's.
            pub fn outputs(&self, tree: &Tree) -> Vec<u64> {
                self.table.outputs(tree)
            }

            /// The number of cases the tree gets wrong: the fitness, 0 at the optimum.
            ///
            /// # Panics
            ///
            /// If the tree has primitives that aren't the problem's.
            pub fn errors(&self, tree: &Tree) -> u64 {
                self.table.errors(tree)
            }
        }

        impl FitnessFunction<Tree> for $name {
            type Output = f64;

            /// The number of cases the tree gets wrong, minimized.
            fn evaluate(&self, tree: &Tree) -> f64 {
                self.errors(tree) as f64
            }
        }
    };
}

/// The Boolean multiplexer of Koza (1992): `k` address bits select one of 2^k data bits, which
/// is the output. The 11-multiplexer (`k` = 3, 2048 cases) is Koza's; the 6-multiplexer (`k` =
/// 2) is a smaller version.
///
/// Its primitives are Koza's: the functions `and`, `or`, `not` and `if` (of three arguments),
/// and the inputs `a0` to `a{k-1}` (the address, `a0` its least significant bit), then `d0` to
/// `d{2^k-1}` (the data). One type: the untyped genetic programming of the paper.
///
/// ```
/// use genoxide::gp::boolean::Multiplexer;
///
/// let problem = Multiplexer::new(3)?; // Koza's 11-multiplexer
/// assert_eq!((problem.inputs(), problem.cases()), (11, 2048));
/// let set = problem.primitives();
/// // the right output everywhere: the data bit that the address selects
/// let tree = set.parse(
///     "if(a2, if(a1, if(a0, d7, d6), if(a0, d5, d4)), if(a1, if(a0, d3, d2), if(a0, d1, d0)))",
/// )?;
/// assert_eq!(problem.errors(&tree), 0);
/// // one data bit is right in the cases that select it, and in half the others
/// assert_eq!(problem.errors(&set.parse("d0")?), 896);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct Multiplexer {
    address_bits: usize,
    table: Table,
}

impl Multiplexer {
    /// The multiplexer of `address_bits` address bits, 1 to 4: the 3-, 6-, 11- and
    /// 20-multiplexer.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] (setting `address_bits`) for 0 or more than 4 address bits.
    pub fn new(address_bits: usize) -> Result<Self> {
        if !(1..=4).contains(&address_bits) {
            return Err(Error::InvalidSetting {
                setting: "address_bits",
                reason: format!("must be 1 to 4, got {address_bits}"),
            });
        }
        let data = 1usize << address_bits;
        let inputs = address_bits + data;
        check_inputs("address_bits", inputs, 3)?;
        let mut set = PrimitiveSet::builder();
        let boolean = set.new_type("bool");
        set.function("and", Logic::And, [boolean, boolean], boolean)
            .function("or", Logic::Or, [boolean, boolean], boolean)
            .function("not", Logic::Not, [boolean], boolean)
            .function("if", Logic::If, [boolean, boolean, boolean], boolean);
        for bit in 0..address_bits {
            set.terminal(format!("a{bit}"), Logic::Input(bit as u16), boolean);
        }
        for bit in 0..data {
            set.terminal(
                format!("d{bit}"),
                Logic::Input((address_bits + bit) as u16),
                boolean,
            );
        }
        let primitives = set.build(boolean)?;
        let mask = (1u64 << address_bits) - 1;
        let table = Table::new(primitives, inputs, |case| {
            let address = case & mask;
            case >> (address_bits as u64 + address) & 1 == 1
        });
        Ok(Self {
            address_bits,
            table,
        })
    }

    /// The number of address bits.
    pub fn address_bits(&self) -> usize {
        self.address_bits
    }

    /// Koza (1992), the source of the problem and its primitives.
    pub fn reference(&self) -> &'static str {
        "Koza, J. R. (1992). Genetic Programming: On the Programming of Computers by Means of \
         Natural Selection. MIT Press."
    }
}

boolean_problem!(Multiplexer);

/// The even-parity function of Koza (1992): true when an even number of the `n` inputs are
/// true. Even-3 to even-5 parity are the usual sizes; each input more doubles the cases and
/// makes it much harder.
///
/// Its primitives are Koza's: the functions `and`, `or`, `nand` and `nor`, and the inputs `d0`
/// to `d{n-1}`. One type.
///
/// ```
/// use genoxide::gp::boolean::EvenParity;
///
/// let problem = EvenParity::new(3)?;
/// assert_eq!(problem.cases(), 8);
/// let set = problem.primitives();
/// // odd parity of two is (a ∨ b) ∧ ¬(a ∧ b), even parity of two (a ∧ b) ∨ ¬(a ∨ b)
/// let odd = "and(or(d0, d1), nand(d0, d1))";
/// let tree = set.parse(&format!("or(and({odd}, d2), nor({odd}, d2))"))?;
/// assert_eq!(problem.errors(&tree), 0);
/// assert_eq!(problem.errors(&set.parse("d0")?), 4);
/// # Ok::<(), genoxide::Error>(())
/// ```
#[derive(Clone, Debug)]
pub struct EvenParity {
    table: Table,
}

impl EvenParity {
    /// Even parity of `inputs` inputs, 2 to 20.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidSetting`] (setting `inputs`) for fewer than 2 or more than 20 inputs.
    pub fn new(inputs: usize) -> Result<Self> {
        check_inputs("inputs", inputs, 2)?;
        let mut set = PrimitiveSet::builder();
        let boolean = set.new_type("bool");
        set.function("and", Logic::And, [boolean, boolean], boolean)
            .function("or", Logic::Or, [boolean, boolean], boolean)
            .function("nand", Logic::Nand, [boolean, boolean], boolean)
            .function("nor", Logic::Nor, [boolean, boolean], boolean);
        for bit in 0..inputs {
            set.terminal(format!("d{bit}"), Logic::Input(bit as u16), boolean);
        }
        let primitives = set.build(boolean)?;
        let table = Table::new(primitives, inputs, |case| case.count_ones() % 2 == 0);
        Ok(Self { table })
    }

    /// Koza (1992), the source of the problem and its primitives.
    pub fn reference(&self) -> &'static str {
        "Koza, J. R. (1992). Genetic Programming: On the Programming of Computers by Means of \
         Natural Selection. MIT Press."
    }
}

boolean_problem!(EvenParity);
