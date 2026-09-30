//! Tree genetic programming: primitive sets, generation, parsing, the evaluators and the operators.

use genoxide::gp::boolean::{EvenParity, Multiplexer};
use genoxide::gp::{
    Columns, ConstantMutation, Constants, Gp, HoistMutation, Init, Mutations, Node,
    OnePointCrossover, PointMutation, PrimitiveSet, ShrinkMutation, Subtree, SubtreeCrossover,
    SubtreeMutation, Tree, Type,
};
use genoxide::math;
use genoxide::prelude::*;
use proptest::prelude::*;

// Koza's function set for symbolic regression, with constants
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum Op {
    Add,
    Sub,
    Mul,
    Div,
    Sin,
    Cos,
    Exp,
    Rlog,
    X,
}

fn koza_set() -> PrimitiveSet<Op> {
    let mut set = PrimitiveSet::builder();
    let real = set.new_type("real");
    set.function("add", Op::Add, [real, real], real)
        .function("sub", Op::Sub, [real, real], real)
        .function("mul", Op::Mul, [real, real], real)
        .function("div", Op::Div, [real, real], real)
        .function("sin", Op::Sin, [real], real)
        .function("cos", Op::Cos, [real], real)
        .function("exp", Op::Exp, [real], real)
        .function("rlog", Op::Rlog, [real], real)
        .terminal("x", Op::X, real)
        .constants(real, Constants::uniform(-1.0..=1.0).unwrap());
    set.build(real).unwrap()
}

// the value of a primitive at a point
fn apply(op: Op, args: &[f64], x: f64) -> f64 {
    match op {
        Op::Add => args[0] + args[1],
        Op::Sub => args[0] - args[1],
        Op::Mul => args[0] * args[1],
        Op::Div => {
            if args[1] == 0.0 {
                1.0
            } else {
                args[0] / args[1]
            }
        }
        Op::Sin => math::sin(args[0]),
        Op::Cos => math::cos(args[0]),
        Op::Exp => math::exp(args[0]),
        Op::Rlog => {
            if args[0] == 0.0 {
                0.0
            } else {
                math::ln(args[0].abs())
            }
        }
        Op::X => x,
    }
}

// the same on columns, point by point
fn apply_columns(op: Op, args: &[&[f64]], output: &mut [f64], xs: &[f64]) {
    for (point, out) in output.iter_mut().enumerate() {
        let mut values = [0.0; 2];
        for (value, arg) in values.iter_mut().zip(args) {
            *value = arg[point];
        }
        *out = apply(op, &values[..args.len()], xs[point]);
    }
}

// a recursive interpreter from the root, as a user writes one
fn walk(subtree: Subtree<'_, Op>, x: f64) -> f64 {
    match subtree.primitive() {
        None => subtree.constant().unwrap(),
        Some(op) => {
            let args: Vec<f64> = subtree.children().map(|child| walk(child, x)).collect();
            apply(op, &args, x)
        }
    }
}

// a typed set: real numbers and Booleans, and a type made of both
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
enum Typed {
    Add,
    Mul,
    Less,
    And,
    If,
    X,
    Y,
}

fn typed_set() -> (PrimitiveSet<Typed>, Type, Type) {
    let mut set = PrimitiveSet::builder();
    let real = set.new_type("real");
    let boolean = set.new_type("bool");
    set.function("add", Typed::Add, [real, real], real)
        .function("mul", Typed::Mul, [real, real], real)
        .function("less", Typed::Less, [real, real], boolean)
        .function("and", Typed::And, [boolean, boolean], boolean)
        .function("if", Typed::If, [boolean, real, real], real)
        .terminal("x", Typed::X, real)
        .terminal("y", Typed::Y, real)
        .constants(real, Constants::integers(-3..=3).unwrap());
    (set.build(real).unwrap(), real, boolean)
}

fn typed_gp(max_depth: usize, max_size: usize) -> Gp<Typed> {
    Gp::builder(typed_set().0)
        .max_depth(max_depth)
        .max_size(max_size)
        .init(Init::RampedHalfAndHalf {
            depths: 1..=max_depth.min(6),
        })
        .build()
        .unwrap()
}

fn setting_error<T: std::fmt::Debug>(result: Result<T>, expected: &str) {
    match result {
        Err(Error::InvalidSetting { setting, .. }) => assert_eq!(setting, expected),
        other => panic!("expected an invalid `{expected}`, got {other:?}"),
    }
}

// a change to a builder with a real type, to check that it makes an invalid set
type Change<'a> = dyn Fn(&mut genoxide::gp::PrimitiveSetBuilder<Op>, Type) + 'a;

#[test]
fn invalid_primitive_sets_are_errors() {
    let build = |f: &Change<'_>| {
        let mut set = PrimitiveSet::builder();
        let real = set.new_type("real");
        set.terminal("x", Op::X, real);
        f(&mut set, real);
        set.build(real)
    };
    let foreign = Type::clone(&{
        let mut other = PrimitiveSet::<Op>::builder();
        other.new_type("a");
        other.new_type("b")
    });
    let cases: [&Change<'_>; 9] = [
        &|set, real| {
            set.function("add", Op::Add, [real, foreign], real);
        },
        &|set, _| {
            set.terminal("y", Op::X, foreign);
        },
        &|set, real| {
            set.terminal("", Op::X, real);
        },
        &|set, real| {
            set.terminal("a b", Op::X, real);
        },
        &|set, real| {
            set.terminal("1.5", Op::X, real);
        },
        &|set, real| {
            set.terminal("inf", Op::X, real);
        },
        &|set, real| {
            set.terminal("x", Op::X, real);
        },
        &|set, real| {
            let c = Constants::uniform(0.0..=1.0).unwrap();
            set.constants(real, c.clone()).constants(real, c);
        },
        &|set, real| {
            // a Boolean that no tree can make
            let boolean = set.new_type("bool");
            set.function("and", Op::Add, [boolean, boolean], boolean)
                .function("if", Op::Mul, [boolean, real, real], real);
        },
    ];
    for case in cases {
        setting_error(build(case), "primitives");
    }
    // no type, a root of another builder, a type name twice
    setting_error(PrimitiveSet::<Op>::builder().build(foreign), "primitives");
    let mut set = PrimitiveSet::builder();
    let real = set.new_type("real");
    set.terminal("x", Op::X, real);
    setting_error(set.build(foreign), "primitives");
    let mut set = PrimitiveSet::builder();
    let real = set.new_type("real");
    set.new_type("real");
    set.terminal("x", Op::X, real);
    setting_error(set.build(real), "primitives");
    // constants
    setting_error(Constants::uniform(1.0..=0.0), "constants");
    setting_error(Constants::uniform(f64::NAN..=0.0), "constants");
    setting_error(Constants::uniform(-f64::MAX..=f64::MAX), "constants");
    setting_error(Constants::integers(0..=(1 << 54)), "constants");
    setting_error(Constants::choice([]), "constants");
    setting_error(Constants::choice([1.0, f64::INFINITY]), "constants");
}

#[test]
fn the_table_of_smallest_trees_by_hand() {
    let mut set = PrimitiveSet::builder();
    let real = set.new_type("real");
    let boolean = set.new_type("bool");
    let pair = set.new_type("pair");
    let big = set.new_type("big");
    set.terminal("x", Op::X, real)
        .function("less", Op::Sub, [real, real], boolean)
        .function("pair", Op::Mul, [real, boolean], pair)
        // `big` has a wide tree of depth 1 and a narrower one of depth 2
        .function("wide", Op::Add, [real, real, real, real], big)
        .function("narrow", Op::Div, [boolean], big)
        .function("first", Op::Sin, [pair], real)
        .function("from_big", Op::Cos, [big], real);
    let set = set.build(real).unwrap();
    assert_eq!(set.min_depth(real), Some(0));
    assert_eq!(set.min_depth(boolean), Some(1));
    assert_eq!(set.min_depth(pair), Some(2));
    assert_eq!(set.min_depth(big), Some(1));
    // less(x, x); pair(x, less(x, x)); wide(x, x, x, x) or narrow(less(x, x))
    assert_eq!(set.min_nodes(boolean, 0), None);
    assert_eq!(set.min_nodes(boolean, 1), Some(3));
    assert_eq!(set.min_nodes(pair, 1), None);
    assert_eq!(set.min_nodes(pair, 2), Some(5));
    assert_eq!(set.min_nodes(big, 1), Some(5));
    assert_eq!(set.min_nodes(big, 2), Some(4));
    assert_eq!(set.min_nodes(big, 100), Some(4));

    // trees of a root that needs depth 2 and 4 nodes
    let mut builder = PrimitiveSet::builder();
    let real = builder.new_type("real");
    let big = builder.new_type("big");
    let boolean = builder.new_type("bool");
    builder
        .terminal("x", Op::X, real)
        .function("less", Op::Sub, [real, real], boolean)
        .function("wide", Op::Add, [real, real, real, real], big)
        .function("narrow", Op::Div, [boolean], big);
    let set = builder.build(big).unwrap();
    setting_error(Gp::builder(set.clone()).max_depth(0).build(), "max_size");
    setting_error(
        Gp::builder(set.clone())
            .max_depth(1)
            .max_size(4)
            .init(Init::Full { depths: 1..=1 })
            .build(),
        "max_size",
    );
    // at depth 1 only `wide` fits, 5 nodes: not in 4
    let gp = Gp::builder(set.clone())
        .max_size(4)
        .init(Init::Grow { depths: 1..=3 })
        .build();
    setting_error(gp, "init");
    let gp = Gp::builder(set)
        .max_size(4)
        .init(Init::Grow { depths: 2..=3 })
        .build()
        .unwrap();
    let mut rng = StreamRng::seed_from_u64(1);
    for _ in 0..100 {
        let tree = gp.random_genome(&mut rng);
        assert!(gp.validate(&tree).is_ok());
        assert_eq!(
            tree.display(gp.primitives()).to_string(),
            "narrow(less(x, x))"
        );
    }
}

#[test]
fn invalid_representations_are_errors() {
    let set = koza_set;
    setting_error(Gp::builder(set()).max_size(0).build(), "max_size");
    setting_error(Gp::builder(set()).max_size(1 << 25).build(), "max_size");
    setting_error(Gp::builder(set()).max_depth(1 << 25).build(), "max_depth");
    setting_error(Gp::builder(set()).max_depth(5).build(), "init");
    #[allow(clippy::reversed_empty_ranges)]
    let empty = Init::Grow { depths: 3..=2 };
    setting_error(Gp::builder(set()).init(empty).build(), "init");
    let gp = Gp::builder(set()).build().unwrap();
    let mut rng = StreamRng::seed_from_u64(0);
    setting_error(gp.full(18, &mut rng), "depth");
    setting_error(gp.ramped_half_and_half(1 << 25, &mut rng), "count");
    setting_error(SubtreeMutation::with_max_depth(1 << 25), "max_depth");
    setting_error(SubtreeCrossover::with_internal_rate(1.5), "internal_rate");
}

#[test]
fn parse_errors_and_validation() {
    let (set, _, _) = typed_set();
    for text in [
        "",
        "add(x)",
        "add(x, x",
        "add x",
        "add(x, x))",
        "add(x, x) x",
        "foo",
        "less(x, x)",
        "if(1, x, x)",
        "add(x, 0.5)",
        "add(x, 7)",
        "add(x, NaN)",
        "x()",
        "add(,x)",
    ] {
        assert!(
            matches!(set.parse(text), Err(Error::InvalidGenome { .. })),
            "{text:?}"
        );
    }
    let tree = set
        .parse(" if( less(x,y) , add(x, -3), mul(y ,2.0))")
        .unwrap();
    assert_eq!(
        tree.display(&set).to_string(),
        "if(less(x, y), add(x, -3.0), mul(y, 2.0))"
    );
    let gp = typed_gp(2, 1024);
    assert!(gp.validate(&tree).is_ok());
    let deep = set.parse("add(x, add(x, add(x, x)))").unwrap();
    assert!(gp.validate(&deep).is_err());
    let small = typed_gp(6, 5);
    assert!(
        small
            .validate(&set.parse("add(x, add(x, x))").unwrap())
            .is_ok()
    );
    assert!(small.validate(&tree).is_err());
    let invalid = |nodes: Vec<Node>| gp.validate(&Tree::from(nodes)).is_err();
    let x = Node::Primitive(set.position("x").unwrap());
    let add = Node::Primitive(set.position("add").unwrap());
    let less = Node::Primitive(set.position("less").unwrap());
    assert!(invalid(vec![]));
    assert!(invalid(vec![add, x]));
    assert!(invalid(vec![x, x]));
    assert!(invalid(vec![less, x, x]));
    assert!(invalid(vec![Node::Primitive(99)]));
    let real = set.root();
    assert!(invalid(vec![Node::Constant {
        ty: real,
        value: 0.5
    }]));
    assert!(!invalid(vec![Node::Constant {
        ty: real,
        value: -3.0
    }]));
}

fn is_sorted_multiset(mut nodes: Vec<Node>) -> Vec<(u8, u32, u64)> {
    let mut keys: Vec<(u8, u32, u64)> = nodes
        .drain(..)
        .map(|node| match node {
            Node::Primitive(index) => (0, index, 0),
            Node::Constant { ty, value } => (1, ty.index() as u32, value.to_bits()),
        })
        .collect();
    keys.sort_unstable();
    keys
}

// the depth of every leaf
fn leaf_depths<P: Copy>(tree: &Tree, set: &PrimitiveSet<P>) -> Vec<usize> {
    fn walk<P: Copy>(subtree: Subtree<'_, P>, depth: usize, out: &mut Vec<usize>) {
        if subtree.arity() == 0 {
            out.push(depth);
        }
        for child in subtree.children() {
            walk(child, depth + 1, out);
        }
    }
    let mut depths = Vec::new();
    walk(tree.root(set), 0, &mut depths);
    depths
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn generated_trees_are_typed_and_within_the_limits(
        seed: u64, max_depth in 2usize..10, max_size in 10usize..200,
    ) {
        let gp = typed_gp(max_depth, max_size);
        let mut rng = StreamRng::seed_from_u64(seed);
        for depth in 0..=max_depth {
            for tree in [gp.full(depth, &mut rng).unwrap(), gp.grow(depth, &mut rng).unwrap()] {
                prop_assert!(gp.validate(&tree).is_ok());
                prop_assert!(tree.depth(gp.primitives()) <= depth);
            }
        }
        for tree in gp.ramped_half_and_half(50, &mut rng).unwrap() {
            prop_assert!(gp.validate(&tree).is_ok());
        }
    }

    #[test]
    fn full_trees_have_every_leaf_at_the_depth(seed: u64, depth in 0usize..8) {
        let gp = Gp::builder(koza_set()).max_size(1 << 10).build().unwrap();
        let mut rng = StreamRng::seed_from_u64(seed);
        let tree = gp.full(depth, &mut rng).unwrap();
        prop_assert!(leaf_depths(&tree, gp.primitives()).iter().all(|&d| d == depth));
        let tree = gp.grow(depth, &mut rng).unwrap();
        prop_assert!(leaf_depths(&tree, gp.primitives()).iter().all(|&d| d <= depth));
        // the root is a function (Koza)
        prop_assert!(depth == 0 || tree.root(gp.primitives()).arity() > 0);
    }

    #[test]
    fn parse_reads_what_display_writes(seed: u64) {
        let gp = typed_gp(8, 300);
        let koza = Gp::builder(koza_set()).build().unwrap();
        let mut rng = StreamRng::seed_from_u64(seed);
        for _ in 0..10 {
            let tree = gp.random_genome(&mut rng);
            let text = tree.display(gp.primitives()).to_string();
            prop_assert_eq!(gp.primitives().parse(&text).unwrap(), tree);
            let tree = koza.random_genome(&mut rng);
            let text = tree.display(koza.primitives()).to_string();
            prop_assert_eq!(koza.primitives().parse(&text).unwrap(), tree);
        }
    }

    #[test]
    fn subtree_crossover_keeps_trees_typed_within_limits_and_conserves_nodes(
        seed: u64, max_depth in 3usize..9, max_size in 15usize..120,
    ) {
        let gp = typed_gp(max_depth, max_size);
        let mut rng = StreamRng::seed_from_u64(seed);
        for rate in [0.0, 0.9, 1.0] {
            let crossover = SubtreeCrossover::with_internal_rate(rate).unwrap();
            for _ in 0..10 {
                let (mut a, mut b) = (gp.random_genome(&mut rng), gp.random_genome(&mut rng));
                let before = is_sorted_multiset([a.nodes(), b.nodes()].concat());
                crossover.crossover(&gp, &mut a, &mut b, &mut rng);
                prop_assert!(gp.validate(&a).is_ok(), "{:?}", gp.validate(&a));
                prop_assert!(gp.validate(&b).is_ok(), "{:?}", gp.validate(&b));
                prop_assert_eq!(is_sorted_multiset([a.nodes(), b.nodes()].concat()), before);
            }
        }
    }

    #[test]
    fn subtree_mutation_always_changes_the_tree_within_the_limits(
        seed: u64, max_depth in 3usize..9, max_size in 15usize..120, depth in 0usize..6,
    ) {
        let gp = typed_gp(max_depth, max_size);
        let mutation = SubtreeMutation::with_max_depth(depth).unwrap();
        let mut rng = StreamRng::seed_from_u64(seed);
        for _ in 0..20 {
            let mut tree = gp.random_genome(&mut rng);
            let before = tree.clone();
            mutation.mutate(&gp, &mut tree, &mut rng);
            prop_assert!(gp.validate(&tree).is_ok(), "{:?}", gp.validate(&tree));
            prop_assert_ne!(tree, before);
        }
    }

    #[test]
    fn the_three_evaluators_agree_to_the_bit(seed: u64) {
        let gp = Gp::builder(koza_set()).build().unwrap();
        let set = gp.primitives();
        let mut rng = StreamRng::seed_from_u64(seed);
        let xs: Vec<f64> = (0..50).map(|i| f64::from(i) / 12.5 - 2.0).collect();
        let (mut stack, mut columns) = (Vec::new(), Columns::new(xs.len()));
        for _ in 0..20 {
            let tree = gp.random_genome(&mut rng);
            let column = tree
                .evaluate_columns(set, &mut columns, |op, args, out| apply_columns(op, args, out, &xs))
                .to_vec();
            for (point, &x) in xs.iter().enumerate() {
                let value = tree.evaluate(set, &mut stack, |op, args| apply(op, args, x), |_, c| c);
                prop_assert_eq!(value.to_bits(), column[point].to_bits());
                prop_assert_eq!(walk(tree.root(set), x).to_bits(), value.to_bits());
            }
        }
    }
}

#[test]
fn evaluation_by_hand() {
    let set = koza_set();
    let tree = set.parse("add(mul(x, x), div(sin(x), sub(x, x)))").unwrap();
    // x² + 1 (protected division by 0)
    let value = tree.evaluate(
        &set,
        &mut Vec::new(),
        |op, args| apply(op, args, 0.5),
        |_, c| c,
    );
    assert_eq!(value, 1.25);
    let tree = set.parse("sub(rlog(x), exp(-0.5))").unwrap();
    let mut columns = Columns::new(2);
    let xs = [-2.0, 0.0];
    let column = tree.evaluate_columns(&set, &mut columns, |op, args, out| {
        apply_columns(op, args, out, &xs)
    });
    assert_eq!(column, [math::ln(2.0) - math::exp(-0.5), -math::exp(-0.5)]);
    // a function of arity 5, beyond the inline arguments
    #[derive(Clone, Copy, Debug)]
    enum Sum {
        Five,
        X,
    }
    let mut builder = PrimitiveSet::builder();
    let real = builder.new_type("real");
    builder
        .function("sum", Sum::Five, [real; 5], real)
        .terminal("x", Sum::X, real);
    let set = builder.build(real).unwrap();
    let tree = set.parse("sum(x, x, sum(x, x, x, x, x), x, x)").unwrap();
    let mut columns = Columns::new(3);
    let column = tree.evaluate_columns(&set, &mut columns, |op, args, out| match op {
        Sum::Five => {
            for (point, out) in out.iter_mut().enumerate() {
                *out = args.iter().map(|arg| arg[point]).sum();
            }
        }
        Sum::X => out.copy_from_slice(&[1.0, 2.0, 3.0]),
    });
    assert_eq!(column, [9.0, 18.0, 27.0]);
}

#[test]
fn subtree_crossover_prefers_function_nodes_nine_times_in_ten() {
    // parents of distinct primitives: f and x in the first, g and y in the second
    #[derive(Clone, Copy, Debug)]
    enum Mark {
        F,
        G,
        X,
        Y,
    }
    let mut set = PrimitiveSet::builder();
    let real = set.new_type("real");
    set.function("f", Mark::F, [real, real], real)
        .function("g", Mark::G, [real, real], real)
        .terminal("x", Mark::X, real)
        .terminal("y", Mark::Y, real);
    let gp = Gp::builder(set.build(real).unwrap()).build().unwrap();
    let a = gp.primitives().parse("f(f(x, x), f(x, f(x, x)))").unwrap();
    let b = gp.primitives().parse("g(y, g(g(y, y), y))").unwrap();
    let (f, g) = (Node::Primitive(0), Node::Primitive(1));
    let mut rng = StreamRng::seed_from_u64(3);
    let (mut first, mut second) = (0, 0);
    let trials = 10_000;
    for _ in 0..trials {
        let (mut child_a, mut child_b) = (a.clone(), b.clone());
        SubtreeCrossover::new().crossover(&gp, &mut child_a, &mut child_b, &mut rng);
        // the subtree that came from the other parent starts at its first foreign node
        let from_a = child_b
            .nodes()
            .iter()
            .find(|&&node| node != g && node != Node::Primitive(3));
        let from_b = child_a
            .nodes()
            .iter()
            .find(|&&node| node != f && node != Node::Primitive(2));
        first += usize::from(from_a == Some(&f));
        second += usize::from(from_b == Some(&g));
    }
    // 0.9 within 4 standard deviations (0.003 each)
    for count in [first, second] {
        let share = count as f64 / f64::from(trials);
        assert!((share - 0.9).abs() < 0.012, "{share}");
    }
}

#[test]
fn ramped_half_and_half_divides_evenly_without_duplicates() {
    let gp = Gp::builder(koza_set()).build().unwrap();
    let trees = gp
        .ramped_half_and_half(500, &mut StreamRng::seed_from_u64(1))
        .unwrap();
    assert_eq!(trees.len(), 500);
    let unique: std::collections::HashSet<&Tree> = trees.iter().collect();
    assert_eq!(unique.len(), 500);
    for (index, tree) in trees.iter().enumerate() {
        let depth = 2 + index % 5;
        let leaves = leaf_depths(tree, gp.primitives());
        if (index / 5) % 2 == 0 {
            // full
            assert!(leaves.iter().all(|&d| d == depth));
        } else {
            assert!(leaves.iter().all(|&d| d <= depth));
        }
    }
}

// deep trees: every operation of genoxide's is a loop, none recursive
#[test]
fn deep_trees_need_no_recursion() {
    #[derive(Clone, Copy, Debug, PartialEq)]
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    enum Unary {
        Neg,
        X,
    }
    let mut set = PrimitiveSet::builder();
    let real = set.new_type("real");
    set.function("neg", Unary::Neg, [real], real)
        .terminal("x", Unary::X, real);
    let set = set.build(real).unwrap();
    let depth = 300_000;
    let mut nodes = vec![Node::Primitive(0); depth];
    nodes.push(Node::Primitive(1));
    let tree = Tree::from(nodes);
    let gp = Gp::builder(set.clone())
        .max_depth(1 << 24)
        .max_size(1 << 24)
        .build()
        .unwrap();
    assert!(gp.validate(&tree).is_ok());
    assert_eq!(tree.depth(&set), depth);
    let copy = tree.clone();
    assert_eq!(copy, tree);
    let hash = |tree: &Tree| {
        use std::hash::{BuildHasher, BuildHasherDefault};
        BuildHasherDefault::<std::collections::hash_map::DefaultHasher>::default().hash_one(tree)
    };
    assert_eq!(hash(&copy), hash(&tree));
    let text = tree.display(&set).to_string();
    assert_eq!(set.parse(&text).unwrap(), tree);
    let value = tree.evaluate(
        &set,
        &mut Vec::new(),
        |op, args: &[f64]| match op {
            Unary::Neg => -args[0],
            Unary::X => 2.0,
        },
        |_, c| c,
    );
    assert_eq!(value, 2.0);
    let mut columns = Columns::new(1);
    let column = tree.evaluate_columns(&set, &mut columns, |op, args, out| match op {
        Unary::Neg => out[0] = -args[0][0],
        Unary::X => out[0] = 2.0,
    });
    assert_eq!(column, [2.0]);
    let mut other = tree.clone();
    SubtreeMutation::new().mutate(&gp, &mut other, &mut StreamRng::seed_from_u64(0));
    assert!(gp.validate(&other).is_ok());
    #[cfg(feature = "serde")]
    {
        let json = serde_json::to_string(&tree).unwrap();
        assert_eq!(serde_json::from_str::<Tree>(&json).unwrap(), tree);
    }
}

#[cfg(feature = "serde")]
#[test]
fn sets_and_representations_deserialize_as_built() {
    let gp = typed_gp(8, 300);
    let json = serde_json::to_string(&gp).unwrap();
    let back: Gp<Typed> = serde_json::from_str(&json).unwrap();
    assert_eq!(back, gp);
    let tree = gp.random_genome(&mut StreamRng::seed_from_u64(1));
    let mut rng = StreamRng::seed_from_u64(2);
    assert_eq!(back.random_genome(&mut rng), {
        let mut rng = StreamRng::seed_from_u64(2);
        gp.random_genome(&mut rng)
    });
    assert!(back.validate(&tree).is_ok());
    // a name used twice, and constants that can't be
    let twice = json.replacen("\"mul\"", "\"add\"", 1);
    assert!(serde_json::from_str::<Gp<Typed>>(&twice).is_err());
    let empty = json.replacen("\"high\":3", "\"high\":-4", 1);
    assert_ne!(empty, json);
    assert!(serde_json::from_str::<Gp<Typed>>(&empty).is_err());
    let limits = json.replacen("\"max_size\":300", "\"max_size\":0", 1);
    assert!(serde_json::from_str::<Gp<Typed>>(&limits).is_err());
}

// the cubic x³ + x² + x: a small run finds it
fn cubic_ga(
    seed: u64,
    parallel_breeding: bool,
) -> Ga<Gp<Op>, Tournament, SubtreeCrossover, SubtreeMutation> {
    let builder = Ga::builder(Gp::builder(koza_set()).build().unwrap())
        .population_size(200)
        .select(Tournament::new(7).unwrap())
        .crossover(SubtreeCrossover::new())
        .mutate(SubtreeMutation::new())
        .mutation_rate(0.1)
        .minimize()
        .seed(seed);
    #[cfg(feature = "parallel")]
    let builder = builder.parallel_breeding(parallel_breeding);
    #[cfg(not(feature = "parallel"))]
    assert!(!parallel_breeding);
    builder.build().unwrap()
}

fn cubic_error(tree: &Tree) -> Option<f64> {
    thread_local! {
        static STACK: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let set = koza_set_cached();
    STACK.with_borrow_mut(|stack| {
        let mut sum = 0.0;
        for i in 0..20 {
            let x = f64::from(i) / 10.0 - 0.95;
            let value = tree.evaluate(set, stack, |op, args| apply(op, args, x), |_, c| c);
            let error = value - (x * x * x + x * x + x);
            sum += error * error;
        }
        sum.is_finite().then_some(sum)
    })
}

fn koza_set_cached() -> &'static PrimitiveSet<Op> {
    static SET: std::sync::OnceLock<PrimitiveSet<Op>> = std::sync::OnceLock::new();
    SET.get_or_init(koza_set)
}

#[cfg(feature = "parallel")]
#[test]
fn seeded_runs_are_the_same_on_any_number_of_threads() {
    let run = |parallel: bool, breeding: bool| {
        let mut engine = Engine::new(cubic_ga(5, breeding), cubic_error)
            .parallel(parallel)
            .stop_when(Stop::generations(15));
        let outcome = engine.run().unwrap();
        (outcome.into_best(), engine.algorithm().population().clone())
    };
    let sequential = run(false, false);
    assert_eq!(run(true, false), sequential);
    let bred = run(false, true);
    let on_threads = |threads: usize, parallel: bool| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| run(parallel, true))
    };
    for threads in [1, 8] {
        assert_eq!(on_threads(threads, false), bred);
        assert_eq!(on_threads(threads, true), bred);
    }
}

#[test]
fn a_ga_finds_the_cubic() {
    let outcome = Engine::new(cubic_ga(1, false), cubic_error)
        .stop_when(Stop::target(1e-20).or(Stop::generations(100)))
        .run()
        .unwrap();
    assert_eq!(
        outcome.stop_reason(),
        StopReason::Target,
        "{:?}",
        outcome.best_fitness()
    );
}

// ---- point, hoist, shrink and constant mutation, one-point crossover (batch G2) -------------

// the positions where two trees of the same shape differ
fn differences(a: &Tree, b: &Tree) -> Vec<usize> {
    assert_eq!(a.len(), b.len());
    (0..a.len())
        .filter(|&i| a.nodes()[i] != b.nodes()[i])
        .collect()
}

// the nodes of a tree of the typed set that point mutation can change: every node but the
// functions alone of their signature (`less`, `and`, `if`)
fn replaceable(gp: &Gp<Typed>, tree: &Tree) -> usize {
    let set = gp.primitives();
    tree.nodes()
        .iter()
        .filter(|node| match node {
            Node::Primitive(index) => !matches!(
                set.primitives()[*index as usize].value(),
                Typed::Less | Typed::And | Typed::If
            ),
            Node::Constant { .. } => true,
        })
        .count()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn point_mutation_changes_exactly_the_picked_nodes_keeping_types_and_shape(
        seed: u64, max_depth in 3usize..9, max_size in 15usize..120, count in 1usize..5,
    ) {
        let gp = typed_gp(max_depth, max_size);
        let set = gp.primitives();
        let mut rng = StreamRng::seed_from_u64(seed);
        for _ in 0..20 {
            let tree = gp.random_genome(&mut rng);
            let mut mutated = tree.clone();
            PointMutation::count(count).unwrap().mutate(&gp, &mut mutated, &mut rng);
            prop_assert!(gp.validate(&mutated).is_ok(), "{:?}", gp.validate(&mutated));
            // a picked node always changes: exactly `count` nodes, or all that can
            let changed = differences(&tree, &mutated);
            prop_assert_eq!(changed.len(), count.min(replaceable(&gp, &tree)));
            for position in changed {
                let (before, after) = (tree.nodes()[position], mutated.nodes()[position]);
                let arity = |node: &Node| match node {
                    Node::Primitive(index) => set.primitives()[*index as usize].arity(),
                    Node::Constant { .. } => 0,
                };
                prop_assert_eq!(arity(&before), arity(&after));
            }
            let mut per_node = tree.clone();
            PointMutation::per_node(0.3).unwrap().mutate(&gp, &mut per_node, &mut rng);
            prop_assert!(gp.validate(&per_node).is_ok());
            prop_assert!(differences(&tree, &per_node).len() <= replaceable(&gp, &tree));
        }
    }

    #[test]
    fn hoist_and_shrink_always_shrink_within_the_limits(
        seed: u64, max_depth in 3usize..9, max_size in 15usize..120,
    ) {
        let gp = typed_gp(max_depth, max_size);
        let mut rng = StreamRng::seed_from_u64(seed);
        for _ in 0..20 {
            let tree = gp.random_genome(&mut rng);
            let mut hoisted = tree.clone();
            HoistMutation.mutate(&gp, &mut hoisted, &mut rng);
            prop_assert!(gp.validate(&hoisted).is_ok());
            // smaller, or unchanged without a function of the root's type below the root
            let set = gp.primitives();
            let hoistable = tree.nodes()[1..].iter().any(|node| match node {
                Node::Primitive(index) => {
                    let primitive = &set.primitives()[*index as usize];
                    primitive.arity() > 0 && primitive.returns() == set.root()
                }
                Node::Constant { .. } => false,
            });
            let expected = if hoistable { hoisted.len() < tree.len() } else { hoisted == tree };
            prop_assert!(expected);
            let mut shrunk = tree.clone();
            ShrinkMutation.mutate(&gp, &mut shrunk, &mut rng);
            prop_assert!(gp.validate(&shrunk).is_ok());
            prop_assert!(shrunk.len() < tree.len() || tree.len() == 1);
        }
    }

    #[test]
    fn constant_mutation_moves_one_constant_within_its_range(seed: u64, sigma in 0.001f64..2.0) {
        let gp = Gp::builder(koza_set()).build().unwrap();
        let typed = typed_gp(6, 100);
        let mut rng = StreamRng::seed_from_u64(seed);
        let mutation = ConstantMutation::gaussian(sigma).unwrap();
        let constants = |tree: &Tree| {
            tree.nodes().iter().filter(|n| matches!(n, Node::Constant { .. })).count()
        };
        for _ in 0..20 {
            // uniform constants in [-1, 1], and integers -3 to 3
            let tree = gp.random_genome(&mut rng);
            let mut mutated = tree.clone();
            mutation.mutate(&gp, &mut mutated, &mut rng);
            prop_assert!(gp.validate(&mutated).is_ok());
            let changed = differences(&tree, &mutated);
            prop_assert_eq!(changed.len(), usize::from(constants(&tree) > 0));
            for position in changed {
                let is_constant = matches!(mutated.nodes()[position], Node::Constant { .. });
                prop_assert!(is_constant);
            }
            let tree = typed.random_genome(&mut rng);
            let mut mutated = tree.clone();
            mutation.mutate(&typed, &mut mutated, &mut rng);
            prop_assert!(typed.validate(&mutated).is_ok());
            prop_assert_eq!(differences(&tree, &mutated).len(), usize::from(constants(&tree) > 0));
        }
    }

    #[test]
    fn a_mix_of_mutations_always_changes_the_tree_within_the_limits(
        seed: u64, max_depth in 3usize..9, max_size in 15usize..120,
    ) {
        let gp = typed_gp(max_depth, max_size);
        let mutations = Mutations::builder()
            .subtree(1.0)
            .point(1.0)
            .hoist(1.0)
            .shrink(1.0)
            .with(1.0, ConstantMutation::gaussian(0.1).unwrap())
            .build()
            .unwrap();
        let mut rng = StreamRng::seed_from_u64(seed);
        for _ in 0..20 {
            let mut tree = gp.random_genome(&mut rng);
            let before = tree.clone();
            mutations.mutate(&gp, &mut tree, &mut rng);
            prop_assert!(gp.validate(&tree).is_ok());
            prop_assert_ne!(tree, before);
        }
    }

    #[test]
    fn one_point_crossover_keeps_trees_typed_within_limits_and_conserves_nodes(
        seed: u64, max_depth in 3usize..9, max_size in 15usize..120,
    ) {
        let gp = typed_gp(max_depth, max_size);
        let mut rng = StreamRng::seed_from_u64(seed);
        for _ in 0..20 {
            let (mut a, mut b) = (gp.random_genome(&mut rng), gp.random_genome(&mut rng));
            let depth = a.depth(gp.primitives()).max(b.depth(gp.primitives()));
            let before = is_sorted_multiset([a.nodes(), b.nodes()].concat());
            OnePointCrossover.crossover(&gp, &mut a, &mut b, &mut rng);
            prop_assert!(gp.validate(&a).is_ok(), "{:?}", gp.validate(&a));
            prop_assert!(gp.validate(&b).is_ok(), "{:?}", gp.validate(&b));
            prop_assert_eq!(is_sorted_multiset([a.nodes(), b.nodes()].concat()), before);
            // exchanged at the same depth: no child deeper than both parents
            prop_assert!(a.depth(gp.primitives()) <= depth && b.depth(gp.primitives()) <= depth);
        }
    }
}

#[test]
fn point_mutation_exchanges_primitives_of_one_signature() {
    let (set, ..) = typed_set();
    let gp = Gp::builder(set).build().unwrap();
    let set = gp.primitives();
    let mut rng = StreamRng::seed_from_u64(1);
    let mut seen = std::collections::BTreeSet::new();
    for _ in 0..500 {
        // add can only become mul; less and if never change; x becomes y or a constant, and the
        // constant another constant, x or y
        let mut tree = set.parse("if(less(x, 1.0), add(x, y), x)").unwrap();
        PointMutation::count(1)
            .unwrap()
            .mutate(&gp, &mut tree, &mut rng);
        seen.insert(tree.display(set).to_string().replace(char::is_numeric, "#"));
    }
    let expected: std::collections::BTreeSet<String> = [
        "if(less(#.#, #.#), add(x, y), x)",
        "if(less(-#.#, #.#), add(x, y), x)",
        "if(less(y, #.#), add(x, y), x)",
        "if(less(x, #.#), add(x, y), x)",
        "if(less(x, -#.#), add(x, y), x)",
        "if(less(x, x), add(x, y), x)",
        "if(less(x, y), add(x, y), x)",
        "if(less(x, #.#), add(#.#, y), x)",
        "if(less(x, #.#), add(-#.#, y), x)",
        "if(less(x, #.#), add(y, y), x)",
        "if(less(x, #.#), add(x, #.#), x)",
        "if(less(x, #.#), add(x, -#.#), x)",
        "if(less(x, #.#), add(x, x), x)",
        "if(less(x, #.#), mul(x, y), x)",
        "if(less(x, #.#), add(x, y), #.#)",
        "if(less(x, #.#), add(x, y), -#.#)",
        "if(less(x, #.#), add(x, y), y)",
    ]
    .map(String::from)
    .into();
    assert!(seen.is_subset(&expected), "{seen:?}");
    // the constant 1.0 was replaced by another value: "less(x, #.#)" with another number
    assert!(seen.contains("if(less(x, y), add(x, y), x)"));
    assert!(seen.contains("if(less(x, #.#), mul(x, y), x)"));
}

#[test]
fn hoist_and_shrink_by_hand() {
    let gp = Gp::builder(koza_set()).build().unwrap();
    let set = gp.primitives();
    let mut rng = StreamRng::seed_from_u64(1);
    let mut hoisted = std::collections::BTreeSet::new();
    let mut shrunk = std::collections::BTreeSet::new();
    for _ in 0..500 {
        let tree = set.parse("add(sin(x), mul(x, 0.5))").unwrap();
        let mut hoist = tree.clone();
        HoistMutation.mutate(&gp, &mut hoist, &mut rng);
        hoisted.insert(hoist.display(set).to_string());
        let mut shrink = tree.clone();
        ShrinkMutation.mutate(&gp, &mut shrink, &mut rng);
        shrunk.insert(
            shrink
                .display(set)
                .to_string()
                .replace(|c: char| c.is_numeric() || c == '-' || c == '.', "#"),
        );
    }
    // the subtrees of the functions below the root
    let expected: std::collections::BTreeSet<String> =
        ["sin(x)", "mul(x, 0.5)"].map(String::from).into();
    assert_eq!(hoisted, expected);
    // a function's subtree (add, sin or mul) replaced by x or a new constant
    for text in &shrunk {
        let leaf = |text: &str| text == "x" || text.chars().all(|c| c == '#');
        let allowed = leaf(text)
            || text
                .strip_prefix("add(")
                .and_then(|rest| rest.strip_suffix(", mul(x, ###))"))
                .is_some_and(leaf)
            || text
                .strip_prefix("add(sin(x), ")
                .and_then(|rest| rest.strip_suffix(')'))
                .is_some_and(leaf);
        assert!(allowed, "{text}");
    }
    assert!(shrunk.contains("x") && shrunk.contains("add(sin(x), x)"));
    // a single leaf has no subtree to hoist or shrink
    let leaf = set.parse("x").unwrap();
    for mutation in [
        Mutations::builder().hoist(1.0).build().unwrap(),
        Mutations::builder().shrink(1.0).build().unwrap(),
    ] {
        let mut tree = leaf.clone();
        mutation.mutate(&gp, &mut tree, &mut rng);
        assert_eq!(tree, leaf);
    }
    // with point mutation beside them, the mix changes the leaf
    let mix = Mutations::builder()
        .hoist(10.0)
        .shrink(10.0)
        .point(1.0)
        .build()
        .unwrap();
    let mut tree = leaf.clone();
    mix.mutate(&gp, &mut tree, &mut rng);
    assert_ne!(tree, leaf);
}

#[test]
fn invalid_mutations_are_errors() {
    setting_error(PointMutation::per_node(0.0), "point_mutation_rate");
    setting_error(PointMutation::per_node(1.5), "point_mutation_rate");
    setting_error(PointMutation::count(0), "point_mutation_count");
    setting_error(ConstantMutation::gaussian(0.0), "constant_sigma");
    setting_error(ConstantMutation::gaussian(f64::NAN), "constant_sigma");
    setting_error(Mutations::builder().build(), "mutations");
    setting_error(Mutations::builder().subtree(0.0).build(), "mutations");
    setting_error(
        Mutations::builder().subtree(-1.0).point(2.0).build(),
        "mutations",
    );
    setting_error(Mutations::builder().subtree(f64::NAN).build(), "mutations");
    setting_error(
        Mutations::builder()
            .subtree(f64::MAX)
            .point(f64::MAX)
            .build(),
        "mutations",
    );
    assert!(Mutations::builder().subtree(0.0).point(1.0).build().is_ok());
}

#[test]
fn a_mix_chooses_by_weight() {
    let gp = Gp::builder(koza_set()).build().unwrap();
    let set = gp.primitives();
    // hoist makes the tree smaller, point keeps its size: 3 to 1
    let mix = Mutations::builder().hoist(3.0).point(1.0).build().unwrap();
    let tree = set.parse("add(x, sin(x))").unwrap();
    let mut rng = StreamRng::seed_from_u64(1);
    let draws = 20_000;
    let mut hoists = 0u32;
    for _ in 0..draws {
        let mut mutated = tree.clone();
        mix.mutate(&gp, &mut mutated, &mut rng);
        hoists += u32::from(mutated.len() < tree.len());
    }
    let (p, n) = (0.75, f64::from(draws));
    let deviation = (n * p * (1.0 - p)).sqrt();
    assert!(
        (f64::from(hoists) - n * p).abs() < 4.0 * deviation,
        "{hoists}"
    );
}

// ---- Boolean problems -----------------------------------------------------------------------

#[test]
fn multiplexer_cases_by_hand() {
    let problem = Multiplexer::new(3).unwrap();
    assert_eq!((problem.inputs(), problem.cases()), (11, 2048));
    assert_eq!(problem.address_bits(), 3);
    let set = problem.primitives();
    let target = |case: u64| problem.targets()[(case / 64) as usize] >> (case % 64) & 1 == 1;
    // case bits: a0 a1 a2 (the address, a0 its lowest bit), then d0 to d7
    let case = |address: u64, data: u64| address | data << 3;
    assert!(target(case(5, 1 << 5)));
    assert!(!target(case(5, !(1 << 5) & 0xff)));
    assert!(target(case(0, 1)));
    assert!(!target(case(7, 0x7f)));
    // each data bit alone is right where it's selected (256 cases) and in half the rest
    for data in 0..8 {
        let tree = set.parse(&format!("d{data}")).unwrap();
        assert_eq!(problem.errors(&tree), 896);
    }
    assert_eq!(problem.errors(&set.parse("a0").unwrap()), 1024);
    // the multiplexer itself, and its negation
    let right =
        "if(a2, if(a1, if(a0, d7, d6), if(a0, d5, d4)), if(a1, if(a0, d3, d2), if(a0, d1, d0)))";
    assert_eq!(problem.errors(&set.parse(right).unwrap()), 0);
    let wrong = set.parse(&format!("not({right})")).unwrap();
    assert_eq!(problem.errors(&wrong), 2048);
    assert_eq!(problem.evaluate(&wrong), 2048.0);
    let outputs = problem.outputs(&set.parse(right).unwrap());
    assert_eq!(outputs, problem.targets());
    // the 6-multiplexer: 64 cases, one word; the 3-multiplexer: 8 cases, the rest of the word 0
    let six = Multiplexer::new(2).unwrap();
    assert_eq!((six.inputs(), six.cases(), six.targets().len()), (6, 64, 1));
    let three = Multiplexer::new(1).unwrap();
    // cases 0 to 7, bits a0 d0 d1: the output is d0 (bit 1) if a0 is 0, else d1 (bit 2)
    assert_eq!(three.targets(), [0b1110_0100]);
    let tree = three.primitives().parse("not(d0)").unwrap();
    assert_eq!(three.outputs(&tree)[0] >> 8, 0);
}

#[test]
fn even_parity_by_hand() {
    for inputs in [2, 3, 5, 7] {
        let problem = EvenParity::new(inputs).unwrap();
        let cases = 1u64 << inputs;
        assert_eq!((problem.inputs(), problem.cases()), (inputs, cases));
        for case in 0..cases {
            let bit = problem.targets()[(case / 64) as usize] >> (case % 64) & 1;
            assert_eq!(bit == 1, case.count_ones() % 2 == 0);
        }
        let set = problem.primitives();
        // one input is right in half the cases, and so is its negation
        assert_eq!(problem.errors(&set.parse("d0").unwrap()), cases / 2);
        assert_eq!(
            problem.errors(&set.parse("nor(d1, d1)").unwrap()),
            cases / 2
        );
    }
    let problem = EvenParity::new(2).unwrap();
    let set = problem.primitives();
    let even = set.parse("or(and(d0, d1), nor(d0, d1))").unwrap();
    assert_eq!(problem.errors(&even), 0);
    let odd = set.parse("and(or(d0, d1), nand(d0, d1))").unwrap();
    assert_eq!(problem.errors(&odd), 4);
}

#[test]
fn invalid_boolean_problems_are_errors() {
    setting_error(Multiplexer::new(0), "address_bits");
    setting_error(Multiplexer::new(5), "address_bits");
    setting_error(EvenParity::new(1), "inputs");
    setting_error(EvenParity::new(21), "inputs");
}

// ---- bloat control: lexicographic parsimony, double tournament, Tarpeian --------------------

// individuals of the given trees and scores
fn trees_with_scores(set: &PrimitiveSet<Op>, trees: &[(&str, f64)]) -> Population<Tree> {
    trees
        .iter()
        .map(|&(text, score)| {
            let mut individual = Individual::new(set.parse(text).unwrap());
            individual.set_fitness(Fitness::new(score));
            individual
        })
        .collect()
}

// how often each individual is selected, one call per selection or all in one call
fn selected<S: Select>(
    select: &S,
    population: &Population<Tree>,
    draws: u32,
    one_per_call: bool,
) -> Vec<u32> {
    let mut rng = StreamRng::seed_from_u64(1);
    let mut counts = vec![0; population.len()];
    let picks = if one_per_call {
        (0..draws)
            .flat_map(|_| select.select(population, Objective::Maximize, 1, &mut rng))
            .collect()
    } else {
        select.select(population, Objective::Maximize, draws as usize, &mut rng)
    };
    for index in picks {
        counts[index] += 1;
    }
    counts
}

// asserts that `count` of `draws` is within 4 standard deviations of probability `p`
fn assert_frequency(count: u32, draws: u32, p: f64) {
    let n = f64::from(draws);
    let deviation = (n * p * (1.0 - p)).sqrt();
    assert!(
        (f64::from(count) - n * p).abs() <= 4.0 * deviation.max(0.5),
        "{count} of {draws}, expected {p}"
    );
}

#[test]
fn lexicographic_tournaments_give_ties_to_the_smaller() {
    let set = koza_set();
    let draws = 40_000;
    // equal fitness: of two drawn, the smaller wins; the same one drawn twice half the time
    let equal = trees_with_scores(&set, &[("add(x, x)", 1.0), ("x", 1.0)]);
    let select = LexicographicTournament::new(2).unwrap();
    for one_per_call in [true, false] {
        let counts = selected(&select, &equal, draws, one_per_call);
        assert_frequency(counts[1], draws, 0.75);
    }
    // fitness first: the larger, fitter one wins whenever drawn
    let fitter = trees_with_scores(&set, &[("add(x, x)", 2.0), ("x", 1.0)]);
    assert_frequency(selected(&select, &fitter, draws, false)[0], draws, 0.75);
    // equal fitness and size: the first drawn wins, each half the time
    let same = trees_with_scores(&set, &[("add(x, x)", 1.0), ("mul(x, x)", 1.0)]);
    assert_frequency(selected(&select, &same, draws, false)[0], draws, 0.5);
    setting_error(LexicographicTournament::new(0), "tournament_size");
    // bucketed, near fitness values tie: 1.0 and 1.5 share the lowest bucket, where the smaller
    // wins; both lose to the others, so each is picked only by a tournament of the two alone
    let near = trees_with_scores(
        &set,
        &[("add(x, x)", 1.5), ("x", 1.0), ("x", 5.0), ("x", 6.0)],
    );
    let select = LexicographicTournament::new(2)
        .unwrap()
        .ratio_buckets(0.5)
        .unwrap();
    let counts = selected(&select, &near, draws, false);
    assert_frequency(counts[0], draws, 1.0 / 16.0);
    assert_frequency(counts[1], draws, 3.0 / 16.0);
    let bucketed = LexicographicTournament::new(2).unwrap();
    setting_error(bucketed.ratio_buckets(0.0), "bucket_ratio");
    setting_error(bucketed.ratio_buckets(1.5), "bucket_ratio");
}

#[test]
fn double_tournaments_pick_the_smaller_of_two_with_probability_d_over_2() {
    let set = koza_set();
    let draws = 40_000;
    // equal fitness, so each fitness tournament of 1 is a random individual: the size
    // tournament meets two different individuals half the time, where the smaller wins with
    // probability D / 2, and the same one twice the other half
    let population = trees_with_scores(&set, &[("add(x, x)", 1.0), ("x", 1.0)]);
    for parsimony in [1.0, 1.2, 1.4, 1.7, 2.0] {
        let p = 0.25 + 0.5 * parsimony / 2.0;
        let fitness_first = DoubleTournament::new(1, parsimony).unwrap();
        let size_first = fitness_first.size_first();
        for select in [fitness_first, size_first] {
            let counts = selected(&select, &population, draws, false);
            assert_frequency(counts[1], draws, p);
        }
    }
    // fitness first, fitness tournaments of 2: the fitter, larger one wins each fitness
    // tournament with probability 3/4; the size tournament then meets it and the smaller one
    // (probability 2 * 3/4 * 1/4) or two copies of either
    let fitter = trees_with_scores(&set, &[("add(x, x)", 2.0), ("x", 1.0)]);
    let parsimony = 1.4;
    let select = DoubleTournament::new(2, parsimony).unwrap();
    let p_small = 0.25 * 0.25 + 2.0 * 0.75 * 0.25 * (parsimony / 2.0);
    assert_frequency(selected(&select, &fitter, draws, false)[1], draws, p_small);
    setting_error(DoubleTournament::new(0, 1.4), "tournament_size");
    setting_error(DoubleTournament::new(7, 0.9), "parsimony");
    setting_error(DoubleTournament::new(7, 2.1), "parsimony");
    setting_error(DoubleTournament::new(7, f64::NAN), "parsimony");
}

#[test]
fn tarpeian_marks_genomes_above_the_mean_size_at_its_rate() {
    let set = koza_set();
    let draws = 20_000;
    // the larger, fitter one is above the mean size: truncation to the best half selects it
    // unless it's marked, with probability `rate`
    let population = trees_with_scores(&set, &[("add(x, x)", 2.0), ("x", 1.0)]);
    for rate in [0.1, 0.3, 0.5, 1.0] {
        let select = Tarpeian::new(Truncation::new(0.5).unwrap(), rate).unwrap();
        let counts = selected(&select, &population, draws, true);
        assert_frequency(counts[1], draws, rate);
        // marks are per call: one call selects from one marking
        let counts = selected(&select, &population, draws, false);
        assert!(counts.contains(&0), "{counts:?}");
    }
    // nothing above the mean: nothing marked
    let equal = trees_with_scores(&set, &[("add(x, x)", 2.0), ("mul(x, x)", 1.0)]);
    let select = Tarpeian::new(Truncation::new(0.5).unwrap(), 1.0).unwrap();
    assert_eq!(selected(&select, &equal, 100, true), [100, 0]);
    // above the mean by the exact mean: 3 nodes against a mean of 7/3
    let three = trees_with_scores(&set, &[("x", 1.0), ("add(x, x)", 3.0), ("add(x, x)", 2.0)]);
    let select = Tarpeian::new(Truncation::new(0.3).unwrap(), 1.0).unwrap();
    assert_eq!(selected(&select, &three, 100, true), [100, 0, 0]);
    setting_error(
        Tarpeian::new(Tournament::new(2).unwrap(), 0.0),
        "tarpeian_rate",
    );
    setting_error(
        Tarpeian::new(Tournament::new(2).unwrap(), 1.1),
        "tarpeian_rate",
    );
}

// the squared error on Koza's quartic at 20 points
fn quartic_error(tree: &Tree) -> Option<f64> {
    thread_local! {
        static STACK: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let set = koza_set_cached();
    STACK.with_borrow_mut(|stack| {
        let mut sum = 0.0;
        for i in 0..20 {
            let x = f64::from(i) / 10.0 - 0.95;
            let value = tree.evaluate(set, stack, |op, args| apply(op, args, x), |_, c| c);
            let error = value - (x * x * x * x + x * x * x + x * x + x);
            sum += error * error;
        }
        sum.is_finite().then_some(sum)
    })
}

// the mean size of the trees after 50 generations on the quartic
fn mean_size<S: Select>(select: S, seed: u64) -> f64 {
    let ga = Ga::builder(Gp::builder(koza_set()).build().unwrap())
        .population_size(100)
        .select(select)
        .crossover(SubtreeCrossover::new())
        .mutate(SubtreeMutation::new())
        .mutation_rate(0.1)
        .minimize()
        .seed(seed)
        .build()
        .unwrap();
    let mut engine = Engine::new(ga, quartic_error).stop_when(Stop::generations(50));
    engine.run().unwrap();
    let population = engine.algorithm().population();
    let total: usize = population.iter().map(|i| i.genome().len()).sum();
    total as f64 / population.len() as f64
}

#[test]
fn double_tournaments_keep_trees_smaller_than_tournaments() {
    // Luke and Panait's setting, D = 1.4 with fitness tournaments of 7, against tournaments of
    // 7, over 20 seeds: smaller on average, and on most seeds (a sign test: 15 or more of 20 has
    // probability 0.021 if neither were smaller)
    let (mut smaller, mut plain_sum, mut double_sum) = (0, 0.0, 0.0);
    for seed in 1..=20 {
        let plain = mean_size(Tournament::new(7).unwrap(), seed);
        let double = mean_size(DoubleTournament::new(7, 1.4).unwrap(), seed);
        smaller += usize::from(double < plain);
        plain_sum += plain;
        double_sum += double;
    }
    assert!(smaller >= 15, "{smaller} of 20");
    assert!(double_sum < plain_sum, "{double_sum} against {plain_sum}");
}

#[cfg(feature = "parallel")]
#[test]
fn seeded_runs_with_bloat_control_and_mixed_mutations_are_the_same_on_any_number_of_threads() {
    let run = |parallel: bool| {
        let ga = Ga::builder(Gp::builder(koza_set()).build().unwrap())
            .population_size(100)
            .select(Tarpeian::new(DoubleTournament::new(7, 1.4).unwrap(), 0.3).unwrap())
            .crossover(OnePointCrossover)
            .mutate(
                Mutations::builder()
                    .subtree(1.0)
                    .point(1.0)
                    .hoist(1.0)
                    .shrink(1.0)
                    .with(1.0, ConstantMutation::gaussian(0.1).unwrap())
                    .build()
                    .unwrap(),
            )
            .mutation_rate(0.3)
            .minimize()
            .parallel_breeding(parallel)
            .seed(3)
            .build()
            .unwrap();
        let mut engine = Engine::new(ga, quartic_error)
            .parallel(parallel)
            .stop_when(Stop::generations(15));
        let outcome = engine.run().unwrap();
        (outcome.into_best(), engine.algorithm().population().clone())
    };
    let sequential = run(false);
    assert_eq!(run(false), sequential);
    let bred = run(true);
    for threads in [1, 8] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        assert_eq!(pool.install(|| run(true)), bred);
    }
}

#[cfg(feature = "serde")]
#[test]
fn mixes_and_bloat_control_deserialize_as_built() {
    let mutations = Mutations::builder()
        .subtree(0.5)
        .point(0.3)
        .with(0.2, ConstantMutation::gaussian(0.1).unwrap())
        .build()
        .unwrap();
    let json = serde_json::to_string(&mutations).unwrap();
    assert_eq!(serde_json::from_str::<Mutations>(&json).unwrap(), mutations);
    // weights are checked as `build` checks them
    let negative = json.replacen("0.5", "-0.5", 1);
    assert_ne!(negative, json);
    assert!(serde_json::from_str::<Mutations>(&negative).is_err());
    let select = Tarpeian::new(DoubleTournament::new(7, 1.4).unwrap().size_first(), 0.3).unwrap();
    let json = serde_json::to_string(&select).unwrap();
    assert_eq!(
        serde_json::from_str::<Tarpeian<DoubleTournament>>(&json).unwrap(),
        select
    );
    let select = LexicographicTournament::new(2)
        .unwrap()
        .ratio_buckets(0.5)
        .unwrap();
    let json = serde_json::to_string(&select).unwrap();
    assert_eq!(
        serde_json::from_str::<LexicographicTournament>(&json).unwrap(),
        select
    );
}

// Keijzer's normal constants: drawn with the mean and deviation, any finite value, moved by
// constant mutation and drawn anew by point mutation
#[test]
fn normal_constants() {
    for (mean, deviation) in [
        (0.0, 0.0),
        (f64::NAN, 1.0),
        (0.0, f64::INFINITY),
        (0.0, -1.0),
    ] {
        setting_error(Constants::normal(mean, deviation), "constants");
    }
    let constants = Constants::normal(2.0, 5.0).unwrap();
    let mut rng = StreamRng::seed_from_u64(1);
    let values: Vec<f64> = (0..20_000).map(|_| constants.sample(&mut rng)).collect();
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let deviation =
        (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt();
    assert!((mean - 2.0).abs() < 0.1, "{mean}");
    assert!((deviation - 5.0).abs() < 0.1, "{deviation}");
    assert!(constants.contains(1e300) && !constants.contains(f64::INFINITY));

    let mut set = PrimitiveSet::builder();
    let real = set.new_type("real");
    set.function("mul", Op::Mul, [real, real], real)
        .terminal("x", Op::X, real)
        .constants(real, constants);
    let gp = Gp::builder(set.build(real).unwrap()).build().unwrap();
    let tree = gp.primitives().parse("mul(x, 0.5)").unwrap();
    for seed in 0..50 {
        let mut rng = StreamRng::seed_from_u64(seed);
        let mut moved = tree.clone();
        ConstantMutation::gaussian(0.1)
            .unwrap()
            .mutate(&gp, &mut moved, &mut rng);
        let Node::Constant { value, .. } = moved.nodes()[2] else {
            panic!("a constant")
        };
        assert!(value != 0.5 && value.is_finite() && gp.validate(&moved).is_ok());
        let mut redrawn = tree.clone();
        PointMutation::count(1)
            .unwrap()
            .mutate(&gp, &mut redrawn, &mut rng);
        assert!(redrawn != tree && gp.validate(&redrawn).is_ok());
    }
    #[cfg(feature = "serde")]
    {
        let json = serde_json::to_string(&gp).unwrap();
        assert_eq!(serde_json::from_str::<Gp<Op>>(&json).unwrap(), gp);
        let invalid = json.replacen("\"deviation\":5.0", "\"deviation\":0.0", 1);
        assert_ne!(invalid, json);
        assert!(serde_json::from_str::<Gp<Op>>(&invalid).is_err());
    }
}
