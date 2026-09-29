//! Tree genetic programming: primitive sets, generation, parsing, the evaluators and the operators.

use genoxide::gp::{
    Columns, Constants, Gp, Init, Node, PrimitiveSet, Subtree, SubtreeCrossover, SubtreeMutation,
    Tree, Type,
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
