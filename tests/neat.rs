//! NEAT: networks and their evaluation, speciation, reproduction and the ask / tell protocol.

use genoxide::neat::{Initial, Neat, Network, NodeKind, Sharing};
use genoxide::nn::Activation;
use genoxide::prelude::*;
use proptest::prelude::*;

const CASES: [([f64; 2], f64); 4] = [
    ([0.0, 0.0], 0.0),
    ([0.0, 1.0], 1.0),
    ([1.0, 0.0], 1.0),
    ([1.0, 1.0], 0.0),
];

// the paper's XOR fitness, (4 − Σ|error|)²
fn xor(network: &Network) -> f64 {
    let mut evaluator = network.feed_forward().expect("feed-forward");
    let mut output = [0.0];
    let error: f64 = CASES
        .iter()
        .map(|(input, target)| {
            evaluator.activate(input, &mut output);
            (output[0] - target).abs()
        })
        .sum();
    (4.0 - error).powi(2)
}

fn setting_error<T: std::fmt::Debug>(result: Result<T>, expected: &str) {
    match result {
        Err(Error::InvalidSetting { setting, .. }) => assert_eq!(setting, expected),
        other => panic!("expected an invalid {expected}, got {other:?}"),
    }
}

#[test]
fn a_network_by_hand() {
    let network = Network::fully_connected(2, 1, &[1.0, -2.0, 0.5]);
    assert_eq!(
        (network.inputs(), network.outputs(), network.hidden()),
        (2, 1, 0)
    );
    let kinds: Vec<NodeKind> = network.nodes().iter().map(|n| n.kind()).collect();
    assert_eq!(
        kinds,
        [
            NodeKind::Input,
            NodeKind::Input,
            NodeKind::Bias,
            NodeKind::Output
        ]
    );
    let innovations: Vec<u32> = network
        .connections()
        .iter()
        .map(|c| c.innovation())
        .collect();
    assert_eq!(innovations, [0, 1, 2]);
    let mut evaluator = network.feed_forward().unwrap();
    let mut output = [0.0];
    evaluator.activate(&[0.25, 0.5], &mut output);
    // 0.25 - 1 + 0.5, through the steepened sigmoid
    let expected = Activation::SteepSigmoid.apply(1.0 * 0.25 - 2.0 * 0.5 + 0.5);
    assert_eq!(output[0].to_bits(), expected.to_bits());
    assert_eq!(network.len(), 3);
}

#[test]
fn invalid_settings_are_errors() {
    setting_error(Neat::builder(0, 1).build(), "neat");
    setting_error(Neat::builder(2, 0).build(), "neat");
    setting_error(
        Neat::builder(2, 1).population_size(0).build(),
        "population_size",
    );
    setting_error(
        Neat::builder(2, 1)
            .compatibility(1.0, 1.0, 0.4, 0.0)
            .build(),
        "compatibility",
    );
    setting_error(
        Neat::builder(2, 1)
            .compatibility(-1.0, 1.0, 0.4, 3.0)
            .build(),
        "compatibility",
    );
    setting_error(
        Neat::builder(2, 1).weight_mutation(1.5, 0.1).build(),
        "weight_mutation",
    );
    setting_error(
        Neat::builder(2, 1).weight_deviations(0.0, 1.0).build(),
        "weight_deviations",
    );
    setting_error(
        Neat::builder(2, 1).structural_mutation(0.1, -0.1).build(),
        "structural_mutation",
    );
    setting_error(
        Neat::builder(2, 1)
            .reproduction(0.25, 0.001, f64::NAN)
            .build(),
        "reproduction",
    );
    setting_error(Neat::builder(2, 1).selection(5, 0.0).build(), "selection");
    setting_error(Neat::builder(2, 1).stagnation(0).build(), "stagnation");
    setting_error(
        Neat::builder(2, 1).sharing(Sharing::Raw).minimize().build(),
        "sharing",
    );
}

#[test]
fn the_ask_tell_protocol() {
    let mut neat = Neat::builder(2, 1)
        .population_size(20)
        .seed(1)
        .build()
        .unwrap();
    assert!(matches!(neat.tell(&[]), Err(Error::TellWithoutAsk)));
    let asked = neat.ask().len();
    assert_eq!(asked, 20);
    assert!(matches!(
        neat.tell(&[Fitness::new(1.0)]),
        Err(Error::FitnessCount {
            expected: 20,
            got: 1
        })
    ));
    let fitness: Vec<Fitness> = neat.ask().iter().map(|n| Fitness::new(xor(n))).collect();
    neat.tell(&fitness).unwrap();
    assert_eq!((neat.generation(), neat.evaluations()), (0, 20));
    assert!(neat.best().is_some() && !neat.species().is_empty());
    // the species partition the population
    let mut members: Vec<usize> = neat
        .species()
        .iter()
        .flat_map(|s| s.members().to_vec())
        .collect();
    members.sort_unstable();
    assert_eq!(members, (0..20).collect::<Vec<_>>());
    // a re-evaluation asks the whole population again, without a new generation
    neat.reevaluate().unwrap();
    let asked = neat.ask().len();
    assert!(matches!(
        neat.reevaluate(),
        Err(Error::ReevaluationOutOfTurn)
    ));
    neat.tell(&vec![Fitness::new(0.0); asked]).unwrap();
    assert_eq!((asked, neat.generation()), (20, 0));
}

// raw sharing, the paper's, needs valid, feasible and non-negative scores
#[test]
fn raw_sharing_rejects_negative_scores() {
    let mut neat = Neat::builder(2, 1)
        .population_size(10)
        .sharing(Sharing::Raw)
        .seed(1)
        .build()
        .unwrap();
    let asked = neat.ask().len();
    let mut fitness = vec![Fitness::new(1.0); asked];
    fitness[3] = Fitness::new(-1.0);
    assert!(matches!(
        neat.tell(&fitness),
        Err(Error::InvalidFitness { .. })
    ));
    // nothing changed: the same genomes can be told again
    assert_eq!(neat.ask().len(), asked);
    neat.tell(&vec![Fitness::new(1.0); asked]).unwrap();
}

#[test]
fn xor_is_solved_with_a_hidden_node() {
    let neat = Neat::builder(2, 1)
        .sharing(Sharing::Raw)
        .seed(3)
        .build()
        .unwrap();
    let outcome = Engine::new(neat, xor)
        .stop_when(Stop::target(15.0).or(Stop::generations(300)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    assert!(outcome.best_genome().hidden() >= 1);
}

#[test]
fn unconnected_networks_grow_connections() {
    let neat = Neat::builder(2, 1)
        .initial(Initial::Unconnected)
        .structural_mutation(0.03, 0.3)
        .seed(1)
        .build()
        .unwrap();
    assert!(
        neat.population()
            .iter()
            .all(|n| n.genome().connections().is_empty())
    );
    let outcome = Engine::new(neat, xor)
        .stop_when(Stop::generations(20))
        .run()
        .unwrap();
    assert!(outcome.best_genome().enabled() > 0);
}

#[cfg(feature = "serde")]
#[test]
fn a_resumed_run_matches_an_uninterrupted_one() {
    let build = || {
        Neat::builder(2, 1)
            .population_size(50)
            .seed(5)
            .build()
            .unwrap()
    };
    let uninterrupted = Engine::new(build(), xor)
        .stop_when(Stop::generations(30))
        .run()
        .unwrap();
    // 12 generations by hand, saved, loaded and run on
    let mut neat = build();
    for _ in 0..13 {
        let fitness: Vec<Fitness> = neat.ask().iter().map(|n| Fitness::new(xor(n))).collect();
        neat.tell(&fitness).unwrap();
    }
    let json = serde_json::to_string(&neat).unwrap();
    let resumed: Neat = serde_json::from_str(&json).unwrap();
    let resumed = Engine::new(resumed, xor)
        .stop_when(Stop::generations(30))
        .run()
        .unwrap();
    assert_eq!(resumed.best_genome(), uninterrupted.best_genome());
    assert_eq!(resumed.evaluations(), uninterrupted.evaluations());
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    // the population size stays exact, every network stays feed-forward and valid, the species
    // partition the population, and a seed gives the same run
    #[test]
    fn generations_keep_their_invariants(seed: u64, size in 10usize..60) {
        let run = |seed| {
            let mut neat = Neat::builder(3, 2)
                .population_size(size)
                .structural_mutation(0.2, 0.3)
                .seed(seed)
                .build()
                .unwrap();
            let mut trace = Vec::new();
            for _ in 0..15 {
                let fitness: Vec<Fitness> = neat
                    .ask()
                    .iter()
                    .map(|n| Fitness::new(n.enabled() as f64 - n.hidden() as f64))
                    .collect();
                neat.tell(&fitness).unwrap();
                trace.push(neat.population().iter().map(|i| i.genome().clone()).collect::<Vec<_>>());
            }
            (neat, trace)
        };
        let (neat, trace) = run(seed);
        prop_assert_eq!(&trace, &run(seed).1);
        prop_assert_eq!(neat.population().len(), size);
        for individual in neat.population() {
            let network = individual.genome();
            prop_assert!(network.feed_forward().is_ok());
            // innovations increasing, node ids increasing, every connection between nodes of
            // the network, into a hidden node or an output
            let innovations: Vec<u32> = network.connections().iter().map(|c| c.innovation()).collect();
            prop_assert!(innovations.windows(2).all(|w| w[0] < w[1]));
            let ids: Vec<u32> = network.nodes().iter().map(|n| n.id()).collect();
            prop_assert!(ids.windows(2).all(|w| w[0] < w[1]));
            for c in network.connections() {
                prop_assert!(ids.contains(&c.from()) && ids.contains(&c.to()));
                let to = network.nodes().iter().find(|n| n.id() == c.to()).unwrap();
                prop_assert!(matches!(to.kind(), NodeKind::Hidden | NodeKind::Output));
            }
        }
        let mut members: Vec<usize> = neat.species().iter().flat_map(|s| s.members().to_vec()).collect();
        members.sort_unstable();
        prop_assert_eq!(members, (0..size).collect::<Vec<_>>());
        // the same structure has the same innovation number in every network
        let mut seen = std::collections::BTreeMap::new();
        for individual in neat.population() {
            for c in individual.genome().connections() {
                let previous = seen.insert((c.from(), c.to()), c.innovation());
                prop_assert!(previous.is_none_or(|p| p == c.innovation()));
            }
        }
    }

    // the compatibility distance: 0 for a network and itself, symmetric
    #[test]
    fn the_distance_is_symmetric(seed: u64) {
        let mut neat = Neat::builder(2, 2).population_size(20).structural_mutation(0.3, 0.5).seed(seed).build().unwrap();
        for _ in 0..5 {
            let fitness: Vec<Fitness> = neat.ask().iter().map(|n| Fitness::new(xor_like(n))).collect();
            neat.tell(&fitness).unwrap();
        }
        let networks: Vec<Network> = neat.population().iter().map(|i| i.genome().clone()).collect();
        for a in &networks {
            prop_assert_eq!(neat.distance(a, a), 0.0);
            for b in &networks {
                prop_assert_eq!(neat.distance(a, b), neat.distance(b, a));
            }
        }
    }
}

fn xor_like(network: &Network) -> f64 {
    network.enabled() as f64
}

// the distance by hand: two networks of 3 genes, one weight differing by 1
#[test]
fn the_distance_by_hand() {
    let neat = Neat::builder(2, 1).seed(1).build().unwrap();
    let a = Network::fully_connected(2, 1, &[1.0, 1.0, 1.0]);
    let b = Network::fully_connected(2, 1, &[1.0, 2.0, 1.0]);
    // no excess or disjoint genes, N = 1 below 20 genes: c3 × 1/3
    assert!((neat.distance(&a, &b) - 0.4 / 3.0).abs() < 1e-15);
}

// networks of a run: `feed_forward` false allows cycles
fn evolved(seed: u64, feed_forward: bool) -> Vec<Network> {
    let mut neat = Neat::builder(3, 2)
        .population_size(30)
        .structural_mutation(0.2, 0.5)
        .feed_forward(feed_forward)
        .seed(seed)
        .build()
        .unwrap();
    for _ in 0..20 {
        let fitness: Vec<Fitness> = neat
            .ask()
            .iter()
            .map(|n| Fitness::new(n.enabled() as f64))
            .collect();
        neat.tell(&fitness).unwrap();
    }
    neat.population()
        .iter()
        .map(|i| i.genome().clone())
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(8))]

    // on an acyclic network, the recurrent evaluator reaches the feed-forward outputs, to the bit,
    // once a signal had time to cross every node, and repeats itself after a reset
    #[test]
    fn recurrent_evaluation_settles_on_feed_forward_networks(seed: u64) {
        for network in evolved(seed, true) {
            let mut feed_forward = network.feed_forward().unwrap();
            let mut recurrent = network.recurrent().unwrap();
            let input = [0.3, -0.7, 1.1];
            let (mut expected, mut output) = ([0.0; 2], [0.0; 2]);
            feed_forward.activate(&input, &mut expected);
            let mut first = Vec::new();
            for _ in 0..network.nodes().len() {
                recurrent.activate(&input, &mut output);
                first.push(output);
            }
            prop_assert_eq!(output.map(f64::to_bits), expected.map(f64::to_bits));
            recurrent.reset();
            for &step in &first {
                recurrent.activate(&input, &mut output);
                prop_assert_eq!(output.map(f64::to_bits), step.map(f64::to_bits));
            }
        }
    }
}

// a run allowing cycles makes some: feed-forward evaluation refuses them, recurrent evaluation
// takes them
#[test]
fn recurrent_runs_make_cycles() {
    let networks = evolved(2, false);
    let cyclic: Vec<&Network> = networks
        .iter()
        .filter(|n| n.feed_forward().is_err())
        .collect();
    assert!(!cyclic.is_empty());
    for network in cyclic {
        let mut recurrent = network.recurrent().unwrap();
        let mut output = [0.0; 2];
        recurrent.activate(&[1.0, 0.0, -1.0], &mut output);
        assert!(output.iter().all(|value| value.is_finite()));
    }
}
