use genoxide::prelude::*;

/// Fixed values: these must never change for the same major version, on any platform.
#[test]
fn portable_random_genomes() {
    let mut rng = StreamRng::seed_from_u64(42);
    let integers = Integer::new([0..=9, -5..=5, i64::MIN..=i64::MAX])
        .unwrap()
        .random_genome(&mut rng);
    let reals = Real::uniform(3, -1.0..=1.0)
        .unwrap()
        .random_genome(&mut rng);
    let order = Permutation::new(6).unwrap().random_genome(&mut rng);
    assert_eq!(integers.into_vec(), [6, 5, -1337086366047644788]);
    assert_eq!(
        reals.into_vec(),
        [
            0.25472104239468063,
            -0.42281224171763476,
            -0.7000822594193501
        ]
    );
    assert_eq!(order.into_vec(), [2, 5, 0, 3, 4, 1]);
}

#[test]
fn integer_reaches_the_bounds() {
    // maximize the sum: every gene at its upper bound
    let integer = Integer::new((0..20).map(|gene| -gene..=gene)).unwrap();
    let ga = Ga::builder(integer)
        .population_size(50)
        .select(Tournament::new(3).unwrap())
        .crossover(UniformCrossover::new())
        .mutate(UniformMutation::per_gene(0.05).unwrap())
        .seed(0)
        .build()
        .unwrap();
    let outcome = Engine::new(ga, |genome: &Integers| genome.iter().sum::<i64>() as f64)
        .stop_when(Stop::target(190.0).or(Stop::generations(2_000)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    assert_eq!(
        outcome.best_genome().to_vec(),
        (0..20).collect::<Vec<i64>>()
    );
}

#[test]
fn real_minimizes_the_sphere() {
    let ga = Ga::builder(Real::uniform(5, -5.0..=5.0).unwrap())
        .population_size(100)
        .select(Tournament::new(3).unwrap())
        .crossover(PointCrossover::one_point())
        .mutate(UniformMutation::count(1).unwrap())
        .minimize()
        .seed(0)
        .build()
        .unwrap();
    let outcome = Engine::new(ga, |genome: &Reals| {
        genome.iter().map(|x| x * x).sum::<f64>()
    })
    .stop_when(Stop::target(0.1).or(Stop::generations(2_000)))
    .run()
    .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
}

#[test]
fn permutation_sorts() {
    // minimize the number of inversions: the identity has none
    let inversions = |order: &Order| {
        let mut count = 0;
        for i in 0..order.len() {
            for j in i + 1..order.len() {
                count += usize::from(order[i] > order[j]);
            }
        }
        count as f64
    };
    let ga = Ga::builder(Permutation::new(12).unwrap())
        .population_size(20)
        .select(Tournament::new(2).unwrap())
        .crossover(NoCrossover)
        .mutate(SwapMutation::new())
        .scheme(Scheme::MuPlusLambda { lambda: 20 })
        .minimize()
        .seed(0)
        .build()
        .unwrap();
    let outcome = Engine::new(ga, inversions)
        .stop_when(Stop::target(0.0).or(Stop::generations(2_000)))
        .run()
        .unwrap();
    assert_eq!(outcome.best_genome(), &Order::identity(12));
}
