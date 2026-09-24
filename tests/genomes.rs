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

#[test]
fn polynomial_mutation_solves_rastrigin() {
    let rastrigin = |x: &Reals| {
        10.0 * x.len() as f64
            + x.iter()
                .map(|xi| xi * xi - 10.0 * (std::f64::consts::TAU * xi).cos())
                .sum::<f64>()
    };
    let ga = Ga::builder(Real::uniform(10, -5.12..=5.12).unwrap())
        .population_size(100)
        .select(Tournament::new(3).unwrap())
        .crossover(UniformCrossover::new())
        .mutate(PolynomialMutation::per_gene(0.1, 20.0).unwrap())
        .scheme(Scheme::Generational { elitism: 2 })
        .minimize()
        .seed(0)
        .build()
        .unwrap();
    let outcome = Engine::new(ga, rastrigin)
        .stop_when(Stop::target(0.01).or(Stop::evaluations(400_000)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
}

#[test]
fn sbx_and_polynomial_mutation_solve_rastrigin() {
    // the NSGA-II pair of real-valued operators
    let rastrigin = |x: &Reals| {
        10.0 * x.len() as f64
            + x.iter()
                .map(|xi| xi * xi - 10.0 * (std::f64::consts::TAU * xi).cos())
                .sum::<f64>()
    };
    let ga = Ga::builder(Real::uniform(10, -5.12..=5.12).unwrap())
        .population_size(100)
        .select(Tournament::new(3).unwrap())
        .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
        .mutate(PolynomialMutation::per_gene(0.1, 20.0).unwrap())
        .scheme(Scheme::Generational { elitism: 2 })
        .minimize()
        .seed(0)
        .build()
        .unwrap();
    let outcome = Engine::new(ga, rastrigin)
        .stop_when(Stop::target(0.01).or(Stop::evaluations(400_000)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
}

#[test]
fn permutation_operators_solve_a_tour() {
    // 20 cities on a circle: the shortest tour visits them in circle order
    const CITIES: usize = 20;
    let position = |city: usize| {
        let angle = std::f64::consts::TAU * city as f64 / CITIES as f64;
        (angle.cos(), angle.sin())
    };
    let length = |tour: &Order| {
        (0..CITIES)
            .map(|i| {
                let (a, b) = (position(tour[i]), position(tour[(i + 1) % CITIES]));
                ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
            })
            .sum::<f64>()
    };
    let optimum = length(&Order::identity(CITIES));
    for seed in 0..3 {
        let ga = Ga::builder(Permutation::new(CITIES).unwrap())
            .population_size(50)
            .select(Tournament::new(3).unwrap())
            .crossover(EdgeRecombinationCrossover)
            .mutate(InversionMutation)
            .mutation_rate(0.3)
            .minimize()
            .seed(seed)
            .build()
            .unwrap();
        let outcome = Engine::new(ga, length)
            .stop_when(Stop::target(optimum + 1e-9).or(Stop::generations(1_000)))
            .run()
            .unwrap();
        assert_eq!(outcome.stop_reason(), StopReason::Target, "seed {seed}");
    }
}

#[test]
fn simulated_annealing_with_two_opt_solves_a_tour() {
    // 30 cities on a circle: the shortest tour visits them in circle order
    const CITIES: usize = 30;
    let position = |city: usize| {
        let angle = std::f64::consts::TAU * city as f64 / CITIES as f64;
        (angle.cos(), angle.sin())
    };
    let length = |tour: &Order| {
        (0..CITIES)
            .map(|i| {
                let (a, b) = (position(tour[i]), position(tour[(i + 1) % CITIES]));
                ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
            })
            .sum::<f64>()
    };
    let optimum = length(&Order::identity(CITIES));
    let search = LocalSearch::builder(Permutation::new(CITIES).unwrap())
        .neighbor(InversionMutation)
        .acceptance(Acceptance::Annealing {
            initial_temperature: 1.0,
            cooling: 0.999,
        })
        .minimize()
        .seed(0)
        .build()
        .unwrap();
    let mut hall_of_fame = HallOfFame::new(5).unwrap();
    let outcome = Engine::new(search, length)
        .stop_when(Stop::target(optimum + 1e-9).or(Stop::generations(50_000)))
        .observe(&mut hall_of_fame)
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    assert_eq!(
        hall_of_fame.best().unwrap().fitness(),
        outcome.best().fitness()
    );
}

#[test]
fn tabu_search_solves_n_queens() {
    let conflicts = |order: &Order| {
        let n = order.len();
        let mut count = 0;
        for i in 0..n {
            for j in i + 1..n {
                count += usize::from(order[i].abs_diff(order[j]) == j - i);
            }
        }
        count as f64
    };
    let search = LocalSearch::builder(Permutation::new(32).unwrap())
        .neighbor(SwapMutation::new())
        .neighbors(16)
        .acceptance(Acceptance::Tabu { tenure: 20 })
        .minimize()
        .seed(0)
        .build()
        .unwrap();
    let outcome = Engine::new(search, conflicts)
        .stop_when(Stop::target(0.0).or(Stop::generations(20_000)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
}

#[test]
fn iterated_local_search_solves_a_tour() {
    // 40 cities on a circle: the shortest tour visits them in circle order
    const CITIES: usize = 40;
    let position = |city: usize| {
        let angle = std::f64::consts::TAU * city as f64 / CITIES as f64;
        (angle.cos(), angle.sin())
    };
    let length = |tour: &Order| {
        (0..CITIES)
            .map(|i| {
                let (a, b) = (position(tour[i]), position(tour[(i + 1) % CITIES]));
                ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
            })
            .sum::<f64>()
    };
    let optimum = length(&Order::identity(CITIES));
    let search = LocalSearch::builder(Permutation::new(CITIES).unwrap())
        .neighbor(InversionMutation)
        .neighbors(8)
        .restart(100, 3)
        .minimize()
        .seed(0)
        .build()
        .unwrap();
    let outcome = Engine::new(search, length)
        .stop_when(Stop::target(optimum + 1e-9).or(Stop::generations(50_000)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
}

#[test]
fn memetic_ga_solves_a_tour() {
    // 40 cities on a circle: the shortest tour visits them in circle order
    const CITIES: usize = 40;
    let position = |city: usize| {
        let angle = std::f64::consts::TAU * city as f64 / CITIES as f64;
        (angle.cos(), angle.sin())
    };
    let length = |tour: &Order| {
        (0..CITIES)
            .map(|i| {
                let (a, b) = (position(tour[i]), position(tour[(i + 1) % CITIES]));
                ((a.0 - b.0).powi(2) + (a.1 - b.1).powi(2)).sqrt()
            })
            .sum::<f64>()
    };
    let optimum = length(&Order::identity(CITIES));
    let ga = Ga::builder(Permutation::new(CITIES).unwrap())
        .population_size(30)
        .select(Tournament::new(3).unwrap())
        .crossover(OrderCrossover)
        .mutate(InversionMutation)
        .mutation_rate(0.2)
        // the 2 best parents, which survive as elites, try 8 2-opt moves each, every generation
        .scheme(Scheme::Generational { elitism: 2 })
        .memetic(2, 8)
        .minimize()
        .seed(0)
        .build()
        .unwrap();
    let outcome = Engine::new(ga, length)
        .stop_when(Stop::target(optimum + 1e-9).or(Stop::generations(3_000)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
}

#[test]
fn feasibility_rules_solve_a_knapsack() {
    // 18 items; the optimum by brute force
    use rand::RngExt;
    let mut rng = StreamRng::seed_from_u64(5);
    let items: Vec<(f64, f64)> = (0..18)
        .map(|_| {
            (
                rng.random_range(1..30) as f64,
                rng.random_range(1..40) as f64,
            )
        })
        .collect();
    let capacity = items.iter().map(|item| item.0).sum::<f64>() / 3.0;
    let totals = |chosen: &dyn Fn(usize) -> bool| {
        (0..items.len())
            .filter(|&item| chosen(item))
            .fold((0.0, 0.0), |(weight, value), item| {
                (weight + items[item].0, value + items[item].1)
            })
    };
    let optimum = (0u32..1 << items.len())
        .map(|mask| totals(&|item| mask >> item & 1 == 1))
        .filter(|&(weight, _)| weight <= capacity)
        .map(|(_, value)| value)
        .fold(0.0, f64::max);
    let ga = Ga::builder(Binary::new(items.len()).unwrap())
        .population_size(60)
        .select(Tournament::new(3).unwrap())
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::per_gene(1.0 / 18.0).unwrap())
        .seed(0)
        .build()
        .unwrap();
    // (value, how much the weight exceeds the capacity)
    let fitness = |genome: &Bits| {
        let (weight, value) = totals(&|item| genome.get(item) == Some(true));
        (value, constraint::at_most(weight, capacity))
    };
    let outcome = Engine::new(ga, fitness)
        .stop_when(Stop::target(optimum).or(Stop::generations(2_000)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    assert!(outcome.best_fitness().is_feasible());
}

#[test]
fn self_adaptation_fine_tunes_where_a_fixed_step_cannot() {
    let sphere = |x: &[f64]| x.iter().map(|xi| xi * xi).sum::<f64>();
    // a (15,100)-ES with self-adaptation
    let es = Ga::builder(AdaptiveReal::new(Real::uniform(10, -5.0..=5.0).unwrap(), 0.3).unwrap())
        .population_size(15)
        .select(Tournament::new(2).unwrap())
        .crossover(NoCrossover)
        .mutate(SelfAdaptiveMutation::new())
        .scheme(Scheme::MuCommaLambda { lambda: 100 })
        .minimize()
        .seed(0)
        .build()
        .unwrap();
    let outcome = Engine::new(es, |genome: &AdaptiveReals| sphere(genome))
        .stop_when(Stop::target(1e-8).or(Stop::evaluations(50_000)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Target);
    // the step size shrank by itself, from 0.3
    assert!(outcome.best_genome().step() < 1e-4);

    // the same with a fixed step: stuck far from 1e-8
    let fixed = Ga::builder(Real::uniform(10, -5.0..=5.0).unwrap())
        .population_size(15)
        .select(Tournament::new(2).unwrap())
        .crossover(NoCrossover)
        .mutate(GaussianMutation::per_gene(1.0, 0.03).unwrap())
        .scheme(Scheme::MuCommaLambda { lambda: 100 })
        .minimize()
        .seed(0)
        .build()
        .unwrap();
    let outcome = Engine::new(fixed, |genome: &Reals| sphere(genome))
        .stop_when(Stop::target(1e-8).or(Stop::evaluations(50_000)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Evaluations);
}

#[test]
fn bits_display_takes_a_width_and_swap_uniform_clamps_its_rate() {
    use genoxide::genome::SwapGenes;
    let bits: Bits = [true, true].into_iter().collect();
    assert_eq!(format!("{bits:>5}"), "   11");
    assert_eq!(format!("{bits:<5}|"), "11   |");
    assert_eq!(format!("{bits}"), "11");
    let mut rng = StreamRng::seed_from_u64(0);
    let (mut a, mut b) = (Bits::zeros(10), Bits::ones(10));
    a.swap_uniform(&mut b, 1.5, &mut rng);
    assert_eq!(a, Bits::ones(10));
    a.swap_uniform(&mut b, f64::NAN, &mut rng);
    assert_eq!(a, Bits::ones(10));
}
