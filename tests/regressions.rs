//! Regression tests from the review before 0.7: differential evolution's restarts with observers,
//! constraints and islands, its defaults on edge cases, and stochastic universal sampling.

use genoxide::prelude::*;

fn sphere(x: &Reals) -> f64 {
    x.iter().map(|xi| xi * xi).sum()
}

fn de_builder(seed: u64) -> genoxide::algorithm::de::DeBuilder {
    De::builder(Real::uniform(8, -5.0..=5.0).unwrap())
        .population_size(20)
        .minimize()
        .seed(seed)
}

// after every generation, restarts included, the population is evaluated, and every individual
// evaluated in it is in the population or discarded
#[test]
fn de_restarts_keep_every_evaluated_individual_visible() {
    let mut de = de_builder(3)
        .restarts(de::Restarts::OnStagnation {
            tolerance: 1.0,
            patience: 1_000,
        })
        .build()
        .unwrap();
    for _ in 0..1_000 {
        let asked: Vec<Reals> = de.ask().iter().cloned().collect();
        let fitness: Vec<Fitness> = asked.iter().map(|x| Fitness::new(sphere(x))).collect();
        de.tell(&fitness).unwrap();
        assert!(de.population().iter().all(|x| x.is_evaluated()));
        let lost = asked
            .iter()
            .filter(|genome| {
                !de.population()
                    .iter()
                    .chain(de.discarded())
                    .any(|x| x.genome() == *genome)
            })
            .count();
        assert_eq!(
            lost, 0,
            "evaluated individuals in neither the population nor discarded"
        );
    }
    assert!(de.restart_count() > 1);
}

// a population that differs only in its constraint violations hasn't converged
#[test]
fn de_constrained_population_is_not_converged_by_equal_scores() {
    let mut de = de_builder(1).build().unwrap();
    // a constraint satisfaction problem: a constant score, and the violation to reduce
    let fitness = |x: &Reals| Fitness::constrained(0.0, sphere(x));
    for _ in 0..10 {
        let told: Vec<Fitness> = de.ask().iter().map(fitness).collect();
        de.tell(&told).unwrap();
    }
    let violations: Vec<f64> = de
        .population()
        .iter()
        .filter_map(|x| x.fitness())
        .map(|f| f.violation())
        .collect();
    assert!(violations.iter().any(|&v| v > 1.0), "{violations:?}");
    assert_eq!(de.restart_count(), 0, "restarted every other generation");
}

// migrants that arrive while a restart's new individuals wait for their evaluation are evaluated
// already, and aren't evaluated again
#[test]
fn de_migrants_during_a_restart_are_not_evaluated_again() {
    let mut de = de_builder(3)
        .restarts(de::Restarts::OnStagnation {
            tolerance: 1.0,
            patience: 1_000,
        })
        .build()
        .unwrap();
    let step = |de: &mut De| {
        let told: Vec<Fitness> = de.ask().iter().map(|x| Fitness::new(sphere(x))).collect();
        de.tell(&told).unwrap();
    };
    // generations until one ends converged: its restart is at the next ask
    let restarts = de.restart_count();
    loop {
        step(&mut de);
        let asked_before = de.restart_count();
        // look ahead without asking: a clone restarts at its ask if one is due
        let mut probe = de.clone();
        probe.ask();
        if probe.restart_count() > asked_before {
            break;
        }
    }
    let genome = Reals::from(vec![0.25; 8]);
    let mut migrant = Individual::new(genome.clone());
    migrant.set_fitness(Fitness::new(sphere(&genome)));
    de.immigrate(vec![migrant]).unwrap();
    let asked: Vec<Reals> = de.ask().iter().cloned().collect();
    assert!(de.restart_count() > restarts);
    assert!(
        !asked.contains(&genome),
        "the evaluated migrant is asked for again ({} asked)",
        asked.len()
    );
    // the migrant is kept through the restart
    assert!(de.population().iter().any(|x| x.genome() == &genome));
}

// smoke: the defaults on edge cases run without panics and repeat with a seed
#[test]
fn de_defaults_edge_cases_smoke() {
    let reals = [
        Real::uniform(1, -1.0..=1.0).unwrap(),
        Real::uniform(3, 2.0..=2.0).unwrap(),
        Real::new([0.0..=0.0, -1.0..=1.0]).unwrap(),
        Real::uniform(2, -8e307..=8e307).unwrap(),
    ];
    for real in reals {
        for objective in [Objective::Minimize, Objective::Maximize] {
            for kind in 0..4 {
                let run = || {
                    let de = De::builder(real.clone())
                        .objective(objective)
                        .seed(5)
                        .build()
                        .unwrap();
                    let f = move |x: &Reals| -> Fitness {
                        let s: f64 = x.iter().map(|v| v * v).sum();
                        match kind {
                            0 => Fitness::new(s),
                            1 => Fitness::invalid(),
                            2 => Fitness::constrained(s, x[0].abs()),
                            _ => Fitness::new(if x[0] > 0.0 { f64::INFINITY } else { -s }),
                        }
                    };
                    let outcome = Engine::new(de, f)
                        .stop_when(Stop::generations(300))
                        .run()
                        .unwrap();
                    (outcome.evaluations(), outcome.best().genome().clone())
                };
                assert_eq!(run(), run());
            }
        }
    }
}

// SUS: each individual gets floor or ceil of its expected count, and the mean is unbiased
#[test]
fn sus_counts_are_floor_or_ceil_and_unbiased() {
    for (scores, objective) in [
        (
            vec![-3.0, 0.5, 7.0, 7.0, -1.0, 2.25, 100.0],
            Objective::Maximize,
        ),
        (
            vec![-3.0, 0.5, 7.0, 7.0, -1.0, 2.25, 100.0],
            Objective::Minimize,
        ),
        (vec![1.0, 2.0], Objective::Maximize),
    ] {
        let mut population: Population<Reals> =
            Population::from_genomes(scores.iter().map(|&s| Reals::from(vec![s])));
        for (i, &s) in scores.iter().enumerate() {
            population[i].set_fitness(Fitness::new(s));
        }
        let worst = match objective {
            Objective::Maximize => scores.iter().cloned().fold(f64::INFINITY, f64::min),
            Objective::Minimize => scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        };
        let weights: Vec<f64> = scores.iter().map(|s| (s - worst).abs()).collect();
        let total: f64 = weights.iter().sum();
        let mut rng = StreamRng::seed_from_u64(0);
        for count in [1usize, 2, 5, 13] {
            let mut sums = vec![0usize; scores.len()];
            let draws = 20_000;
            for _ in 0..draws {
                let picks =
                    StochasticUniversalSampling.select(&population, objective, count, &mut rng);
                assert_eq!(picks.len(), count);
                let mut counts = vec![0usize; scores.len()];
                for p in picks {
                    counts[p] += 1;
                }
                for (i, &c) in counts.iter().enumerate() {
                    let expected = count as f64 * weights[i] / total;
                    assert!(
                        c as f64 >= expected.floor() && c as f64 <= expected.ceil(),
                        "{i}: {c} vs {expected}"
                    );
                    sums[i] += c;
                }
            }
            for (i, &sum) in sums.iter().enumerate() {
                let expected = count as f64 * weights[i] / total;
                let mean = sum as f64 / draws as f64;
                assert!(
                    (mean - expected).abs() < 0.02 * count as f64 + 0.01,
                    "{i}: {mean} vs {expected}"
                );
            }
        }
    }
}

// islands of restarting DEs: after every generation, every island's population is evaluated, and
// the evaluations are what was asked
#[test]
fn de_islands_with_restarts_keep_evaluated_populations() {
    let islands: Vec<De> = (0..2)
        .map(|seed| {
            De::builder(Real::uniform(4, -5.0..=5.0).unwrap())
                .minimize()
                .restarts(de::Restarts::OnStagnation {
                    tolerance: 1e-2,
                    patience: 1_000,
                })
                .seed(seed)
                .build()
                .unwrap()
        })
        .collect();
    let mut islands = Islands::builder(islands)
        .interval(1)
        .migrants(2)
        .seed(0)
        .build()
        .unwrap();
    let mut asked = 0;
    for _ in 0..2_000 {
        let told: Vec<Fitness> = islands
            .ask()
            .iter()
            .map(|x| Fitness::new(sphere(x)))
            .collect();
        asked += told.len() as u64;
        islands.tell(&told).unwrap();
        for island in islands.islands() {
            assert!(island.population().iter().all(|x| x.is_evaluated()));
        }
    }
    let restarts: u64 = islands.islands().iter().map(|i| i.restart_count()).sum();
    assert!(restarts > 0);
    assert_eq!(islands.evaluations(), asked);
}
