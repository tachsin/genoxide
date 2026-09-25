//! Edge cases of the engines and the builders: runs that could never end, runs continued after
//! their budget, and sizes too large to allocate.

use genoxide::Objective::Minimize;
use genoxide::engine::STALL_GENERATIONS;
use genoxide::prelude::*;
use std::cell::Cell;
use std::sync::mpsc;
use std::thread;
use std::time::Duration;

fn one_max(genome: &Bits) -> f64 {
    genome.count_ones() as f64
}

// runs `run` on another thread, and fails if it doesn't return within a minute
fn within_a_minute<T: Send + 'static>(run: impl FnOnce() -> T + Send + 'static) -> T {
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let _ = sender.send(run());
    });
    receiver
        .recv_timeout(Duration::from_secs(60))
        .expect("the run never returned")
}

// a GA whose children are all copies of their parents: a permutation of 1 gene
fn single_genome_ga() -> Ga<Permutation, Tournament, OrderCrossover, SwapMutation> {
    Ga::builder(Permutation::new(1).unwrap())
        .population_size(4)
        .select(Tournament::new(2).unwrap())
        .crossover(OrderCrossover)
        .mutate(SwapMutation::new())
        .seed(0)
        .build()
        .unwrap()
}

// a GA without mutation, whose population converges to copies of one genome
fn converging_ga() -> Ga<Binary, Tournament, UniformCrossover, BitFlip> {
    Ga::builder(Binary::new(16).unwrap())
        .population_size(20)
        .select(Tournament::new(3).unwrap())
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::count(1).unwrap())
        .mutation_rate(0.0)
        .seed(1)
        .build()
        .unwrap()
}

#[test]
fn an_evaluation_limit_that_copies_never_reach_stops_the_run_as_stalled() {
    let outcome = within_a_minute(|| {
        Engine::new(single_genome_ga(), |_: &Order| 1.0)
            .stop_when(Stop::evaluations(100))
            .run()
            .unwrap()
    });
    assert_eq!(outcome.stop_reason(), StopReason::Stalled);
    assert_eq!(outcome.evaluations(), 4);
    assert_eq!(outcome.generations(), STALL_GENERATIONS);
}

#[test]
fn an_unreachable_target_with_copies_stops_the_run_as_stalled() {
    let outcome = within_a_minute(|| {
        Engine::new(single_genome_ga(), |_: &Order| 1.0)
            .stop_when(Stop::target(2.0))
            .run()
            .unwrap()
    });
    assert_eq!(outcome.stop_reason(), StopReason::Stalled);
}

#[test]
fn a_converged_ga_without_mutation_stops_as_stalled() {
    let outcome = within_a_minute(|| {
        Engine::new(converging_ga(), one_max)
            .stop_when(Stop::evaluations(2_000))
            .run()
            .unwrap()
    });
    assert_eq!(outcome.stop_reason(), StopReason::Stalled);
    assert!(outcome.evaluations() < 2_000);
}

#[test]
fn a_stop_condition_that_copies_can_meet_still_ends_the_run() {
    // generations, stagnation, time and custom conditions end the run as before: stalling
    // doesn't change runs that ended
    let outcome = Engine::new(converging_ga(), one_max)
        .stop_when(Stop::evaluations(2_000).or(Stop::generations(3 * STALL_GENERATIONS)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Generations);
    assert_eq!(outcome.generations(), 3 * STALL_GENERATIONS);
    let outcome = Engine::new(single_genome_ga(), |_: &Order| 1.0)
        .stop_when(Stop::target(2.0).or(Stop::stagnation(2 * STALL_GENERATIONS)))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Stagnation);
}

#[test]
fn a_stalled_run_returns_at_once_when_run_again() {
    let mut observed = 0;
    let mut engine = Engine::new(single_genome_ga(), |_: &Order| 1.0)
        .stop_when(Stop::evaluations(100))
        .on_generation(|_| observed += 1);
    let first = engine.run().unwrap();
    let again = engine.run().unwrap();
    drop(engine);
    assert_eq!(again.stop_reason(), StopReason::Stalled);
    assert_eq!(again.generations(), first.generations());
    assert_eq!(observed, STALL_GENERATIONS + 1);
}

#[test]
fn a_multi_objective_run_of_copies_stops_as_stalled() {
    let nsga2 = Nsga2::builder(Permutation::new(1).unwrap(), [Minimize, Minimize])
        .population_size(4)
        .crossover(OrderCrossover)
        .mutate(SwapMutation::new())
        .eliminate_duplicates(false)
        .seed(0)
        .build()
        .unwrap();
    let outcome = within_a_minute(|| {
        MultiEngine::new(nsga2, |_: &Order| [0.0, 1.0])
            .stop_when(Stop::evaluations(100))
            .run()
            .unwrap()
    });
    assert_eq!(outcome.stop_reason(), StopReason::Stalled);
    assert_eq!(outcome.evaluations(), 4);
    assert_eq!(outcome.generations(), STALL_GENERATIONS);
}

fn setting<T: std::fmt::Debug>(result: Result<T>) -> &'static str {
    match result {
        Err(Error::InvalidSetting { setting, .. }) => setting,
        other => panic!("expected a setting error, got {other:?}"),
    }
}

#[test]
fn no_crossover_without_mutation_is_a_setting_error() {
    let builder = || {
        Ga::builder(Binary::new(8).unwrap())
            .population_size(10)
            .select(Tournament::new(2).unwrap())
            .crossover(NoCrossover)
            .mutate(BitFlip::count(1).unwrap())
            .mutation_rate(0.0)
    };
    assert_eq!(setting(builder().build()), "mutation_rate");
    assert_eq!(setting(builder().build_steady()), "mutation_rate");
    assert!(builder().mutation_rate(0.1).build().is_ok());
    let nsga2 = Nsga2::builder(Binary::new(8).unwrap(), [Minimize, Minimize])
        .population_size(10)
        .crossover(NoCrossover)
        .mutate(BitFlip::count(1).unwrap())
        .mutation_rate(0.0)
        .build();
    assert_eq!(setting(nsga2), "mutation_rate");
}

// an asynchronous run whose budget is smaller than the population returns at once when run
// again, like one whose budget is larger
#[test]
fn an_asynchronous_run_with_its_budget_spent_before_a_generation_does_nothing_again() {
    let steady = Ga::builder(Binary::new(16).unwrap())
        .population_size(10)
        .select(Tournament::new(2).unwrap())
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::count(1).unwrap())
        .seed(4)
        .build_steady()
        .unwrap();
    let mut statistics = Statistics::new();
    let saves = Cell::new(0);
    let mut engine = AsyncEngine::new(steady, one_max)
        .workers(1)
        .stop_when(Stop::evaluations(5))
        .observe(&mut statistics)
        .checkpoint_every(1, |_| {
            saves.set(saves.get() + 1);
            Ok(())
        });
    let first = engine.run().unwrap();
    let again = engine.run().unwrap();
    drop(engine);
    assert_eq!((first.evaluations(), again.evaluations()), (5, 5));
    assert_eq!(again.stop_reason(), StopReason::Evaluations);
    assert_eq!(saves.get(), 1, "the checkpoint closure was called again");
    assert_eq!(
        statistics.records().len(),
        1,
        "the observers were notified again"
    );
}

// the largest size the builders accept
const MAX: usize = 1 << 24;

#[test]
fn huge_sizes_are_setting_errors() {
    for size in [MAX + 1, usize::MAX] {
        let ga = || {
            Ga::builder(Binary::new(8).unwrap())
                .population_size(size)
                .select(Tournament::new(2).unwrap())
                .crossover(UniformCrossover::new())
                .mutate(BitFlip::count(1).unwrap())
        };
        assert_eq!(setting(ga().build()), "population_size");
        assert_eq!(setting(ga().build_steady()), "population_size");
        let memetic = ga().population_size(10).memetic(1, size).build();
        assert_eq!(setting(memetic), "memetic");

        assert_eq!(setting(Tournament::new(size)), "tournament_size");

        let real = || Real::uniform(2, -1.0..=1.0).unwrap();
        let es = Es::builder(real()).parents(size).offspring(size).build();
        assert_eq!(setting(es), "parents");
        let es = Es::builder(real()).parents(2).offspring(size).build();
        assert_eq!(setting(es), "offspring");
        let de = De::builder(real()).population_size(size).build();
        assert_eq!(setting(de), "population_size");
        let de = De::builder(real())
            .control(de::Control::Shade { memory: size })
            .build();
        assert_eq!(setting(de), "memory");
        let pso = Pso::builder(real()).population_size(size).build();
        assert_eq!(setting(pso), "population_size");
        let cmaes = Cmaes::builder(real()).population_size(size).build();
        assert_eq!(setting(cmaes), "population_size");

        let search = || {
            LocalSearch::builder(Permutation::new(8).unwrap())
                .neighbor(SwapMutation::new())
                .seed(0)
        };
        assert_eq!(setting(search().neighbors(size).build()), "neighbors");
        assert_eq!(setting(search().restart(10, size).build()), "restart");

        let objectives = [Minimize, Minimize];
        let nsga2 = Nsga2::builder(real(), objectives)
            .population_size(size)
            .crossover(UniformCrossover::new())
            .mutate(UniformMutation::count(1).unwrap())
            .build();
        assert_eq!(setting(nsga2), "population_size");
        let nsga3 = Nsga3::builder(real(), objectives, vec![[1.0, 0.0], [0.0, 1.0]])
            .population_size(size)
            .crossover(UniformCrossover::new())
            .mutate(UniformMutation::count(1).unwrap())
            .build();
        assert_eq!(setting(nsga3), "population_size");
        let spea2 = Spea2::builder(real(), objectives)
            .population_size(size)
            .crossover(UniformCrossover::new())
            .mutate(UniformMutation::count(1).unwrap())
            .build();
        assert_eq!(setting(spea2), "population_size");
        let sms_emoa = || {
            SmsEmoa::builder(real(), objectives)
                .crossover(UniformCrossover::new())
                .mutate(UniformMutation::count(1).unwrap())
        };
        assert_eq!(
            setting(sms_emoa().population_size(size).build()),
            "population_size"
        );
        assert_eq!(
            setting(sms_emoa().population_size(10).offspring(size).build()),
            "offspring"
        );
    }
    assert!(Tournament::new(MAX).is_ok());
}

#[test]
fn a_huge_hall_of_fame_needs_no_memory_ahead() {
    let hall_of_fame = HallOfFame::<Bits>::new(usize::MAX).unwrap();
    assert!(hall_of_fame.individuals().is_empty());
}

// the default population of a CMA-ES counts the genes that can take more than one value
#[test]
fn the_default_cma_es_population_counts_the_genes_that_can_change() {
    let real = Real::new((0..10).map(|gene| if gene < 2 { 0.0..=1.0 } else { 1.0..=1.0 })).unwrap();
    let mut cmaes = Cmaes::builder(real).seed(0).build().unwrap();
    assert_eq!(cmaes.ask().len(), Cmaes::default_population_size(2));
}
