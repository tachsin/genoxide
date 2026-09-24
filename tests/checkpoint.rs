//! Checkpoints: a resumed run is exactly the run without the interruption.
#![cfg(feature = "serde")]

use genoxide::Objective::Minimize;
use genoxide::algorithm::islands::Topology;
use genoxide::checkpoint;
use genoxide::multi::problems::{Dtlz2, TestProblem, Zdt1};
use genoxide::multi::{Decomposition, MultiObjectiveAlgorithm, SmsEmoa, das_dennis};
use genoxide::prelude::*;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::f64::consts::TAU;

fn rastrigin(x: &Reals) -> f64 {
    10.0 * x.len() as f64
        + x.iter()
            .map(|xi| xi * xi - 10.0 * (TAU * xi).cos())
            .sum::<f64>()
}

fn one_max(genome: &Bits) -> f64 {
    genome.count_ones() as f64
}

fn bytes<A: Serialize>(algorithm: &A) -> Vec<u8> {
    let mut bytes = Vec::new();
    checkpoint::save(algorithm, &mut bytes).unwrap();
    bytes
}

// runs `split` generations, saves, loads and runs to `total`; the same state as `total` at once
fn resumes<A, F>(make: impl Fn() -> A, fitness: F, split: u64, total: u64, exact_bytes: bool)
where
    A: Algorithm + Serialize + DeserializeOwned,
    F: FitnessFunction<A::Genome> + Copy,
{
    let mut whole = Engine::new(make(), fitness).stop_when(Stop::generations(total));
    let expected = whole.run().unwrap();

    let mut first = Engine::new(make(), fitness).stop_when(Stop::generations(split));
    first.run().unwrap();
    let resumed: A = checkpoint::load(bytes(first.algorithm()).as_slice()).unwrap();
    assert_eq!(resumed.generation(), split);
    let mut second = Engine::new(resumed, fitness).stop_when(Stop::generations(total));
    let outcome = second.run().unwrap();

    assert_eq!(outcome.best(), expected.best());
    assert_eq!(outcome.generations(), expected.generations());
    assert_eq!(outcome.evaluations(), expected.evaluations());
    assert_eq!(
        second.algorithm().population(),
        whole.algorithm().population()
    );
    if exact_bytes {
        assert_eq!(bytes(second.algorithm()), bytes(whole.algorithm()));
    }
}

fn resumes_multi<A, F, const M: usize>(make: impl Fn() -> A, fitness: F, split: u64, total: u64)
where
    A: MultiObjectiveAlgorithm<M> + Serialize + DeserializeOwned,
    F: genoxide::multi::MultiFitnessFunction<A::Genome, M> + Copy,
{
    let mut whole = MultiEngine::new(make(), fitness).stop_when(Stop::generations(total));
    let expected = whole.run().unwrap();

    let mut first = MultiEngine::new(make(), fitness).stop_when(Stop::generations(split));
    first.run().unwrap();
    let resumed: A = checkpoint::load(bytes(first.algorithm()).as_slice()).unwrap();
    let mut second = MultiEngine::new(resumed, fitness).stop_when(Stop::generations(total));
    let outcome = second.run().unwrap();

    assert_eq!(outcome.front(), expected.front());
    assert_eq!(outcome.evaluations(), expected.evaluations());
    assert_eq!(bytes(second.algorithm()), bytes(whole.algorithm()));
}

type OneMaxGa = Ga<Binary, Tournament, UniformCrossover, BitFlip>;

fn one_max_ga(seed: u64) -> OneMaxGa {
    Ga::builder(Binary::new(64).unwrap())
        .population_size(30)
        .select(Tournament::new(3).unwrap())
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::per_gene(1.0 / 64.0).unwrap())
        .seed(seed)
        .build()
        .unwrap()
}

fn real_ga(
    scheme: Scheme,
    seed: u64,
) -> Ga<Real, Tournament, SimulatedBinaryCrossover, PolynomialMutation> {
    Ga::builder(Real::uniform(10, -5.12..=5.12).unwrap())
        .population_size(20)
        .select(Tournament::new(3).unwrap())
        .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
        .mutate(PolynomialMutation::per_gene(0.1, 20.0).unwrap())
        .scheme(scheme)
        .minimize()
        .seed(seed)
        .build()
        .unwrap()
}

fn de(seed: u64) -> De {
    // L-SHADE: an archive, adaptive parameters and a shrinking population
    De::l_shade(Real::uniform(10, -5.12..=5.12).unwrap(), 3_000)
        .minimize()
        .seed(seed)
        .build()
        .unwrap()
}

#[test]
fn genetic_algorithms_resume_exactly() {
    resumes(|| one_max_ga(1), one_max, 7, 20, true);
    for scheme in [
        Scheme::Generational { elitism: 1 },
        Scheme::SteadyState { replacements: 5 },
        Scheme::MuPlusLambda { lambda: 20 },
        Scheme::MuCommaLambda { lambda: 40 },
    ] {
        resumes(|| real_ga(scheme, 2), rastrigin, 9, 25, true);
    }
    // memetic, on permutations
    let tour = |order: &Order| {
        (0..order.len())
            .map(|i| order[i].abs_diff(order[(i + 1) % order.len()]) as f64)
            .sum::<f64>()
    };
    let memetic = || {
        Ga::builder(Permutation::new(20).unwrap())
            .population_size(20)
            .select(Tournament::new(3).unwrap())
            .crossover(OrderCrossover)
            .mutate(InversionMutation)
            .scheme(Scheme::Generational { elitism: 2 })
            .memetic(2, 4)
            .minimize()
            .seed(3)
            .build()
            .unwrap()
    };
    resumes(memetic, tour, 5, 15, true);
}

#[test]
fn other_algorithms_resume_exactly() {
    resumes(|| de(4), rastrigin, 10, 40, true);
    let pso = || {
        Pso::builder(Real::uniform(10, -5.12..=5.12).unwrap())
            .population_size(20)
            .topology(pso::Topology::Ring { neighbors: 1 })
            .minimize()
            .seed(5)
            .build()
            .unwrap()
    };
    resumes(pso, rastrigin, 10, 30, true);
    let es = || {
        Es::builder(Real::uniform(10, -5.12..=5.12).unwrap())
            .parents(5)
            .offspring(35)
            .recombination(es::Recombination::Dominant { rho: 2 })
            .minimize()
            .seed(6)
            .build()
            .unwrap()
    };
    resumes(es, rastrigin, 10, 30, true);
    // across a BIPOP restart
    let cmaes = || {
        Cmaes::builder(Real::uniform(10, -5.12..=5.12).unwrap())
            .restarts(cmaes::Restarts::Bipop)
            .minimize()
            .seed(7)
            .build()
            .unwrap()
    };
    resumes(cmaes, rastrigin, 150, 400, true);
    let diagonal = || {
        Cmaes::builder(Real::uniform(10, -5.12..=5.12).unwrap())
            .covariance(cmaes::Covariance::Diagonal)
            .minimize()
            .seed(8)
            .build()
            .unwrap()
    };
    resumes(diagonal, rastrigin, 20, 60, true);
}

#[test]
fn local_search_resumes_exactly() {
    let conflicts = |order: &Order| {
        let mut count = 0;
        for i in 0..order.len() {
            for j in i + 1..order.len() {
                count += usize::from(order[i].abs_diff(order[j]) == j - i);
            }
        }
        count as f64
    };
    // the tabu set is a hash set, whose order differs between runs: compare the populations
    let tabu = || {
        LocalSearch::builder(Permutation::new(16).unwrap())
            .neighbor(SwapMutation::new())
            .neighbors(8)
            .acceptance(Acceptance::Tabu { tenure: 10 })
            .minimize()
            .seed(9)
            .build()
            .unwrap()
    };
    resumes(tabu, conflicts, 20, 60, false);
    let annealing = || {
        LocalSearch::builder(Binary::new(64).unwrap())
            .neighbor(BitFlip::count(1).unwrap())
            .acceptance(Acceptance::Annealing {
                initial_temperature: 2.0,
                cooling: 0.99,
            })
            .restart(50, 3)
            .seed(10)
            .build()
            .unwrap()
    };
    resumes(annealing, one_max, 70, 200, true);
}

#[test]
fn islands_resume_exactly() {
    for topology in [Topology::Ring, Topology::Random] {
        let islands = || {
            Islands::builder(
                (0..3)
                    .map(|seed| real_ga(Scheme::default(), seed))
                    .collect(),
            )
            .topology(topology)
            .interval(4)
            .seed(11)
            .build()
            .unwrap()
        };
        // resumed between migrations and at one
        resumes(islands, rastrigin, 6, 20, true);
        resumes(islands, rastrigin, 8, 20, true);
    }
    let islands = || {
        Islands::builder((0..3).map(de).collect())
            .interval(5)
            .seed(12)
            .build()
            .unwrap()
    };
    resumes(islands, rastrigin, 7, 20, true);
}

#[test]
fn multi_objective_algorithms_resume_exactly() {
    let zdt1 = Zdt1::new(10);
    let dtlz2 = Dtlz2::<3>::default();
    let sbx = || SimulatedBinaryCrossover::new(15.0).unwrap();
    let mutation = |n: usize| PolynomialMutation::per_gene(1.0 / n as f64, 20.0).unwrap();
    let nsga2 = || {
        Nsga2::builder(zdt1.real(), [Minimize; 2])
            .population_size(20)
            .crossover(sbx())
            .mutate(mutation(10))
            .seed(12)
            .build()
            .unwrap()
    };
    resumes_multi(nsga2, zdt1, 8, 20);
    let nsga3 = || {
        Nsga3::builder(dtlz2.real(), [Minimize; 3], das_dennis::<3>(4))
            .population_size(16)
            .crossover(sbx())
            .mutate(mutation(12))
            .seed(13)
            .build()
            .unwrap()
    };
    resumes_multi(nsga3, dtlz2, 8, 20);
    let spea2 = || {
        Spea2::builder(zdt1.real(), [Minimize; 2])
            .population_size(20)
            .crossover(sbx())
            .mutate(mutation(10))
            .seed(14)
            .build()
            .unwrap()
    };
    resumes_multi(spea2, zdt1, 8, 20);
    let moead = || {
        Moead::builder(dtlz2.real(), [Minimize; 3], das_dennis::<3>(4))
            .decomposition(Decomposition::Pbi { theta: 5.0 })
            .crossover(sbx())
            .mutate(mutation(12))
            .seed(15)
            .build()
            .unwrap()
    };
    resumes_multi(moead, dtlz2, 8, 20);
    let sms_emoa = || {
        SmsEmoa::builder(zdt1.real(), [Minimize; 2])
            .population_size(20)
            .crossover(sbx())
            .mutate(mutation(10))
            .seed(16)
            .build()
            .unwrap()
    };
    resumes_multi(sms_emoa, zdt1, 30, 80);
}

#[test]
fn checkpoint_every_saves_on_schedule_and_at_the_end() {
    let mut saved: Vec<(u64, Vec<u8>)> = Vec::new();
    let expected = Engine::new(one_max_ga(17), one_max)
        .stop_when(Stop::generations(25))
        .checkpoint_every(10, |ga| {
            saved.push((ga.generation(), bytes(ga)));
            Ok(())
        })
        .run()
        .unwrap();
    let generations: Vec<u64> = saved.iter().map(|(generation, _)| *generation).collect();
    assert_eq!(generations, [0, 10, 20, 25]);
    // resuming from generation 20 ends like the run
    let ga: OneMaxGa = checkpoint::load(saved[2].1.as_slice()).unwrap();
    let outcome = Engine::new(ga, one_max)
        .stop_when(Stop::generations(25))
        .run()
        .unwrap();
    assert_eq!(outcome.best(), expected.best());
    assert_eq!(saved[3].1, {
        let mut engine = Engine::new(
            checkpoint::load::<OneMaxGa>(saved[2].1.as_slice()).unwrap(),
            one_max,
        )
        .stop_when(Stop::generations(25));
        engine.run().unwrap();
        bytes(engine.algorithm())
    });

    // an error stops the run
    let error = Engine::new(one_max_ga(17), one_max)
        .stop_when(Stop::generations(25))
        .checkpoint_every(10, |ga| {
            if ga.generation() == 10 {
                Err(Error::Checkpoint {
                    reason: "disk full".to_string(),
                })
            } else {
                Ok(())
            }
        })
        .run()
        .unwrap_err();
    assert_eq!(
        error,
        Error::Checkpoint {
            reason: "disk full".to_string()
        }
    );
    // every 0 generations is an error, for both engines
    let error = Engine::new(one_max_ga(17), one_max)
        .stop_when(Stop::generations(5))
        .checkpoint_every(0, |_| Ok(()))
        .run()
        .unwrap_err();
    assert!(matches!(
        error,
        Error::InvalidSetting {
            setting: "checkpoint_every",
            ..
        }
    ));
    let zdt1 = Zdt1::new(10);
    let nsga2 = Nsga2::builder(zdt1.real(), [Minimize; 2])
        .population_size(8)
        .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
        .mutate(PolynomialMutation::per_gene(0.1, 20.0).unwrap())
        .build()
        .unwrap();
    let error = MultiEngine::new(nsga2, zdt1)
        .stop_when(Stop::generations(5))
        .checkpoint_every(0, |_| Ok(()))
        .run()
        .unwrap_err();
    assert!(matches!(
        error,
        Error::InvalidSetting {
            setting: "checkpoint_every",
            ..
        }
    ));
}

#[test]
fn files_are_replaced_atomically() {
    let directory =
        std::env::temp_dir().join(format!("genoxide-checkpoint-{}", std::process::id()));
    std::fs::create_dir_all(&directory).unwrap();
    let path = directory.join("run.ckpt");
    let outcome = Engine::new(one_max_ga(18), one_max)
        .stop_when(Stop::generations(12))
        .checkpoint_every(5, |ga| checkpoint::save_file(ga, &path))
        .run()
        .unwrap();
    let ga: OneMaxGa = checkpoint::load_file(&path).unwrap();
    assert_eq!(ga.generation(), 12);
    assert_eq!(ga.best().unwrap(), outcome.best());
    // only the checkpoint is left
    let files: Vec<_> = std::fs::read_dir(&directory).unwrap().collect();
    assert_eq!(files.len(), 1);
    // a missing file
    let error = checkpoint::load_file::<OneMaxGa>(directory.join("missing.ckpt")).unwrap_err();
    assert!(matches!(error, Error::Checkpoint { .. }));
    std::fs::remove_dir_all(&directory).unwrap();
}

fn reason(error: Error) -> String {
    match error {
        Error::Checkpoint { reason } => reason,
        error => panic!("{error:?}"),
    }
}

#[test]
fn damaged_checkpoints_are_errors() {
    let mut engine = Engine::new(one_max_ga(19), one_max).stop_when(Stop::generations(3));
    engine.run().unwrap();
    let good = bytes(engine.algorithm());
    let load = |bytes: &[u8]| checkpoint::load::<OneMaxGa>(bytes).map_err(reason);
    assert!(load(&good).is_ok());

    assert_eq!(load(b"").unwrap_err(), "not a genoxide checkpoint");
    assert_eq!(
        load(b"{\"generation\": 3}").unwrap_err(),
        "not a genoxide checkpoint"
    );
    // every truncation
    for len in 8..good.len() {
        let error = load(&good[..len]).unwrap_err();
        assert!(
            error == "truncated" || error.starts_with("corrupted"),
            "{len}: {error}"
        );
    }
    // every flipped bit after the version
    let version_end = 9 + usize::from(good[8]);
    for byte in version_end..good.len() {
        for bit in 0..8 {
            let mut bad = good.clone();
            bad[byte] ^= 1 << bit;
            assert!(load(&bad).is_err(), "byte {byte}, bit {bit}");
        }
    }
    // extra bytes
    let mut long = good.clone();
    long.push(0);
    assert!(load(&long).unwrap_err().starts_with("corrupted"));
    // another version
    let mut other = good.clone();
    other[9] = b'9';
    let error = load(&other).unwrap_err();
    assert!(error.starts_with("saved by genoxide 9"), "{error}");
    assert!(error.contains(env!("CARGO_PKG_VERSION")), "{error}");
    // another algorithm: an error, not a panic or a wrong run
    let error = checkpoint::load::<De>(good.as_slice())
        .map_err(reason)
        .unwrap_err();
    assert!(
        error.starts_with("doesn't hold an algorithm of this type"),
        "{error}"
    );
    let error = checkpoint::load::<Ga<Binary, Rank, UniformCrossover, BitFlip>>(good.as_slice())
        .map_err(reason)
        .unwrap_err();
    assert!(
        error.starts_with("doesn't hold an algorithm of this type"),
        "{error}"
    );
}

#[test]
fn deserializing_validates() {
    use genoxide::multi::Scores;
    // valid values round trip, in a self-describing format too
    let json = |value: &str| value.to_string();
    let bits: Bits = [true, false, true].into_iter().collect();
    assert_eq!(
        serde_json::from_str::<Bits>(&serde_json::to_string(&bits).unwrap()).unwrap(),
        bits
    );
    let scores = Scores::constrained([1.0, -2.5, 3.0], 0.5);
    assert_eq!(
        serde_json::from_str::<Scores<3>>(&serde_json::to_string(&scores).unwrap()).unwrap(),
        scores
    );
    let ga = one_max_ga(20);
    let text = serde_json::to_string(&ga).unwrap();
    assert_eq!(
        bytes(&serde_json::from_str::<OneMaxGa>(&text).unwrap()),
        bytes(&ga)
    );

    // invalid values are errors
    let invalid = [
        serde_json::from_str::<Bits>(&json(r#"{"words":[8],"len":3}"#)).is_err(),
        serde_json::from_str::<Bits>(&json(r#"{"words":[],"len":3}"#)).is_err(),
        serde_json::from_str::<Order>(&json(r#"{"genes":[0,0,1]}"#)).is_err(),
        serde_json::from_str::<Fitness>(&json(r#"{"score":1.0,"violation":-1.0}"#)).is_err(),
        serde_json::from_str::<Scores<2>>(&json(r#"{"values":[1.0,2.0],"violation":-1.0}"#))
            .is_err(),
        serde_json::from_str::<Scores<2>>(&json(r#"{"values":[1.0],"violation":0.0}"#)).is_err(),
        serde_json::from_str::<Scores<2>>(&json(r#"{"values":[1.0,2.0,3.0],"violation":0.0}"#))
            .is_err(),
        serde_json::from_str::<Binary>(&json(r#"{"len":0}"#)).is_err(),
        serde_json::from_str::<Permutation>(&json(r#"{"len":0}"#)).is_err(),
        serde_json::from_str::<Real>(&json(r#"{"bounds":[{"start":1.0,"end":0.0}]}"#)).is_err(),
        serde_json::from_str::<Integer>(&json(r#"{"bounds":[]}"#)).is_err(),
    ];
    assert_eq!(invalid, [true; 11]);
    // -0.0 becomes 0.0, as in the constructors
    let fitness: Fitness = serde_json::from_str(r#"{"score":-0.0,"violation":0.0}"#).unwrap();
    assert!(fitness.score().unwrap().is_sign_positive());
}
