//! Checkpoints: a resumed run is exactly the run without the interruption.
#![cfg(feature = "serde")]

use genoxide::Objective::Minimize;
use genoxide::algorithm::islands::Topology;
use genoxide::checkpoint;
use genoxide::multi::problems::{Dtlz2, MultiProblem, Zdt1};
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

// minimize Σ xⱼ subject to Σ xⱼ² ≤ 1 and x₀ ≥ −0.5, with the derivatives
fn ball(x: &Reals, gradient: &mut [f64], g: &mut [f64], jacobian: &mut [f64]) -> f64 {
    let n = x.len();
    gradient.fill(1.0);
    g[0] = x.iter().map(|xi| xi * xi).sum::<f64>() - 1.0;
    g[1] = -0.5 - x[0];
    for j in 0..n {
        jacobian[j] = 2.0 * x[j];
    }
    jacobian[n] = -1.0;
    x.iter().sum()
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
fn resumes<A, F>(make: impl Fn() -> A, fitness: F, split: u64, total: u64)
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
    assert_eq!(bytes(second.algorithm()), bytes(whole.algorithm()));
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
    resumes(|| one_max_ga(1), one_max, 7, 20);
    for scheme in [
        Scheme::Generational { elitism: 1 },
        Scheme::SteadyState { replacements: 5 },
        Scheme::MuPlusLambda { lambda: 20 },
        Scheme::MuCommaLambda { lambda: 40 },
    ] {
        resumes(|| real_ga(scheme, 2), rastrigin, 9, 25);
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
    resumes(memetic, tour, 5, 15);
}

#[cfg(feature = "parallel")]
#[test]
fn genetic_algorithms_with_parallel_breeding_resume_exactly() {
    let parallel = |scheme, seed| {
        let ga = Ga::builder(Real::uniform(10, -5.12..=5.12).unwrap())
            .population_size(21)
            .select(Tournament::new(3).unwrap())
            .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
            .mutate(PolynomialMutation::per_gene(0.1, 20.0).unwrap())
            .scheme(scheme)
            .parallel_breeding(true)
            .minimize()
            .seed(seed)
            .build()
            .unwrap();
        assert!(ga.parallel_breeding());
        ga
    };
    for scheme in [
        Scheme::Generational { elitism: 1 },
        Scheme::SteadyState { replacements: 5 },
        Scheme::MuPlusLambda { lambda: 21 },
        Scheme::MuCommaLambda { lambda: 41 },
    ] {
        resumes(|| parallel(scheme, 2), rastrigin, 9, 25);
    }
    // the setting is saved: without it, the resumed run would breed differently
    let mut engine =
        Engine::new(parallel(Scheme::default(), 3), rastrigin).stop_when(Stop::generations(5));
    engine.run().unwrap();
    let resumed: Ga<Real, Tournament, SimulatedBinaryCrossover, PolynomialMutation> =
        checkpoint::load(bytes(engine.algorithm()).as_slice()).unwrap();
    assert!(resumed.parallel_breeding());
}

#[cfg(feature = "parallel")]
#[test]
fn de_and_es_with_parallel_breeding_resume_exactly() {
    let de = |seed| {
        let de = De::l_shade(Real::uniform(10, -5.12..=5.12).unwrap(), 3_000)
            .parallel_breeding(true)
            .minimize()
            .seed(seed)
            .build()
            .unwrap();
        assert!(de.parallel_breeding());
        de
    };
    resumes(|| de(4), rastrigin, 10, 40);
    // SHADE, with restarts
    let restarting = || {
        De::builder(Real::uniform(10, -5.12..=5.12).unwrap())
            .population_size(20)
            .restarts(de::Restarts::OnStagnation {
                tolerance: 1e-3,
                patience: 3,
            })
            .parallel_breeding(true)
            .minimize()
            .seed(5)
            .build()
            .unwrap()
    };
    resumes(restarting, rastrigin, 12, 40);
    let es = |seed| {
        Es::builder(Real::uniform(10, -5.12..=5.12).unwrap())
            .parents(5)
            .offspring(35)
            .parallel_breeding(true)
            .minimize()
            .seed(seed)
            .build()
            .unwrap()
    };
    resumes(|| es(6), rastrigin, 10, 30);
    // islands of DEs with parallel breeding
    let islands = || {
        Islands::builder((0..3).map(de).collect())
            .interval(5)
            .seed(12)
            .build()
            .unwrap()
    };
    resumes(islands, rastrigin, 7, 20);
    // the setting is saved: without it, the resumed run would breed differently
    let mut engine = Engine::new(de(3), rastrigin).stop_when(Stop::generations(5));
    engine.run().unwrap();
    let resumed: De = checkpoint::load(bytes(engine.algorithm()).as_slice()).unwrap();
    assert!(resumed.parallel_breeding());
    let mut engine = Engine::new(es(3), rastrigin).stop_when(Stop::generations(5));
    engine.run().unwrap();
    let resumed: Es = checkpoint::load(bytes(engine.algorithm()).as_slice()).unwrap();
    assert!(resumed.parallel_breeding());
}

#[test]
fn other_algorithms_resume_exactly() {
    resumes(|| de(4), rastrigin, 10, 40);
    let pso = || {
        Pso::builder(Real::uniform(10, -5.12..=5.12).unwrap())
            .population_size(20)
            .topology(pso::Topology::Ring { neighbors: 1 })
            .minimize()
            .seed(5)
            .build()
            .unwrap()
    };
    resumes(pso, rastrigin, 10, 30);
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
    resumes(es, rastrigin, 10, 30);
    // across a BIPOP restart
    let cmaes = || {
        Cmaes::builder(Real::uniform(10, -5.12..=5.12).unwrap())
            .restarts(cmaes::Restarts::Bipop)
            .minimize()
            .seed(7)
            .build()
            .unwrap()
    };
    resumes(cmaes, rastrigin, 150, 400);
    let diagonal = || {
        Cmaes::builder(Real::uniform(10, -5.12..=5.12).unwrap())
            .covariance(cmaes::Covariance::Diagonal)
            .minimize()
            .seed(8)
            .build()
            .unwrap()
    };
    resumes(diagonal, rastrigin, 20, 60);
    // the simplex, an iteration under way and random restarts, one point a round or four
    for speculative in [false, true] {
        let nelder_mead = || {
            NelderMead::builder(Real::uniform(4, -5.12..=5.12).unwrap())
                .restarts(local::Restarts::Random { times: 1_000 })
                .speculative(speculative)
                .minimize()
                .seed(11)
                .build()
                .unwrap()
        };
        resumes(nelder_mead, rastrigin, 150, 400);
    }
    // L-BFGS-B: its pairs, a line search under way and random restarts, with finite differences
    // and with a supplied gradient
    let lbfgsb = || {
        Lbfgsb::builder(Real::uniform(4, -5.12..=5.12).unwrap())
            .restarts(local::Restarts::Random { times: 1_000 })
            .minimize()
            .seed(12)
            .build()
            .unwrap()
    };
    resumes(lbfgsb, rastrigin, 37, 150);
    let rosenbrock = Differentiable(|x: &Reals, gradient: &mut [f64]| {
        let mut value = 0.0;
        for i in 0..x.len() - 1 {
            let (a, b) = (x[i + 1] - x[i] * x[i], 1.0 - x[i]);
            gradient[i] += -400.0 * x[i] * a - 2.0 * b;
            gradient[i + 1] += 200.0 * a;
            value += 100.0 * a * a + b * b;
        }
        value
    });
    resumes(lbfgsb, rosenbrock, 23, 80);
    // MMA's asymptotes and iterates, and GCMMA's inner iterations under way, to convergence
    for method in [mma::Method::Mma, mma::Method::Gcmma] {
        let mma = || {
            Mma::builder(Real::uniform(3, -5.0..=5.0).unwrap())
                .method(method)
                .minimize()
                .seed(12)
                .build()
                .unwrap()
        };
        for split in [1, 5, 9] {
            resumes(mma, Constrained::differentiable(2, ball), split, 200);
        }
    }
    // the mean and Adam's moments, with the mean evaluated and parallel breeding
    for parallel in [false, true] {
        let open_es = || {
            OpenEs::builder(Real::uniform(10, -5.12..=5.12).unwrap())
                .population_size(20)
                .sigma(0.01)
                .evaluate_mean(true)
                .parallel_breeding(parallel)
                .minimize()
                .seed(9)
                .build()
                .unwrap()
        };
        resumes(open_es, rastrigin, 10, 30);
    }
    let sgd = || {
        OpenEs::builder(Real::uniform(10, -5.12..=5.12).unwrap())
            .population_size(20)
            .optimizer(genoxide::algorithm::open_es::Optimizer::sgd(0.01, 0.9))
            .minimize()
            .seed(10)
            .build()
            .unwrap()
    };
    resumes(sgd, rastrigin, 10, 30);
    // a first-order method's memory: with finite differences and random restarts, each rule
    for step in [
        first_order::Step::gradient(0.001),
        first_order::Step::momentum(0.001, 0.9),
        first_order::Step::nesterov(0.001, 0.9),
        first_order::Step::adam(0.05),
        first_order::Step::adamw(0.05, 0.01),
    ] {
        let first_order = || {
            FirstOrder::builder(Real::uniform(4, -5.12..=5.12).unwrap())
                .step(step)
                .restarts(local::Restarts::Random { times: 1_000 })
                .minimize()
                .seed(12)
                .build()
                .unwrap()
        };
        resumes(first_order, rastrigin, 150, 400);
    }
    // and with the gradients supplied
    let problem = genoxide::problems::Rosenbrock::new(10);
    let first_order = || {
        use genoxide::problems::Problem;
        FirstOrder::builder(problem.representation())
            .step(first_order::Step::adam(0.01))
            .minimize()
            .seed(13)
            .build()
            .unwrap()
    };
    resumes(first_order, problem, 100, 300);
}

#[test]
fn a_cmaes_that_stops_where_it_converges_resumes() {
    let cmaes = || {
        Cmaes::builder(Real::uniform(10, -5.12..=5.12).unwrap())
            .restarts(cmaes::Restarts::Stop)
            .minimize()
            .seed(12)
            .build()
            .unwrap()
    };
    let mut whole = Engine::new(cmaes(), rastrigin).stop_when(Stop::generations(100_000));
    let expected = whole.run().unwrap();
    assert_eq!(expected.stop_reason(), StopReason::Converged);
    let converged = expected.generations();
    assert!(converged > 100, "{converged}");

    // saved before the convergence, resumed to it
    resumes(cmaes, rastrigin, converged / 2, 100_000);
    let mut first = Engine::new(cmaes(), rastrigin).stop_when(Stop::generations(converged / 2));
    first.run().unwrap();
    let resumed: Cmaes = checkpoint::load(bytes(first.algorithm()).as_slice()).unwrap();
    let outcome = Engine::new(resumed, rastrigin)
        .stop_when(Stop::generations(100_000))
        .run()
        .unwrap();
    assert_eq!(outcome.stop_reason(), StopReason::Converged);
    assert_eq!(outcome.generations(), converged);

    // saved when it has finished: it stays finished
    let finished: Cmaes = checkpoint::load(bytes(whole.algorithm()).as_slice()).unwrap();
    assert!(finished.is_finished());
    assert_eq!(finished.converged(), whole.algorithm().converged());
    let again = Engine::new(finished, rastrigin)
        .stop_when(Stop::generations(100_000))
        .run()
        .unwrap();
    assert_eq!(again.stop_reason(), StopReason::Converged);
    assert_eq!(again.evaluations(), expected.evaluations());
    assert_eq!(again.best(), expected.best());
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
    resumes(tabu, conflicts, 20, 60);
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
    resumes(annealing, one_max, 70, 200);
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
        resumes(islands, rastrigin, 6, 20);
        resumes(islands, rastrigin, 8, 20);
    }
    let islands = || {
        Islands::builder((0..3).map(de).collect())
            .interval(5)
            .seed(12)
            .build()
            .unwrap()
    };
    resumes(islands, rastrigin, 7, 20);
}

#[test]
fn multi_objective_algorithms_resume_exactly() {
    let zdt1 = Zdt1::new(10);
    let dtlz2 = Dtlz2::<3>::default();
    let sbx = || SimulatedBinaryCrossover::new(15.0).unwrap();
    let mutation = |n: usize| PolynomialMutation::per_gene(1.0 / n as f64, 20.0).unwrap();
    let nsga2 = || {
        Nsga2::builder(zdt1.representation(), [Minimize; 2])
            .population_size(20)
            .crossover(sbx())
            .mutate(mutation(10))
            .seed(12)
            .build()
            .unwrap()
    };
    resumes_multi(nsga2, zdt1, 8, 20);
    let nsga3 = || {
        Nsga3::builder(dtlz2.representation(), [Minimize; 3], das_dennis::<3>(4))
            .population_size(16)
            .crossover(sbx())
            .mutate(mutation(12))
            .seed(13)
            .build()
            .unwrap()
    };
    resumes_multi(nsga3, dtlz2, 8, 20);
    let spea2 = || {
        Spea2::builder(zdt1.representation(), [Minimize; 2])
            .population_size(20)
            .crossover(sbx())
            .mutate(mutation(10))
            .seed(14)
            .build()
            .unwrap()
    };
    resumes_multi(spea2, zdt1, 8, 20);
    let moead = || {
        Moead::builder(dtlz2.representation(), [Minimize; 3], das_dennis::<3>(4))
            .decomposition(Decomposition::Pbi { theta: 5.0 })
            .crossover(sbx())
            .mutate(mutation(12))
            .seed(15)
            .build()
            .unwrap()
    };
    resumes_multi(moead, dtlz2, 8, 20);
    let sms_emoa = || {
        SmsEmoa::builder(zdt1.representation(), [Minimize; 2])
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
    let nsga2 = Nsga2::builder(zdt1.representation(), [Minimize; 2])
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
            error == "truncated" || error.starts_with("corrupted or truncated"),
            "{len}: {error}"
        );
    }
    // every flipped bit after the magic bytes
    for byte in 8..good.len() {
        for bit in 0..8 {
            let mut bad = good.clone();
            bad[byte] ^= 1 << bit;
            let error = load(&bad).unwrap_err();
            assert!(
                error.starts_with("corrupted"),
                "byte {byte}, bit {bit}: {error}"
            );
        }
    }
    // extra bytes
    let mut long = good.clone();
    long.push(0);
    assert!(load(&long).unwrap_err().starts_with("corrupted"));

    // with a valid checksum: another version
    let mut other = good[..good.len() - 8].to_vec();
    other[9] = b'9';
    let error = load(&summed(other)).unwrap_err();
    assert!(error.starts_with("saved by genoxide 9"), "{error}");
    assert!(error.contains(env!("CARGO_PKG_VERSION")), "{error}");
    // another algorithm, even one with the same layout
    let error = checkpoint::load::<De>(good.as_slice())
        .map_err(reason)
        .unwrap_err();
    assert!(
        error.starts_with("holds a genoxide::algorithm::ga::Ga<"),
        "{error}"
    );
    assert!(
        error.contains(", not a genoxide::algorithm::de::De"),
        "{error}"
    );
    let roulette = Ga::builder(Binary::new(8).unwrap())
        .population_size(4)
        .select(Roulette)
        .crossover(NoCrossover)
        .mutate(BitFlip::count(1).unwrap())
        .seed(0)
        .build()
        .unwrap();
    let error = checkpoint::load::<Ga<Binary, StochasticUniversalSampling, NoCrossover, BitFlip>>(
        bytes(&roulette).as_slice(),
    )
    .map_err(reason)
    .unwrap_err();
    assert!(error.contains("Roulette"), "{error}");
    // a payload that doesn't hold what the header says: an error, not a panic
    let header = good.len() - 8 - payload_len(&good);
    let mut wrong = good[..header].to_vec();
    wrong.truncate(header - 8);
    wrong.extend_from_slice(&3u64.to_le_bytes());
    wrong.extend_from_slice(&[1, 2, 3]);
    let error = load(&summed(wrong)).unwrap_err();
    assert!(
        error.starts_with("doesn't hold an algorithm of this type"),
        "{error}"
    );
}

// the length of a checkpoint's payload
fn payload_len(bytes: &[u8]) -> usize {
    let version_end = 9 + usize::from(bytes[8]);
    let kind_len = u16::from_le_bytes([bytes[version_end], bytes[version_end + 1]]);
    let len_start = version_end + 2 + usize::from(kind_len);
    u64::from_le_bytes(bytes[len_start..len_start + 8].try_into().unwrap()) as usize
}

// `bytes` with the checksum of a checkpoint: 64-bit FNV-1a of everything after the magic bytes
fn summed(mut bytes: Vec<u8>) -> Vec<u8> {
    let sum = bytes[8..]
        .iter()
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, &byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3)
        });
    bytes.extend_from_slice(&sum.to_le_bytes());
    bytes
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

// what a hand-edited or damaged checkpoint can hold, and would hang or panic a run
#[test]
fn deserializing_checks_what_would_hang_or_panic() {
    fn sphere(x: &Reals) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }

    // a De of 3 individuals never finds three others for a trial
    let de = De::builder(Real::uniform(3, -5.0..=5.0).unwrap())
        .population_size(10)
        .strategy(genoxide::algorithm::de::Strategy::Rand1)
        .minimize()
        .seed(1)
        .build()
        .unwrap();
    let mut engine = Engine::new(de, sphere).stop_when(Stop::generations(2));
    engine.run().unwrap();
    let mut json = serde_json::to_value(engine.algorithm()).unwrap();
    json["population"]["individuals"]
        .as_array_mut()
        .unwrap()
        .truncate(4);
    serde_json::from_value::<De>(json.clone()).unwrap();
    json["population"]["individuals"]
        .as_array_mut()
        .unwrap()
        .truncate(3);
    let message = serde_json::from_value::<De>(json).unwrap_err().to_string();
    assert!(
        message.contains("differential evolution needs at least 4 individuals, got 3"),
        "{message}"
    );

    // islands without an island have no objective
    let islands = (0..2)
        .map(|seed| {
            De::builder(Real::uniform(3, -5.0..=5.0).unwrap())
                .population_size(10)
                .minimize()
                .seed(seed)
                .build()
                .unwrap()
        })
        .collect::<Vec<_>>();
    let islands = Islands::builder(islands)
        .topology(Topology::Ring)
        .build()
        .unwrap();
    let mut json = serde_json::to_value(&islands).unwrap();
    assert_eq!(
        bytes(&serde_json::from_value::<Islands<De>>(json.clone()).unwrap()),
        bytes(&islands)
    );
    json["islands"].as_array_mut().unwrap().clear();
    let message = serde_json::from_value::<Islands<De>>(json)
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("islands need at least 1 island"),
        "{message}"
    );
}

#[test]
fn a_controlled_run_resumes() {
    use genoxide::engine::Progress;
    // the mutation rate falls over the run, and the population is scored again at generation 7:
    // the generation the first part stops at, so the checkpoint holds a pending re-evaluation
    fn control(ga: &mut OneMaxGa, progress: &Progress) -> genoxide::Result<()> {
        ga.set_mutation_rate(1.0 - progress.generation() as f64 / 40.0)?;
        if progress.generation() == 7 {
            ga.reevaluate()?;
        }
        Ok(())
    }
    let mut whole = Engine::new(one_max_ga(1), one_max)
        .stop_when(Stop::generations(20))
        .control(control);
    let expected = whole.run().unwrap();

    let mut first = Engine::new(one_max_ga(1), one_max)
        .stop_when(Stop::generations(7))
        .control(control);
    first.run().unwrap();
    let resumed: OneMaxGa = checkpoint::load(bytes(first.algorithm()).as_slice()).unwrap();
    let mut second = Engine::new(resumed, one_max)
        .stop_when(Stop::generations(20))
        .control(control);
    let outcome = second.run().unwrap();

    assert_eq!(outcome.best(), expected.best());
    assert_eq!(outcome.evaluations(), expected.evaluations());
    assert_eq!(bytes(second.algorithm()), bytes(whole.algorithm()));
}

// genetic programming: trees, the typed set with its constants, and the tree operators
#[test]
fn genetic_programs_resume_exactly() {
    use genoxide::gp::{Constants, Gp, PrimitiveSet, SubtreeCrossover, SubtreeMutation, Tree};

    #[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
    enum Op {
        Add,
        Mul,
        Less,
        If,
        X,
    }
    fn gp() -> Gp<Op> {
        let mut set = PrimitiveSet::builder();
        let real = set.new_type("real");
        let boolean = set.new_type("bool");
        set.function("add", Op::Add, [real, real], real)
            .function("mul", Op::Mul, [real, real], real)
            .function("less", Op::Less, [real, real], boolean)
            .function("if", Op::If, [boolean, real, real], real)
            .terminal("x", Op::X, real)
            .constants(real, Constants::uniform(-1.0..=1.0).unwrap());
        Gp::builder(set.build(real).unwrap()).build().unwrap()
    }
    // |x| at 11 points
    fn error(tree: &Tree) -> f64 {
        let set = gp().primitives().clone();
        let mut stack = Vec::new();
        (0..=10)
            .map(|i| {
                let x = f64::from(i) / 5.0 - 1.0;
                let value = tree.evaluate(
                    &set,
                    &mut stack,
                    |op, args: &[f64]| match op {
                        Op::Add => args[0] + args[1],
                        Op::Mul => args[0] * args[1],
                        Op::Less => f64::from(u8::from(args[0] < args[1])),
                        Op::If => {
                            if args[0] != 0.0 {
                                args[1]
                            } else {
                                args[2]
                            }
                        }
                        Op::X => x,
                    },
                    |_, value| value,
                );
                (value - x.abs()).abs()
            })
            .sum()
    }
    for breeding in [false, true] {
        let make = || {
            let builder = Ga::builder(gp())
                .population_size(30)
                .select(Tournament::new(3).unwrap())
                .crossover(SubtreeCrossover::new())
                .mutate(SubtreeMutation::new())
                .mutation_rate(0.2)
                .minimize()
                .seed(4);
            #[cfg(feature = "parallel")]
            let builder = builder.parallel_breeding(breeding);
            builder.build().unwrap()
        };
        resumes(make, error, 6, 15);
    }
}

#[test]
fn an_lbfgsb_saved_between_an_ask_and_its_tell_resumes() {
    let sphere = |x: &[f64]| Fitness::new(x.iter().map(|xi| xi * xi).sum());
    let mut lbfgsb = Lbfgsb::builder(Real::uniform(3, -5.0..=5.0).unwrap())
        .minimize()
        .seed(3)
        .build()
        .unwrap();
    for _ in 0..4 {
        let fitness: Vec<Fitness> = lbfgsb.ask().iter().map(|x| sphere(x)).collect();
        lbfgsb.tell(&fitness).unwrap();
    }
    let asked: Vec<Vec<f64>> = lbfgsb.ask().iter().map(|x| x.to_vec()).collect();
    let mut resumed: Lbfgsb = checkpoint::load(bytes(&lbfgsb).as_slice()).unwrap();
    let again: Vec<Vec<f64>> = resumed.ask().iter().map(|x| x.to_vec()).collect();
    assert_eq!(again, asked);
    let fitness: Vec<Fitness> = asked.iter().map(|x| sphere(x)).collect();
    lbfgsb.tell(&fitness).unwrap();
    resumed.tell(&fitness).unwrap();
    assert_eq!(bytes(&resumed), bytes(&lbfgsb));
}

// Branin's function, for Bayesian optimization
fn branin(x: &Reals) -> f64 {
    use genoxide::problems::Branin;
    Branin.evaluate(x)
}

fn bo(seed: u64) -> Bo {
    use genoxide::problems::{Branin, Problem};
    Bo::builder(Branin.representation())
        .minimize()
        .seed(seed)
        .build()
        .unwrap()
}

#[test]
fn bayesian_optimization_resumes() {
    // the warm start of the hyperparameters is saved; the model is fitted again on the next ask
    resumes(|| bo(1), branin, 4, 12);
    let log = || {
        use genoxide::problems::{Branin, Problem};
        Bo::builder(Branin.representation())
            .output(bo::Output::Log)
            .acquisition(bo::Acquisition::UpperConfidenceBound { beta: 2.0 })
            .minimize()
            .seed(2)
            .build()
            .unwrap()
    };
    resumes(log, branin, 0, 8);
}

#[test]
fn bayesian_optimization_saved_between_an_ask_and_its_tell_resumes() {
    let mut bo = bo(3);
    for _ in 0..4 {
        let fitness: Vec<Fitness> = bo.ask().iter().map(|x| Fitness::new(branin(x))).collect();
        bo.tell(&fitness).unwrap();
    }
    let asked: Vec<Vec<f64>> = bo.ask().iter().map(|x| x.to_vec()).collect();
    let mut resumed: Bo = checkpoint::load(bytes(&bo).as_slice()).unwrap();
    let again: Vec<Vec<f64>> = resumed.ask().iter().map(|x| x.to_vec()).collect();
    assert_eq!(again, asked);
    let fitness: Vec<Fitness> = asked
        .iter()
        .map(|x| Fitness::new(branin(&Reals::from(x.clone()))))
        .collect();
    bo.tell(&fitness).unwrap();
    resumed.tell(&fitness).unwrap();
    assert_eq!(bytes(&resumed), bytes(&bo));
    for _ in 0..3 {
        let a: Vec<Vec<f64>> = bo.ask().iter().map(|x| x.to_vec()).collect();
        let b: Vec<Vec<f64>> = resumed.ask().iter().map(|x| x.to_vec()).collect();
        assert_eq!(a, b);
        let fitness: Vec<Fitness> = a
            .iter()
            .map(|x| Fitness::new(branin(&Reals::from(x.clone()))))
            .collect();
        bo.tell(&fitness).unwrap();
        resumed.tell(&fitness).unwrap();
    }
    assert_eq!(bytes(&resumed), bytes(&bo));
}

// a continuation: Adam through 3 stages of a smoothed Σ |xᵢ − cᵢ|, its ε shared by an atomic
mod continuation {
    use super::bytes;
    use genoxide::checkpoint;
    use genoxide::gradient::Differentiable;
    use genoxide::prelude::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    const EPSILON: [f64; 3] = [1.0, 0.1, 0.01];

    type Staged = Continuation<FirstOrder>;

    // the stage's ε, set from its index
    fn on_stage(epsilon: &Arc<AtomicU64>) -> impl Fn(usize, &mut FirstOrder) -> Result<()> + use<> {
        let epsilon = Arc::clone(epsilon);
        move |stage, _| {
            epsilon.store(EPSILON[stage].to_bits(), Ordering::Relaxed);
            Ok(())
        }
    }

    fn staged(epsilon: &Arc<AtomicU64>) -> Staged {
        let adam = FirstOrder::builder(Real::uniform(3, -5.0..=5.0).unwrap())
            .step(first_order::Step::adam(0.05))
            .minimize()
            .seed(4)
            .build()
            .unwrap();
        Continuation::builder(adam)
            .stages(EPSILON.len())
            .generations(10)
            .on_stage(on_stage(epsilon))
            .build()
            .unwrap()
    }

    fn smoothed(
        epsilon: &Arc<AtomicU64>,
    ) -> Differentiable<impl Fn(&Reals, &mut [f64]) -> f64 + Sync + use<>> {
        let epsilon = Arc::clone(epsilon);
        Differentiable(move |x: &Reals, gradient: &mut [f64]| {
            let e = f64::from_bits(epsilon.load(Ordering::Relaxed));
            let mut value = 0.0;
            for (i, c) in [0.5, -1.0, 2.0].into_iter().enumerate() {
                let root = ((x[i] - c) * (x[i] - c) + e * e).sqrt();
                gradient[i] = (x[i] - c) / root;
                value += root;
            }
            value
        })
    }

    #[test]
    fn a_continuation_resumes_in_its_stage() {
        let epsilon = Arc::new(AtomicU64::new(0));
        let mut whole =
            Engine::new(staged(&epsilon), smoothed(&epsilon)).stop_when(Stop::generations(1_000));
        let expected = whole.run().unwrap();
        assert_eq!(expected.stop_reason(), StopReason::Converged);
        assert_eq!(expected.generations(), 30);

        // within a stage, at its end (a re-evaluation pending), and in the last one
        for split in [7, 10, 15, 29] {
            // a fresh ε, as in another process: the resumed run sets its stage's
            let epsilon = Arc::new(AtomicU64::new(0));
            let mut first = Engine::new(staged(&epsilon), smoothed(&epsilon))
                .stop_when(Stop::generations(split));
            first.run().unwrap();
            let saved = bytes(first.algorithm());
            let epsilon = Arc::new(AtomicU64::new(0));
            let mut resumed: Staged = checkpoint::load(saved.as_slice()).unwrap();
            assert_eq!(resumed.generation(), split);
            assert_eq!(resumed.stage(), first.algorithm().stage());
            assert_eq!(resumed.stages(), first.algorithm().stages());
            // without its closure, the run can't start
            let mut missing = Engine::new(resumed.clone(), smoothed(&epsilon))
                .stop_when(Stop::generations(1_000));
            assert!(matches!(
                missing.run(),
                Err(Error::MissingSetting {
                    setting: "on_stage"
                })
            ));
            resumed.set_on_stage(on_stage(&epsilon));
            let mut second =
                Engine::new(resumed, smoothed(&epsilon)).stop_when(Stop::generations(1_000));
            let outcome = second.run().unwrap();
            assert_eq!(outcome.best(), expected.best(), "{split}");
            assert_eq!(outcome.evaluations(), expected.evaluations());
            assert_eq!(second.algorithm().stages(), whole.algorithm().stages());
            assert_eq!(bytes(second.algorithm()), bytes(whole.algorithm()));
        }
    }

    #[test]
    fn a_damaged_continuation_is_an_error() {
        let epsilon = Arc::new(AtomicU64::new(0));
        let mut engine =
            Engine::new(staged(&epsilon), smoothed(&epsilon)).stop_when(Stop::generations(12));
        engine.run().unwrap();
        let value = serde_json::to_value(engine.algorithm()).unwrap();
        let loads = |change: &dyn Fn(&mut serde_json::Value)| {
            let mut value = value.clone();
            change(&mut value);
            serde_json::from_value::<Staged>(value).is_ok()
        };
        assert!(loads(&|_| {}));
        assert!(!loads(&|value| value["stage"] = 3.into()));
        assert!(!loads(&|value| value["stage_count"] = 0.into()));
        assert!(!loads(&|value| value["generations"] = 0.into()));
        assert!(!loads(&|value| {
            let stage = value["stages"][0].clone();
            value["stages"] = vec![stage; 4].into();
        }));
    }
}

#[test]
fn bayesian_optimization_in_batches_resumes() {
    use genoxide::algorithm::bo::{Fantasy, Lie};
    use genoxide::problems::{Branin, Problem};
    for fantasy in [Fantasy::KrigingBeliever, Fantasy::ConstantLiar(Lie::Max)] {
        let batch = || {
            Bo::builder(Branin.representation())
                .batch(3)
                .fantasy(fantasy)
                .minimize()
                .seed(4)
                .build()
                .unwrap()
        };
        resumes(batch, branin, 3, 7);
    }
}

// Gramacy et al.'s (2016) toy problem, its two constraints' values one by one
fn toy(x: &Reals, g: &mut [f64]) -> f64 {
    let wave = genoxide::math::sin(TAU * (x[0] * x[0] - 2.0 * x[1]));
    g[0] = 1.5 - x[0] - 2.0 * x[1] - 0.5 * wave;
    g[1] = x[0] * x[0] + x[1] * x[1] - 1.5;
    x[0] + x[1]
}

#[test]
fn constrained_bayesian_optimization_resumes() {
    use genoxide::constraint::Constrained;
    // the constraints' values of every point and their models' warm starts are saved
    let toy_bo = || {
        Bo::builder(Real::uniform(2, 0.0..=1.0).unwrap())
            .batch(2)
            .minimize()
            .seed(5)
            .build()
            .unwrap()
    };
    resumes(toy_bo, Constrained::new(2, toy), 3, 8);
}

#[test]
fn asynchronous_bayesian_optimization_resumes_with_its_pending_points() {
    use genoxide::algorithm::Incremental;
    use genoxide::engine::Provided;
    // prepared at the start of its run, as an engine does
    let mut bo = bo(6);
    Incremental::prepare(&mut bo, Provided::NOTHING).unwrap();
    let design: Vec<Reals> = (0..6).map(|_| bo.propose()).collect();
    for x in &design {
        bo.receive(x.clone(), Fitness::new(branin(x))).unwrap();
    }
    // three points being evaluated, then one result
    let pending: Vec<Reals> = (0..3).map(|_| bo.propose()).collect();
    bo.receive(pending[0].clone(), Fitness::new(branin(&pending[0])))
        .unwrap();
    // the checkpoint holds the two points still being evaluated
    let mut resumed: Bo = checkpoint::load(bytes(&bo).as_slice()).unwrap();
    assert_eq!(resumed.proposed(), &pending[1..]);
    // a new run's engine has lost their evaluations: they are proposed again first
    Incremental::prepare(&mut resumed, Provided::NOTHING).unwrap();
    assert!(resumed.proposed().is_empty());
    assert_eq!(resumed.propose(), pending[1]);
    assert_eq!(resumed.propose(), pending[2]);
    // then the run goes on as the uninterrupted one, with the same points fantasized
    assert_eq!(resumed.propose(), bo.propose());
    assert_eq!(bytes(&resumed), bytes(&bo));
    for x in &pending[1..] {
        bo.receive(x.clone(), Fitness::new(branin(x))).unwrap();
        resumed.receive(x.clone(), Fitness::new(branin(x))).unwrap();
    }
    assert_eq!(resumed.propose(), bo.propose());
    assert_eq!(bytes(&resumed), bytes(&bo));
}

#[test]
fn integer_bayesian_optimization_resumes() {
    let quadratic = |x: &Integers| {
        let x: Vec<f64> = x.iter().map(|&v| v as f64).collect();
        (x[0] - 2.6) * (x[0] - 2.6) + 2.0 * (x[1] + 1.3) * (x[1] + 1.3) + x[2] * x[2]
    };
    let integer_bo = || {
        Bo::builder(Integer::uniform(3, -5..=5).unwrap())
            .minimize()
            .seed(2)
            .build()
            .unwrap()
    };
    resumes(integer_bo, quadratic, 3, 9);
}
