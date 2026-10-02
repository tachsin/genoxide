//! Latin hypercube samples of real genomes: one genome per stratum of every gene.

use genoxide::prelude::*;

#[test]
fn every_stratum_of_every_gene_holds_one_genome() {
    let real = Real::new([-5.0..=5.0, 0.0..=1.0, 100.0..=1e6, -1e-3..=0.0]).unwrap();
    for (n, seed) in [(1, 1), (2, 2), (7, 3), (100, 4), (1_000, 5)] {
        let sample = real
            .latin_hypercube(n, &mut StreamRng::seed_from_u64(seed))
            .unwrap();
        assert_eq!(sample.len(), n);
        for (gene, range) in real.bounds().iter().enumerate() {
            let (low, high) = (*range.start(), *range.end());
            let mut strata: Vec<usize> = sample
                .iter()
                .map(|genome| {
                    let x = genome[gene];
                    assert!((low..=high).contains(&x));
                    // the stratum, with a point on a boundary in the one below
                    let position = (x - low) / (high - low) * n as f64;
                    (position as usize).min(n - 1)
                })
                .collect();
            strata.sort();
            assert_eq!(strata, (0..n).collect::<Vec<_>>(), "n = {n}, gene {gene}");
        }
    }
}

#[test]
fn fixed_genes_take_their_value() {
    let real = Real::new([2.5..=2.5, 0.0..=1.0]).unwrap();
    let sample = real
        .latin_hypercube(20, &mut StreamRng::seed_from_u64(6))
        .unwrap();
    assert!(sample.iter().all(|genome| genome[0] == 2.5));
}

#[test]
fn a_seed_gives_the_same_sample_on_every_platform() {
    let real = Real::uniform(3, -1.0..=1.0).unwrap();
    let sample = |seed| {
        real.latin_hypercube(4, &mut StreamRng::seed_from_u64(seed))
            .unwrap()
    };
    assert_eq!(sample(7), sample(7));
    assert_ne!(sample(7), sample(8));
    let bits: Vec<u64> = sample(7)[0].iter().map(|x| x.to_bits()).collect();
    assert_eq!(
        bits,
        [
            4600217523584051144,
            13829652665615299316,
            4585620556182390272
        ],
        "the sample changed, which breaks reproducibility"
    );
}

#[test]
fn a_sample_of_none_or_too_many_is_an_error() {
    let real = Real::uniform(2, 0.0..=1.0).unwrap();
    let mut rng = StreamRng::seed_from_u64(1);
    for n in [0, (1 << 24) + 1] {
        assert!(matches!(
            real.latin_hypercube(n, &mut rng),
            Err(Error::InvalidSetting { setting: "n", .. })
        ));
    }
}

#[test]
fn it_seeds_an_algorithm() {
    // a Latin hypercube as a GA's initial population
    let real = Real::uniform(5, -5.12..=5.12).unwrap();
    let initial = real
        .latin_hypercube(40, &mut StreamRng::seed_from_u64(9))
        .unwrap();
    let ga = Ga::builder(real)
        .population_size(40)
        .initial_genomes(initial.clone())
        .select(Tournament::new(3).unwrap())
        .crossover(SimulatedBinaryCrossover::new(15.0).unwrap())
        .mutate(PolynomialMutation::per_gene(0.2, 20.0).unwrap())
        .minimize()
        .seed(9)
        .build()
        .unwrap();
    let mut engine = Engine::new(ga, |x: &Reals| x.iter().map(|xi| xi * xi).sum::<f64>())
        .stop_when(Stop::generations(0));
    engine.run().unwrap();
    let population: Vec<&Reals> = engine
        .algorithm()
        .population()
        .iter()
        .map(Individual::genome)
        .collect();
    for genome in &initial {
        assert!(population.contains(&genome));
    }
}
