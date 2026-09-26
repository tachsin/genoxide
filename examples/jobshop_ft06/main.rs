//! Job shop scheduling: Fisher and Thompson's 6×6 instance (ft06), whose shortest makespan is 55.
//!
//! Six jobs each go through the six machines in their own order, and a machine does one operation
//! at a time. A permutation of the 36 operations, where operation `k` counts for job `k / 6`, is
//! read as a sequence of jobs (a permutation with repetition): the n-th time a job appears, its
//! n-th operation starts as early as its job and its machine allow (a semi-active schedule). A
//! genetic algorithm with order crossover and swap mutation searches the sequences.
//!
//! ```text
//! cargo run --release --example jobshop_ft06
//! ```

use genoxide::prelude::*;

const JOBS: usize = 6;
const MACHINES: usize = 6;
// each job's operations in order: (machine, duration)
const INSTANCE: [[(usize, u32); MACHINES]; JOBS] = [
    [(2, 1), (0, 3), (1, 6), (3, 7), (5, 3), (4, 6)],
    [(1, 8), (2, 5), (4, 10), (5, 10), (0, 10), (3, 4)],
    [(2, 5), (3, 4), (5, 8), (0, 9), (1, 1), (4, 7)],
    [(1, 5), (0, 5), (2, 5), (3, 3), (4, 8), (5, 9)],
    [(2, 9), (1, 3), (4, 5), (5, 4), (0, 3), (3, 1)],
    [(1, 3), (3, 3), (5, 9), (0, 10), (4, 4), (2, 1)],
];
const OPTIMUM: f64 = 55.0;

// the start time of each job's operations in the semi-active schedule of `order`
fn schedule(order: &Order) -> [[u32; MACHINES]; JOBS] {
    let mut starts = [[0; MACHINES]; JOBS];
    let mut next = [0; JOBS];
    let mut job_free = [0; JOBS];
    let mut machine_free = [0; MACHINES];
    for &operation in order.iter() {
        let job = operation / MACHINES;
        let (machine, duration) = INSTANCE[job][next[job]];
        let start = job_free[job].max(machine_free[machine]);
        starts[job][next[job]] = start;
        next[job] += 1;
        job_free[job] = start + duration;
        machine_free[machine] = start + duration;
    }
    starts
}

// when the last operation ends
fn makespan(order: &Order) -> f64 {
    let starts = schedule(order);
    let last = MACHINES - 1;
    let end = |job: usize| starts[job][last] + INSTANCE[job][last].1;
    f64::from((0..JOBS).map(end).max().unwrap_or(0))
}

fn main() -> Result<()> {
    let ga = Ga::builder(Permutation::new(JOBS * MACHINES)?)
        .population_size(100)
        .select(Tournament::new(3)?)
        .crossover(OrderCrossover)
        .mutate(SwapMutation::new())
        .minimize()
        .seed(1)
        .build()?;

    let outcome = Engine::new(ga, makespan)
        .stop_when(Stop::target(OPTIMUM).or(Stop::generations(1_000)))
        .run()?;

    println!(
        "makespan {} after {} generations (the optimum: {OPTIMUM})",
        outcome.best_fitness(),
        outcome.generations()
    );
    // each machine's jobs, in the order it does them
    let starts = schedule(outcome.best_genome());
    for machine in 0..MACHINES {
        let mut jobs: Vec<(u32, usize)> = Vec::new();
        for (job, operations) in INSTANCE.iter().enumerate() {
            for (step, &(on, _)) in operations.iter().enumerate() {
                if on == machine {
                    jobs.push((starts[job][step], job));
                }
            }
        }
        jobs.sort_unstable();
        let jobs: Vec<usize> = jobs.into_iter().map(|(_, job)| job).collect();
        println!("machine {machine}: jobs {jobs:?}");
    }
    Ok(())
}
