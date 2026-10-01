import site from "@/genoxide-site.json";

/**
 * genoxide's static facts: names, links and the landing page's copy.
 * Everything that changes with the repository (examples, versions) is
 * fetched instead, see examples.js and versions.js.
 */

export const GENOXIDE_REPO = "tachsin/genoxide";
export const GENOXIDE_BRANCH = "main";

/**
 * The commit these pages were synced from (the app's genoxide-site.json).
 * The examples are read at it too, so the pages and the files they expect
 * always come from the same commit, and a new pin is new URLs: nothing
 * cached from an older commit is ever served for it.
 */
export const GENOXIDE_COMMIT = site.commit;

export const GENOXIDE_PATH = "/projects/genoxide";
export const GENOXIDE_OG_IMAGE = "/projects/genoxide/opengraph-image";

export const GENOXIDE_TAGLINE = "Optimization for Rust and Python";

export const GENOXIDE_DESCRIPTION =
  "genoxide is an evolutionary computation library for Rust, with a Python package: genetic algorithms, evolution strategies, CMA-ES, differential evolution, particle swarms, local search and multi-objective optimization in one library.";

export const GENOXIDE_LICENSE = "MIT OR Apache-2.0";

export const GENOXIDE_LINKS = {
  github: "https://github.com/tachsin/genoxide",
  crates: "https://crates.io/crates/genoxide",
  pypi: "https://pypi.org/project/genoxide/",
  docsRs: "https://docs.rs/genoxide",
  pythonApi: "https://tachsin.github.io/genoxide/api/python/",
  docsSite: "https://tachsin.github.io/genoxide/",
  features: "https://github.com/tachsin/genoxide/blob/main/docs/features.md",
  cli: "https://github.com/tachsin/genoxide/blob/main/docs/cli.md",
  roadmap: "https://github.com/tachsin/genoxide/blob/main/ROADMAP.md",
  changelog: "https://github.com/tachsin/genoxide/blob/main/CHANGELOG.md",
  examplesDir: "https://github.com/tachsin/genoxide/tree/main/examples",
  benchmarksDir: "https://github.com/tachsin/genoxide/tree/main/docs/benchmarks",
  benchmarkMethodology: "https://github.com/tachsin/genoxide/blob/main/benchmarks/README.md",
  benchmarkRules: "https://github.com/tachsin/genoxide/blob/main/docs/benchmarks/rules.md",
  benchmarkNotes: "https://github.com/tachsin/genoxide/blob/main/docs/benchmarks/notes.md",
  benchmarkResults: "https://github.com/tachsin/genoxide/blob/main/docs/benchmarks/results.md",
  benchmarkLibraries: "https://github.com/tachsin/genoxide/tree/main/docs/benchmarks/libraries",
};

export const GENOXIDE_KEYWORDS = [
  "genoxide",
  "genetic algorithm",
  "evolutionary computation",
  "evolution strategies",
  "CMA-ES",
  "differential evolution",
  "particle swarm optimization",
  "NSGA-II",
  "NSGA-III",
  "multi-objective optimization",
  "metaheuristics",
  "Rust",
  "Python",
];

/** Landing page highlights, from docs/features.md. `icon` is a key of HIGHLIGHT_ICONS in the page. */
export const GENOXIDE_HIGHLIGHTS = [
  {
    icon: "genomes",
    title: "Genomes",
    body: "Binary (bit-packed), integer and real bounded per gene, permutation, and real with a self-adaptive step size.",
  },
  {
    icon: "ga",
    title: "Genetic algorithms",
    body: "Generational with elitism, steady-state, (μ+λ), (μ,λ) and memetic schemes, with the classic selection, crossover and mutation operators.",
  },
  {
    icon: "es",
    title: "ES and CMA-ES",
    body: "(μ/ρ +, λ) evolution strategies with self-adapted step sizes, and CMA-ES with IPOP and BIPOP restarts and sep-CMA-ES for high dimensions.",
  },
  {
    icon: "de",
    title: "Differential evolution and PSO",
    body: "rand/1, best/1 and current-to-pbest/1 with fixed, dithered or adaptive parameters (JADE, SHADE, L-SHADE), and particle swarms with global or ring topology.",
  },
  {
    icon: "local",
    title: "Local search",
    body: "Hill climbing, simulated annealing, tabu search and iterated local search, with any mutation as the neighborhood.",
  },
  {
    icon: "multi",
    title: "Multi-objective",
    body: "NSGA-II, NSGA-III, SPEA2, MOEA/D and SMS-EMOA, with constraints, a Pareto archive, and hypervolume, IGD, IGD+, GD and spread.",
  },
  {
    icon: "engine",
    title: "Engine",
    body: "Parallel, batch and asynchronous evaluation, island models, constraints, stop conditions, checkpoints and observers.",
  },
  {
    icon: "python",
    title: "Python and the command line",
    body: "pip install genoxide for numpy genomes and vectorized fitness functions, and a genoxide program for fitness functions in any language.",
  },
];

/** The properties table of docs/features.md. */
export const GENOXIDE_PROPERTIES = [
  {
    title: "Checked settings",
    body: "Invalid settings are errors from build(), before anything runs. An operator that doesn't fit the genome is a compile error.",
  },
  {
    title: "Reproducible",
    body: "The same seed gives the same result, on any number of threads and on 32- or 64-bit machines.",
  },
  {
    title: "Safe",
    body: "#![forbid(unsafe_code)]: memory safety from the compiler, and parallel code without data races.",
  },
  {
    title: "Fast",
    body: "Compiled Rust, bit-packed binary genomes, and parallel or batch evaluation.",
  },
];

/** The first example on the landing page: OneMax, as in the repository's README. */
export const GENOXIDE_FIRST_EXAMPLE = {
  rust: `use genoxide::prelude::*;

fn main() -> genoxide::Result<()> {
    // OneMax: find the 100-bit string with the most ones
    let ga = Ga::builder(Binary::new(100)?)
        .population_size(100)
        .select(Tournament::new(3)?)
        .crossover(UniformCrossover::new())
        .mutate(BitFlip::per_gene(0.01)?)
        .seed(42)
        .build()?;

    let outcome = Engine::new(ga, |genome: &Bits| genome.count_ones() as f64)
        .stop_when(Stop::target(100.0).or(Stop::generations(1_000)))
        .run()?;

    println!("best: {} after {} generations", outcome.best_fitness(), outcome.generations());
    Ok(())
}`,
  python: `import genoxide as gx

# OneMax: find the 100-bit string with the most ones
ga = gx.Ga(
    gx.Binary(100),
    population_size=100,
    select=gx.Tournament(3),
    crossover=gx.UniformCrossover(),
    mutation=gx.BitFlip(rate=0.01),
    seed=42,
)
result = ga.run(lambda bits: bits.sum(), target=100, generations=1_000)
print(result.best_fitness, result.generations)`,
};
