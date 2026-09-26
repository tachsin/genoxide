//! genoxide's errors about settings, with the names the Python package gives the settings: a
//! wrong `Tournament(0)` is `Tournament.size`, not genoxide's `tournament_size`.

use genoxide::Error;

type Result<T> = std::result::Result<T, String>;

/// The result of a genoxide constructor or builder, with its error as a message that names the
/// Python setting.
pub fn setting<T>(result: genoxide::Result<T>) -> Result<T> {
    result.map_err(|error| message(error, None))
}

/// [`setting`], for the constructor of a genome: its `len` and `bounds` are `length` and `bounds`
/// of `genome`, e.g. `Binary.length`.
pub fn genome_setting<T>(result: genoxide::Result<T>, genome: &str) -> Result<T> {
    result.map_err(|error| message(error, Some(genome)))
}

fn message(error: Error, genome: Option<&str>) -> String {
    match error {
        Error::MissingSetting { setting } => format!("`{}` is needed", python(setting, genome)),
        Error::InvalidSetting { setting, reason } => {
            // the schemes' `lambda` is `offspring` in Python: MuPlusLambda(offspring)
            let reason = if setting == "scheme" {
                reason.replace("lambda", "offspring")
            } else {
                reason
            };
            format!("invalid setting `{}`: {reason}", python(setting, genome))
        }
        error => error.to_string(),
    }
}

// the Python name of a genoxide setting; the settings with the same name in both, e.g.
// `population_size`, and those Python can't set, keep their name
fn python(setting: &str, genome: Option<&str>) -> String {
    match (setting, genome) {
        ("len", Some(genome)) => format!("{genome}.length"),
        ("bounds", Some(genome)) => format!("{genome}.bounds"),
        (setting, _) => python_name(setting).to_string(),
    }
}

fn python_name(setting: &str) -> &str {
    match setting {
        // CMA-ES without a gene that has more than one value
        "real" => "Real.bounds",
        "tournament_size" => "Tournament.size",
        "rank_pressure" => "Rank.pressure",
        "truncation_fraction" => "Truncation.fraction",
        "crossover_points" => "PointCrossover.points",
        "sbx_eta" => "SimulatedBinaryCrossover.eta",
        "blend_alpha" => "BlendCrossover.alpha",
        "bit_flip_rate" => "BitFlip.rate",
        "bit_flip_count" => "BitFlip.count",
        "uniform_mutation_rate" => "UniformMutation.rate",
        "uniform_mutation_count" => "UniformMutation.count",
        "gaussian_mutation_rate" => "GaussianMutation.rate",
        "gaussian_mutation_count" => "GaussianMutation.count",
        "gaussian_sigma" => "GaussianMutation.sigma",
        "polynomial_mutation_rate" => "PolynomialMutation.rate",
        "polynomial_mutation_count" => "PolynomialMutation.count",
        "polynomial_eta" => "PolynomialMutation.eta",
        "swap_count" => "SwapMutation.count",
        "initial_temperature" => "Annealing.initial_temperature",
        "cooling" => "Annealing.cooling",
        "tenure" => "Tabu.tenure",
        "theta" => "Pbi.theta",
        // Pso's ring is genoxide's ring topology
        "topology" => "ring",
        // De's l_shade is genoxide's linear population size reduction
        "linear_reduction" => "l_shade",
        setting => setting,
    }
}
