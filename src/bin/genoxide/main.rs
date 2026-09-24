//! `genoxide`: runs an optimization described in a TOML or JSON file, with any program as the
//! fitness function. See `genoxide --help`.

mod builtin;
mod config;
mod operators;
mod process;
mod run;

use std::path::Path;
use std::process::ExitCode;

const HELP: &str = "\
genoxide: evolutionary optimization, described in a TOML or JSON run file

Usage:
  genoxide run <file> [--resume]   run it; the result goes to stdout as JSON, progress to stderr
  genoxide check <file>            check the run file without running it
  genoxide fitness [<name>]        a built-in fitness program, or the list of them
  genoxide --help | --version

The fitness program (`fitness.command` in the run file) runs once per worker, in the run file's
directory. It reads a genome per line on stdin, the genes separated by spaces (bits as 0 and 1),
and writes a line per genome on stdout: its objective values, then optionally a constraint
violation (0 when feasible), separated by spaces. `nan` marks a genome that can't be scored.

A minimal run file:

  [genome]
  type = \"real\"
  length = 10
  bounds = [-5.12, 5.12]

  [fitness]
  builtin = \"rastrigin\"          # or: command = [\"python3\", \"fitness.py\"]
  objectives = [\"minimize\"]

  [algorithm]
  type = \"cmaes\"

  [stop]
  target = 1e-6
  evaluations = 100000

See https://github.com/tachsin/genoxide/blob/main/docs/cli.md for every setting.";

fn main() -> ExitCode {
    let arguments: Vec<String> = std::env::args().skip(1).collect();
    match execute(&arguments) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("genoxide: {message}");
            ExitCode::FAILURE
        }
    }
}

fn execute(arguments: &[String]) -> Result<(), String> {
    let arguments: Vec<&str> = arguments.iter().map(String::as_str).collect();
    match arguments.as_slice() {
        [] | ["help" | "--help" | "-h"] => {
            println!("{HELP}");
            Ok(())
        }
        ["--version" | "-V"] => {
            println!("genoxide {}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        ["run", file, options @ ..] => {
            let resume = match options {
                [] => false,
                ["--resume"] => true,
                _ => return Err(format!("unknown options {options:?}; see genoxide --help")),
            };
            let path = Path::new(file);
            let result = run::run(
                load(path)?,
                path,
                run::Options {
                    resume,
                    check: false,
                },
            )?;
            println!("{}", format_result(&result));
            Ok(())
        }
        ["check", file] => {
            let path = Path::new(file);
            run::run(
                load(path)?,
                path,
                run::Options {
                    resume: false,
                    check: true,
                },
            )?;
            println!("{file}: ok");
            Ok(())
        }
        ["fitness"] => {
            println!(
                "Built-in fitness programs (genoxide fitness <name>, or `builtin = \"<name>\"`):"
            );
            for function in builtin::FUNCTIONS {
                println!("  {:<12} {}", function.name, function.description);
            }
            Ok(())
        }
        ["fitness", name] => builtin::serve(name),
        _ => Err(format!(
            "unknown command `{}`; see genoxide --help",
            arguments.join(" ")
        )),
    }
}

// reads a run file: JSON for `.json`, TOML otherwise
fn load(path: &Path) -> Result<config::Run, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("can't read {}: {error}", path.display()))?;
    let json = path
        .extension()
        .is_some_and(|extension| extension.eq_ignore_ascii_case("json"));
    let run = if json {
        serde_json::from_str(&text).map_err(|error| error.to_string())
    } else {
        toml::from_str(&text).map_err(|error| error.to_string())
    };
    run.map_err(|error| format!("{}: {error}", path.display()))
}

// a result: a field per line, and a line per member of a list of objects (the front)
fn format_result(result: &serde_json::Value) -> String {
    let serde_json::Value::Object(fields) = result else {
        return result.to_string();
    };
    let mut text = String::from(
        "{
",
    );
    for (index, (key, value)) in fields.iter().enumerate() {
        text.push_str(&format!("  {}: ", serde_json::Value::from(key.as_str())));
        match value {
            serde_json::Value::Array(items) if items.iter().all(serde_json::Value::is_object) => {
                text.push_str(
                    "[
",
                );
                for (index, item) in items.iter().enumerate() {
                    let comma = if index + 1 < items.len() { "," } else { "" };
                    text.push_str(&format!(
                        "    {item}{comma}
"
                    ));
                }
                text.push_str("  ]");
            }
            value => text.push_str(&value.to_string()),
        }
        text.push_str(if index + 1 < fields.len() {
            ",
"
        } else {
            "
"
        });
    }
    text.push('}');
    text
}
