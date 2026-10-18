//! Evaluate a saved annotated JPEG corpus and emit strict machine-readable metrics.

use std::{env, fs, path::PathBuf, process::ExitCode};

/// Shared immutable fixture loader and scorer used by the integration tests.
#[path = "../tests/support/mod.rs"]
pub mod scene_evaluation;

use scene_evaluation::{
    EvaluationOptions, EvaluationResult, default_manifest_path, evaluate_manifest,
};

/// Runs the batch evaluator, distinguishing scoring failures from invalid inputs.
fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("scene evaluation failed: {error}");
            ExitCode::from(2)
        }
    }
}

/// Parses selection flags, writes JSON even for failed scoring, and returns success.
///
/// # Errors
///
/// Returns an error for invalid arguments, invalid fixtures, or an unwritable output.
fn run() -> EvaluationResult<bool> {
    let mut manifest = default_manifest_path();
    let mut output = None;
    let mut options = EvaluationOptions::default();
    let mut arguments = env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--manifest" => {
                manifest = PathBuf::from(arguments.next().ok_or("--manifest needs a path")?)
            }
            "--output" => {
                output = Some(PathBuf::from(
                    arguments.next().ok_or("--output needs a path")?,
                ))
            }
            "--include-challenges" => options.include_challenges = true,
            "--scene" => {
                options.scene_id = Some(arguments.next().ok_or("--scene needs an identifier")?)
            }
            "--help" | "-h" => {
                println!(
                    "Usage: evaluate_scenes [--manifest PATH] [--output PATH] [--include-challenges] [--scene ID]\nDefaults to the permanent regression corpus. Exit status: 0 pass, 1 scored failures, 2 invalid input/I/O."
                );
                return Ok(true);
            }
            _ => return Err(format!("unknown argument: {argument}").into()),
        }
    }
    let report = evaluate_manifest(&manifest, &options)?;
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(output) = output {
        fs::write(output, format!("{json}\n"))?;
    } else {
        println!("{json}");
    }
    Ok(report.summary.failed == 0)
}
