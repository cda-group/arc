use clap::Parser;
use std::ffi::OsString;
use std::path::PathBuf;

fn history() -> OsString {
    std::env::temp_dir()
        .join("arc-lang")
        .join("history.txt")
        .into_os_string()
}

#[derive(Debug, Clone, Parser)]
pub struct Config {
    /// Read source from file
    pub file: Option<PathBuf>,
    /// Loads file statement-by-statement into the REPL.
    #[clap(long)]
    pub interactive: bool,
    /// The history file to use
    #[clap(long, default_value = history())]
    pub history: PathBuf,
    /// Clear build caches
    #[clap(long)]
    pub clear_caches: bool,
    /// Print version
    #[clap(long)]
    pub version: bool,
    #[clap(flatten)]
    pub show: Show,
    /// Fail on first error
    #[clap(long)]
    pub failfast: bool,
}

#[derive(Debug, Default, Clone, Copy, Parser)]
pub struct Show {
    /// Show build caches
    #[clap(long = "show-caches")]
    pub caches: bool,
    /// Show IR after appending prelude
    #[clap(long = "show-prelude")]
    pub prelude: bool,
    /// Show IR after parsing
    #[clap(long = "show-parsed")]
    pub parsed: bool,
    /// Show IR after name resolution
    #[clap(long = "show-resolved")]
    pub resolved: bool,
    /// Show IR after type inference
    #[clap(long = "show-inferred")]
    pub inferred: bool,
    /// Show IR after pattern compilation
    #[clap(long = "show-patcomped")]
    pub patcomped: bool,
    /// Show IR after monomorphisation
    #[clap(long = "show-monomorphised")]
    pub monomorphised: bool,
    /// Show IR after interpretation
    #[clap(long = "show-optimised")]
    pub dataflow: bool,
    /// Show IR after reachability analysis
    #[clap(long = "show-reachable")]
    pub reachable: bool,
    /// Show Rust
    #[clap(long = "show-mlir")]
    pub mlir: bool,
    /// Show MLIR
    #[clap(long = "show-rust")]
    pub rust: bool,
    /// Show type annotations of expressions and patterns
    #[clap(long = "show-types")]
    pub types: bool,
    /// Show warnings
    #[clap(long = "show-warnings")]
    pub warnings: bool,
    /// Show hints
    #[clap(long = "show-hints")]
    pub hints: bool,
    /// Show backtrace of the location in the compiler where errors are created. Used for debugging
    #[clap(long = "show-backtrace")]
    pub backtrace: bool,
    /// Show colors
    #[clap(long = "show-colors", default_value_t = true)]
    pub colors: bool,
}
