use derive_more::Constructor as New;
use derive_more::From;

use std::path::PathBuf;

pub use clap::{ArgEnum, Clap};
use strum::EnumString;

/// Specification of the Arc-script command-line interface (CLI).
#[derive(Clap, Debug, Clone, Default)]
pub struct Opt {
    /// Activate DEBUG mode
    #[clap(short, long)]
    pub debug: bool,

    /// Fail after the first pass which produces an error.
    #[clap(short, long)]
    pub fail_fast: bool,

    /// Mute all diagnostics messages.
    #[clap(short, long)]
    pub suppress_diags: bool,

    /// Print AST with type information and parentheses
    #[clap(short, long, parse(from_occurrences))]
    pub verbosity: i32,

    /// Sub-command
    #[clap(subcommand)]
    pub subcmd: SubCmd,
}

/// Sub-commands of the CLI.
#[derive(Clap, Debug, Clone, From)]
pub enum SubCmd {
    /// Run in REPL mode
    #[cfg(feature = "repl")]
    #[cfg_attr(feature = "repl", clap(name = "repl"))]
    Repl,

    /// Run in LSP mode
    #[cfg(feature = "lsp")]
    #[cfg_attr(feature = "lsp", clap(name = "lsp"))]
    Lsp,

    /// Compile and execute source file
    #[clap(name = "run")]
    Run(Run),
}

impl Default for SubCmd {
    fn default() -> Self {
        Self::Run(Run::default())
    }
}

/// Configuration parameters for the `run` subcommand.
#[derive(Clap, Default, Debug, Clone, New)]
pub struct Run {
    /// Path to main file
    #[clap(parse(from_os_str))]
    pub main: Option<PathBuf>,
    /// Output mode: [AST|HIR|DFG|Rust|MLIR]
    #[clap(long)]
    pub output: Output,
}

#[derive(ArgEnum, Debug, Clone, EnumString)]
pub enum Output {
    AST,
    HIR,
    DFG,
    Rust,
    MLIR,
}

impl Default for Output {
    fn default() -> Self {
        Self::MLIR
    }
}