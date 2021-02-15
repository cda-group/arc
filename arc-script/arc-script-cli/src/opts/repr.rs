//! Types for deriving a Clap command-line argument parser.

pub use clap::ArgEnum;
pub use clap::Clap;

use std::path::PathBuf;

/// Specification of the Arc-script command-line interface (CLI).
#[derive(Clap, Debug, Clone)]
pub struct Opt {
    /// Activate DEBUG mode
    #[clap(short, long)]
    pub debug: bool,

    /// Activate PROFILING mode.
    #[clap(short, long)]
    pub profile: bool,

    /// Fail after the first pass which produces an error.
    #[clap(long)]
    pub fail_fast: bool,

    /// Mute all diagnostics messages.
    #[clap(short, long)]
    pub suppress_diags: bool,

    /// Print AST with type information and parentheses
    #[clap(short, long, parse(from_occurrences))]
    pub verbosity: i32,

    /// Print result even if there are errors.
    #[clap(long)]
    pub force_output: bool,

    /// Sub-command
    #[clap(subcommand)]
    pub subcmd: SubCmd,
}

/// Sub-commands of the CLI.
#[derive(Clap, Debug, Clone)]
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

    /// Generate command-line completions
    #[clap(name = "completions")]
    Completions(Completions),
}

/// Configuration parameters for the `run` subcommand.
#[derive(Clap, Debug, Clone)]
pub struct Run {
    /// Path to main file.
    #[clap(parse(from_os_str))]
    pub main: Option<PathBuf>,
    /// Select output mode.
    #[clap(long, short, arg_enum, value_name = "FORMAT")]
    pub output: Output,
}

/// An output mode.
#[derive(ArgEnum, Debug, Clone)]
pub enum Output {
    /// Output AST.
    AST,
    /// Output HIR.
    HIR,
    /// Output DFG.
    DFG,
    /// Output Rust.
    Rust,
    /// Output MLIR (Default).
    MLIR,
}

/// Configuration parameters for the `completions` command.
#[derive(Clap, Debug, Clone)]
pub struct Completions {
    /// Shell to generate completions for.
    #[clap(long, arg_enum, value_name = "SHELL")]
    pub shell: Shell,
}

/// Different types of shells.
#[derive(ArgEnum, Debug, Clone)]
pub enum Shell {
    /// Bourne Again Shell (Default).
    Bash,
    /// Z Shell.
    Zsh,
    /// Fish Shell.
    Fish,
    /// Elvish Shell.
    Elvish,
    /// Power Shell.
    PowerShell,
}
