//! Types for deriving a Clap command-line argument parser.

pub use clap::ArgEnum;
pub use clap::Clap;

use std::path::PathBuf;

/// Specification of the Arc-script command-line interface (CLI).
#[derive(Clap, Debug, Clone)]
pub struct Opt {
    /// Set LANGUAGE mode.
    #[clap(
        long,
        arg_enum,
        case_insensitive(true),
        value_name = "LANG",
        default_value("arc")
    )]
    pub lang: Lang,

    /// Activate DEBUG mode.
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

    /// Print AST with type information and parentheses.
    #[clap(short, long, parse(from_occurrences))]
    pub verbosity: u8,

    /// Print result even if there are errors.
    #[clap(long)]
    pub force_output: bool,

    /// Skip type inference pass.
    #[clap(long)]
    pub no_infer: bool,

    /// Do not include prelude.
    #[clap(long)]
    pub no_prelude: bool,

    /// Sub-command.
    #[clap(subcommand)]
    pub subcmd: SubCmd,
}

/// Sub-commands of the CLI.
#[derive(Clap, Debug, Clone)]
pub enum SubCmd {
    /// Run in REPL mode.
    #[cfg(feature = "repl")]
    #[cfg_attr(feature = "repl", clap(name = "repl"))]
    Repl,

    /// Run in LSP mode.
    #[cfg(feature = "lsp")]
    #[cfg_attr(feature = "lsp", clap(name = "lsp"))]
    Lsp,

    /// Compile and execute source file.
    #[clap(name = "run")]
    Run(Run),

    /// Generate command-line completions.
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
    #[clap(
        long,
        short,
        arg_enum,
        case_insensitive(true),
        value_name = "FORMAT",
        default_value("rust")
    )]
    pub output: Output,
}

/// Mode which selects which language to use.
#[derive(ArgEnum, Debug, Clone, Copy)]
pub enum Lang {
    /// Arc-Query Language. A higher-level language which translates into Arc-Script.
    Arq,
    /// Arc-Script Language. A dataflow-language which translates into Arcon.
    Arc,
}

/// An output mode.
#[derive(ArgEnum, Debug, Clone)]
#[allow(clippy::upper_case_acronyms)]
pub enum Output {
    /// Output AST.
    AST,
    /// Output HIR.
    HIR,
    /// Output Rust.
    Rust,
    /// Output Rust via MLIR.
    RustMLIR,
    /// Output MLIR (Default).
    MLIR,
}

/// Configuration parameters for the `completions` command.
#[derive(Clap, Debug, Clone)]
pub struct Completions {
    /// Shell to generate completions for.
    #[clap(long, arg_enum, case_insensitive(true), value_name = "SHELL")]
    pub shell: Shell,
}

/// Different types of shells.
#[allow(clippy::enum_variant_names)]
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
