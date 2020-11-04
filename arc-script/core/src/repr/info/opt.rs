use crate::repr::info::connector::Connector;
use std::path::PathBuf;

#[cfg(feature = "cli")]
pub use clap::Clap;

#[cfg_attr(feature = "cli", derive(Clap))]
#[derive(Debug)]
pub struct Opt {
    /// Activate DEBUG mode
    #[cfg_attr(feature = "cli", clap(short, long))]
    pub debug: bool,

    /// Emit MLIR to stdout
    #[cfg_attr(feature = "cli", clap(short, long))]
    pub mlir: bool,

    /// Only run up until typechecking
    #[cfg_attr(feature = "cli", clap(short, long))]
    pub check: bool,

    /// Print AST with type information and parentheses
    #[cfg_attr(feature = "cli", clap(short, long))]
    pub verbose: bool,

    /// Data interface to the outside world
    #[cfg_attr(feature = "cli", clap(short = 'C', parse(try_from_str = serde_json::from_str), number_of_values = 1))]
    pub connectors: Vec<Connector>,

    #[cfg_attr(feature = "cli", clap(subcommand))]
    pub subcmd: SubCmd,
}

#[cfg_attr(feature = "cli", derive(Clap))]
#[derive(Debug)]
pub enum SubCmd {
    /// Run in REPL mode
    #[cfg(feature = "repl")]
    #[cfg_attr(feature = "cli", clap(name = "repl"))]
    Repl,

    /// Run in LSP mode
    #[cfg(feature = "lsp")]
    #[cfg_attr(feature = "cli", clap(name = "lsp"))]
    Lsp,

    /// Compile source code
    #[cfg_attr(feature = "cli", clap(name = "code"))]
    Code(Code),

    /// Compile source file
    #[cfg_attr(feature = "cli", clap(name = "file"))]
    File(File),

    #[cfg_attr(feature = "cli", clap(name = ""))]
    Lib,
}

#[cfg_attr(feature = "cli", derive(Clap))]
#[derive(Debug)]
pub struct Code {
    /// String of source code
    pub code: String,
}

#[cfg_attr(feature = "cli", derive(Clap))]
#[derive(Debug)]
pub struct File {
    /// Path to file
    #[cfg_attr(feature = "cli", clap(parse(from_os_str)))]
    pub path: PathBuf,
}

impl Default for Opt {
    fn default() -> Self {
        Self {
            debug: false,
            mlir: false,
            check: true,
            verbose: false,
            connectors: Vec::new(),
            subcmd: SubCmd::Lib,
        }
    }
}
