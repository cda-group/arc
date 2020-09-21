pub use clap::Clap;
use derive_more::Constructor;
use serde::Deserialize;
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Clap, Debug)]
pub struct Opt {
    /// Activate DEBUG mode
    #[clap(short, long)]
    pub debug: bool,

    /// Emit MLIR to stdout
    #[clap(short, long)]
    pub mlir: bool,

    /// Only run up until typechecking
    #[clap(short, long)]
    pub check: bool,

    /// Print AST with type information and parentheses
    #[clap(short, long)]
    pub verbose: bool,

    /// Data interface to the outside world
    #[clap(short = 'C', parse(try_from_str = serde_json::from_str), number_of_values = 1)]
    pub connectors: Vec<Connector>,

    #[clap(subcommand)]
    pub subcmd: SubCmd,
}

#[derive(Deserialize, Debug, Constructor)]
pub struct Connector {
    endpoint: Endpoint,
    name: String,
    provider: Provider,
}

#[derive(Deserialize, Debug)]
pub enum Endpoint {
    Source,
    Sink,
}

#[derive(Deserialize, Debug)]
pub enum Provider {
    Socket(SocketAddr),
    File(PathBuf),
}

#[derive(Clap, Debug)]
pub enum SubCmd {
    /// Run in REPL mode
    #[cfg(feature = "repl")]
    #[clap(name = "repl")]
    Repl,

    /// Run in LSP mode
    #[cfg(feature = "lsp")]
    #[clap(name = "lsp")]
    Lsp,

    /// Compile source code
    #[clap(name = "code")]
    Code(Code),

    /// Compile source file
    #[clap(name = "file")]
    File(File),

    #[clap(name = "")]
    Lib,
}

#[derive(clap::Clap, Debug)]
pub struct Code {
    /// String of source code
    pub code: String,
}

#[derive(clap::Clap, Debug)]
pub struct File {
    /// Path to file
    #[clap(parse(from_os_str))]
    pub path: PathBuf,
}
