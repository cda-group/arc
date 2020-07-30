use std::path::PathBuf;

/// A basic example
#[cfg_attr(feature = "cli", derive(clap::Clap, Debug))]
pub struct Opt {
    /// Activate DEBUG mode
    #[cfg_attr(feature = "cli", clap(short, long))]
    pub debug: bool,

    /// Emit mlir
    #[cfg_attr(feature = "cli", clap(short, long))]
    pub mlir: bool,

    /// Print AST with type information and parentheses
    #[cfg_attr(feature = "cli", clap(short, long))]
    pub verbose: bool,

    #[cfg_attr(feature = "cli", clap(subcommand))]
    pub subcmd: SubCmd,
}

impl Opt {
    #[cfg(feature = "cli")]
    pub fn get() -> Opt { <Opt as ::clap::Clap>::parse() }

    #[cfg(not(feature = "cli"))]
    pub fn get() -> Opt {
        if cfg!(feature = "lsp") {
            Opt {
                debug: false,
                mlir: false,
                verbose: false,
                subcmd: SubCmd::Lsp,
            }
        } else if cfg!(feature = "repl") {
            Opt {
                debug: false,
                mlir: false,
                verbose: false,
                subcmd: SubCmd::Repl,
            }
        } else {
            let mut args = std::env::args().collect::<Vec<String>>().into_iter();
            match (args.nth(1), args.next()) {
                (Some(arg), None) => Opt {
                    debug: true,
                    mlir: true,
                    verbose: false,
                    subcmd: SubCmd::Code(Code { code: arg }),
                },
                (Some(_), Some(_)) => panic!("Found multiple arguments"),
                (..) => panic!("Expected source string literal"),
            }
        }
    }
}

#[cfg_attr(feature = "cli", derive(clap::Clap, Debug))]
pub enum SubCmd {
    /// Run in REPL mode
    #[cfg_attr(feature = "cli", clap(name = "repl"))]
    Repl,

    /// Run in LSP mode
    #[cfg_attr(feature = "cli", clap(name = "lsp"))]
    Lsp,

    /// Compile source code
    #[cfg_attr(feature = "cli", clap(name = "code"))]
    Code(Code),

    /// Compile source file
    #[cfg_attr(feature = "cli", clap(name = "file"))]
    File(File),

    /// Compile source file
    #[cfg_attr(feature = "cli", clap(name = "file"))]
    Bench,

    /// Run as a library (Not CLI)
    Lib,
}

#[cfg_attr(feature = "cli", derive(clap::Clap, Debug))]
pub struct Code {
    /// String of source code
    pub code: String,
}

#[cfg_attr(feature = "cli", derive(clap::Clap, Debug))]
pub struct File {
    /// Path to file
    #[cfg_attr(feature = "cli", clap(parse(from_os_str)))]
    pub path: PathBuf,
}
