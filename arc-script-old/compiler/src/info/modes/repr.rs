use arc_script_compiler_shared::Educe;

/// Compiler settings.
#[derive(Debug, Clone, Default)]
pub struct Mode {
    /// Activates debug mode when true. Debug mode will cause the compiler to emit internal messages
    /// that are useful for pinning down and squashing bugs within the compiler.
    pub debug: bool,
    /// If true, activate profiling mode. Profiling mode measures and emits the execution time of each
    /// compilation phase.
    pub profile: bool,
    /// Configures the verbosity level of the compiler. The levels correspond to [`tracing::Level`].
    pub verbosity: Verbosity,
    /// Suppresses diagnostics from being printed. Can be useful when using the compiler as a library.
    pub suppress_diags: bool,
    /// Fails compilation as soon as a compilation phase emits an error.
    pub fail_fast: bool,
    /// Skip type inference pass.
    pub no_infer: bool,
    /// Do not include prelude.
    pub no_prelude: bool,
    /// Forces the compiler to emit its internal representation (e.g., AST) even if it has errors.
    pub force_output: bool,
    /// Configures what kind of language the compiler expects its input source code to be written in.
    pub lang: Lang,
    /// Configures what kind of input the compiler reads.
    pub input: Input,
    /// Configures what kind of output the compiler writes.
    pub output: Output,
}

#[derive(Debug, Clone)]
/// Language modes.
pub enum Lang {
    /// Arc-Query
    Arq,
    /// Arc-Script
    Arc,
}

impl Default for Lang {
    fn default() -> Self {
        Self::Arc
    }
}

/// Verbosity levels
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum Verbosity {
    /// Designates very serious errors.
    Error,
    /// Designates hazardous situations.
    Warn,
    /// Designates useful information.
    Info,
    /// Designates lower priority information.
    Debug,
    /// Designates very low priority information.
    Trace,
}

impl Default for Verbosity {
    fn default() -> Self {
        Self::Error
    }
}

/// The compiler can read input from different sources.
#[derive(Debug, Clone, Educe)]
#[educe(Default)]
pub enum Input {
    /// Read code directly from a source string.
    Code(String),
    /// Read input from a source file or root directory.
    #[cfg(not(target_arch = "wasm32"))]
    File(Option<std::path::PathBuf>),
    /// Input will be added later.
    #[educe(Default)]
    Empty,
}

/// The compiler can write
#[derive(Debug, Clone, Copy, Educe)]
#[educe(Default)]
pub enum Output {
    /// Emit [`crate::ast::AST`] as output (by first parsing it from source code).
    AST,
    /// Emit [`crate::hir::HIR`] as output (by first lowering the `AST` into it).
    HIR,
    /// Emit [`crate::rust::Rust`] as output (by first lowering the `HIR` into it).
    Rust,
    /// Emit Rust as output via the MLIR route.
    RustMLIR,
    /// Emit [`crate::mlir::MLIR`] as output (by first lowering the `HIR` into it).
    #[educe(Default)]
    MLIR,
    /// Emit no output.
    Silent,
}

/// Consult environment variables to determine the backend to use for
/// rust output.
#[must_use]
pub fn get_rust_backend() -> Output {
    let use_mlir_backend = match std::env::var("ARCSCRIPT_MLIR_BACKEND") {
        Ok(val) => val.parse().unwrap_or(0) != 0,
        Err(e) => false,
    };

    if use_mlir_backend {
        Output::RustMLIR
    } else {
        Output::Rust
    }
}
