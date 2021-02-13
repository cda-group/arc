use educe::Educe;

/// Compiler settings.
#[derive(Debug, Clone, Default)]
pub struct Mode {
    /// Activates debug mode when true. Debug mode will cause the compiler to emit internal messages
    /// that are useful for pinning down and squashing bugs within the compiler.
    pub debug: bool,
    /// If true, activate profiling mode. Profiling mode measures and emits the execution time of each
    /// compilation phase.
    pub profile: bool,
    /// Configures the verbosity level of the compiler. The levels correspond to [`tracing::Level`]
    /// and are:
    ///
    /// 5 (Trace) - Designates very low priority, often extremely verbose, information.
    /// 4 (Debug) - Designates lower priority information.
    /// 3 (Info)  - Designates useful information.
    /// 2 (Warn)  - Designates hazardous situations.
    /// 1 (Error) - Designates very serious errors.
    pub verbosity: i32,
    /// Suppresses diagnostics from being printed. Can be useful when using the compiler as a library.
    pub suppress_diags: bool,
    /// Fails compilation as soon as a compilation phase emits an error.
    pub fail_fast: bool,
    /// Forces the compiler to emit its internal representation (e.g., AST) even if it has errors.
    pub force_output: bool,
    /// Configures what kind of input the compiler reads.
    pub input: Input,
    /// Configures what kind of output the compiler writes.
    pub output: Output,
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
    /// Emit [`crate::compiler::ast::AST`] as output (by first parsing it from source code).
    AST,
    /// Emit [`crate::compiler::hir::HIR`] as output (by first lowering the `AST` into it).
    HIR,
    /// Emit [`crate::compiler::dfg::DFG`] as output (by first evaluating the `HIR` into it).
    DFG,
    /// Emit [`crate::compiler::rust::Rust`] as output (by first lowering the `HIR` and `DFG` into it).
    Rust,
    /// Emit [`crate::compiler::mlir::MLIR`] as output (by first lowering the `HIR` and `DFG` into it).
    #[educe(Default)]
    MLIR,
    /// Emit no output.
    Silent,
}
