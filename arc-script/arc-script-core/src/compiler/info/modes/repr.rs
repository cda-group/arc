/// Compiler settings.
#[derive(Debug, Clone, Default)]
pub struct Mode {
    pub debug: bool,
    pub verbosity: i32,
    pub suppress_diags: bool,
    pub fail_fast: bool,
    pub force_output: bool,
    pub input: Input,
    pub output: Output,
}

#[derive(Debug, Clone)]
pub enum Input {
    /// Process source string.
    Code(String),
    /// Process input file or root directory.
    #[cfg(not(target_arch = "wasm32"))]
    File(Option<std::path::PathBuf>),
    /// Input will be added later.
    Empty,
}

impl Default for Input {
    fn default() -> Self {
        Self::File(None)
    }
}

/// Run until
#[derive(Debug, Clone, Copy)]
pub enum Output {
    AST,
    HIR,
    DFG,
    Rust,
    MLIR,
    Silent,
}

impl Default for Output {
    fn default() -> Self {
        Self::MLIR
    }
}
