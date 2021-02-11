//! Exports for users who use the compiler as a library.

pub use crate::compiler;
pub use crate::compiler::ast;
pub use crate::compiler::info::diags;
pub use crate::compiler::info::logger;
pub use crate::compiler::info::modes;

pub use anyhow::Result;
pub use half::bf16;
pub use half::f16;
