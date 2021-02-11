/// Module for representing diagnostics.
pub mod repr;
/// Module which exposes structs which diagnostics can be written to.
pub mod sink;
/// Module for transforming [`repr::Diagnostic`] into [`codespan_reporting::diagnostic::Diagnostic`].
pub mod to_codespan;

pub use repr::*;
pub use sink::*;
