#[cfg(feature = "backend_local")]
pub use arc_script_codegen_local as backend;

#[cfg(feature = "backend_arcon")]
pub use arc_script_codegen_arcon as backend;

pub use arc_script_codegen_shared as shared;

pub use backend::rewrite;

pub use backend::prelude::Stream;

pub use shared::paste::paste;
pub use shared::*;
