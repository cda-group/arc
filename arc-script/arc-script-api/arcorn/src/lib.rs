/// State abstraction
/// Enum macros
pub mod enums;

pub mod backends {
    #[cfg(feature = "backend_arcon")]
    pub mod arcon {
        pub mod operators {
            pub mod convert;
        }
        pub mod state;
    }
    #[cfg(feature = "backend_arctime")]
    pub mod arctime { }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "backend_arcon")] {
        pub use backends::arcon::*;
        pub use arcon;
        pub use prost;
        pub use arcon_macros;
    } else if #[cfg(feature = "backend_arctime")] {
        pub use backends::arctime::*;
        pub use arctime;
    }
}

pub use arcorn_macros::rewrite;

/// Exports
pub use derive_more;

pub use types::*;

pub use paste::paste;
pub use half::bf16;
pub use half::f16;
