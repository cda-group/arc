#![feature(unboxed_closures)]

// #[macro_use(match_cfg)]
// extern crate match_cfg;

// match_cfg::match_cfg! {
//     #[cfg(feature = "backend_arcon")] => {
//         pub mod backends {
//             pub mod arcon {
//                 pub mod operators {
//                     pub mod convert;
//                 }
//                 pub mod state;
//             }
//         }
//         pub use backends::arcon::*;
//         pub use arcon;
//         pub use prost;
//         pub use arcon_macros;
//         pub use arcon::prelude::Stream;
//     }
//     #[cfg(feature = "backend_arctime")] => {
pub mod backends {
    pub mod arctime {}
}
pub use arctime;
pub use arctime::prelude::Stream;
pub use backends::arctime::*;
//     }
//     _ => {
//         compile_err!("No backend found");
//     }
// }

/// Enum macros
pub mod enums;

/// Derive macros for enums, structs, and tasks.
pub use arcorn_macros::rewrite;

/// Exports
pub use derive_more;

pub use paste::paste;
pub use half::bf16;
pub use half::f16;

/// Functions must be cloneable
use dyn_clone::DynClone;

pub trait ArcornFn<Args>: FnOnce<Args> + Send + DynClone + 'static {}
impl<Args, F> ArcornFn<Args> for F where F: FnOnce<Args> + Send + DynClone + 'static {}

dyn_clone::clone_trait_object!(<I, O> ArcornFn<I, Output=O>);
