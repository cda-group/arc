#![allow(unused)]
pub mod declarations;

use ast::*;
use im_rc::vector;
use im_rc::Vector;
use info::Info;

pub const MLIR_PRELUDE: &str = include_str!("prelude.mlir");

pub use declarations::prelude;
