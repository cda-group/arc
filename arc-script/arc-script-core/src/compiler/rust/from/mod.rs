mod lower;

use crate::compiler::hir::HIR;
use crate::compiler::info::Info;
use crate::compiler::rust::Rust;
use crate::compiler::shared::{Lower, Map};

impl Rust {
    pub(crate) fn from(hir: &HIR, info: &Info) -> Self {
        let mut rust = Self::default();
        let buf = String::default();
        let mangled = Map::default();
        let ctx = &mut lower::Context::new(info, hir, &mut rust, buf, mangled);
        hir.lower(ctx);
        rust
    }
}
