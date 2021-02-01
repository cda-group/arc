mod lower;

use crate::compiler::dfg::DFG;
use crate::compiler::hir;
use crate::compiler::hir::HIR;
use crate::compiler::info::Info;
use crate::compiler::rust::Rust;
use crate::compiler::shared::{Lower, Map, New};

impl Rust {
    pub(crate) fn from(hir: &HIR, dfg: &DFG, info: &Info) -> Self {
        let mut rust = Rust::default();
        let buf = String::default();
        let mangled = Map::default();
        let ctx = &mut lower::Context::new(info, &mut rust, buf, mangled);
        hir.lower(ctx);
        rust
    }
}
