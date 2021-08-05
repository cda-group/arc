//! Functions for comparing types for alpha-equivalence.

use crate::hir;
use crate::info;

use std::collections::hash_map::DefaultHasher;
use std::hash::Hash;
use std::hash::Hasher;

use arc_script_compiler_shared::From;
use arc_script_compiler_shared::Into;

struct Context<'i> {
    hir: &'i hir::HIR,
    info: &'i info::Info,
    hasher: &'i mut DefaultHasher,
}

#[derive(From, Into, Eq, PartialEq)]
pub(crate) struct TypeHash(u64);

impl hir::Type {
    fn hash(self, ctx: &mut Context<'_>) -> TypeHash {
        let mut hasher = DefaultHasher::new();
        hasher.write(format!("{}", ctx.hir.pretty(&self, ctx.info)).as_bytes());
        hasher.finish().into()
    }
}
