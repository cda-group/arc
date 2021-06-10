use crate::compiler::hir;
use crate::compiler::hir::infer::unify::Unify;

use super::Context;

impl Context<'_> {
    fn constrain(&mut self, t: hir::Type, c: hir::ConstraintKind) {
        self
    }
}

