use crate::hir;
use crate::hir::infer::unify::Unify;

use super::Context;

impl Context<'_> {
    fn constrain(&mut self, t: hir::Type, c: hir::ConstraintKind) {
        self
    }
}

