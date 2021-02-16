use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::info::paths::PathId;
use crate::compiler::info::types::TypeId;

impl Default for hir::Expr {
    /// Returns an error-expression. This method is solely meant to be used for
    /// [`std::mem::replace`] to make in-place modifications to the HIR, for
    /// example in [`crate::passes::pruner`].
    fn default() -> Self {
        Self::new(hir::ExprKind::Err, TypeId::default(), None)
    }
}

impl Default for hir::Type {
    fn default() -> Self {
        Self::new(hir::TypeKind::Unknown)
    }
}

impl From<hir::TypeKind> for hir::Type {
    fn from(kind: hir::TypeKind) -> Self {
        Self::new(kind)
    }
}

impl From<hir::ScalarKind> for hir::Type {
    fn from(kind: hir::ScalarKind) -> Self {
        hir::TypeKind::Scalar(kind).into()
    }
}

impl From<hir::ScalarKind> for hir::TypeKind {
    fn from(kind: hir::ScalarKind) -> Self {
        Self::Scalar(kind)
    }
}

impl From<ast::Path> for hir::Path {
    fn from(path: ast::Path) -> Self {
        Self::new(path.id, path.loc)
    }
}

impl From<PathId> for hir::Path {
    fn from(id: PathId) -> Self {
        Self::new(id, None)
    }
}
