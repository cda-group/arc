use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::info::paths::PathId;
use crate::compiler::info::types::TypeId;
use crate::compiler::info::Info;

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
        hir::Type::new(hir::TypeKind::Unknown)
    }
}

impl From<hir::TypeKind> for hir::Type {
    fn from(kind: hir::TypeKind) -> Self {
        Self::new(kind)
    }
}

impl Into<hir::Type> for hir::ScalarKind {
    fn into(self) -> hir::Type {
        hir::TypeKind::Scalar(self).into()
    }
}

impl Into<hir::TypeKind> for hir::ScalarKind {
    fn into(self) -> hir::TypeKind {
        hir::TypeKind::Scalar(self)
    }
}

impl From<ast::Path> for hir::Path {
    fn from(path: ast::Path) -> hir::Path {
        hir::Path::new(path.id, path.loc)
    }
}

impl From<PathId> for hir::Path {
    fn from(id: PathId) -> hir::Path {
        hir::Path::new(id, None)
    }
}
