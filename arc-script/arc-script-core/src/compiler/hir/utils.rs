use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::info::paths::PathId;
use crate::compiler::info::types::TypeId;

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

impl From<Vec<(hir::Name, TypeId)>> for hir::Type {
    fn from(vec: Vec<(hir::Name, TypeId)>) -> Self {
        hir::TypeKind::Struct(vec.into_iter().collect()).into()
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
