use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::info::files::Loc;
use crate::compiler::info::paths::PathId;
use crate::compiler::info::types::TypeId;
use crate::compiler::info::Info;

use arc_script_core_shared::itertools::Itertools;
use arc_script_core_shared::VecMap;

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
        Self::new(id, Loc::Fake)
    }
}

pub(crate) trait SortFields: Sized {
    /// Sorts the fields of a record-type.
    fn sort_fields(self, ctx: &mut Info) -> Self;
}

impl<T> SortFields for VecMap<hir::Name, T> {
    fn sort_fields(mut self, info: &mut Info) -> Self {
        self.drain()
            .sorted_by_key(|(x, _)| info.names.resolve(x.id))
            .collect()
    }
}
