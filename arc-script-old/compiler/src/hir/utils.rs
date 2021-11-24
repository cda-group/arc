use crate::ast;
use crate::hir;
use crate::info::files::Loc;
use crate::info::paths::PathId;
use crate::info::Info;
use bitmaps::Bitmap;

use arc_script_compiler_shared::itertools::Itertools;
use arc_script_compiler_shared::VecMap;

impl hir::HIR {
    pub(crate) fn intern(&mut self, path: impl Into<PathId>, item: impl Into<hir::Item>) {
        self.defs.insert(path.into(), item.into());
    }

    pub(crate) fn resolve(&self, path: impl Into<PathId>) -> &hir::Item {
        self.defs.get(&path.into()).unwrap()
    }
}

impl Default for hir::TypeKind {
    fn default() -> Self {
        hir::TypeKind::Unknown(hir::Constraint::default())
    }
}

impl From<hir::ScalarKind> for hir::TypeKind {
    fn from(kind: hir::ScalarKind) -> Self {
        Self::Scalar(kind)
    }
}

impl From<Vec<(hir::Name, hir::Type)>> for hir::TypeKind {
    fn from(vec: Vec<(hir::Name, hir::Type)>) -> Self {
        hir::TypeKind::Struct(vec.into_iter().collect())
    }
}

impl From<hir::ConstraintKind> for hir::TypeKind {
    fn from(kind: hir::ConstraintKind) -> Self {
        let mut b = Bitmap::new();
        b.set(kind as usize, true);
        hir::TypeKind::Unknown(b.into())
    }
}

impl hir::ItemKind {
    pub(crate) const fn get_path(&self) -> hir::Path {
        match self {
            hir::ItemKind::TypeAlias(item) => item.path,
            hir::ItemKind::Enum(item) => item.path,
            hir::ItemKind::Fun(item) => item.path,
            hir::ItemKind::Task(item) => item.path,
            hir::ItemKind::ExternFun(item) => item.path,
            hir::ItemKind::ExternType(item) => item.path,
            hir::ItemKind::Variant(item) => item.path,
        }
    }
}

pub(crate) trait SortFields: Sized {
    /// Sorts the fields of a record-type.
    fn sort_fields(self, ctx: &mut Info) -> Self;
}

impl<T> SortFields for VecMap<hir::Name, T> {
    fn sort_fields(mut self, info: &mut Info) -> Self {
        self.drain()
            .sorted_by_key(|(x, _)| info.names.resolve(x))
            .collect()
    }
}

impl hir::Path {
    pub(crate) fn is_enum(self, hir: &hir::HIR) -> bool {
        matches!(&hir.resolve(self).kind, hir::ItemKind::Enum(_))
    }
}
