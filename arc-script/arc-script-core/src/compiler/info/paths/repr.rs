use crate::compiler::hir::{Name, Path};
use crate::compiler::info::files::Loc;
use crate::compiler::shared::{Map, New};

use derive_more::From;
use shrinkwraprs::Shrinkwrap;

/// The product of interning a `Path`.
#[derive(Debug, Clone, Copy, Shrinkwrap, Eq, PartialEq, Hash)]
pub struct PathId(usize);

pub type PathBuf = Vec<Name>;

/// An interner for interning `Path`s into `PathId`s, and resolving the other way around.
#[derive(Default, Debug)]
pub(crate) struct PathInterner {
    pub(crate) store: Map<PathBuf, PathId>,
}

impl PathInterner {
    /// Interns a path and returns an id mapping to id.
    pub(crate) fn intern(&mut self, path: impl AsRef<PathBuf>) -> PathId {
        self.store
            .get(path.as_ref())
            .map(|id| *id)
            .unwrap_or_else(|| {
                let id = PathId(self.store.len());
                self.store.insert(path.as_ref().clone().into(), id);
                id
            })
    }

    /// Resolves a `PathId` to a `Path` (`Vec<NameId>`).
    pub(crate) fn resolve(&self, id: PathId) -> &PathBuf {
        self.store.get_index(*id).map(|(buf, _)| buf).unwrap()
    }
}

pub(crate) fn extend(path: &PathBuf, name: Name) -> PathBuf {
    let mut path = path.clone();
    path.push(name);
    path
}
