use crate::compiler::hir::{Name, Path};
use crate::compiler::info::files::Loc;
use crate::compiler::shared::New;

use derive_more::From;
use shrinkwraprs::Shrinkwrap;

use std::borrow::Borrow;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, Shrinkwrap, Eq, PartialEq, Hash)]
pub struct PathId(usize);

#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub struct PathBuf {
    pub(crate) pred: Option<PathId>,
    pub(crate) name: Name,
}

/// An interner for interning `Path`s into `PathId`s, and resolving the other way around.
#[derive(Debug)]
pub(crate) struct PathInterner {
    pub(crate) path_to_id: HashMap<PathBuf, PathId>,
    pub(crate) id_to_path: Vec<PathBuf>,
    pub(crate) root: PathId,
}

impl From<Name> for PathInterner {
    fn from(name: Name) -> Self {
        let mut path_to_id = HashMap::new();
        let mut id_to_path = Vec::new();
        let mut path_buf = PathBuf { pred: None, name };
        let path_id = PathId(id_to_path.len());

        id_to_path.push(path_buf);
        id_to_path.insert(path_id.0, path_buf);

        let mut interner = Self {
            path_to_id,
            id_to_path,
            root: path_id,
        };
        interner
    }
}

impl PathInterner {
    /// Interns a path and returns an id mapping to it.
    pub(crate) fn intern(&mut self, path: PathBuf) -> PathId {
        self.path_to_id.get(&path).map(|id| *id).unwrap_or_else(|| {
            let id = PathId(self.id_to_path.len());
            self.id_to_path.push(path);
            self.path_to_id.insert(path, id);
            id
        })
    }

    pub(crate) fn intern_child(&mut self, pred: PathId, name: Name) -> PathId {
        self.intern(PathBuf {
            pred: Some(pred),
            name,
        })
    }

    fn join_rec(&mut self, pred: PathId, child_path: PathId) -> PathId {
        let child_path_buf = *self.resolve(child_path);
        if let Some(child_pred) = child_path_buf.pred {
            let child_pred = self.join_rec(pred, child_pred);
            self.intern_child(child_pred, child_path_buf.name)
        } else {
            self.intern_child(pred, child_path_buf.name)
        }
    }

    pub(crate) fn join(&mut self, pred: PathId, path: PathId) -> PathId {
        self.join_rec(pred, path)
    }

    fn intern_vec(&mut self, pred: Option<PathId>, path: Vec<Name>) -> PathId {
        if path.is_empty() {
            return pred.unwrap();
        }
        let mut iter = path.into_iter();
        // Intern root
        let mut path = PathBuf {
            pred,
            name: iter.next().unwrap(),
        };
        let mut id = self.intern(path);
        // Intern children
        while let Some(name) = iter.next() {
            path = PathBuf {
                pred: Some(id),
                name,
            };
            id = self.intern(path);
        }
        id
    }

    pub(crate) fn intern_abs_vec(&mut self, path: Vec<Name>) -> PathId {
        self.intern_vec(Some(self.root), path)
    }

    /// Interns a vector of names and returns an id mapping to it.
    pub(crate) fn intern_rel_vec(&mut self, mut path: Vec<Name>) -> PathId {
        self.intern_vec(None, path)
    }

    /// Resolves a `PathId` to a `Path` (`Vec<NameId>`).
    pub(crate) fn resolve(&self, id: impl Borrow<PathId>) -> &PathBuf {
        self.id_to_path.get(**id.borrow()).unwrap()
    }
}
