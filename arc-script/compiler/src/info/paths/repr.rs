use crate::hir::Name;
use crate::info::files::Loc;
use crate::info::files::Spanned;

use arc_script_compiler_macros::Loc;
use arc_script_compiler_macros::GetId;
use arc_script_compiler_shared::Educe;
use arc_script_compiler_shared::From;
use arc_script_compiler_shared::Into;
use arc_script_compiler_shared::Map;
use arc_script_compiler_shared::New;
use arc_script_compiler_shared::Shrinkwrap;

use std::borrow::Borrow;

/// Path to an item (or variable).
#[derive(Debug, Copy, Clone, New, Loc, GetId, Shrinkwrap)]
pub struct Path {
    #[shrinkwrap(main_field)]
    pub(crate) id: PathId,
    pub(crate) loc: Loc,
}

/// Identifier for paths.
#[derive(Debug, Clone, Copy, Shrinkwrap, Eq, PartialEq, Hash, From, Into)]
pub struct PathId(usize);
 
#[derive(Debug, Clone, Copy, Eq, PartialEq, Hash)]
pub(crate) struct PathKind {
    pub(crate) pred: Option<PathId>,
    pub(crate) name: Name,
}     
     
/// An interner for interning `PathKind`s into `PathId`s, and resolving the other way around.
#[derive(Debug)]
pub(crate) struct PathInterner {
    pub(crate) kind_to_id: Map<PathKind, PathId>,
    pub(crate) id_to_kind: Vec<PathKind>,
    pub(crate) root: PathId,
}      
      
impl From<Name> for PathInterner {
    fn from(name: Name) -> Self {
        let kind_to_id = Map::default();
        let mut id_to_kind = Vec::new();
        let kind = PathKind { pred: None, name };
        let id = PathId(0);
      
        id_to_kind.push(kind);
        id_to_kind.insert(id.into(), kind);
       
        Self {
            kind_to_id,
            id_to_kind,
            root: id,
        }  
    }  
} 

impl PathInterner {
    /// Interns a path and returns an id mapping to it.
    pub(crate) fn intern(&mut self, kind: PathKind) -> PathId {
        self.kind_to_id.get(&kind).copied().unwrap_or_else(|| {
            let id = PathId(self.id_to_kind.len());
            self.id_to_kind.push(kind);
            self.kind_to_id.insert(kind, id);
            id
        })
    }

    /// Extends a path with a child, i.e., `<path>::<name>`.
    pub(crate) fn intern_child(&mut self, id: impl Into<PathId>, name: Name) -> PathId {
        self.intern(PathKind {
            pred: Some(id.into()),
            name,
        })
    }

    /// Returns a relative path which starts with `name`.
    pub(crate) fn intern_orphan(&mut self, name: Name) -> PathId {
        self.intern(PathKind { pred: None, name })
    }

    /// Joins two paths i.e., `<a>::<b>`.
    pub(crate) fn join(&mut self, a: impl Into<PathId>, b: impl Into<PathId>) -> PathId {
        self.join_rec(a.into(), b.into())
    }

    fn join_rec(&mut self, a: PathId, b: PathId) -> PathId {
        let b = *self.resolve(b);
        if let Some(b_pred) = b.pred {
            let ab = self.join_rec(a, b_pred);
            self.intern_child(ab, b.name)
        } else {
            self.intern_child(a, b.name)
        }
    }

    fn intern_vec(&mut self, path: Option<PathId>, names: Vec<Name>) -> PathId {
        if names.is_empty() {
            return path.unwrap();
        }
        let mut iter = names.into_iter();
        // Intern root
        let mut kind = PathKind {
            pred: path,
            name: iter.next().unwrap(),
        };
        let mut path = self.intern(kind);
        // Intern children
        for name in iter {
            kind = PathKind {
                pred: Some(path),
                name,
            };
            path = self.intern(kind);
        }
        path
    }

    /// Interns a vector of names as an absolute path.
    pub(crate) fn intern_abs_vec(&mut self, names: Vec<Name>) -> PathId {
        self.intern_vec(Some(self.root), names)
    }

    /// Interns a vector of names and returns an id mapping to it.
    pub(crate) fn intern_rel_vec(&mut self, names: Vec<Name>) -> PathId {
        self.intern_vec(None, names)
    }

    /// Resolves a `Path` to a `PathBuf` (`Vec<NameId>`).
    pub(crate) fn resolve(&self, path: impl Into<PathId>) -> &PathKind {
        self.id_to_kind.get(path.into().0).unwrap()
    }
}
