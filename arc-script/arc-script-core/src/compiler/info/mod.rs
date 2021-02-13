use modes::Mode;

use crate::compiler::hir::{Name, Path};

use crate::compiler::info::diags::DiagInterner;
use crate::compiler::info::files::FileInterner;
use crate::compiler::info::names::NameInterner;
use crate::compiler::info::paths::{PathId, PathInterner};

use std::str;
use types::TypeInterner;

pub(crate) mod utils;

/// Module for representing and interning diagnostics.
/// NB: Public because it is used by LSP.
pub mod diags;
/// Module for interning files.
pub mod files;
/// Module for logging debug information.
pub mod logger;
/// Module for representing modes of compilation.
pub mod modes;
/// Module for interning names.
pub(crate) mod names;
/// Module for interning paths.
pub(crate) mod paths;
/// Module for interning and unifying types.
pub(crate) mod types;
// Module for interning locations.
// pub(crate) mod locs;
// Module for interning expressions.
// pub(crate) mod exprs;

/// Info which is shared between the AST, HIR, and DFG.
#[derive(Debug)]
pub struct Info {
    /// Command-line options
    pub mode: Mode,
    /// Interner for diagnostics.
    pub(crate) diags: DiagInterner,
    /// Interner for files.
    pub files: FileInterner,
    /// Interner for names.
    pub(crate) names: NameInterner,
    /// Interner for paths.
    pub(crate) paths: PathInterner,
    /// Interner for types.
    pub(crate) types: TypeInterner,
    // Interner for expressions.
    //     pub(crate) exprs: ExprInterner,
}

impl From<Mode> for Info {
    fn from(mode: Mode) -> Self {
        let names = NameInterner::default();
        let root: Name = names.root.into();
        let paths = PathInterner::from(root);
        Self {
            mode,
            names,
            paths,
            diags: Default::default(),
            files: Default::default(),
            types: Default::default(),
        }
    }
}

impl Info {
    /// Converts an OS-Path into a syntactic path and then interns it.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn intern_ospath(&mut self, buf: &std::path::Path) -> PathId {
        let buf = buf
            .components()
            .map(|c| self.names.intern(c.as_os_str().to_str().unwrap()).into())
            .collect::<Vec<_>>();
        self.paths.intern_abs_vec(buf)
    }

    /// Resolves a `PathId` to a `Vec<NameBuf>`.
    pub(crate) fn resolve_to_names(&self, mut id: PathId) -> Vec<&str> {
        let mut names = Vec::new();
        loop {
            let path = self.paths.resolve(id);
            let name = self.names.resolve(path.name.id);
            names.push(name);
            if let Some(pred) = path.pred {
                id = pred;
            } else {
                break;
            }
        }
        names.reverse();
        names
    }

    /// Generates a fresh `PathId`.
    pub(crate) fn fresh_name_path(&mut self) -> (Name, Path) {
        let name = self.names.fresh();
        let path = self.paths.intern_child(self.paths.root, name).into();
        (name, path)
    }
}
