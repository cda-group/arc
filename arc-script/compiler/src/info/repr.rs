use crate::hir::Name;
use crate::hir::Path;
use crate::info::diags::DiagInterner;
use crate::info::files::FileInterner;
use crate::info::modes::Mode;
use crate::info::names::NameInterner;
use crate::info::paths::PathId;
use crate::info::paths::PathInterner;
use crate::info::types::TypeInterner;

use std::str;
use tracing::instrument;

/// Info which is shared between the AST and HIR.
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
}

impl Info {
    #[instrument(name = "Mode => Info", level = "debug")]
    pub(crate) fn from(mode: Mode) -> Self {
        tracing::debug!("\n{:?}", mode);
        let names = NameInterner::default();
        let root = names.common.root;
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

    /// Resolves a `Path` to a `Vec<NameBuf>`.
    pub(crate) fn resolve_to_names(&self, mut path: impl Into<PathId>) -> Vec<&str> {
        let mut path = path.into();
        let mut names = Vec::new();
        loop {
            let kind = self.paths.resolve(path);
            let name = self.names.resolve(kind.name);
            names.push(name);
            if let Some(pred) = kind.pred {
                path = pred;
            } else {
                break;
            }
        }
        names.reverse();
        names
    }

    /// Generates a fresh `Name` and `Path`.
    pub(crate) fn fresh_name_path(&mut self) -> (Name, Path) {
        let name = self.names.fresh();
        let path = self.paths.intern_child(self.paths.root, name).into();
        (name, path)
    }

    /// Generates a fresh `Path`.
    pub(crate) fn fresh_path(&mut self) -> Path {
        let name = self.names.fresh();
        self.paths.intern_child(self.paths.root, name).into()
    }
}
