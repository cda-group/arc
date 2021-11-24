mod debug;
mod stack;
mod table;
pub(crate) mod declare;

use crate::ast;
use crate::ast::AST;
use crate::hir;
use crate::hir::Name;
use crate::info::diags::Error;
use crate::info::paths::PathId;
use crate::info::Info;

use arc_script_compiler_shared::VecDeque;

use stack::SymbolStack;
use table::SymbolTable;

pub(crate) use table::ItemDeclKind;

/// A data structure for disambiguating names and paths of the AST.
#[derive(Debug)]
pub(crate) struct Resolver {
    /// Table which stores information about items and namespaces.
    pub(crate) table: SymbolTable,
    /// Stack which stores information about variables and scopes.
    pub(crate) stack: SymbolStack,
    /// Path to the current namespace.
    pub(crate) path: PathId,
}

#[derive(Debug)]
pub(crate) enum DeclKind {
    Item(hir::Path, ItemDeclKind),
    Var(hir::Name, hir::ScopeKind),
}

impl Resolver {
    pub(crate) fn from(ast: &AST, info: &mut Info) -> Self {
        Self {
            table: SymbolTable::from(ast, info),
            stack: SymbolStack::default(),
            path: info.paths.root,
        }
    }
}

impl Resolver {
    /// Returns the declaration kind of a path.
    pub(crate) fn resolve(&mut self, path: &ast::Path, info: &mut Info) -> Option<DeclKind> {
        tracing::trace!("Resolving path: {}", path.id.debug(info));
        // First check if the symbol is stored as a variable on the symbol stack
        let kind = *info.paths.resolve(path);
        if kind.pred.is_none() {
            if let Some((name, kind)) = self.stack.resolve(kind.name) {
                return Some(DeclKind::Var(name, kind));
            }
        }

        // Otherwise it might be stored in the symbol table
        let absolute_path = if path.is_absolute(info) {
            path.id
        } else {
            let relative_path = info.paths.join(self.path, *path);
            self.table.absolute(relative_path)
        };
        tracing::trace!("Resolving absolute path: {}", absolute_path.debug(info));

        self.table
            .get_decl(absolute_path)
            .map(|decl| DeclKind::Item(absolute_path.into(), decl))
            .or_else(|| {
                tracing::trace!("Path not found");
                info.diags.intern(Error::PathNotFound {
                    path: absolute_path.into(),
                    loc: path.loc,
                });
                None
            })
    }

    /// Pushes a namespace onto the path.
    pub(crate) fn push_namespace(&mut self, name: Name, info: &mut Info) {
        self.path = info.paths.intern_child(self.path, name);
    }

    /// Pops a namespace off the stack
    pub(crate) fn pop_namespace(&mut self, info: &mut Info) {
        self.path = info.paths.resolve(self.path).pred.unwrap();
    }

    pub(crate) fn set_path(&mut self, path: PathId) {
        self.path = path;
    }
}
