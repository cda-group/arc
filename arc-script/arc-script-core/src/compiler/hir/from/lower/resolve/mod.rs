mod stack;
mod table;
mod display;

use crate::compiler::ast::{self, AST};
use crate::compiler::hir::{self, Name};
use crate::compiler::info::names::NameId;
use crate::compiler::info::paths::{PathBuf, PathId};
use crate::compiler::info::{self, Info};

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
    pub(crate) path_id: PathId,
}

#[derive(Debug)]
pub(crate) enum DeclKind {
    Item(hir::Path, ItemDeclKind),
    Var(hir::Name),
}

impl Resolver {
    /// Constructs a Resolver from a &mut AST, a mutable borrow is required to
    /// intern paths along the way.
    pub(crate) fn from(ast: &AST, info: &mut Info) -> Self {
        Self {
            table: SymbolTable::from(ast, info),
            stack: SymbolStack::default(),
            path_id: info.paths.root,
        }
    }
}

impl Resolver {
    /// Returns the declaration kind of a path.
    pub(crate) fn resolve(&mut self, path: &ast::Path, info: &mut Info) -> Option<DeclKind> {
        // First check if the symbol is stored as a variable on the symbol stack
        let path_buf = *info.paths.resolve(path.id);
        if path_buf.pred.is_none() {
            if let Some(name) = self.stack.resolve(path_buf.name) {
                return Some(DeclKind::Var(name));
            }
        }
        // Otherwise it might be stored in the symbol table
        let path = info.paths.join(self.path_id, path.id);
        let true_path = self.table.resolve(path);
        Some(DeclKind::Item(true_path.into(), self.table.get(true_path)?))
    }

    /// Pushes a namespace onto the path.
    pub(crate) fn push_namespace(&mut self, name: Name, info: &mut Info) {
        self.path_id = info.paths.intern_child(self.path_id, name);
        self.stack.push_scope();
    }

    /// Pops a namespace off the stack
    pub(crate) fn pop_namespace(&mut self, info: &mut Info) {
        self.path_id = info.paths.resolve(self.path_id).pred.unwrap();
        self.stack.pop_scope();
    }
}
