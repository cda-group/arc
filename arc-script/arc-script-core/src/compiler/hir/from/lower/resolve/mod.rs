mod stack;
mod table;

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
    pub(crate) path: PathBuf,
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
            path: PathBuf::default(),
        }
    }
}

impl Resolver {
    /// Returns the full path of a symbol.
    pub(crate) fn resolve(&mut self, path: &ast::Path, info: &mut Info) -> Option<DeclKind> {
        // First check if the symbol is stored as a variable on the symbol stack
        let names = info.paths.resolve(path.id);
        if let [name] = names.as_slice() {
            if let Some(name) = self.stack.resolve(*name) {
                return Some(DeclKind::Var(name));
            }
        }
        // Otherwise it might be stored in the symbol table
        let mut item_path = self.path.clone();
        item_path.extend(names);
        let item_path_id = info.paths.intern(item_path);
        let true_item_path_id = self.table.resolve(item_path_id);
        Some(DeclKind::Item(
            true_item_path_id.into(),
            self.table.get(true_item_path_id)?,
        ))
    }

    /// Pushes a namespace onto the path.
    pub(crate) fn push_namespace(&mut self, name: Name) {
        self.path.push(name);
        self.stack.push_scope();
    }

    /// Pops a namespace off the stack
    pub(crate) fn pop_namespace(&mut self) {
        self.path.pop();
        self.stack.pop_scope();
    }
}
