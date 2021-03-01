mod debug;
mod stack;
mod table;

use crate::compiler::ast;
use crate::compiler::ast::AST;
use crate::compiler::hir;
use crate::compiler::hir::Name;
use crate::compiler::info::diags::Error;

use crate::compiler::info::paths::PathId;
use crate::compiler::info::Info;

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
    Var(hir::Name, hir::VarKind),
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
        tracing::trace!("{}", path.id.debug(info));
        // First check if the symbol is stored as a variable on the symbol stack
        let path_buf = *info.paths.resolve(path.id);
        if path_buf.pred.is_none() {
            if let Some((name, kind)) = self.stack.resolve(path_buf.name) {
                return Some(DeclKind::Var(name, kind));
            }
        }
        // Otherwise it might be stored in the symbol table
        let rel_path = info.paths.join(self.path_id, path.id);
        let abs_path = self.table.absolute(rel_path);
        self.table
            .get_decl(abs_path)
            .map(|decl| DeclKind::Item(abs_path.into(), decl))
            .or_else(|| {
                info.diags.intern(Error::PathNotFound {
                    path: rel_path.into(),
                    loc: path.loc,
                });
                None
            })
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
