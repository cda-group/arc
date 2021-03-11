use crate::compiler::ast;
use crate::compiler::ast::AST;

use crate::compiler::hir::Name;
use crate::compiler::info::diags::Error;
use crate::compiler::info::paths::PathId;
use crate::compiler::info::Info;
use arc_script_core_shared::Map;
use arc_script_core_shared::Set;

/// Every symbol is some kind of declaration. The symbol's declaration kind
/// determines in which table of the HIR the definition of the symbol is stored.
/// Note that variable declarations are not included here since they live on the
/// stack rather than the namespace.
#[derive(Debug, Clone, Copy)]
pub(crate) enum ItemDeclKind {
    Alias,
    Enum,
    Fun,
    Task,
    Extern,
    Variant,
    State,
}

/// A symbol table which stores both an import graph, which can be used to track
/// down dependencies, and a table containing all declarations of the AST.
#[derive(Debug, Default)]
pub(crate) struct SymbolTable {
    /// A set of imports which have already been compressed.
    pub(crate) compressed: Set<PathId>,
    /// An import graph.
    pub(crate) imports: Map<PathId, PathId>,
    /// A table of declarations.
    pub(crate) declarations: Map<PathId, ItemDeclKind>,
}

impl SymbolTable {
    /// Constructs a `SymbolTable` from an `AST`.
    pub(crate) fn from(ast: &AST, info: &mut Info) -> Self {
        let mut table = Self::default();
        for (path, module) in &ast.modules {
            module.declare(*path, &mut table, info);
        }
        tracing::debug!("{}", table.debug(info));
        table
    }
}

impl SymbolTable {
    /// Returns the final target destination of a path in the import graph, while
    /// performing path compression. By keeping a `visited` set, every node is
    /// visited at most once, which means that cycles will be skipped.
    ///
    /// For example, when resolving node `B` in the following graph:
    ///   A -> B -> C -> D
    ///   E -> F -------/
    /// The result is `D` and a graph:
    ///   A -> B -> D
    ///   C -------/
    ///   E -> F --/
    pub(crate) fn absolute(&mut self, path: PathId) -> PathId {
        if self.compressed.contains(&path) {
            // Path has already been compressed
            self.imports.get(&path).cloned().unwrap_or(path)
        } else {
            // Mark path as compressed to avoid infinite cycles.
            self.compressed.insert(path);
            self.imports.remove(&path).map_or(path, |next| {
                // `path` is an alias for `next`, keep compressing
                let real = self.absolute(next);
                self.imports.insert(path, real);
                real
            })
        }
    }
    /// Returns the declaration kind of a path.
    pub(crate) fn get_decl(&mut self, path: PathId) -> Option<ItemDeclKind> {
        let real = self.imports.get(&path).unwrap_or(&path);
        self.declarations.get(real).cloned()
    }
}
