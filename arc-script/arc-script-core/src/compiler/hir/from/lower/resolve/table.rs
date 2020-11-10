use crate::compiler::ast::{self, AST};
use crate::compiler::hir;
use crate::compiler::info::paths::{PathBuf, PathId};
use crate::compiler::info::{self, Info};

use std::collections::{HashMap as Map, HashSet as Set};

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
    /// Constructs a SymbolTable from an AST.
    pub(crate) fn from(ast: &AST, info: &mut Info) -> Self {
        let mut table = SymbolTable::default();
        for (path_id, module) in &ast.modules {
            let path = info.paths.resolve(*path_id).clone();
            module.declare(&path, &mut table, info);
        }
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
    pub(crate) fn resolve(&mut self, path: PathId) -> PathId {
        if !self.compressed.contains(&path) {
            // Mark path as compressed to avoid infinite cycles.
            self.compressed.insert(path);
            if let Some(next) = self.imports.remove(&path) {
                // `path` is an alias for `next`, keep compressing
                let real = self.resolve(next);
                self.imports.insert(path, real);
                real
            } else {
                // Final destination reached
                path
            }
        } else {
            // Path has already been compressed
            self.imports.get(&path).cloned().unwrap_or(path)
        }
    }
    /// Returns the declaration kind of a path.
    pub(crate) fn get(&mut self, path: PathId) -> Option<ItemDeclKind> {
        let real = self.imports.get(&path).unwrap_or(&path);
        self.declarations.get(real).cloned()
    }
}

/// `Declare` adds all item-declarations and imports in `&self` to the symbol table
/// with resolved names.
trait Declare {
    fn declare(&self, path: &PathBuf, table: &mut SymbolTable, info: &mut Info);
}

/// Declare all items within a module.
impl Declare for ast::Module {
    fn declare(&self, path: &PathBuf, table: &mut SymbolTable, info: &mut Info) {
        for item in &self.items {
            item.declare(path, table, info);
        }
    }
}

/// Declare an item and its sub-items.
impl Declare for ast::Item {
    fn declare(&self, path: &PathBuf, table: &mut SymbolTable, info: &mut Info) {
        match &self.kind {
            ast::ItemKind::Fun(item) => {
                let mut path = path.clone();
                path.push(item.name);
                table
                    .declarations
                    .insert(info.paths.intern(path), ItemDeclKind::Fun);
            }
            ast::ItemKind::Alias(item) => {
                let mut path = path.clone();
                path.push(item.name);
                table
                    .declarations
                    .insert(info.paths.intern(path), ItemDeclKind::Alias);
            }
            ast::ItemKind::Enum(item) => {
                let mut path = path.clone();
                path.push(item.name);
                table
                    .declarations
                    .insert(info.paths.intern(path), ItemDeclKind::Enum);
            }
            ast::ItemKind::Task(item) => {
                let mut path = path.clone();
                path.push(item.name);
                for item in &item.items {
                    item.declare(&path, table, info);
                }
                table
                    .declarations
                    .insert(info.paths.intern(path), ItemDeclKind::Task);
            }
            ast::ItemKind::Use(item) => {
                let name = if let Some(alias) = item.alias {
                    alias.clone()
                } else {
                    info.paths.resolve(item.path.id).last().unwrap().clone()
                };
                let mut use_path = path.clone();
                use_path.push(name);
                table
                    .imports
                    .insert(info.paths.intern(use_path), item.path.id);
            }
            ast::ItemKind::Err => {}
            ast::ItemKind::Extern(item) => {
                let mut path = path.clone();
                path.push(item.name);
                table
                    .declarations
                    .insert(info.paths.intern(path), ItemDeclKind::Extern);
            }
        }
    }
}

/// Declare a task-item and its sub-items.
impl Declare for ast::TaskItem {
    fn declare(&self, path: &PathBuf, table: &mut SymbolTable, info: &mut Info) {
        match &self.kind {
            ast::TaskItemKind::Fun(item) => {
                let mut path = path.clone();
                path.push(item.name);
                table
                    .declarations
                    .insert(info.paths.intern(path), ItemDeclKind::Fun);
            }
            ast::TaskItemKind::Alias(item) => {
                let mut path = path.clone();
                path.push(item.name);
                table
                    .declarations
                    .insert(info.paths.intern(path), ItemDeclKind::Alias);
            }
            ast::TaskItemKind::Use(item) => {
                let name = if let Some(alias) = item.alias {
                    alias.clone()
                } else {
                    info.paths.resolve(item.path.id).last().unwrap().clone()
                };
                let mut use_path = path.clone();
                use_path.push(name);
                table
                    .imports
                    .insert(info.paths.intern(use_path), item.path.id);
            }
            ast::TaskItemKind::Enum(id) => todo!(),
            ast::TaskItemKind::State(item) => todo!(),
            ast::TaskItemKind::On(_) => {}
            ast::TaskItemKind::Err => {}
        }
    }
}
