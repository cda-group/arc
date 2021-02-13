use crate::compiler::ast::{self, AST};
use crate::compiler::hir;
use crate::compiler::hir::Name;
use crate::compiler::info::diags::Error;
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
    pub(crate) fn resolve(&mut self, path: PathId) -> PathId {
        if self.compressed.contains(&path) {
            // Path has already been compressed
            self.imports.get(&path).cloned().unwrap_or(path)
        } else {
            // Mark path as compressed to avoid infinite cycles.
            self.compressed.insert(path);
            self.imports.remove(&path).map_or(path, |next| {
                // `path` is an alias for `next`, keep compressing
                let real = self.resolve(next);
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

/// `Declare` adds all item-declarations and imports in `&self` to the symbol table
/// with resolved names.
trait Declare {
    fn declare(&self, path: PathId, table: &mut SymbolTable, info: &mut Info);
}

/// Declare all items within a module.
impl Declare for ast::Module {
    fn declare(&self, path: PathId, table: &mut SymbolTable, info: &mut Info) {
        for item in &self.items {
            item.declare(path, table, info);
        }
    }
}

/// Declare an item and its sub-items.
impl Declare for ast::Item {
    #[rustfmt::skip]
    fn declare(&self, path: PathId, table: &mut SymbolTable, info: &mut Info) {
        match &self.kind {
            ast::ItemKind::Fun(item)    => item.declare(path, table, info),
            ast::ItemKind::Alias(item)  => item.declare(path, table, info),
            ast::ItemKind::Enum(item)   => item.declare(path, table, info),
            ast::ItemKind::Task(item)   => item.declare(path, table, info),
            ast::ItemKind::Use(item)    => item.declare(path, table, info),
            ast::ItemKind::Extern(item) => item.declare(path, table, info),
            ast::ItemKind::Err          => {}
        }
    }
}

/// Declare a task-item and its sub-items.
impl Declare for ast::TaskItem {
    #[rustfmt::skip]
    fn declare(&self, path: PathId, table: &mut SymbolTable, info: &mut Info) {
        match &self.kind {
            ast::TaskItemKind::Fun(item)   => item.declare(path, table, info),
            ast::TaskItemKind::Alias(item) => item.declare(path, table, info),
            ast::TaskItemKind::Use(item)   => item.declare(path, table, info),
            ast::TaskItemKind::Enum(item)  => item.declare(path, table, info),
            ast::TaskItemKind::State(item) => item.declare(path, table, info),
            ast::TaskItemKind::On(_)       => {}
            ast::TaskItemKind::Err         => {}
        }
    }
}

impl Declare for ast::Fun {
    fn declare(&self, path: PathId, table: &mut SymbolTable, info: &mut Info) {
        let path = info.paths.intern_child(path, self.name);
        if table.declarations.insert(path, ItemDeclKind::Fun).is_some() {
            info.diags.intern(Error::NameClash { name: self.name })
        }
    }
}

impl Declare for ast::Alias {
    fn declare(&self, path: PathId, table: &mut SymbolTable, info: &mut Info) {
        let path = info.paths.intern_child(path, self.name);
        if table
            .declarations
            .insert(path, ItemDeclKind::Alias)
            .is_some()
        {
            info.diags.intern(Error::NameClash { name: self.name })
        }
    }
}

impl Declare for ast::Enum {
    fn declare(&self, path: PathId, table: &mut SymbolTable, info: &mut Info) {
        let path = info.paths.intern_child(path, self.name);
        if table
            .declarations
            .insert(path, ItemDeclKind::Enum)
            .is_some()
        {
            info.diags.intern(Error::NameClash { name: self.name })
        } else {
            for variant in &self.variants {
                variant.declare(path, table, info)
            }
        }
    }
}

impl Declare for ast::Task {
    fn declare(&self, path: PathId, table: &mut SymbolTable, info: &mut Info) {
        let path = info.paths.intern_child(path, self.name);
        if table
            .declarations
            .insert(path, ItemDeclKind::Task)
            .is_some()
        {
            info.diags.intern(Error::NameClash { name: self.name })
        } else {
            for item in &self.items {
                item.declare(path, table, info);
            }
            self.ihub.declare("Source", path, table, info);
            self.ohub.declare("Sink", path, table, info);
        }
    }
}

impl ast::Hub {
    fn declare(&self, name: &str, path: PathId, table: &mut SymbolTable, info: &mut Info) {
        match &self.kind {
            ast::HubKind::Tagged(ports) => {
                // Declare the enum of a tagged hub
                let hub_name = info.names.intern(name).into();
                let hub_path = info.paths.intern_child(path, hub_name);
                table.declarations.insert(hub_path, ItemDeclKind::Enum);
                for variant in ports {
                    let alias_path = info.paths.intern_child(path, variant.name);
                    let port_path = info.paths.intern_child(hub_path, variant.name);
                    table.imports.insert(alias_path, port_path);
                    variant.declare(hub_path, table, info);
                }
            }
            ast::HubKind::Single(_) => {}
        }
    }
}

impl Declare for ast::Use {
    fn declare(&self, path: PathId, table: &mut SymbolTable, info: &mut Info) {
        let name = self
            .alias
            .map_or_else(|| info.paths.resolve(self.path.id).name, |alias| alias);
        let use_path = info.paths.intern_child(path, name);
        if table.declarations.contains_key(&use_path) {
            info.diags.intern(Error::NameClash { name })
        } else {
            table.imports.insert(use_path, self.path.id);
        }
    }
}

impl Declare for ast::Extern {
    fn declare(&self, path: PathId, table: &mut SymbolTable, info: &mut Info) {
        let path = info.paths.intern_child(path, self.name);
        if table
            .declarations
            .insert(path, ItemDeclKind::Extern)
            .is_some()
        {
            info.diags.intern(Error::NameClash { name: self.name })
        }
    }
}

impl Declare for ast::Variant {
    fn declare(&self, path: PathId, table: &mut SymbolTable, info: &mut Info) {
        let path = info.paths.intern_child(path, self.name);
        if table
            .declarations
            .insert(path, ItemDeclKind::Variant)
            .is_some()
        {
            info.diags.intern(Error::NameClash { name: self.name })
        }
    }
}

impl Declare for ast::Port {
    fn declare(&self, path: PathId, table: &mut SymbolTable, info: &mut Info) {
        let path = info.paths.intern_child(path, self.name);
        if table
            .declarations
            .insert(path, ItemDeclKind::Variant)
            .is_some()
        {
            info.diags.intern(Error::NameClash { name: self.name })
        }
    }
}

impl Declare for ast::State {
    fn declare(&self, path: PathId, table: &mut SymbolTable, info: &mut Info) {
        let path = info.paths.intern_child(path, self.name);
        if table
            .declarations
            .insert(path, ItemDeclKind::State)
            .is_some()
        {
            info.diags.intern(Error::NameClash { name: self.name })
        }
    }
}
