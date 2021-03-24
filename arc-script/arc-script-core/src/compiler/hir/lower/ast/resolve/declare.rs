use crate::compiler::ast;
use crate::compiler::ast::AST;
use crate::compiler::hir::lower::ast::resolve::ItemDeclKind;
use crate::compiler::hir::lower::ast::resolve::SymbolTable;
use crate::compiler::hir::Name;
use crate::compiler::info::diags::Error;
use crate::compiler::info::paths::PathId;
use crate::compiler::info::Info;

use arc_script_core_shared::map;
use arc_script_core_shared::Map;
use arc_script_core_shared::Set;

/// `Declare` adds all item-declarations and imports in `&self` to the symbol table
/// with resolved names.
pub(crate) trait Declare {
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
            ast::TaskItemKind::Fun(item)     => item.declare(path, table, info),
            ast::TaskItemKind::Extern(item)  => item.declare(path, table, info),
            ast::TaskItemKind::Alias(item)   => item.declare(path, table, info),
            ast::TaskItemKind::Use(item)     => item.declare(path, table, info),
            ast::TaskItemKind::Enum(item)    => item.declare(path, table, info),
            ast::TaskItemKind::Startup(_)    => {}
            ast::TaskItemKind::State(_)      => {}
            ast::TaskItemKind::On(_)         => {}
            ast::TaskItemKind::Timeout(_)    => {},
            ast::TaskItemKind::Timer(_)      => {},
            ast::TaskItemKind::Err           => {}
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
            self.ihub
                .declare(info.names.common.source.into(), path, table, info);
            self.ohub
                .declare(info.names.common.sink.into(), path, table, info);
        }
    }
}

impl ast::Hub {
    fn declare(&self, hub_name: Name, path: PathId, table: &mut SymbolTable, info: &mut Info) {
        match &self.kind {
            ast::HubKind::Tagged(ports) => {
                // Declare the enum of a tagged hub
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
