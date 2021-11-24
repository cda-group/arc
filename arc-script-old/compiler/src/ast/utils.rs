use crate::ast;

use crate::hir::Name;
use crate::info::files::Loc;
use crate::info::names::NameId;
use crate::info::Info;

impl ast::Path {
    /// Converts the path into a path which points to an entry in the file system.
    /// The last component, which is assumed to be an item, is therefore stripped from the path.
    pub(crate) fn path_buf(self, info: &Info) -> std::path::PathBuf {
        let names = info.resolve_to_names(self);
        let path = names[..names.len() - 1].join("/");
        std::path::PathBuf::from(path)
    }
    /// TODO: Improve name resolution algorithm so this code is not needed
    pub(crate) fn is_absolute(self, info: &Info) -> bool {
        let names = info.resolve_to_names(self);
        names[0] == "crate"
    }
}

impl ast::Module {
    /// Returns the imports of a module in the form of OS-paths which are
    /// relative to the project root.
    pub(crate) fn imports(&self, info: &Info) -> Vec<ast::Path> {
        let mut imports = Vec::new();
        for item in &self.items {
            match &item.kind {
                ast::ItemKind::Use(item) if item.path.is_absolute(info) => imports.push(item.path),
                ast::ItemKind::Task(item) => {
                    for item in &item.items {
                        if let ast::TaskItemKind::Use(item) = &item.kind {
                            if item.path.is_absolute(info) {
                                imports.push(item.path)
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        imports
    }
}

impl From<NameId> for Name {
    fn from(id: NameId) -> Self {
        Self::new(id, Loc::Fake)
    }
}

impl From<usize> for ast::Index {
    fn from(id: usize) -> Self {
        Self::new(id, Loc::Fake)
    }
}
