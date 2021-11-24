use crate::ast;
use crate::ast::AST;
use crate::hir::lower::ast::resolve::ItemDeclKind;
use crate::hir::lower::ast::resolve::SymbolTable;
use crate::hir::Name;
use crate::info::diags::Error;
use crate::info::paths::PathId;
use crate::info::Info;

use arc_script_compiler_shared::map;
use arc_script_compiler_shared::Map;
use arc_script_compiler_shared::New;
use arc_script_compiler_shared::Set;
use arc_script_compiler_shared::Shrinkwrap;

/// `Declare` adds all item-declarations and imports in `&self` to the symbol table
/// with resolved names.
pub(crate) trait Declare {
    fn declare(&self, path: PathId, ctx: &mut Context<'_>);
}

#[derive(New, Shrinkwrap)]
#[shrinkwrap(mutable)]
pub(crate) struct Context<'i> {
    pub(crate) table: &'i mut SymbolTable,
    #[shrinkwrap(main_field)]
    pub(crate) info: &'i mut Info,
    pub(crate) ast: &'i AST,
}

/// Macro for implementing the `Declare` trait.
macro_rules! declare {
    {
        [$node:ident, $path:ident, $ctx:ident]
        $($ty:path => $expr:expr ,)*
    } => {
        $(
            impl Declare for $ty {
                fn declare(&self, $path: PathId, $ctx: &mut Context<'_>) {
                    let $node = self;
                    tracing::trace!("{}: {} ⊢ {}",
                        stringify!($ty),
                        $ctx.ast.pretty(&$path, $ctx.info),
                        $ctx.ast.pretty(self, $ctx.info)
                    );
                    $expr;
                }
            }
        )*
    };
}

declare! {
    [node, path, ctx]

    ast::Module => node.items.iter().for_each(|item| item.declare(path, ctx)),
    ast::Item => match &node.kind {
        ast::ItemKind::Fun(item)        => item.declare(path, ctx),
        ast::ItemKind::TypeAlias(item)  => item.declare(path, ctx),
        ast::ItemKind::Enum(item)       => item.declare(path, ctx),
        ast::ItemKind::Task(item)       => item.declare(path, ctx),
        ast::ItemKind::Use(item)        => item.declare(path, ctx),
        ast::ItemKind::Assign(item)     => item.declare(path, ctx),
        ast::ItemKind::ExternFun(item)  => item.declare(path, ctx),
        ast::ItemKind::ExternType(item) => item.declare(path, ctx),
        ast::ItemKind::Err              => {}
    },
    ast::TaskItem => match &node.kind {
        ast::TaskItemKind::Fun(item)       => {
            let path = ctx.paths.intern_child(path, item.name);
            if ctx.table.declarations.insert(path, ItemDeclKind::Method).is_some() {
                ctx.diags.intern(Error::NameClash { name: item.name })
            }
        },
        ast::TaskItemKind::ExternFun(item) => item.declare(path, ctx),
        ast::TaskItemKind::TypeAlias(item) => item.declare(path, ctx),
        ast::TaskItemKind::Use(item)       => item.declare(path, ctx),
        ast::TaskItemKind::Enum(item)      => item.declare(path, ctx),
        ast::TaskItemKind::Stmt(_)         => {} // TODO! Handle variables
        ast::TaskItemKind::Err             => {}
    },
    ast::Fun => {
        let path = ctx.paths.intern_child(path, node.name);
        if ctx.table.declarations.insert(path, ItemDeclKind::Fun).is_some() {
            ctx.diags.intern(Error::NameClash { name: node.name })
        }
    },
    ast::Assign => node.param.pat.declare(path, ctx),
    ast::Pat => match ctx.ast.pats.resolve(node) {
        ast::PatKind::Ignore => {}
        ast::PatKind::Or(p0, p1) => {
            p0.declare(path, ctx);
            p1.declare(path, ctx);
        }
        ast::PatKind::Struct(fs) => fs.iter().for_each(|f| {
            if let Some(p) = &f.val {
                p.declare(path, ctx)
            } else {
                let path = ctx.paths.intern_child(path, f.name);
                if ctx.table.declarations.insert(path, ItemDeclKind::Alias).is_some() {
                    ctx.diags.intern(Error::NameClash { name: f.name })
                }
            }
        }),
        ast::PatKind::Tuple(ps) => ps.iter().for_each(|p| p.declare(path, ctx)),
        ast::PatKind::Const(_) => {}
        ast::PatKind::Var(x) => {
            let path = ctx.paths.intern_child(path, *x);
            if ctx.table.declarations.insert(path, ItemDeclKind::Alias).is_some() {
                ctx.diags.intern(Error::NameClash { name: *x })
            }
        }
        ast::PatKind::Variant(_, p) => p.declare(path, ctx),
        ast::PatKind::By(p0, p1) => {
            p0.declare(path, ctx);
            p1.declare(path, ctx);
        },
        ast::PatKind::Err => {}
    },
    ast::TypeAlias => {
        let path = ctx.paths.intern_child(path, node.name);
        if ctx.table.declarations.insert(path, ItemDeclKind::Alias).is_some() {
            ctx.diags.intern(Error::NameClash { name: node.name })
        }
    },
    ast::Enum => {
        let path = ctx.paths.intern_child(path, node.name);
        if ctx.table.declarations.insert(path, ItemDeclKind::Enum).is_some() {
            ctx.diags.intern(Error::NameClash { name: node.name })
        } else {
            for variant in &node.variants {
                variant.declare(path, ctx)
            }
        }
    },
    ast::Task => {
        let path = ctx.paths.intern_child(path, node.name);
        if ctx.table.declarations.insert(path, ItemDeclKind::Task).is_some() {
            ctx.diags.intern(Error::NameClash { name: node.name })
        } else {
            for item in &node.items {
                item.declare(path, ctx);
            }
            let iname = ctx.names.common.iinterface;
            let oname = ctx.names.common.ointerface;
            for (node, name) in [(&node.iinterface, iname), (&node.ointerface, oname)] {
                match &node.kind {
                    ast::InterfaceKind::Tagged(ports) => {
                        // Declare the enum of a tagged interface
                        let interface_path = ctx.paths.intern_child(path, name);
                        ctx.table
                            .declarations
                            .insert(interface_path, ItemDeclKind::Enum);
                        for variant in ports {
                            let alias_path = ctx.paths.intern_child(path, variant.name);
                            let port_path = ctx.paths.intern_child(interface_path, variant.name);
                            ctx.table.imports.insert(alias_path, port_path);
                            variant.declare(interface_path, ctx);
                        }
                    }
                    ast::InterfaceKind::Single(_) => {}
                }
            }
        }
    },
    ast::Use => {
        let name = node.alias.map_or_else(|| ctx.paths.resolve(node.path).name, |alias| alias);
        let use_path = ctx.paths.intern_child(path, name);
        if ctx.table.declarations.contains_key(&use_path) {
            ctx.diags.intern(Error::NameClash { name })
        } else {
            ctx.table.import(use_path, node.path);
        }
    },
    ast::ExternFun => {
        let path = ctx.paths.intern_child(path, node.decl.name);
        if ctx.table.declarations.insert(path, ItemDeclKind::ExternFun).is_some() {
            ctx.diags.intern(Error::NameClash { name: node.decl.name })
        }
    },
    ast::ExternType => {
        let path = ctx.paths.intern_child(path, node.name);
        if ctx.table.declarations.insert(path, ItemDeclKind::ExternType).is_some() {
            ctx.diags.intern(Error::NameClash { name: node.name })
        }
        let self_path = ctx.paths.intern_child(path, node.name);
        ctx.table.import(self_path, path);
    },
    ast::Variant => {
        let path = ctx.paths.intern_child(path, node.name);
        if ctx.table.declarations.insert(path, ItemDeclKind::Variant).is_some() {
            ctx.diags.intern(Error::NameClash { name: node.name })
        }
    },
    ast::Port => {
        let path = ctx.paths.intern_child(path, node.name);
        if ctx.table.declarations.insert(path, ItemDeclKind::Variant).is_some() {
            ctx.diags.intern(Error::NameClash { name: node.name })
        }
    },
}
