use super::Context;

use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::hir::lower::ast::resolve;
use crate::compiler::info::diags::Error;
use crate::compiler::info::files::Loc;

use arc_script_core_shared::get;
use arc_script_core_shared::map;
use arc_script_core_shared::Lower;
use arc_script_core_shared::VecDeque;
use arc_script_core_shared::VecMap;

impl ast::Interface {
    /// Generates an enum for an interface, e.g.:
    /// ```skip
    /// task Foo(): (X(~A), Y(~B)) -> ~C {
    ///     # ...
    /// }
    /// ```
    /// becomes
    /// ```skip
    /// enum Foo_IInterface { X(A), Y(B) }
    /// enum Foo_OInterface { __(C), }
    /// task Foo(): (X(~A), Y(~B)) -> ~C {
    ///     # ...
    /// }
    /// ```
    pub(crate) fn lower(&self, name: hir::Name, ctx: &mut Context<'_>) -> hir::Interface {
        let task_path = ctx.res.path;
        // Interior is the enum of streams inside the task
        let enum_path: hir::Path = ctx.paths.intern_child(task_path, name).into();
        // Exterior is the list of parameters for calling the task
        let (variant_paths, param_ts) = match &self.kind {
            ast::InterfaceKind::Tagged(ports) => self.lower_tagged(ports, enum_path, ctx),
            ast::InterfaceKind::Single(t) => self.lower_single(t, enum_path, ctx),
        };
        let enum_item = hir::Item::syn(hir::Enum::new(enum_path, variant_paths).into());
        ctx.hir.intern(enum_path, enum_item);
        let key_ts = param_ts.iter().map(|t| ctx.types.fresh()).collect();
        hir::Interface {
            interior: enum_path,
            exterior: param_ts,
            keys: key_ts,
            loc: self.loc,
        }
    }

    /// Returns a list of variants and parameters for an interface of multiple ports
    fn lower_tagged(
        &self,
        ports: &Vec<ast::Port>,
        enum_path: hir::Path,
        ctx: &mut Context<'_>,
    ) -> (Vec<hir::Path>, Vec<hir::Type>) {
        ports
            .iter()
            .map(|port| {
                let path = ctx.paths.intern_child(enum_path, port.name).into();
                let param_t = port.t.lower(ctx);
                let variant_t = ctx.types.fresh();
                let variant = hir::Variant::new(path, variant_t, port.loc).into();
                ctx.hir.intern(path, hir::Item::new(variant, self.loc));
                (path, param_t)
            })
            .unzip()
    }

    /// Returns a list of variants and parameters for an interface of a single port
    fn lower_single(
        &self,
        t: &ast::Type,
        enum_path: hir::Path,
        ctx: &mut Context<'_>,
    ) -> (Vec<hir::Path>, Vec<hir::Type>) {
        let variant_name = ctx.names.common.dummy;
        let variant_path = ctx.paths.intern_child(enum_path.id, variant_name).into();
        let param_t = t.lower(ctx);
        let variant_t = ctx.types.fresh();
        let variant_item = hir::Item::syn(hir::Variant::syn(variant_path, variant_t).into());
        ctx.hir.intern(variant_path, variant_item);
        let variants = vec![variant_path];
        (variants, vec![param_t])
    }
}

// fn flatten() {
//     if !cases.is_empty() {
//         let fun_params = std::mem::replace(&mut task_params, special::pattern::params(&node.params, ctx));
//         let task_args = special::pattern::params_to_args(&task_params, hir::BindingKind::Local, ctx);
//
//         let path = *ctx.paths.resolve(task_path);
//         let task_name = ctx.names.fresh_with_base(path.name);
//         let fun_path = std::mem::replace(
//             &mut task_path,
//             ctx.paths
//                 .intern_child(path.pred.unwrap(), task_name)
//                 .into(),
//         );
//
//         let tt = ctx.types.fresh();
//         let rt = ctx.types.fresh();
//         let ft = ctx.types.fresh();
//
//         let call = hir::Expr::syn(
//             hir::ExprKind::Call(
//                 hir::Expr::syn(hir::ExprKind::Item(task_path), tt).into(),
//                 task_args,
//             ),
//             rt,
//         );
//         let body = special::pattern::fold_cases(call, None, cases);
//         let fun = hir::Fun::new(Global, fun_path, fun_params, None, body, ft, rt);
//         let item = hir::Item::syn(hir::ItemKind::Fun(fun));
//         ctx.hir.intern(fun_path, item);
//         ctx.hir.items.push(fun_path);
//     }
// }

/// Lowers a `emit e` into a `emit crate::path::to::Task::OInterface::__(e)` in the context of a
/// task with an untagged port.
pub(crate) fn lower_emit(e0: &ast::Expr, ctx: &mut Context<'_>) -> hir::ExprKind {
    let v0 = e0.lower(ctx);
    if let Some(enum_path) = ctx.generated_ointerface_interior {
        let dummy_x = ctx.names.common.dummy;
        let variant_path = ctx.paths.intern_child(enum_path, dummy_x);
        let v1 = ctx.new_expr_enwrap(variant_path.into(), v0).into_ssa(ctx);
        hir::ExprKind::Emit(v1)
    } else {
        hir::ExprKind::Emit(v0)
    }
}

/// Lowers an `on { (<pattern> => <block>,)* }` into a finite state machine.
pub(crate) fn lower_on(cs: &[ast::Case], ctx: &mut Context<'_>) -> hir::ExprKind {
    todo!()
}
