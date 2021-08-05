use super::Context;

use crate::ast;
use crate::hir;
use crate::hir::lower::ast::resolve;
use crate::info::diags::Error;
use crate::info::files::Loc;

use arc_script_compiler_shared::get;
use arc_script_compiler_shared::map;
use arc_script_compiler_shared::Lower;
use arc_script_compiler_shared::VecDeque;
use arc_script_compiler_shared::VecMap;

/// Takes the statements inside a task and extracts
/// 1. All assignment statements. These are converted into state-fields of the task.
/// 2. All expression statements. These are lifted into an on_start-function which is run
///    when the task is instantiated.
/// 3. A top-level on-event statement. This is lifted into an on_event-function which is run on 
///    each event.
///
/// ```ignore
/// task Foo(param: i32): ~i32 -> ~i32 {
///     val state: Cell[i32] = Cell(param * 2);
///     emit state.get();
///     on event => emit state.get() + event;
/// }
/// ```
///
/// Which creates two functions:
///
/// ```ignore
/// task Foo(param: i32): ~i32 -> ~i32 {
///     val state: Cell[i32] = uninitialised;
///     fun startup(param: i32) {
///         initialise[state](Cell(param * 2));
///         emit state.get();
///     }
///     fun handler(event: i32) {
///         emit state.get() + event;
///     }
/// }
/// ```
///
/// Then, this generates roughly the following code:
///
/// ```ignore
/// #[codegen::rewrite]
/// mod foo {
///     struct Foo {
///         param: i32,
///         #[state]
///         bar: State<Cell<i32>>,
///     }
///     struct IInterface { __(i32) }
///     struct OInterface { __(i32) }
/// }
/// impl foo::Foo {
///     fn startup(&mut self, param: i32) {
///         self.state.initialise(param * 2);
///         let x0 = self.bar.unwrap();
///         let x1 = enwrap!(OInterface::__, x0);
///         self.emit(x1);
///     }
///     fn handler(&mut self, event: IInterface) {
///         let x0 = unwrap!(IInterface::__, event);
///         let x1 = self.bar.unwrap();
///         let x2 = x0 + x1;
///         let x3 = enwrap!(OInterface::__, x2);
///         self.emit(x3);
///     }
/// }
/// ```
///
pub(crate) fn lower_items(
    items: &[ast::TaskItem],
    ctx: &mut Context<'_>,
) -> (
    VecMap<hir::Name, hir::Type>,
    hir::OnStart,
    hir::OnEvent,
    Vec<hir::PathId>,
) {
    ctx.res.stack.push_scope();
    let mut state_fields = VecMap::new();
    let mut on_event = None;
    let mut namespace = Vec::new();
    for item in items {
        if let ast::TaskItemKind::Stmt(s) = &item.kind {
            lower_stmt(s, &mut state_fields, &mut on_event, ctx)
        } else {
            namespace.extend(item.lower(ctx))
        }
    }
    let v = ctx.new_expr_unit().into_ssa(ctx);
    let stmts = ctx.res.stack.pop_scope();
    let body = hir::Block::syn(stmts, v);
    let name = ctx.names.common.on_start;
    let path = ctx.new_path(name);
    let f = ctx.new_method(path, vec![], body);
    ctx.hir.intern(path, hir::Item::syn(hir::ItemKind::Fun(f)));
    let on_start = hir::OnStart::syn(path);
    (state_fields, on_start, on_event.unwrap(), namespace)
}

fn lower_stmt(
    s: &ast::Stmt,
    state_fields: &mut VecMap<hir::Name, hir::Type>,
    on_event: &mut Option<hir::OnEvent>,
    ctx: &mut Context<'_>,
) {
    match &s.kind {
        ast::StmtKind::Empty => {}
        ast::StmtKind::Assign(a) => {
            // TODO: Add support for patterns in state fields.
            let x = *get!(ctx.ast.pats.resolve(a.param.pat), ast::PatKind::Var(x));
            let rhs_v = a.expr.lower(ctx);
            let member_x = ctx
                .res
                .stack
                .rename_to_unique(x, hir::ScopeKind::Member, ctx.info)
                .unwrap();
            // State-fields must be lazily initialised
            let member_t = ctx.new_type_fresh_if_none(&a.param.t);
            ctx.new_expr(hir::ExprKind::Initialise(member_x, rhs_v)).into_ssa(ctx);
            state_fields.insert(member_x, member_t);
        }
        ast::StmtKind::Expr(e) => {
            // TODO: Event handler must have exactly one case for the moment
            if let ast::ExprKind::On(cases) = ctx.ast.exprs.resolve(e) {
                // TODO: Event handler must be at the top level for the moment
                assert_eq!(cases.len(), 1);
                *on_event = Some(lower_on_event(cases[0], ctx));
            } else {
                e.lower(ctx);
            }
        }
    }
}

/// Lowers an `on { (<pattern> => <expr>)* }`  statement-expression.
/// TODO: Currently we assume there is only one case-pattern.
fn lower_on_event(case: ast::Case, ctx: &mut Context<'_>) -> hir::OnEvent {
    ctx.res.stack.push_scope();
    // For now assume there is only one case
    let (event_param, cases) = flatten_handler(&case.pat, ctx);
    let return_var = case.body.lower(ctx);
    let stmts = ctx.res.stack.pop_scope();
    let then_block = hir::Block::syn(stmts, return_var);
    let else_block = ctx.new_expr_unreachable().into_block(ctx);
    let block = ctx.fold_cases(then_block, else_block, cases);
    let name = ctx.names.common.on_event;
    let path = ctx.new_path(name);
    let f = hir::Fun {
        path,
        kind: hir::FunKind::Method,
        params: vec![event_param],
        body: block,
        t: ctx.types.fresh(),
        rt: ctx.types.fresh(),
    };
    ctx.hir.intern(path, hir::Item::syn(hir::ItemKind::Fun(f)));
    hir::OnEvent::syn(path)
}

/// Flattens the event handler pattern of an `on { (<pattern> => <expr>)* }` structure.
fn flatten_handler(
    pat: &ast::Pat,
    ctx: &mut Context<'_>,
) -> (hir::Param, VecDeque<super::pattern::Case>) {
    let (event_param, mut cases) = super::pattern::lower_pat(pat, true, hir::ScopeKind::Local, ctx);
    if let Some(iinterface_interior) = ctx.generated_iinterface_interior {
        // By default all events are wrapped inside enums when entering and exiting a task. We
        // therefore wrap the parameter inside a `crate::MyTask::IInterface::__(<expr>)` when the
        // event is not discriminated by a tag.
        let (wrapped_event_param, v) = ctx.new_fresh_param_var();
        let x = ctx.new_path_dummy(iinterface_interior);
        let s = ctx.new_expr_unwrap(x, v).into_stmt(event_param, ctx);
        cases.push_front(super::pattern::Case::Stmt(s));
        (wrapped_event_param, cases)
    } else {
        (event_param, cases)
    }
}

impl ast::Interface {
    /// Generates an enum for an interface, e.g.:
    /// ```skip
    /// fun Foo(): (X(~A), Y(~B)) -> ~C {
    ///     # ...
    /// }
    /// ```
    /// becomes
    /// ```skip
    /// fun Foo(): (X(~A), Y(~B)) -> ~C {
    ///     enum IInterface { X(A), Y(B) }
    ///     enum OInterface { __(C), }
    ///     # ...
    /// }
    /// ```
    pub(crate) fn lower(&self, name: hir::Name, ctx: &mut Context<'_>) -> hir::Interface {
        let task_path = ctx.res.path;
        let enum_path: hir::Path = ctx.paths.intern_child(task_path, name).into();
        let param_ts = match &self.kind {
            ast::InterfaceKind::Tagged(ports) => {
                // Construct enum for tagged ports
                let (variants, param_ts): (Vec<_>, Vec<_>) = ports
                    .iter()
                    .map(|port| {
                        let path = ctx.paths.intern_child(enum_path, port.name).into();
                        let exterior_t = port.t.lower(ctx);
                        let interior_t = ctx.types.fresh();
                        ctx.hir.intern(
                            path,
                            hir::Item::new(
                                hir::Variant::new(path, interior_t, port.loc).into(),
                                self.loc,
                            ),
                        );
                        (path, exterior_t)
                    })
                    .unzip();
                let enum_item = hir::Item::syn(hir::Enum::new(enum_path, variants).into());
                ctx.hir.intern(enum_path, enum_item);
                param_ts
            }
            ast::InterfaceKind::Single(t) => {
                // Construct enum for untagged port
                let variant_name = ctx.names.common.dummy;
                let variant_path = ctx.paths.intern_child(enum_path.id, variant_name).into();
                let exterior_t = t.lower(ctx);
                let interior_t = ctx.types.fresh();
                let variant_item =
                    hir::Item::syn(hir::Variant::syn(variant_path, interior_t).into());
                ctx.hir.intern(variant_path, variant_item);
                let variants = vec![variant_path];
                let enum_item = hir::Item::syn(hir::Enum::new(enum_path, variants).into());
                ctx.hir.intern(enum_path, enum_item);
                vec![exterior_t]
            }
        };
        let key_ts = param_ts.iter().map(|t| ctx.types.fresh()).collect();
        hir::Interface {
            interior: enum_path,
            exterior: param_ts,
            keys: key_ts,
            loc: self.loc,
        }
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
