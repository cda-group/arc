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

/// Lowers a block inside a task into a finite state machine (FSM).
pub(crate) fn lower_fsm(block: &ast::Block, ctx: &mut Context<'_>) -> hir::FSM {
    ctx.res.stack.push_scope();
    let v = ctx.new_expr_unit().into_ssa(ctx);
    let stmts = ctx.res.stack.pop_scope();
    let body = hir::Block::syn(stmts, v);
    let name = ctx.names.common.on_start;
    let path = ctx.new_path(name);
    let f = ctx.new_method(path, vec![], body);
    ctx.hir.intern(path, hir::Item::syn(hir::ItemKind::Fun(f)));
    let fsm = hir::FSM {
        state_enum: todo!(),
        event_funs: todo!(),
    };
    fsm
}

/// Lowers an `on { (<pattern> => <expr>)* }` statement-expression.
/// TODO: Currently we assume there is only one case-pattern.
fn lower_on_event(case: ast::Case, ctx: &mut Context<'_>) -> hir::Expr {
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
    todo!()
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
