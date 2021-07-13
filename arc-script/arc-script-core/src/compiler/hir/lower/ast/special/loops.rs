use super::Context;

use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::hir::lower::ast::utils::is_async::IsAsync;

use arc_script_core_shared::Lower;
use arc_script_core_shared::VecMap;

/// Lowers a loop into a FSM if it contains an async operation. Otherwise the loop is lowered into
/// a regular loop.
pub(crate) fn lower_loop(b: &ast::Block, ctx: &mut Context<'_>) -> hir::ExprKind {
    if b.is_async(ctx) {
        let state = ctx.state_enum.unwrap();
        let vars = ctx
            .res
            .stack
            .capture()
            .into_iter()
            .map(|x| {
                let t = ctx.types.fresh();
                hir::Var::syn(hir::VarKind::Ok(x, hir::ScopeKind::Local), t)
            })
            .collect::<Vec<_>>();
        let tuple = ctx.new_expr_tuple(vars).into_ssa(ctx);

        let entry_state_t = ctx.types.fresh();
        let entry_state = ctx.new_variant(state, entry_state_t);
        let entry_state_x = entry_state.path;
        let entry_state_item = hir::Item::syn(ctx.new_variant(state, entry_state_t).into());
        ctx.hir.intern(entry_state_x, entry_state_item);
        let entry_state_v = ctx.new_expr(hir::ExprKind::Enwrap(entry_state_x, tuple));

        let exit_state_t = ctx.types.fresh();
        let exit_state = hir::Item::syn(ctx.new_variant(state, entry_state_t).into());
        let exit_state_x = entry_state.path;
        let exit_state_item = hir::Item::syn(ctx.new_variant(state, exit_state_t).into());
        ctx.hir.intern(exit_state_x, exit_state_item);
        let exit_state_v = ctx.new_expr(hir::ExprKind::Enwrap(exit_state_x, tuple));
        todo!()
        // Construct loop-entry state of all captured variables
        // Construct loop-exit state of all captured variables
        // The first statement must construct and return the loop-entry state
        // All continue statements must construct and return the loop-entry state
        // All break statements must construct and return the loop-exit state
    } else {
        hir::ExprKind::Loop(b.lower(ctx))
    }
}
