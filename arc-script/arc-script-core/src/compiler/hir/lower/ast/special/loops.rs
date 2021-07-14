use super::Context;

use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::hir::lower::ast::utils::is_async::IsAsync;

use arc_script_core_shared::Lower;
use arc_script_core_shared::VecMap;

/// Lowers a loop into a FSM if it contains an async operation. Otherwise the loop is lowered into
/// a regular loop. For example:
/// ```skip
/// task Foo(): ~i32 -> (~i32, i32) {
///     var x: i32 = 0;
///     loop { if x < 100 {         x += 1; } else { break } };
///     loop { if x < 200 { on y => x += y; } else { break } };
///     x += 1;
///     return x;
/// }
/// ```
/// becomes
/// ```skip
/// task Foo(): ~i32 -> ~i32 {
///     enum State { S0(unit), S1((i32,)), S2((i32,)), S3((i32,)), R(i32) }
///     fun on_start(): State {
///         State::S0(unit)
///     }
///     fun on_event(state: State): State {
///         if is[S0](state) {
///             val values = unwrap[S0](state);
///             var x: i32 = 0;
///             loop { if x < 100 { x += 1; } else { break } };
///             S1((x,))
///         } else {
///             if is[S1](state) {
///                 val values = unwrap[S3](state);
///                 val x = values.0;
///                 if x < 200 { S2((x,)) } else { S3((x,)) }
///             } else {
///                 if is[S2](state) {
///                     val values = unwrap[S3](state);
///                     var x = values.0;
///                     val y = unwrap[Some](event);
///                     x += y;
///                     S1((x,))
///                 } else {
///                     if is[S3](state) {
///                         val values = unwrap[S3](state);
///                         var x = values.0;
///                         x += 1;
///                         R(x)
///                     } else {
///                         panic
///                     }
///                 }
///             }
///         }
///     }
///     fun handle(state)
///     return State0(x);
///     fun action0(state: State)
/// }
/// ```
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
