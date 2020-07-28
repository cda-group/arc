use crate::ast::*;
use ExprKind::*;

impl Script {
    pub fn assign_scope(&mut self) {
        self.body.assign_scope();
    }
}

impl Expr {
    pub fn assign_scope(&mut self) {
        self.for_each_expr(|expr, stack| match &mut expr.kind {
            Let(let_id, ..) => let_id.scope = Some(stack.len()),
            Var(var_id) => {
                if let Some((let_id, _)) = var_id.lookup_with_name(stack) {
                    var_id.scope = let_id.scope
                }
            }
            _ => {}
        })
    }
}
