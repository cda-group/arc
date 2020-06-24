use crate::ast::*;

impl Expr {
    /// Give each variable a unique identifier
    /// Requires that variables can be distinguished by their `name`
    /// Should be run directly after building the AST.
    pub fn assign_uid(&mut self) {
        Uid::reset();
        self.for_each_expr(|expr, stack| match &mut expr.kind {
            ExprKind::Let(id, ..) => id.uid = Some(Uid::new()),
            ExprKind::Var(id) => {
                if let Some((_, _, uid)) = id.lookup_with_name(stack) {
                    id.uid = uid
                }
            }
            _ => {}
        })
    }

    /// Give each variable a scope
    /// Requires that variables can be distinguished by their `uid`
    /// Should be run directly after transforming the AST is transformed.
    pub fn assign_scope(&mut self) {
        self.for_each_expr(|expr, stack| match &mut expr.kind {
            ExprKind::Let(id, ..) => id.scope = Some(stack.len()),
            ExprKind::Var(id) => {
                if let Some((_, scope, _)) = id.lookup_with_uid(stack) {
                    id.scope = Some(scope)
                }
            }
            _ => {}
        })
    }
}
