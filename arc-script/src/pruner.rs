use ExprKind::*;
use {crate::ast::*, std::collections::HashMap};

impl Script<'_> {
    pub fn prune(&mut self) {
        let aliases = &mut HashMap::new();
        self.ast.for_each_expr(|expr| expr.prune_rec(aliases));
    }
}

impl Expr {
    /// Prune `let id1 = id2 in body` expressions
    /// Unwrapping is OK since let-binding variables are
    /// guaranteed to have a uid, expression-variables must be
    /// checked, since they might not be bound.
    pub fn prune_rec(&mut self, aliases: &mut HashMap<Ident, Ident>) {
        match &mut self.kind {
            Let(let_id, v, b) => {
                if let Var(var_id) = &mut v.kind {
                    aliases.insert(*let_id, *var_id);
                    *self = std::mem::take(b);
                    self.prune_rec(aliases);
                }
            }
            Var(var_id) => {
                if let Some(mut alias) = aliases.get(&var_id) {
                    while let Some(tmp) = aliases.get(&alias) {
                        alias = tmp;
                    }
                    *var_id = *alias;
                    self.prune_rec(aliases);
                }
            }
            _ => {}
        }
    }
}
