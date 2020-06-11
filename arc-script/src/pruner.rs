use {crate::ast::*, std::collections::HashMap};

impl Expr {
    pub fn prune(&mut self) {
        let aliases = &mut HashMap::new();
        self.for_each_expr(|expr, _____| expr.prune_rec(aliases))
    }

    /// Prune `let id1 = id2 in body` expressions
    /// Unwrapping is justified since let-binding variables are
    /// guaranteed to have a uid, expression-variables must be
    /// checked, since they might not be bound.
    pub fn prune_rec(&mut self, aliases: &mut HashMap<Uid, Ident>) {
        match &mut self.kind {
            ExprKind::Let(let_id, _, v, b) => match &mut v.kind {
                ExprKind::Var(var_id) if var_id.uid.is_some() => {
                    aliases.insert(let_id.uid.unwrap(), var_id.clone());
                    *self = std::mem::take(b);
                    self.prune_rec(aliases);
                }
                _ => {}
            },
            ExprKind::Var(var_id) if var_id.uid.is_some() => {
                if let Some(mut alias) = aliases.get(&var_id.uid.unwrap()) {
                    while let Some(tmp) = aliases.get(&alias.uid.unwrap()) {
                        alias = tmp;
                    }
                    *var_id = alias.clone();
                    self.prune_rec(aliases);
                }
            }
            _ => {}
        }
    }
}
