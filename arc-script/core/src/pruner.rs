use {crate::prelude::*, std::collections::HashMap};

impl Script<'_> {
    pub fn prune(&mut self) {
        let aliases = &mut HashMap::new();
        self.ast.for_each_expr(|expr| expr.prune_rec(aliases));
    }
}

impl Expr {
    /// Remove `let id1 = id2` expressions
    pub fn prune_rec(&mut self, aliases: &mut HashMap<Ident, Ident>) {
        match &mut self.kind {
            BinOp(lhs, Seq, rhs) => {
                if let Let(let_id, v) = &mut lhs.kind {
                    if let Var(var_id) = &mut v.kind {
                        aliases.insert(*let_id, *var_id);
                        *self = rhs.take();
                        self.prune_rec(aliases);
                    }
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
