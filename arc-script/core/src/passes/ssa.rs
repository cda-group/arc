use crate::prelude::*;

type Context = Vec<(Ident, Expr)>;

trait Ssa {
    fn flatten(self, info: &mut Info) -> (Context, Self);
}

impl Script<'_> {
    pub fn into_ssa(mut self) -> Self {
        let info = &mut self.info;
        self.ast.fundefs = self
            .ast
            .fundefs
            .into_iter()
            .map(|(ident, mut fundef)| {
                fundef.body = fundef.body.into_ssa(info);
                (ident, fundef)
            })
            .collect();
        self
    }
}

impl Expr {
    /// Reverse-folds a vec of SSA values into an AST of let-expressions
    pub fn into_ssa(self, info: &mut Info) -> Self {
        let (ctx, var) = self.flatten(info);
        ctx.into_iter().rev().fold(var, |acc, (id, expr)| Expr {
            tv: acc.tv,
            span: acc.span,
            kind: Let(id, expr.into(), acc.into()),
        })
    }
}

impl Ssa for Expr {
    /// Turns an expression into a flat list of assignments and a variable
    fn flatten(self, info: &mut Info) -> (Context, Self) {
        let Self { kind, tv, span } = self;
        let (mut ctx, kind) = match kind {
            Let(id, e1, e2) => {
                let (mut vc, v) = if e1.is_imm() {
                    (vec![], *e1)
                } else {
                    e1.flatten(info)
                };
                let (rc, r) = e2.flatten(info);
                vc.push((id, v));
                return (merge(vc, rc), r);
            }
            BinOp(l, op, r) => {
                let ((lc, l), (rc, r)) = (l.flatten(info), r.flatten(info));
                (merge(lc, rc), BinOp(l.into(), op, r.into()))
            }
            UnOp(o, e) => {
                let (oc, o) = o.flatten(info);
                let (ec, e) = e.flatten(info);
                (merge(oc, ec), UnOp(o, e.into()))
            }
            If(c, t, e) => {
                let ((cc, c), t, e) = (c.flatten(info), t.into_ssa(info), e.into_ssa(info));
                (cc, If(c.into(), t.into(), e.into()))
            }
            Closure(ps, e) => {
                let e = e.into_ssa(info);
                (Vec::new(), Closure(ps, e.into()))
            }
            ConsArray(e) => {
                let (ec, e) = e.flatten(info);
                (ec, ConsArray(e))
            }
            ConsStruct(a) => {
                let (ac, a) = a.flatten(info);
                (ac, ConsStruct(a))
            }
            kind @ Var(_) => return (vec![], Expr { kind, tv, span }),
            kind => (vec![], kind),
        };
        let id = info.table.genvar(tv);
        let var = Self::new(Var(id), tv, span);
        let expr = Self::new(kind, tv, span);
        ctx.push((id, expr));
        (ctx, var)
    }
}

impl Expr {
    /// Returns true if an expression has no subexpressions
    fn is_imm(&self) -> bool {
        matches!(&self.kind, Var(_) | Lit(_))
    }
}

impl Ssa for UnOpKind {
    fn flatten(self, info: &mut Info) -> (Context, Self) {
        match self {
            Call(a) => {
                let (ac, a) = a.flatten(info);
                (ac, Call(a))
            }
            _ => (Vec::new(), self),
        }
    }
}

impl Ssa for Vec<Expr> {
    fn flatten(self, info: &mut Info) -> (Context, Self) {
        let mut ctx = vec![];
        let exprs = self
            .into_iter()
            .map(|e| {
                let (mut ec, e) = e.flatten(info);
                ctx.append(&mut ec);
                e
            })
            .collect();
        (ctx, exprs)
    }
}

impl Ssa for VecMap<Symbol, Expr> {
    fn flatten(self, info: &mut Info) -> (Context, Self) {
        let mut ctx = vec![];
        let exprs = self
            .into_iter()
            .map(|(f, e)| {
                let (mut ec, e) = e.flatten(info);
                ctx.append(&mut ec);
                (f, e)
            })
            .collect();
        (ctx, exprs)
    }
}
