use crate::{prelude::*, info::*, utils::*};

type Context = Vec<(Ident, Expr)>;

trait Ssa {
    fn flatten(self, info: &mut Info) -> (Context, Self);
}

impl Script<'_> {
    pub fn into_ssa(mut self) -> Self {
        let info = &mut self.info;
        self.ast.body = self.ast.body.into_ssa(info);
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
        let typer = &mut info.typer.borrow_mut();
        ctx.into_iter().rev().fold(var, |acc, (id, expr)| Expr {
            tv: acc.tv,
            span: acc.span,
            kind: BinOp(
                Box::new(Expr {
                    tv: typer.intern(Scalar(Unit)),
                    span: acc.span,
                    kind: Let(id, Box::new(expr)),
                }),
                Seq,
                Box::new(acc),
            ),
        })
    }
}

impl Ssa for Expr {
    /// Turns an expression into a flat list of assignments and a variable
    fn flatten(self, info: &mut Info) -> (Context, Self) {
        let Self { kind, tv, span } = self;
        let (mut ctx, kind) = match kind {
            BinOp(l, op, r) => {
                if let Seq = op {
                    if let Let(id, v) = l.kind {
                        let (mut vc, v) = match v.is_imm() {
                            true => (vec![], *v),
                            false => v.flatten(info),
                        };
                        let (rc, r) = r.flatten(info);
                        vc.push((id, v));
                        return (merge(vc, rc), r);
                    }
                }
                let ((lc, l), (rc, r)) = (l.flatten(info), r.flatten(info));
                (merge(lc, rc), BinOp(Box::new(l), op, Box::new(r)))
            }
            UnOp(o, e) => {
                let (oc, o) = o.flatten(info);
                let (ec, e) = e.flatten(info);
                (merge(oc, ec), UnOp(o, Box::new(e)))
            }
            If(c, t, e) => {
                let ((cc, c), t, e) = (c.flatten(info), t.into_ssa(info), e.into_ssa(info));
                (cc, If(Box::new(c), Box::new(t), Box::new(e)))
            }
            Closure(ps, e) => {
                let e = e.into_ssa(info);
                (Vec::new(), Closure(ps, Box::new(e)))
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
        match &self.kind {
            Var(_) | Lit(_) => true,
            _ => false,
        }
    }
}

impl Ssa for UnOpKind {
    fn flatten(self, info: &mut Info) -> (Context, Self) {
        match self {
            FunCall(a) => {
                let (ac, a) = a.flatten(info);
                (ac, FunCall(a))
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

impl<T> Ssa for Vec<(T, Expr)> {
    fn flatten(self, info: &mut Info) -> (Context, Self) {
        let mut ctx = vec![];
        let exprs = self
            .into_iter()
            .map(|(x, e)| {
                let (mut ec, e) = e.flatten(info);
                ctx.append(&mut ec);
                (x, e)
            })
            .collect();
        (ctx, exprs)
    }
}
