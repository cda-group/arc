use crate::{ast::*, utils::*};

type Context = Vec<(Ident, Expr)>;

trait Ssa {
    fn flatten(self) -> (Context, Self);
}

impl Expr {
    /// Reverse-folds a vec of SSA values into an AST of let-expressions
    pub fn into_ssa(self) -> Expr {
        let (ctx, var) = self.flatten();
        ctx.into_iter().rev().fold(var, |acc, (id, expr)| Expr {
            ty: acc.ty.clone(),
            span: acc.span.clone(),
            kind: ExprKind::Let(id, expr.ty.clone(), Box::new(expr), Box::new(acc)),
        })
    }
}
impl Ssa for Expr {
    /// Turns an expression into a flat list of assignments and a variable
    fn flatten(self) -> (Context, Expr) {
        let Expr { kind, ty, span } = self;
        use {Bif::*};
        let (mut ctx, kind) = match kind {
            ExprKind::BinOp(l, op, r) => {
                let ((lc, l), (rc, r)) = (l.flatten(), r.flatten());
                (merge(lc, rc), ExprKind::BinOp(Box::new(l), op, Box::new(r)))
            }
            ExprKind::UnOp(op, e) => {
                let (ec, e) = e.flatten();
                (ec, ExprKind::UnOp(op, Box::new(e)))
            }
            ExprKind::If(c, t, e) => {
                let ((cc, c), t, e) = (c.flatten(), t.into_ssa(), e.into_ssa());
                (cc, ExprKind::If(Box::new(c), Box::new(t), Box::new(e)))
            }
            ExprKind::Bif(Fmap(e)) => {
                let (ec, e) = e.flatten();
                (ec, ExprKind::Bif(Bif::Fmap(Box::new(e))))
            }
            ExprKind::Bif(Fold(a, f)) => {
                let ((ac, a), (fc, f)) = (a.flatten(), f.flatten());
                (merge(ac, fc), ExprKind::Bif(Fold(Box::new(a), Box::new(f))))
            }
            ExprKind::Call(id, a) => {
                let (ac, a) = a.flatten();
                (ac, ExprKind::Call(id, a))
            }
            ExprKind::Let(id, _, v, b) => {
                let (mut vc, v) = match v.is_imm() {
                    true => (vec![], *v),
                    false => v.flatten(),
                };
                let (bc, b) = b.flatten();
                vc.push((id, v));
                return (merge(vc, bc), b);
            }
            ExprKind::Array(e) => {
                let (ec, e) = e.flatten();
                (ec, ExprKind::Array(e))
            }
            ExprKind::Struct(a) => {
                let (ac, a) = a.flatten();
                (ac, ExprKind::Struct(a))
            }
            kind @ ExprKind::Var(_) => return (vec![], Expr { kind, ty, span }),
            kind => (vec![], kind),
        };
        let id = Ident::gen();
        let var = Expr::new(ExprKind::Var(id.clone()), ty.clone(), span);
        let expr = Expr { kind, ty, span };
        ctx.push((id, expr));
        (ctx, var)
    }
}

impl Expr {
    /// Returns true if an expression has no subexpressions
    fn is_imm(&self) -> bool {
        match &self.kind {
            ExprKind::Var(_) => true,
            ExprKind::Lit(_) => true,
            _ => false,
        }
    }
}

impl Ssa for Vec<Expr> {
    fn flatten(self) -> (Context, Self) {
        let mut ctx = vec![];
        let exprs = self
            .into_iter()
            .map(|e| {
                let (mut ec, e) = e.flatten();
                ctx.append(&mut ec);
                e
            })
            .collect();
        (ctx, exprs)
    }
}

impl Ssa for Vec<(Ident, Expr)> {
    fn flatten(self) -> (Context, Self) {
        let mut ctx = vec![];
        let exprs = self
            .into_iter()
            .map(|(id, e)| {
                let (mut ec, e) = e.flatten();
                ctx.append(&mut ec);
                (id, e)
            })
            .collect();
        (ctx, exprs)
    }
}
