//! Lowers a let-expression `let p = e0 in e1`
//!
//! For example:
//! ```txt
//! let (a, (_, c)) = (1, (2, 3));
//! a
//! ```
//! should after SSA become something like:
//! ```txt
//! let x0 = 1;
//! let x1 = 2;
//! let x2 = 3;
//! let x4 = (x1, x2);
//! let x5 = (x0, x4);
//! let (a, (_, c)) = x5;
//! a
//! ```
//! then you need to somehow do
//! ```txt
//! // construct
//! let x0 = 1;
//! let x1 = 2;
//! let x2 = 3;
//! let x4 = (x1, x2);
//! let x5 = (x0, x4);
//! // deconstruct
//! let a = x5.0;
//! let x7 = x5.1;
//! let x8 = x7.1;
//! a
//! ```
//!

use super::Context;
use crate::compiler::ast;
use crate::compiler::hir::{Expr, ExprKind, Param, ParamKind};
use crate::compiler::info::diags::Error;
use crate::compiler::info::names::NameId;
use crate::compiler::info::types::TypeId;
use crate::compiler::shared::Lower;

pub(super) fn lower_params(
    ps: &Vec<ast::Param>,
    ctx: &mut Context,
) -> (Vec<Param>, Vec<(Param, Expr)>) {
    let mut env = Vec::new();
    let mut params = Vec::new();
    for p in ps {
        let tv = p.ty.lower(ctx).unwrap_or_else(|| ctx.info.types.fresh());
        if let ast::PatKind::Var(x) = &p.pat.kind {
            let ux = ctx.res.stack.bind(*x, ctx.info).unwrap();
            params.push(Param::syn(ParamKind::Var(ux.into()), tv));
        } else {
            let x = ctx.info.names.fresh();
            let x = ctx.res.stack.bind(x, ctx.info).unwrap();
            let v = Expr::syn(ExprKind::Var(x.into()), tv);
            p.pat.ssa(v, tv, ctx, &mut env);
            params.push(Param::syn(ParamKind::Var(x.into()), tv));
        }
    }
    (params, env)
}

pub(super) fn lower_param_expr(p: &ast::Param, e: Expr, ctx: &mut Context) -> Vec<(Param, Expr)> {
    let mut env = Vec::new();
    let tv = p.ty.lower(ctx).unwrap_or_else(|| ctx.info.types.fresh());
    p.pat.ssa(e, tv, ctx, &mut env);
    env
}

pub(super) fn fold_env(e: Expr, env: Vec<(Param, Expr)>) -> Expr {
    env.into_iter().rev().fold(e, |e1, (p, e0)| Expr {
        tv: e1.tv,
        loc: None,
        kind: ExprKind::Let(p, e0.into(), e1.into()),
    })
}

impl ast::Pat {
    fn ssa(&self, e0: Expr, t0: TypeId, ctx: &mut Context, env: &mut Vec<(Param, Expr)>) {
        match &self.kind {
            ast::PatKind::Struct(fs) => {
                let x0 = ctx.info.names.fresh();
                env.push((Param::syn(ParamKind::Var(x0.into()), t0), e0));
                for f in fs {
                    let v0 = Expr::syn(ExprKind::Var(x0.into()), t0);
                    let t1 = ctx.info.types.fresh();
                    let e1 = Expr::syn(ExprKind::Access(v0.into(), f.name.id.into()), t1);
                    if let Some(p) = &f.val {
                        p.ssa(e1, ctx.info.types.fresh(), ctx, env)
                    } else {
                        env.push((Param::syn(ParamKind::Var(f.name.id.into()), t1), e1));
                    }
                }
            }
            ast::PatKind::Tuple(ps) => {
                let x0 = ctx.info.names.fresh();
                env.push((Param::syn(ParamKind::Var(x0.into()), t0), e0));
                for (i, p) in ps.iter().enumerate() {
                    let v0 = Expr::syn(ExprKind::Var(x0.into()), t0);
                    let t1 = ctx.info.types.fresh();
                    let e1 = Expr::syn(ExprKind::Project(v0.into(), i.into()), t1);
                    p.ssa(e1, ctx.info.types.fresh(), ctx, env)
                }
            }
            ast::PatKind::Var(x) => {
                if let Some(x0) = ctx.res.stack.bind(*x, ctx.info) {
                    env.push((Param::syn(ParamKind::Var(x0.into()), t0), e0));
                } else {
                    ctx.info.diags.intern(Error::NameClash { name: *x });
                }
            }
            ast::PatKind::Ignore => env.push((Param::syn(ParamKind::Ignore, t0), e0)),
            ast::PatKind::Err => env.push((Param::syn(ParamKind::Err, t0), e0)),
            _ => ctx
                .info
                .diags
                .intern(Error::RefutablePattern { loc: self.loc }),
        }
    }
}
