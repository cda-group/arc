use super::Context;
use crate::compiler::ast;
use crate::compiler::hir::{self, Expr, ExprKind, Param, ParamKind, HIR};
use crate::compiler::info::diags::Error;
use crate::compiler::info::names::NameId;
use crate::compiler::info::types::TypeId;
use crate::compiler::shared::Lower;

use crate::compiler::hir::from::lower::path;

/// Lowers parameters but requires them to contain no patterns.
pub(super) fn lower_params_basic(ps: &Vec<ast::Param>, ctx: &mut Context) -> Vec<Param> {
    ps.iter()
        .map(|p| {
            let tv = p.ty.lower(ctx).unwrap_or_else(|| ctx.info.types.fresh());
            if let ast::PatKind::Var(x) = &p.pat.kind {
                let ux = ctx.res.stack.bind(*x, ctx.info).unwrap();
                Param::syn(ParamKind::Var(ux), tv)
            } else {
                Param::syn(ParamKind::Err, tv)
            }
        })
        .collect()
}

pub(super) fn lower_params(ps: &Vec<ast::Param>, ctx: &mut Context) -> (Vec<Param>, Vec<Case>) {
    let mut cases = Vec::new();
    let mut params = ps
        .iter()
        .map(|p| {
            let tv = p.ty.lower(ctx).unwrap_or_else(|| ctx.info.types.fresh());
            if let ast::PatKind::Var(x) = &p.pat.kind {
                let ux = ctx.res.stack.bind(*x, ctx.info).unwrap();
                Param::syn(ParamKind::Var(ux), tv)
            } else {
                let x = ctx.info.names.fresh();
                let ux = ctx.res.stack.bind(x, ctx.info).unwrap();
                let v = Expr::syn(ExprKind::Var(ux), tv);
                ssa(&p.pat, v, tv, false, ctx, &mut cases);
                Param::syn(ParamKind::Var(ux), tv)
            }
        })
        .collect();
    (params, cases)
}

/// Lowers a single `<pat>`, e.g., `on <pat> => <expr>`
pub(super) fn lower_pat(p: &ast::Pat, ctx: &mut Context) -> (Param, Vec<Case>) {
    let mut cases = Vec::new();
    let tv = ctx.info.types.fresh();
    let param = if let ast::PatKind::Var(x) = &p.kind {
        let ux = ctx.res.stack.bind(*x, ctx.info).unwrap();
        Param::syn(ParamKind::Var(ux), tv)
    } else {
        let x = ctx.info.names.fresh();
        let ux = ctx.res.stack.bind(x, ctx.info).unwrap();
        let v = Expr::syn(ExprKind::Var(ux), tv);
        ssa(&p, v, tv, true, ctx, &mut cases);
        Param::syn(ParamKind::Var(ux), tv)
    };
    (param, cases)
}

pub(super) fn lower_param_expr(p: &ast::Param, e: Expr, ctx: &mut Context) -> Vec<Case> {
    let mut cases = Vec::new();
    let tv = p.ty.lower(ctx).unwrap_or_else(|| ctx.info.types.fresh());
    ssa(&p.pat, e, tv, false, ctx, &mut cases);
    cases
}

pub(super) fn lower_branching_pat_expr(p: &ast::Pat, e: Expr, ctx: &mut Context) -> Vec<Case> {
    let mut cases = Vec::new();
    let tv = ctx.info.types.fresh();
    ssa(&p, e, tv, true, ctx, &mut cases);
    cases
}

/// Folds cases into a single expression. The `else_branch` should be specified if the
/// pattern is refutable (i.e., contains guards).
pub(super) fn fold_cases(then_branch: Expr, else_branch: Option<Expr>, cases: Vec<Case>) -> Expr {
    cases
        .into_iter()
        .rev()
        .fold(then_branch, |body, case| match case {
            // Case when the pattern is refutable, but does not bind anything:
            // if let 1 = b { c } else { d }
            Case::Guard { cond } => Expr {
                tv: body.tv,
                loc: body.loc,
                kind: ExprKind::If(
                    cond.into(),
                    body.into(),
                    else_branch.clone().unwrap().into(),
                ),
            },
            // Case when the pattern is irrefutable, but still binds something:
            // let a = b { c } else { d }
            Case::Bind { param, expr } => Expr {
                tv: body.tv,
                loc: None,
                kind: ExprKind::Let(param, expr.into(), body.into()),
            },
            _ => unreachable!(),
        })
}

#[derive(Debug)]
pub(super) enum Case {
    Guard { cond: Expr },
    Bind { param: Param, expr: Expr },
}

pub(super) struct CaseDebug<'a> {
    case: &'a Case,
    info: &'a Info,
    hir: &'a HIR,
}

impl Case {
    pub(super) fn debug<'a>(&'a self, info: &'a Info, hir: &'a HIR) -> CaseDebug<'a> {
        CaseDebug {
            case: self,
            info,
            hir,
        }
    }
}

use crate::compiler::info::Info;
use std::fmt::Display;

impl<'a> Display for CaseDebug<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "CaseDebug: {{")?;
        match self.case {
            Case::Guard { cond } => {
                writeln!(f, "    true == {}", hir::pretty(cond, self.hir, self.info))?
            }
            Case::Bind { param, expr } => match &param.kind {
                ParamKind::Var(name) => writeln!(
                    f,
                    "    {} => {}",
                    self.info.names.resolve(name.id),
                    hir::pretty(expr, self.hir, self.info)
                )?,
                ParamKind::Ignore => {
                    writeln!(f, "    _ => {}", hir::pretty(expr, self.hir, self.info))?
                }
                _ => writeln!(f, "<error>")?,
            },
            _ => {}
        }
        writeln!(f, "}}")?;
        Ok(())
    }
}

/// Transforms
///
///   if let Some(Some(a)) = b {
///       a+a
///   } else {
///       c+c
///   }
///
/// Into:
///
///   let x0 = b.is_some();
///   if x0 {
///       let x1 = b.unwrap();
///       let x2 = x1.is_some();
///       if x2 {
///           let a = x1.unwrap();
///           a
///       } else {
///           x3(c)
///       }
///   } else {
///       x3(c)
///   }
///
///   fn x3(c: i32) -> i32 {
///       c+c
///   }
///
fn ssa(
    // Left-hand-side of an SSA assignment
    p0: &ast::Pat,
    // Right-hand-side of an SSA assignment
    e0: Expr,
    t0: TypeId,
    branching: bool,
    ctx: &mut Context,
    cases: &mut Vec<Case>,
) {
    match &p0.kind {
        // Irrefutable patterns
        ast::PatKind::Struct(fs) => {
            let x0 = ctx.info.names.fresh();
            cases.push(Case::Bind {
                param: Param::syn(ParamKind::Var(x0), t0),
                expr: e0,
            });
            for f in fs {
                let v0 = Expr::syn(ExprKind::Var(x0), t0);
                let t1 = ctx.info.types.fresh();
                let e1 = Expr::syn(ExprKind::Access(v0.into(), f.name), t1);
                if let Some(p1) = &f.val {
                    ssa(p1, e1, ctx.info.types.fresh(), branching, ctx, cases)
                } else {
                    cases.push(Case::Bind {
                        param: Param::syn(ParamKind::Var(f.name), t1),
                        expr: e1,
                    });
                }
            }
        }
        ast::PatKind::Tuple(ps) => {
            let x0 = ctx.info.names.fresh();
            cases.push(Case::Bind {
                param: Param::syn(ParamKind::Var(x0), t0),
                expr: e0,
            });
            for (i, p1) in ps.iter().enumerate() {
                let v0 = Expr::syn(ExprKind::Var(x0), t0);
                let t1 = ctx.info.types.fresh();
                let e1 = Expr::syn(ExprKind::Project(v0.into(), i.into()), t1);
                ssa(p1, e1, ctx.info.types.fresh(), branching, ctx, cases)
            }
        }
        ast::PatKind::Var(x) => {
            if let Some(x0) = ctx.res.stack.bind(*x, ctx.info) {
                cases.push(Case::Bind {
                    param: Param::syn(ParamKind::Var(x0), t0),
                    expr: e0,
                });
            } else {
                ctx.info.diags.intern(Error::NameClash { name: *x });
            }
        }
        ast::PatKind::Ignore => cases.push(Case::Bind {
            param: Param::syn(ParamKind::Ignore, t0),
            expr: e0,
        }),
        ast::PatKind::Err => cases.push(Case::Bind {
            param: Param::syn(ParamKind::Err, t0),
            expr: e0,
        }),
        ast::PatKind::Or(p00, p01) => todo!(),
        ast::PatKind::Val(kind) if branching => cases.push(Case::Guard {
            cond: Expr::syn(
                ExprKind::BinOp(
                    e0.into(),
                    hir::BinOp::syn(hir::BinOpKind::Equ),
                    Expr::syn(ExprKind::Lit(kind.lower(ctx)), ctx.info.types.fresh()).into(),
                ),
                ctx.info.types.fresh(),
            ),
        }),
        ///   if let Some(a) = Some(b) {
        ///       a
        ///   } else {
        ///       1
        ///   }
        ///
        /// Becomes
        ///
        ///   let x0 = Some(b) in
        ///   let x1 = isa[Some](x0) in
        ///   if x1 {
        ///       let a = unwrap[Some](x0) in
        ///       a
        ///   } else {
        ///       1
        ///   }
        ast::PatKind::Variant(x, p) if branching => {
            if let Some(x) = path::lower_variant(x, ctx) {
                let x0 = ctx.info.names.fresh();
                cases.push(Case::Bind {
                    param: Param::syn(ParamKind::Var(x0), t0),
                    expr: e0,
                });
                let v0 = Expr::syn(ExprKind::Var(x0), t0);
                cases.push(Case::Guard {
                    cond: Expr::syn(ExprKind::Is(x, v0.clone().into()), ctx.info.types.fresh()),
                });
                let x1 = ctx.info.names.fresh();
                let t = ctx.info.types.fresh();
                let e = Expr::syn(ExprKind::Unwrap(x, v0.into()), t);
                ssa(p, e, t, false, ctx, cases);
            }
        }
        _ => ctx
            .info
            .diags
            .intern(Error::RefutablePattern { loc: p0.loc }),
    }
}

/// Extracts the parameters of a pattern
pub(super) fn params(ps: &Vec<ast::Param>, ctx: &mut Context) -> Vec<hir::Param> {
    let mut params = Vec::new();
    ps.iter().for_each(|p| p.pat.params(&mut params, ctx));
    params
}

impl ast::Pat {
    fn params(&self, params: &mut Vec<hir::Param>, ctx: &mut Context) {
        match &self.kind {
            ast::PatKind::Ignore => {}
            ast::PatKind::Struct(pfs) => pfs.iter().for_each(|f| {
                if let Some(p) = &f.val {
                    p.params(params, ctx)
                } else {
                    let tv = ctx.info.types.fresh();
                    let param = hir::Param::new(hir::ParamKind::Var(f.name), tv, self.loc);
                    params.push(param);
                }
            }),
            ast::PatKind::Tuple(ps) => ps.iter().for_each(|p| p.params(params, ctx)),
            ast::PatKind::Var(x) => {
                let tv = ctx.info.types.fresh();
                let param = hir::Param::new(hir::ParamKind::Var(*x), tv, self.loc);
                params.push(param);
            }
            ast::PatKind::Val(_) => unreachable!(),
            ast::PatKind::Or(_, _) => unreachable!(),
            ast::PatKind::Variant(_, _) => {}
            ast::PatKind::Err => {}
        }
    }
}

pub(super) fn params_to_args(params: &Vec<hir::Param>) -> Vec<hir::Expr> {
    params
        .iter()
        .map(|p| {
            let tv = p.tv;
            let x = if let hir::ParamKind::Var(x) = p.kind {
                x
            } else {
                unreachable!();
            };
            hir::Expr::syn(hir::ExprKind::Var(x), tv)
        })
        .collect::<Vec<_>>()
}
