//! HIR -> MLIR:
//!   * Patterns are flattened (pattern compilation)
//!   * Functions are flattened (lambda lifting)
//!   * Expressions are flattened (ANF)
#![allow(unused)]

pub mod context;

use std::io::Read;

use context::ExprDecl;
use diagnostics::Error;
use im_rc::ordmap;
use im_rc::vector;
use im_rc::OrdMap;
use im_rc::Vector;
use info::Info;
use mlir::*;
use regex::Matches;
use regex::Regex;
use utils::OptionUtils;
use utils::VectorUtils;
use value::dynamic::Dataflow;

use crate::context::Context;
use crate::context::ScopeKind;
use crate::context::TypeDecl;

pub fn process(ctx: &mut Context, c: Dataflow, hir: Vector<hir::Stmt>) -> Vector<Item> {
    hir.into_iter().for_each(|s| lower_top_stmt(ctx, s));
    std::mem::take(&mut ctx.items)
}

pub fn process_eager(ctx: &mut Context, hir: Vector<hir::Stmt>) -> Vector<Item> {
    hir.into_iter().for_each(|s| lower_top_stmt(ctx, s));
    std::mem::take(&mut ctx.items)
}

fn lower_const(ctx: &mut Context, c: hir::Const) -> Const {
    match c {
        hir::CInt(c) => Const::CInt(c),
        hir::CFloat(c) => Const::CFloat(c),
        hir::CBool(c) => Const::CBool(c),
        hir::CString(c) => Const::CString(c),
        hir::CUnit => Const::CUnit,
        hir::CChar(c) => Const::CChar(c),
    }
}

fn lower_meta(ctx: &mut Context, m: hir::Meta) -> Meta {
    m.into_iter()
        .map(|(x, c)| (x, c.map(|c| lower_const(ctx, c))))
        .collect()
}

fn lower_top_stmt(ctx: &mut Context, s: hir::Stmt) {
    let info = s.info;
    match s.kind {
        hir::SDef(m, x, _, ps, t, b) => {
            let m = lower_meta(ctx, m);
            ctx.stack.bind_expr_decl(x.clone(), ExprDecl::Def);
            ctx.stack.push_scope(ScopeKind::Def);
            let vs = ps.mapm(ctx, lower_pattern);
            let b = lower_block(ctx, b);
            let t = lower_type(ctx, t, info);
            ctx.stack.pop_scope();
            ctx.items.push_back(IDef(x, vs, t, b).with(m));
        }
        hir::SBif(m, x, _, ts, t) => {
            let m = lower_meta(ctx, m);
            if m.get("staged").is_some() {
                return;
            }
            let t = lower_type(ctx, t, info);
            let vs = ts
                .into_iter()
                .enumerate()
                .map(|(i, t)| {
                    let x = format!("__{}", i);
                    let t = lower_type(ctx, t, info);
                    VName(x).with(t)
                })
                .collect();
            if let Some(Some(CString(x1))) = m.get("mlir") {
                ctx.stack
                    .bind_expr_decl(x.clone(), ExprDecl::Bif(x1.clone(), t.clone()));
            } else if let Some(Some(CString(x1))) = m.get("rust") {
                ctx.stack
                    .bind_expr_decl(x.clone(), ExprDecl::Bif(x1.clone(), t.clone()));
                ctx.items.push_back(IExternDef(x1.clone(), vs, t).with(m));
            } else {
                ctx.diagnostics.push_error(Error::UncompileableCode {
                    info,
                    msg: "Expected either `mlir` or `rust` attribute",
                });
            }
        }
        hir::SEnum(m, x, _, xts) => {
            let xts = xts.mapm(ctx, |ctx, (x, t)| (x, lower_type(ctx, t, info)));
            ctx.stack.bind_type_decl(x, TypeDecl::Enum(xts));
        }
        hir::SBit(m, x, _) => {
            let m = lower_meta(ctx, m);
            if m.get("staged").is_some() {
                return;
            }
            if let Some(Some(CString(x1))) = m.get("mlir") {
                ctx.stack.bind_type_decl(x, TypeDecl::Native(x1.clone()));
            } else if let Some(Some(CString(x1))) = m.get("rust") {
                ctx.stack.bind_type_decl(x, TypeDecl::Adt(x1.clone()));
            } else {
                ctx.diagnostics.push_error(Error::UncompileableCode {
                    info,
                    msg: r#"Type must be tagged with "mlir" or "rust""#,
                });
            }
        }
        hir::SVal(..) | hir::SExpr(..) => {
            ctx.diagnostics.push_error(Error::UncompileableCode {
                info,
                msg: "Top level statements are not allowed",
            });
        }
        hir::SRecDef(..) => todo!(),
        hir::SNoop => {}
    }
}

fn lower_row(ctx: &mut Context, t: hir::Type, info: Info, xts: &mut Vector<(Name, Type)>) {
    match t.kind.as_ref().clone() {
        hir::TRowEmpty => {}
        hir::TRowExtend((x, t), r) => {
            let t = lower_type(ctx, t, info);
            xts.push_back((x, t));
            lower_row(ctx, r, info, xts);
        }
        _ => unreachable!(),
    }
}

fn lower_type(ctx: &mut Context, t: hir::Type, info: Info) -> Type {
    match t.kind() {
        hir::TFun(ts, t) => {
            let ts = ts.mapm(ctx, |ctx, t| lower_type(ctx, t, info));
            let t = lower_type(ctx, t, info);
            TFun(ts, t).into()
        }
        hir::TRecord(t) => {
            let mut xts = Vector::new();
            lower_row(ctx, t, info, &mut xts);
            xts.sort_by(|(x0, _), (x1, _)| x0.cmp(x1));
            TRecord(xts).into()
        }
        hir::TArray(_, _) => todo!(),
        hir::TNominal(x, ts) => match ctx.stack.find_type_decl(&x) {
            Some(TypeDecl::Native(x1)) => {
                let ts = ts.mapm(ctx, |ctx, t| lower_type(ctx, t, info));
                TNative(x1, ts).into()
            }
            Some(TypeDecl::Adt(x1)) => {
                let ts = ts.mapm(ctx, |ctx, t| lower_type(ctx, t, info));
                TAdt(x1, ts).into()
            }
            Some(TypeDecl::Enum(xts)) => TEnum(xts).into(),
            None => unreachable!("Type should be declared by now {:?}", x),
        },
        hir::TAlias(_, _, t) => lower_type(ctx, t, info),
        hir::TVar(..)
        | hir::TGeneric(..)
        | hir::TRowEmpty
        | hir::TRowExtend(..)
        | hir::TRecordConcat(..)
        | hir::TTuple(..)
        | hir::TArrayConcat(..) => {
            ctx.diagnostics.push_error(Error::UncompileableCode {
                info,
                msg: "Type should be lowered by now",
            });
            TError.into()
        }
        hir::TUnit => TRecord(vector![]).into(),
        hir::TNever => TEnum(vector![]).into(),
        hir::TError => TError.into(),
    }
}

fn lower_pattern(ctx: &mut Context, p: hir::Pattern) -> Val {
    let t = lower_type(ctx, p.t, p.info);
    if let hir::PVal(x) = p.kind.as_ref().clone() {
        VName(x).with(t)
    } else {
        ctx.diagnostics.push_error(Error::UncompileableCode {
            info: p.info,
            msg: "Pattern not compiled",
        });
        VError.with(t)
    }
}

fn lower_expr_val(ctx: &mut Context, e: hir::Expr) -> Val {
    let t = lower_type(ctx, e.t, e.info);
    if let hir::ExprKind::EVal(x) = e.kind.as_ref().clone() {
        VName(x).with(t)
    } else {
        ctx.diagnostics.push_error(Error::UncompileableCode {
            info: e.info,
            msg: "Code not in ANF",
        });
        VError.with(t)
    }
}

fn lower_expr_stmt(ctx: &mut Context, s: hir::Stmt) -> Stmt {
    let (v, e) = match s.kind {
        hir::SVal(p, e) => {
            let v = lower_pattern(ctx, p);
            (vector![v], e)
        }
        hir::SExpr(e) => (vector![], e),
        e => unreachable!("{:?}", e),
    };
    match e.kind() {
        hir::EConst(c) => {
            let c = lower_const(ctx, c);
            SConst(c).with(v)
        }
        hir::EFunCall(e, es) => {
            let v1 = lower_expr_val(ctx, e);
            let vs = es.mapm(ctx, lower_expr_val);
            SFunCallIndirect(v1, vs).with(v)
        }
        hir::EFunCallDirect(x, ts, es) => match ctx.stack.find_expr_decl(&x) {
            Some(ExprDecl::Bif(x1, _)) => {
                let es = es.mapm(ctx, lower_expr_val);
                SFunCallDirect(x1, es).with(v)
            }
            Some(ExprDecl::Def) => {
                let es = es.mapm(ctx, lower_expr_val);
                SFunCallDirect(x, es).with(v)
            }
            y => unreachable!("{:?} {:?}", x, y),
        },
        hir::EFunReturn(e) => {
            let v1 = lower_expr_val(ctx, e);
            SFunReturn(v1).with(v)
        }
        hir::ELoopBreak(e) => {
            let v1 = lower_expr_val(ctx, e);
            SWhileBreak(vector![v1]).with(v)
        }
        hir::EIfElse(e, b0, b1) => {
            let v1 = lower_expr_val(ctx, e);
            let b0 = lower_block(ctx, b0);
            let b1 = lower_block(ctx, b1);
            SIfElse(v1, b0, b1).with(v)
        }
        hir::ERecord(xes) => {
            let xes = xes.mapm(ctx, |ctx, (x, e)| {
                let v = lower_expr_val(ctx, e);
                (x, v)
            });
            SRecord(xes).with(v)
        }
        hir::ERecordConcat(_, _) => todo!(),
        hir::ERecordAccess(e, x) => {
            let v1 = lower_expr_val(ctx, e);
            SRecordAccess(v1, x).with(v)
        }
        hir::EDef(x, ts) => match ctx.stack.find_expr_decl(&x) {
            Some(ExprDecl::Bif(x1, _)) => {
                let ts = ts.mapm(ctx, |ctx, t| lower_type(ctx, t, s.info));
                SFun(x1).with(v)
            }
            Some(ExprDecl::Def) => {
                let ts = ts.mapm(ctx, |ctx, t| lower_type(ctx, t, s.info));
                SFun(x).with(v)
            }
            _ => unreachable!(),
        },
        hir::EVariant(_, _, x, e) => {
            let v1 = lower_expr_val(ctx, e);
            SVariant(x, v1).with(v)
        }
        hir::EVariantAccess(_, _, x, e) => {
            let v1 = lower_expr_val(ctx, e);
            SVariantAccess(x, v1).with(v)
        }
        hir::EVariantCheck(_, _, x, e) => {
            let v1 = lower_expr_val(ctx, e);
            SVariantCheck(x, v1).with(v)
        }
        hir::EArray(_) => todo!(),
        hir::EArrayConcat(_, _) => todo!(),
        hir::ELoop(_)
        | hir::ELoopContinue
        | hir::EArrayAccess(_, _)
        | hir::EFun(_, _, _)
        | hir::EMatch(_, _)
        | hir::EMut(_, _)
        | hir::EVal(_)
        | hir::EVar(_)
        | hir::EDo(_)
        | hir::ENoop(_)
        | hir::ETuple(_)
        | hir::ETupleAccess(_, _)
        | hir::EFor(_, _, _)
        | hir::EWhile(_, _) => {
            ctx.diagnostics.push_error(Error::UncompileableCode {
                info: e.info,
                msg: "Complex expression",
            });
            SError.with(v)
        }
        hir::EError => SError.with(v),
    }
}

fn lower_block(ctx: &mut Context, b: hir::Block) -> Block {
    ctx.stack.push_scope(ScopeKind::Block);
    let mut ss = b.ss.mapm(ctx, lower_expr_stmt);
    let v = lower_expr_val(ctx, b.e);
    ctx.stack.pop_scope();
    match ctx.stack.iter().next().unwrap().kind {
        ScopeKind::Def => ss.push_back(SFunReturn(v).with([])),
        ScopeKind::While => ss.push_back(SWhileYield(vector![v]).with([])),
        _ => ss.push_back(SBlockResult(v).with([])),
    }
    Block::new(ss)
}
