//! AST => HIR:
//! * Name resolution
//!   * Check that each expression variable and type variable is bound
//!   * Disambiguate variants from functions and variables
//!     * e.g., Foo(p)
//!     * e.g., Foo(1)
//!   * Disambiguate mutable variables from immutable variables
//!   * Generate a unique identifier for each function, enum, type alias, and variable
//! * Generate type variables
//!   * Each expr, pattern, and missing type (e.g., _) gets a type variable
//! * Syntax desugaring
//!   * String interpolation desugaring
//!   * Method call desugaring
//!   * Convert records to rows
//!   * Convert () and ! to nominal types
//!   * Convert operators to functions
//! * Syntactic checks:
//!   * No duplicate fields, variants or attributes
//!   * No return outside of function, no break outside of loop
//!   * No var captured by a function
//!   * Correct number of expression arguments to functions
//!   * Correct number of type arguments to type aliases and enums
//!   * Check that only place expressions are mutated (variables, arrays, records)

//! TODO:
//! * Disambiguate functions from variables
//!   * e.g., foo[i32,i32] and foo[1,2,3]
//!   * e.g., foo.bar() and Foo.Bar()
#![allow(unused)]
pub mod context;

use ast::Body;
use ast::ExprField;
use context::ExprDecl;
use context::ScopeKind;
use context::TypeDecl;
use diagnostics::Diagnostics;
use diagnostics::Error;
use diagnostics::Warning;
use hir::*;
use im_rc::ordmap;
use im_rc::ordmap::Entry;
use im_rc::vector;
use im_rc::OrdMap;
use im_rc::Vector;
use info::Info;
use regex::Matches;
use regex::Regex;
use utils::AssocVectorUtils;
use utils::OptionUtils;
use utils::VectorUtils;

use crate::context::Context;

pub fn process(ctx: &mut Context, ss: Vector<ast::Stmt>) -> Vector<Stmt> {
    ss.map(|s| lower_stmt(ctx, s))
}

pub fn lower_stmt(ctx: &mut Context, s: ast::Stmt) -> Stmt {
    let info = s.info;
    match s.kind {
        ast::SNoop => SNoop.with(info),
        ast::SVal(p, e) => {
            let e = lower_expr(ctx, e);
            let p = lower_pattern(ctx, p, false);
            SVal(p, e).with(info)
        }
        ast::SVar(p, e) => {
            let e = lower_expr(ctx, e);
            let p = lower_pattern(ctx, p, true);
            SVal(p, e).with(info)
        }
        ast::SExpr(e) => {
            let e = lower_expr(ctx, e);
            SExpr(e).with(info)
        }
        ast::SDef(m, x, gs, ps, t, bs, b) => {
            ctx.stack
                .bind_expr_decl(x.clone(), ExprDecl::Def(info, ps.len(), gs.clone()));
            let m = lower_meta(ctx, m);
            ctx.stack.push_scope(ScopeKind::Def);
            gs.clone()
                .into_iter()
                .for_each(|x| lower_generic(ctx, x, info));
            let ps = ps.mapm(ctx, |ctx, p| lower_pattern(ctx, p, false));
            let t = lower_type_or_fresh(ctx, t);
            let b = lower_body(ctx, b);
            ctx.stack.pop_scope();
            SDef(m, x, gs, ps, t, b).with(info)
        }
        // TODO: Fix so that enums are correctly lexically scoped, i.e.,
        //     enum X { A }
        //     val a1 = X::A;
        //     use X::A;
        //     val a = A;
        //  Currently it won't work with generic types
        ast::SEnum(m, x, gs, _bs, xts) => {
            let m = lower_meta(ctx, m);
            gs.clone()
                .into_iter()
                .for_each(|x| lower_generic(ctx, x, info));
            let xts = xts
                .into_iter()
                .filter_map(|v| lower_enum_variant(ctx, x.clone(), gs.clone(), v, info))
                .collect::<Vector<_>>();
            ctx.stack
                .bind_type_decl(x.clone(), TypeDecl::Enum(info, gs.clone(), xts.clone()));
            SEnum(m, x, gs, xts).with(info)
        }
        ast::SType(m, x, gs, t) => {
            let m = lower_meta(ctx, m);
            ctx.stack.push_scope(ScopeKind::Type);
            gs.clone()
                .into_iter()
                .for_each(|x| lower_generic(ctx, x, info));
            let t = lower_type(ctx, t);
            ctx.stack.pop_scope();
            ctx.stack
                .bind_type_decl(x.clone(), TypeDecl::Type(info, gs.clone(), t.clone()));
            SNoop.with(info)
        }
        ast::SBuiltinDef(m, x, gs, ts, t, bs) => {
            let m = lower_meta(ctx, m);
            gs.clone()
                .into_iter()
                .for_each(|x| lower_generic(ctx, x, info));
            let ts = ts.mapm(ctx, lower_type);
            let t = lower_type_or_unit(ctx, t);
            ctx.stack
                .bind_expr_decl(x.clone(), ExprDecl::Def(info, ts.len(), gs.clone()));
            SBif(m, x, gs, ts, t).with(info)
        }
        ast::SBuiltinType(m, x, gs, bs) => {
            let m = lower_meta(ctx, m);
            gs.clone()
                .into_iter()
                .for_each(|x| lower_generic(ctx, x, info));
            ctx.stack
                .bind_type_decl(x.clone(), TypeDecl::Bit(info, gs.clone()));
            SBit(m, x, gs).with(info)
        }
        ast::SBuiltinClass(_, _, _, _) => todo!(),
        ast::SBuiltinInstance(_, _, _, _, _) => todo!(),
        ast::SInject(_, _) => todo!(),
    }
}

fn lower_const(ctx: &mut Context, c: ast::Const) -> Const {
    match c {
        ast::CInt(c) => CInt(c),
        ast::CFloat(c) => CFloat(c),
        ast::CBool(c) => CBool(c),
        ast::CString(c) => CString(c),
        ast::CUnit => CUnit,
        ast::CChar(c) => CChar(c),
    }
}

fn lower_meta(ctx: &mut Context, meta: ast::Meta) -> Meta {
    let mut map = ordmap![];
    for a in meta.into_iter() {
        if map.contains_key(&a.x) {
            ctx.diagnostics.push_error(Error::DuplicateMetaKey {
                key: a.x,
                info: a.info,
            });
        } else {
            let c = a.c.map(|c| lower_const(ctx, c));
            map.insert(a.x, c);
        }
    }
    map
}

fn lower_generic(ctx: &mut Context, x: ast::Generic, info: Info) {
    ctx.stack.bind_type_decl(x, TypeDecl::Generic(info))
}

fn lower_type(ctx: &mut Context, t: ast::Type) -> Type {
    let info = t.info;
    match (*t.kind).clone() {
        ast::TParen(t) => lower_type(ctx, t),
        ast::TFun(ts, t) => {
            let ts = ts.mapm(ctx, lower_type);
            let t = lower_type(ctx, t);
            TFun(ts, t).into()
        }
        ast::TTuple(ts) => {
            let ts = ts.mapm(ctx, lower_type);
            TTuple(ts, true).into()
        }
        ast::TRecord(xts) => {
            let xts = xts.mapm_assoc(ctx, lower_type);
            let t = fields_to_rows(xts, TRowEmpty.into());
            TRecord(t).into()
        }
        ast::TRecordConcat(t0, t1) => {
            let t0 = lower_type(ctx, t0);
            let t1 = lower_type(ctx, t1);
            TRecordConcat(t0, t1).into()
        }
        ast::TArrayConcat(t0, t1) => {
            let t0 = lower_type(ctx, t0);
            let t1 = lower_type(ctx, t1);
            TArrayConcat(t0, t1).into()
        }
        ast::TName(x, ts) => match ctx.stack.find_type_decl(&x) {
            Some(TypeDecl::Enum(info1, gs, _)) => {
                let ts = lower_type_args_strict(ctx, x.clone(), ts, gs.clone(), info, info1);
                TNominal(x, ts).into()
            }
            Some(TypeDecl::Bit(info1, gs)) => {
                let ts = lower_type_args_strict(ctx, x.clone(), ts, gs.clone(), info, info1);
                TNominal(x, ts).into()
            }
            Some(TypeDecl::Type(info1, gs, t)) => {
                let ts = lower_type_args_strict(ctx, x.clone(), ts, gs.clone(), info, info1);
                let ctx = gs
                    .into_iter()
                    .zip(ts)
                    .for_each(|(x, t)| ctx.stack.bind_type_decl(x, TypeDecl::TypeArg(info, t)));
                TAlias(info1, info, t).into()
            }
            Some(TypeDecl::TypeArg(info1, t)) => t.into(),
            Some(TypeDecl::Generic(_)) => {
                if ts.len() > 0 {
                    ctx.diagnostics.push_error(Error::GenericWithArgs {
                        name: x,
                        info: t.info,
                    });
                    TError.into()
                } else {
                    TGeneric(x).into()
                }
            }
            None => {
                ctx.diagnostics.push_error(Error::UnresolvedTypeName {
                    name: x,
                    info: t.info,
                });
                TError.into()
            }
        },
        ast::TArray(t, n) => {
            let t = lower_type(ctx, t);
            TArray(t, n).into()
        }
        ast::TUnit => TUnit.into(),
        ast::TNever => TNever.into(),
        ast::TIgnore => ctx.new_type_var(),
        ast::TError => TError.into(),
    }
}

fn lower_pattern_name(ctx: &mut Context, x: Name, info: Info, m: bool) -> Pattern {
    if m {
        ctx.stack.bind_expr_decl(x.clone(), ExprDecl::Var(info));
        ctx.typed(|t| PVar(x).with(t, info))
    } else {
        ctx.stack.bind_expr_decl(x.clone(), ExprDecl::Val(info));
        ctx.typed(|t| PVal(x).with(t, info))
    }
}

fn lower_pattern(ctx: &mut Context, p: ast::Pattern, m: bool) -> Pattern {
    let info = p.info;
    match (*p.kind).clone() {
        ast::PParen(p) => lower_pattern(ctx, p, m),
        ast::PIgnore => ctx.typed(|t| PIgnore.with(t, info)),
        ast::POr(p0, p1) => {
            let p0 = lower_pattern(ctx, p0, m);
            let p1 = lower_pattern(ctx, p1, m);
            ctx.typed(|t| POr(p0, p1).with(t, info))
        }
        ast::PTypeAnnot(p, t) => {
            let p = lower_pattern(ctx, p, m);
            let t = lower_type(ctx, t);
            PNoop(p).with(t, info)
        }
        ast::PRecord(xps) => {
            let xps = xps.mapm(ctx, |ctx, (x, p)| match p {
                Some(p) => {
                    let p = lower_pattern(ctx, p, m);
                    (x, p)
                }
                None => {
                    let p = lower_pattern_name(ctx, x.clone(), info, m);
                    (x, p)
                }
            });
            ctx.typed(|t| PRecord(xps).with(t, info))
        }
        ast::PRecordConcat(p0, p1) => {
            let p0 = lower_pattern(ctx, p0, m);
            let p1 = lower_pattern(ctx, p1, m);
            ctx.typed(|t| PRecordConcat(p0, p1).with(t, info))
        }
        ast::PTuple(ps) => {
            let ps = ps.mapm(ctx, |ctx, p| lower_pattern(ctx, p, m));
            ctx.typed(|t| PTuple(ps).with(t, info))
        }
        ast::PArray(ps) => {
            let ps = ps.mapm(ctx, |ctx, p| lower_pattern(ctx, p, m));
            ctx.typed(|t| PArray(ps).with(t, info))
        }
        ast::PArrayConcat(p0, p1) => {
            let p0 = lower_pattern(ctx, p0, m);
            let p1 = lower_pattern(ctx, p1, m);
            ctx.typed(|t| PArrayConcat(p0, p1).with(t, info))
        }
        ast::PConst(c) => {
            let c = lower_const(ctx, c);
            ctx.typed(|t| PConst(c).with(t, info))
        }
        ast::PName(x) => match ctx.stack.find_expr_decl(&x) {
            Some(ExprDecl::Variant(info1, x0, gs)) => {
                let ts = lower_type_args(ctx, x.clone(), vector![], gs, info, info1);
                let p = ctx.typed(|t| PConst(CUnit).with(t, info));
                ctx.typed(|t| PVariant(x0, ts, x, p).with(t, info))
            }
            Some(expr_name) => {
                ctx.diagnostics.push_warning(Warning::ShadowedVariable {
                    info0: expr_name.info(),
                    info1: info,
                });
                lower_pattern_name(ctx, x, info, m)
            }
            None => lower_pattern_name(ctx, x, info, m),
        },
        ast::PVariantTuple(x, ps) => {
            let p = lower_pattern(ctx, ast::PTuple(ps).with(info), m);
            lower_pattern_variant(ctx, x, p, info)
        }
        ast::PVariantRecord(x, xps) => {
            let p = lower_pattern(ctx, ast::PRecord(xps).with(info), m);
            lower_pattern_variant(ctx, x, p, info)
        }
        ast::PError => panic!(),
    }
}

fn lower_pattern_variant(ctx: &mut Context, x: Name, p: Pattern, info: Info) -> Pattern {
    match ctx.stack.find_expr_decl(&x) {
        Some(ExprDecl::Variant(info1, x0, gs)) => {
            let ts = lower_type_args(ctx, x.clone(), vector![], gs, info, info1);
            ctx.typed(|t| PVariant(x0, ts, x, p).with(t, info))
        }
        Some(_) => {
            ctx.diagnostics.push_error(Error::ExpectedVariant { info });
            ctx.typed(|t| PError.with(t, info))
        }
        None => {
            ctx.diagnostics.push_error(Error::UnresolvedName { info });
            ctx.typed(|t| PError.with(t, info))
        }
    }
}

fn lower_enum_variant(
    ctx: &mut Context,
    enum_x: String,
    gs: Vector<ast::Generic>,
    v: ast::Variant,
    info: Info,
) -> Option<(Name, Type)> {
    let (x, t) = match v {
        ast::VUnit(x) => (x, TUnit.into()),
        ast::VRecord(x, xts) => {
            let xts = xts.mapm_assoc(ctx, lower_type);
            let t = fields_to_rows(xts, TRowEmpty.into());
            (x, TRecord(t).into())
        }
        ast::VTuple(x, ts) => {
            let ts = ts.mapm(ctx, lower_type);
            (x, TTuple(ts, true).into())
        }
    };
    if let Some(expr_name) = ctx.stack.find_expr_decl(&x) {
        ctx.diagnostics.push_error(Error::NameClash {
            info0: info,
            info1: expr_name.info(),
        });
        None
    } else {
        ctx.stack
            .bind_expr_decl(x.clone(), ExprDecl::Variant(info, enum_x.clone(), gs));
        Some((x, t))
    }
}

fn lower_block(ctx: &mut Context, b: ast::Block, kind: ScopeKind) -> Block {
    ctx.stack.push_scope(kind);
    let ss = b.ss.mapm(ctx, lower_stmt);
    let e = lower_expr_or_unit(ctx, b.info, b.e);
    ctx.stack.pop_scope();
    Block::new(ss, e, b.info)
}

fn lower_call(ctx: &mut Context, e: ast::Expr, es: Vector<ast::Expr>) -> Expr {
    let info = e.info;
    if let ast::EName(x, ts) = e.kind() {
        match ctx.stack.find_expr_decl(&x) {
            Some(ExprDecl::Variant(info1, x0, gs)) => {
                let ts = lower_type_args(ctx, x.clone(), ts, gs, info, info1);
                let e = lower_expr(ctx, ast::ETuple(es).with(info));
                return ctx.typed(|t| EVariant(x0, ts, x, e).with(t, info));
            }
            Some(ExprDecl::Def(_, n, gs)) => {
                if es.len() == n {
                    let ts = lower_type_args(ctx, x.clone(), ts, gs, info, info);
                    let es = es.mapm(ctx, lower_expr);
                    return ctx.typed(|t| EFunCallDirect(x, ts, es).with(t, info));
                } else {
                    ctx.diagnostics.push_error(Error::WrongNumberOfArguments {
                        info,
                        expected: n,
                        found: es.len(),
                    });
                    return ctx.typed(|t| EError.with(t, info));
                }
            }
            _ => {}
        }
    }
    let e = lower_expr(ctx, e);
    let es = es.mapm(ctx, lower_expr);
    ctx.typed(|t| EFunCall(e, es).with(t, info))
}

fn lower_expr(ctx: &mut Context, e: ast::Expr) -> Expr {
    let info = e.info;
    match e.kind() {
        ast::EParen(e) => lower_expr(ctx, e),
        ast::EName(x, ts) => match ctx.stack.find_expr_decl(&x) {
            Some(ExprDecl::Variant(info1, x0, gs)) => {
                let ts = lower_type_args(ctx, x.clone(), ts, gs, info, info1);
                let e = ctx.typed(|t| EConst(CUnit).with(t, info));
                ctx.typed(|t| EVariant(x0, ts, x, e).with(t, info))
            }
            Some(ExprDecl::Def(info1, _, gs)) => {
                let ts = lower_type_args(ctx, x.clone(), ts, gs, info, info1);
                ctx.typed(|t| EDef(x, ts).with(t, info))
            }
            Some(ExprDecl::Val(_)) => {
                if ts.is_empty() {
                    ctx.typed(|t| EVal(x).with(t, info))
                } else {
                    ctx.diagnostics
                        .push_error(Error::UnexpectedTypeArgs { info });
                    ctx.typed(|t| EError.with(t, info))
                }
            }
            Some(ExprDecl::Var(_)) => {
                if ts.is_empty() {
                    ctx.typed(|t| EVar(x).with(t, info))
                } else {
                    ctx.diagnostics
                        .push_error(Error::UnexpectedTypeArgs { info });
                    ctx.typed(|t| EError.with(t, info))
                }
            }
            None => {
                ctx.diagnostics.push_error(Error::UnresolvedName { info });
                ctx.typed(|t| EError.with(t, info))
            }
        },
        ast::EQuery(p, e, qs) => lower_query(ctx, p, e, qs),
        ast::ERecordAccess(e, x) => {
            let e = lower_expr(ctx, e);
            ctx.typed(|t| ERecordAccess(e, x).with(t, info))
        }
        ast::ERecordAccessMulti(_, _) => todo!(),
        ast::ERecordConcat(e0, e1) => {
            let e0 = lower_expr(ctx, e0);
            let e1 = lower_expr(ctx, e1);
            ctx.typed(|t| ERecordConcat(e0, e1).with(t, info))
        }
        ast::EFunCall(e, es) => lower_call(ctx, e, es),
        ast::EMethodCall(e0, x, ts, mut es) => {
            es.push_front(e0);
            lower_expr(
                ctx,
                ast::EFunCall(ast::EName(x, vector![]).with(info), es).with(info),
            )
        }
        ast::ETypeAnnot(e, t) => {
            let e = lower_expr(ctx, e);
            let t = lower_type(ctx, t);
            ENoop(e).with(t, info)
        }
        ast::EIfElse(e, b0, b1) => {
            let e = lower_expr(ctx, e);
            let b0 = lower_block(ctx, b0, ScopeKind::Block);
            let b1 = lower_block_opt(ctx, info, b1);
            ctx.typed(|t| EIfElse(e, b0, b1).with(t, info))
        }
        ast::ELit(l) => lower_lit(ctx, l, e.info),
        ast::ELoop(b) => {
            let b = lower_block(ctx, b, ScopeKind::Loop);
            ctx.typed(|t| ELoop(b).with(t, info))
        }
        ast::ERecord(xes) => {
            let xes = xes.mapm(ctx, |ctx, xe| lower_expr_field(ctx, info, xe));
            ctx.typed(|t| ERecord(xes).with(t, info))
        }
        ast::EVariantRecord(x, xes) => {
            if let Some(ExprDecl::Variant(info1, x0, gs)) = ctx.stack.find_expr_decl(&x) {
                let ts = lower_type_args(ctx, x.clone(), vector![], gs, info, info1);
                let e = lower_expr(ctx, ast::ERecord(xes).with(info));
                return ctx.typed(|t| EVariant(x0, ts, x, e).with(t, info));
            } else {
                ctx.diagnostics.push_error(Error::ExpectedVariant { info });
                return ctx.typed(|t| EError.with(t, info));
            }
        }
        ast::EFunReturn(e) => {
            let e = lower_expr_or_unit(ctx, info, e);
            ctx.typed(|t| EFunReturn(e).with(t, info))
        }
        ast::ELoopBreak(e) => {
            if let Some(e) = e {
                if ctx.is_inside_infinite_loop() {
                    let e = lower_expr(ctx, e);
                    ctx.typed(|t| ELoopBreak(e).with(t, info))
                } else {
                    ctx.diagnostics
                        .push_error(Error::BreakOutsideInfiniteLoop { info });
                    ctx.typed(|t| EError.with(t, info))
                }
            } else {
                if ctx.is_inside_loop() {
                    let e = lower_expr_or_unit(ctx, info, e);
                    ctx.typed(|t| ELoopBreak(e).with(t, info))
                } else {
                    ctx.diagnostics.push_error(Error::BreakOutsideLoop { info });
                    ctx.typed(|t| EError.with(t, info))
                }
            }
        }
        ast::ELoopContinue => {
            if ctx.is_inside_loop() {
                ctx.typed(|t| ELoopContinue.with(t, info))
            } else {
                ctx.diagnostics
                    .push_error(Error::ContinueOutsideLoop { info });
                ctx.typed(|t| EError.with(t, info))
            }
        }
        ast::EArray(es) => {
            let es = es.mapm(ctx, lower_expr);
            ctx.typed(|t| EArray(es).with(t, info))
        }
        ast::EUnop(op, e) => {
            let e = lower_expr(ctx, e);
            lower_unop(ctx, op, e, info)
        }
        ast::EBinop(e0, op, e1) => {
            let e0 = lower_expr(ctx, e0);
            let e1 = lower_expr(ctx, e1);
            lower_binop(ctx, e0, op, e1, info)
        }
        ast::EMut(e0, op, e1) => {
            let e0 = lower_place_expr(ctx, e0);
            let e1 = lower_expr(ctx, e1);
            if let Some(op) = op {
                let e2 = lower_binop(ctx, e0.clone(), op, e1, info);
                ctx.typed(|t| EMut(e0, e2).with(t, info))
            } else {
                ctx.typed(|t| EMut(e0, e1).with(t, info))
            }
        }
        ast::EDo(b) => {
            let b = lower_block(ctx, b, ScopeKind::Block);
            ctx.typed(|t| EDo(b).with(t, info))
        }
        ast::EFor(p, e, b) => {
            ctx.stack.push_scope(ScopeKind::Loop);
            let p = lower_pattern(ctx, p, false);
            let e = lower_expr(ctx, e);
            let b = lower_block(ctx, b, ScopeKind::Block);
            ctx.stack.pop_scope();
            ctx.typed(|t| EFor(p, e, b).with(t, info))
        }
        ast::EFun(ps, t, b) => {
            ctx.stack.push_scope(ScopeKind::Fun);
            let ps = ps.mapm(ctx, |ctx, p| lower_pattern(ctx, p, false));
            let t1 = lower_type_or_fresh(ctx, t);
            let b = lower_body(ctx, b);
            ctx.stack.pop_scope();
            ctx.typed(|t| hir::EFun(ps, t1, b).with(t, info))
        }
        ast::EMatch(e, arms) => {
            let e = lower_expr(ctx, e);
            let arms = arms.mapm(ctx, lower_arm);
            ctx.typed(|t| EMatch(e, arms).with(t, info))
        }
        ast::ETupleAccess(e, i) => {
            let e = lower_expr(ctx, e);
            ctx.typed(|t| ETupleAccess(e, i).with(t, info))
        }
        ast::EArrayAccess(e0, e1) => {
            let e0 = lower_expr(ctx, e0);
            let e1 = lower_expr(ctx, e1);
            ctx.typed(|t| EArrayAccess(e0, e1).with(t, info))
        }
        ast::EThrow(_e) => todo!(),
        ast::ETry(_b0, _arms, _b1) => todo!(),
        ast::ETuple(es) => {
            let es = es.mapm(ctx, lower_expr);
            ctx.typed(|t| ETuple(es).with(t, info))
        }
        ast::EWhile(e, b) => {
            let e = lower_expr(ctx, e);
            let b = lower_block(ctx, b, ScopeKind::Loop);
            ctx.typed(|t| EWhile(e, b).with(t, info))
        }
        ast::EArrayConcat(e0, e1) => {
            let e0 = lower_expr(ctx, e0);
            let e1 = lower_expr(ctx, e1);
            ctx.typed(|t| EArrayConcat(e0, e1).with(t, info))
        }
        ast::EError => ctx.typed(|t| EError.with(t, info)),
    }
}

fn lower_place_expr(ctx: &mut Context, e: ast::Expr) -> Expr {
    let info = e.info;
    match e.kind() {
        ast::EName(x, ts) => match ctx.stack.find_expr_decl(&x) {
            Some(ExprDecl::Var(_)) => {
                if ts.len() == 0 {
                    ctx.typed(|t| EVar(x).with(t, info))
                } else {
                    ctx.diagnostics
                        .push_error(Error::UnexpectedTypeArgs { info });
                    ctx.typed(|t| EError.with(t, info))
                }
            }
            Some(x) => {
                ctx.diagnostics.push_error(Error::ExpectedVar { info });
                ctx.typed(|t| EError.with(t, info))
            }
            None => {
                ctx.diagnostics.push_error(Error::UnresolvedName { info });
                ctx.typed(|t| EError.with(t, info))
            }
        },
        ast::EArrayAccess(e0, e1) => {
            let e0 = lower_place_expr(ctx, e0);
            let e1 = lower_expr(ctx, e1);
            ctx.typed(|t| EArrayAccess(e0, e1).with(t, info))
        }
        ast::ETupleAccess(e, i) => {
            let e = lower_place_expr(ctx, e);
            ctx.typed(|t| ETupleAccess(e, i).with(t, info))
        }
        _ => {
            ctx.diagnostics
                .push_error(Error::ExpectedPlaceExpr { info });
            ctx.typed(|t| EError.with(t, info))
        }
    }
}

fn lower_arm(ctx: &mut Context, (p, e): ast::Arm) -> Arm {
    ctx.stack.push_scope(ScopeKind::Arm);
    let p = lower_pattern(ctx, p, false);
    let e = lower_expr(ctx, e);
    let b = Block::new(vector![], e, p.info);
    ctx.stack.pop_scope();
    (p, b)
}

fn lower_unop(ctx: &mut Context, op: ast::Unop, e: Expr, info: Info) -> Expr {
    let x = ast::ops::unop(op.kind).to_string();
    match ctx.stack.find_expr_decl(&x) {
        Some(ExprDecl::Def(_, _, gs)) => {
            let ts = gs.iter().map(|g| ctx.new_type_var()).collect();
            ctx.typed(|t| EFunCallDirect(x, ts, vector![e]).with(t, info))
        }
        _ => {
            ctx.diagnostics.push_error(Error::UnresolvedName { info });
            ctx.typed(|t| EError.with(t, info))
        }
    }
}

fn lower_binop(ctx: &mut Context, e0: Expr, op: ast::Binop, e1: Expr, info: Info) -> Expr {
    let x = ast::ops::binop(op.kind).to_string();
    match ctx.stack.find_expr_decl(&x) {
        Some(ExprDecl::Def(_, _, gs)) => {
            let ts = gs.iter().map(|g| ctx.new_type_var()).collect();
            ctx.typed(|t| EFunCallDirect(x, ts, vector![e0.clone(), e1]).with(t, info))
        }
        _ => {
            ctx.diagnostics.push_error(Error::UnresolvedName { info });
            ctx.typed(|t| EError.with(t, info))
        }
    }
}

fn lower_expr_or_unit(ctx: &mut Context, info: Info, e: Option<ast::Expr>) -> Expr {
    e.mapm_or_else(ctx, lower_expr, |ctx| {
        ctx.typed(|t| EConst(CUnit).with(t, info))
    })
}

fn lower_type_or_unit(ctx: &mut Context, t: Option<ast::Type>) -> Type {
    t.mapm_or_else(ctx, lower_type, |ctx| TUnit.into())
}

fn lower_type_or_fresh(ctx: &mut Context, t: Option<ast::Type>) -> Type {
    t.mapm_or_else(ctx, lower_type, |ctx| ctx.new_type_var())
}

fn lower_expr_field(ctx: &mut Context, info: Info, f: ExprField) -> (Name, Expr) {
    match f {
        ast::FName(x, e) => {
            if let Some(e) = e {
                let e = lower_expr(ctx, e);
                (x, e)
            } else {
                let e = match ctx.stack.find_expr_decl(&x) {
                    Some(ExprDecl::Val(_)) => ctx.typed(|t| EVal(x.clone()).with(t, info)),
                    Some(ExprDecl::Var(_)) => ctx.typed(|t| EVar(x.clone()).with(t, info)),
                    Some(ExprDecl::Def(info1, _, gs)) => {
                        let ts = lower_type_args(ctx, x.clone(), vector![], gs, info, info1);
                        ctx.typed(|t| EDef(x.clone(), ts).with(t, info))
                    }
                    Some(ExprDecl::Variant(..)) => panic!("Variant not allowed here"),
                    None => panic!("Name not found"),
                };
                (x, e)
            }
        }
        ast::FExpr(e, x) => {
            let info = e.info;
            let e = lower_expr(ctx, e);
            let e = ctx.typed(|t| ERecordAccess(e, x.clone()).with(t, info));
            (x, e)
        }
    }
}

// from p in e => e.map(|p| p)
fn lower_query(
    ctx: &mut Context,
    p: ast::Pattern,
    e: ast::Expr,
    qs: Vector<ast::QueryStmt>,
) -> Expr {
    let map = ast::ExprKind::EName("map".to_string(), vector![]).with(e.info);
    let p = lower_pattern(ctx, p.clone(), false);
    // let es = pattern_to_expr_record(ctx, p);
    // let mapper = ast::ExprKind::EFun(vector![p], Body::BExpr()).with(e.info);
    // ast::ExprKind::EFunCall(map, vector![e])
    todo!()
}

/// Extract all the names from a pattern and place them inside a record expression.
fn pattern_to_expr_record(ctx: &mut Context, p: hir::Pattern) -> hir::Expr {
    fn f(ctx: &mut Context, p: &hir::Pattern, acc: &mut Vector<(Name, hir::Expr)>) {
        match p.kind() {
            hir::PNoop(_) => {}
            hir::PVal(x) => {
                let e = ctx.typed(|t| EVal(x.clone()).with(t, p.info));
                acc.push_back((x, e));
            }
            hir::PVar(_) => {}
            hir::PVariant(_, _, _, p) => f(ctx, &p, acc),
            hir::PIgnore => {}
            hir::POr(p0, _) => f(ctx, &p0, acc),
            hir::PRecord(xps) => xps.iter().for_each(|(_, p)| f(ctx, &p, acc)),
            hir::PRecordConcat(_, _) => todo!(),
            hir::PArray(_) => todo!(),
            hir::PArrayConcat(_, _) => todo!(),
            hir::PTuple(ps) => ps.iter().for_each(|p| f(ctx, &p, acc)),
            hir::PConst(_) => {}
            hir::PError => {}
        }
    }
    let mut acc = vector![];
    f(ctx, &p, &mut acc);
    hir::ExprKind::ERecord(acc).with(p.t, p.info)
}

lazy_static::lazy_static! {
    static ref REGEX: Regex = Regex::new(r#"\$(\{[^}]+}|[[:alpha:]_][[:alnum:]_]*)"#).unwrap();
}

fn lower_lit(ctx: &mut Context, l: ast::Lit, info: Info) -> Expr {
    match l {
        ast::LInt(l, s) => {
            if let Some(s) = s {
                let x = format!("__{}", s);
                let e0 = ast::EName(x, vector![]).with(info);
                let e1 = ast::ELit(ast::LInt(l, None)).with(info);
                lower_call(ctx, e0, vector![e1])
            } else {
                ctx.typed(|t| EConst(CInt(l)).with(t, info))
            }
        }
        ast::LFloat(l, s) => {
            if let Some(s) = s {
                let x = format!("__{}", s);
                let e0 = ast::EName(x, vector![]).with(info);
                let e1 = ast::ELit(ast::LFloat(l, None)).with(info);
                lower_call(ctx, e0, vector![e1])
            } else {
                ctx.typed(|t| EConst(CFloat(l)).with(t, info))
            }
        }
        ast::LBool(b) => ctx.typed(|t| EConst(CBool(b)).with(t, info)),
        ast::LString(s) => lower_lit_str(ctx, s, info),
        ast::LUnit => ctx.typed(|t| EConst(CUnit).with(t, info)),
        ast::LChar(c) => ctx.typed(|t| EConst(CChar(c)).with(t, info)),
    }
}

struct SplitIter<'r, 't> {
    string: &'t str,
    matches: Matches<'r, 't>,
    prev: usize,
    delim: Option<Split<'t>>,
}

impl<'r, 't> SplitIter<'r, 't> {
    fn new(regex: &'r Regex, string: &'t str) -> Self {
        let matches = regex.find_iter(string);
        Self {
            string,
            matches,
            prev: 0,
            delim: None,
        }
    }
}

impl<'r, 't> Iterator for SplitIter<'r, 't> {
    type Item = Split<'t>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(splice) = self.delim.take() {
            Some(splice)
        } else if let Some(m) = self.matches.next() {
            let splice = Split::Text(&self.string[self.prev..m.start()]);
            self.delim = Some(Split::Delim(&self.string[m.start()..m.end()]));
            self.prev = m.end();
            Some(splice)
        } else if self.prev < self.string.len() {
            let splice = Split::Text(&self.string[self.prev..]);
            self.prev = self.string.len();
            Some(splice)
        } else {
            None
        }
    }
}

enum Split<'a> {
    Text(&'a str),
    Delim(&'a str),
}

fn splice_to_expr(ctx: &mut Context, splice: Split, info: Info) -> Expr {
    match splice {
        Split::Text(s) => ctx.typed(|t| EConst(CString(s.to_owned())).with(t, info)),
        Split::Delim(s) => match parser::parse_splice(info, &mut ctx.diagnostics, &s[1..]) {
            Some(ast::SName(x)) => match ctx.stack.find_expr_decl(&x) {
                Some(ExprDecl::Val(_)) => {
                    let e = ctx.typed(|t| EVal(x).with(t, info));
                    call(ctx, "to_string", [], [e], info)
                }
                Some(ExprDecl::Var(_)) => {
                    let e = ctx.typed(|t| EVar(x).with(t, info));
                    call(ctx, "to_string", [], [e], info)
                }
                Some(decl) => {
                    ctx.diagnostics
                        .push_error(Error::ExpectedVarOrVal { info: decl.info() });
                    ctx.typed(|t| EError.with(t, info))
                }
                None => {
                    ctx.diagnostics
                        .push_error(Error::NameNotFound { info, name: x });
                    ctx.typed(|t| EError.with(t, info))
                }
            },
            Some(ast::SBlock(b)) => {
                let b = lower_block(ctx, b, ScopeKind::Block);
                ctx.typed(|t| EDo(b).with(t, info))
            }
            _ => ctx.typed(|t| EError.with(t, info)),
        },
    }
}

fn lower_lit_str(ctx: &mut Context, s: String, info: Info) -> Expr {
    if s.is_empty() {
        return ctx.typed(|t| EConst(CString(s)).with(t, info));
    }
    let mut it = SplitIter::new(&REGEX, &s);
    let s = it.next().unwrap();
    let e = splice_to_expr(ctx, s, info);
    it.fold(e, |e, s| {
        let e1 = splice_to_expr(ctx, s, info);
        call(ctx, "concat", [], [e, e1], info)
    })
}

fn lower_type_args_strict(
    ctx: &mut Context,
    x: Name,
    ts: Vector<ast::Type>,
    gs: Vector<ast::Generic>,
    info0: Info,
    info1: Info,
) -> Vector<Type> {
    if gs.len() == ts.len() {
        ts.mapm(ctx, lower_type)
    } else {
        ctx.diagnostics.push_error(Error::WrongNumberOfTypeArgs {
            name: x,
            expected: gs.len(),
            found: ts.len(),
            info0,
            info1,
        });
        vector![]
    }
}

fn lower_type_args(
    ctx: &mut Context,
    x: Name,
    ts: Vector<ast::Type>,
    gs: Vector<ast::Generic>,
    info0: Info,
    info1: Info,
) -> Vector<Type> {
    match (gs.len(), ts.len()) {
        (gn, tn) if gn == tn => ts.mapm(ctx, lower_type),
        (gn, tn) if tn == 0 => gs.mapm(ctx, |ctx, _| ctx.new_type_var()),
        _ => {
            ctx.diagnostics.push_error(Error::WrongNumberOfTypeArgs {
                name: x,
                expected: gs.len(),
                found: ts.len(),
                info0,
                info1,
            });
            vector![]
        }
    }
}

fn fields_to_rows(x: Vector<(Name, Type)>, t: Type) -> Type {
    x.into_iter()
        .rev()
        .fold(t, |t0, (x, t1)| TRowExtend((x, t1), t0).into())
}

fn lower_block_opt(ctx: &mut Context, info: Info, b: Option<ast::Block>) -> Block {
    b.mapm_or_else(
        ctx,
        |ctx, b| lower_block(ctx, b, ScopeKind::Block),
        |ctx| {
            let e = ctx.typed(|t| EConst(CUnit).with(t, info));
            Block::new(vector![], e, info)
        },
    )
}

fn lower_body(ctx: &mut Context, b: ast::Body) -> Block {
    match b {
        ast::BExpr(e) => {
            let info = e.info;
            let e = lower_expr(ctx, e);
            Block::new(vector![], e, info)
        }
        ast::BBlock(b) => lower_block(ctx, b, ScopeKind::Def),
    }
}

fn empty_expr_env(ctx: &mut Context, info: Info) -> Expr {
    ctx.typed(|t| ERecord(vector![]).with(t, info))
}

fn nominal<const N: usize>(x: impl Into<String>, ts: [Type; N]) -> Type {
    TNominal(x.into(), ts.into_iter().collect()).into()
}

fn call<const N: usize, const M: usize>(
    ctx: &mut Context,
    x: impl Into<String>,
    ts: [Type; N],
    es: [Expr; M],
    info: Info,
) -> Expr {
    let e = ctx.typed(|t| EVal(x.into()).with(t, info));
    ctx.typed(|t| EFunCall(e, es.into_iter().collect()).with(t, info))
}
