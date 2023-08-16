#![allow(unused)]
pub mod context;

use diagnostics::Error;
use hir::*;
use im_rc::vector;
use im_rc::OrdMap;
use im_rc::OrdSet;
use im_rc::Vector;
use info::Info;
use utils::AssocVectorUtils;
use utils::OptionUtils;
use utils::VectorUtils;

use crate::context::*;

pub fn process(ctx: &mut Context, ss: Vector<Stmt>) -> Vector<Stmt> {
    let ss = ss.mapm(ctx, infer_stmt);
    let f = &|t| apply(ctx.get_subst(), t);
    ss.map(|s| s.map_type(&f))
}

fn infer_block(ctx: &mut Context, b: Block) -> Block {
    ctx.stack.push_scope(ScopeKind::Block);
    let ss = b.ss.mapm(ctx, infer_stmt);
    let e = infer_expr(ctx, b.e);
    ctx.stack.pop_scope();
    Block::new(ss, e, b.info)
}

fn infer_stmt(ctx: &mut Context, s: Stmt) -> Stmt {
    let info = s.info;
    match s.clone().kind {
        SVal(p, e) => {
            let p = infer_pattern(ctx, p);
            let e = infer_expr(ctx, e);
            unify(ctx, p.t.clone(), e.t.clone(), info);
            SVal(p, e).with(info)
        }
        SExpr(e) => {
            let e = infer_expr(ctx, e);
            SExpr(e).with(info)
        }
        SBif(m, x, gs, ts, t) => {
            ctx.stack.bind_expr_decl(
                x.clone(),
                ExprDecl::Def(
                    info,
                    TFun(ts.clone(), t.clone()).into(),
                    gs.clone(),
                    vector![],
                ),
            );
            SBif(m, x, gs, ts, t).with(info)
        }
        SDef(m, x, gs0, ps, t, b) => {
            ctx.stack.push_scope(ScopeKind::Def(t.clone()));

            let t_fun: Type = TFun(ps.clone().map(|p| p.t), t.clone()).into();

            ctx.stack.bind_expr_decl(
                x.clone(),
                ExprDecl::Def(info, t_fun.clone(), gs0.clone(), vector![]),
            );

            let ps = ps.mapm(ctx, infer_pattern);
            let b = infer_block(ctx, b);
            unify(ctx, t.clone(), b.e.t.clone(), info);
            ctx.stack.pop_scope();

            // Apply MGU to all types in the function
            // * Consider if this could be changed so that we only apply the MGU to the function type
            let f = &|t| apply(ctx.get_subst(), t);
            let ps = ps.map(|p| p.map_type(f));
            let t = f(t);
            let b = b.map_type(f);
            let t_fun: Type = TFun(ps.clone().map(|p| p.t), t.clone()).into();

            // Generalise all types in the function
            // Create a subsitution from free type variables to generics
            // * We could treat generics and free type variables the same, which would mean we
            //   could get rid of this step.
            let ftv = ftv(t_fun.clone()).into_iter().collect::<Vector<_>>();
            let gs1 = (0..ftv.len())
                .into_iter()
                .map(|i| format!("G{i}"))
                .collect::<Vector<_>>();
            let s = ftv
                .into_iter()
                .zip(gs1.clone())
                .map(|(tv, g)| (tv, TGeneric(g).into()))
                .collect::<Vector<_>>();

            let f = &|t| generalise(s.clone(), t);
            let ps = ps.map(|p| p.map_type(f));
            let t = f(t);
            let b = b.map_type(f);
            let t_fun: Type = TFun(ps.clone().map(|p| p.t), t.clone()).into();

            ctx.stack.bind_expr_decl(
                x.clone(),
                ExprDecl::Def(info, t_fun.clone(), gs0.clone(), gs1.clone()),
            );
            SDef(m, x, gs0.concat(gs1), ps, t, b).with(info)
        }
        SEnum(m, x, gs, xts) => SEnum(m, x, gs, xts).with(info),
        SBit(m, x, gs) => SBit(m, x, gs).with(info),
        SNoop => SNoop.with(info),
        SRecDef(_, _) => todo!(),
    }
}

fn ts_of_es(es: Vector<Expr>) -> Vector<Type> {
    es.into_iter().map(|e| e.t).collect()
}

fn infer_expr(ctx: &mut Context, e: Expr) -> Expr {
    let info = e.info;
    let t = e.t.clone();
    match (*e.kind).clone() {
        EConst(l) => {
            unify(ctx, const_type(&l), t.clone(), info);
            EConst(l).with(t, info)
        }
        EFun(ps, t1, b) => {
            let ps = ps.mapm(ctx, infer_pattern);
            let b = infer_block(ctx, b);
            unify(ctx, t.clone(), b.e.t.clone(), info);
            EFun(ps, t1, b).with(t, info)
        }
        EFunCall(e, es) => {
            let e = infer_expr(ctx, e);
            let es = es.mapm(ctx, infer_expr);
            let t_fun = TFun(ts_of_es(es.clone()), t.clone()).into();
            unify(ctx, e.t.clone(), t_fun, info);
            EFunCall(e, es).with(t, info)
        }
        EDef(x, ts0) => {
            let (t1, ts) = infer_def(ctx, x.clone(), ts0);
            unify(ctx, t.clone(), t1, info);
            EDef(x, ts).with(t, info)
        }
        EFunCallDirect(x, ts0, es) => {
            let (t1, ts1) = infer_def(ctx, x.clone(), ts0);
            let es = es.mapm(ctx, infer_expr);
            let t_fun = TFun(ts_of_es(es.clone()), t.clone()).into();
            unify(ctx, t1, t_fun, info);
            EFunCallDirect(x, ts1, es).with(t, info)
        }
        ELoop(b) => {
            let b = infer_block(ctx, b);
            ELoop(b).with(t, info)
        }
        ELoopBreak(e) => {
            let e = infer_expr(ctx, e);
            unify(ctx, ctx.break_type(), e.t.clone(), info);
            ELoopBreak(e).with(t, info)
        }
        ELoopContinue => ELoopContinue.with(t, info),
        EMatch(e, arms) => {
            let e = infer_expr(ctx, e);
            let arms = arms.mapm(ctx, |ctx, (p, b)| {
                let p = infer_pattern(ctx, p);
                unify(ctx, e.t.clone(), p.t.clone(), info);
                let b = infer_block(ctx, b);
                unify(ctx, t.clone(), b.e.t.clone(), info);
                (p, b)
            });
            EMatch(e, arms).with(t, info)
        }
        ERecord(xes) => {
            let xes = xes.mapm_assoc(ctx, infer_expr);
            let xts = xes.clone().map(|(x, e)| (x, e.t));
            unify(
                ctx,
                t.clone(),
                TRecord(fields_to_rows(xts, TRowEmpty.into())).into(),
                info,
            );
            ERecord(xes).with(t, info)
        }
        ERecordAccess(e, x) => {
            let e = infer_expr(ctx, e);
            let r = ctx.fresh_r();
            let t1 = TRecord(TRowExtend((x.clone(), t.clone()), r).into()).into();
            unify(ctx, t1, e.t.clone(), info);
            ERecordAccess(e, x).with(t, info)
        }
        ERecordConcat(e0, e1) => {
            let e0 = infer_expr(ctx, e0);
            let e1 = infer_expr(ctx, e1);
            let r0 = ctx.fresh_r();
            let r1 = ctx.fresh_r();
            unify(ctx, TRecord(r0.clone()).into(), e0.t.clone(), info);
            unify(ctx, TRecord(r1.clone()).into(), e1.t.clone(), info);
            unify(ctx, TRecordConcat(r0, r1).into(), t.clone(), info);
            ERecordConcat(e0, e1).with(t, info)
        }
        EFunReturn(e) => {
            let e = infer_expr(ctx, e);
            unify(ctx, ctx.return_type(), e.t.clone(), info);
            EFunReturn(e).with(t, info)
        }
        EVariant(x0, ts, x1, e) => {
            let e = infer_expr(ctx, e);
            let t1 = match ctx.stack.find_type_decl(&x0) {
                Some(TypeDecl::Enum(_, gs, xts)) => {
                    let s = gs.zip(&ts);
                    let t = xts.find_assoc(&x1).unwrap().clone();
                    apply(s, t)
                }
                _ => unreachable!(),
            };
            unify(
                ctx,
                t.clone(),
                TNominal(x0.clone(), ts.clone()).into(),
                info,
            );
            unify(ctx, t1.clone(), e.t.clone(), info);
            EVariant(x0, ts, x1, e).with(t, info)
        }
        EIfElse(e, b0, b1) => {
            let e = infer_expr(ctx, e);
            unify(ctx, e.t.clone(), atom("Bool"), info);
            let b0 = infer_block(ctx, b0);
            let b1 = infer_block(ctx, b1);
            unify(ctx, b0.e.t.clone(), b1.e.t.clone(), info);
            unify(ctx, t.clone(), b0.e.t.clone(), info);
            EIfElse(e, b0, b1).with(t, info)
        }
        ENoop(e) => {
            let e = infer_expr(ctx, e);
            unify(ctx, t.clone(), e.t.clone(), info);
            ENoop(e).with(t, info)
        }
        // TODO: Array concatenation
        EArray(es) => {
            let t1 = ctx.fresh_t();
            let es = es.mapm(ctx, |ctx, e2| {
                unify(ctx, e2.t.clone(), t1.clone(), info);
                infer_expr(ctx, e2)
            });
            // let e = infer_expr(ctx, e);
            unify(
                ctx,
                t.clone(),
                TArray(t1.clone(), Some(es.len() as i32)).into(),
                info,
            );
            EArray(es).with(t, info)
        }
        // TODO: Array size as a type variable
        EArrayAccess(e0, e1) => {
            let e0 = infer_expr(ctx, e0);
            let e1 = infer_expr(ctx, e1);
            let t = ctx.fresh_t();
            unify(ctx, e0.t.clone(), TArray(t.clone(), None).into(), info);
            unify(ctx, e1.t.clone(), atom("i32"), info);
            EArrayAccess(e0, e1).with(t, info)
        }
        EArrayConcat(e0, e1) => {
            todo!()
            // let e0 = infer_expr(ctx, e0);
            // let e1 = infer_expr(ctx, e1);
            // let t = ctx.fresh_t();
            // unify(ctx, e0.t.clone(), TArray(t.clone(), None).into(), info);
            // unify(ctx, e1.t.clone(), TArray(t.clone(), None).into(), info);
            // EArrayConcat(e0, e1).with(t, info)
        }
        EMut(e0, e1) => {
            let e0 = infer_expr(ctx, e0);
            let e1 = infer_expr(ctx, e1);
            unify(ctx, e0.t.clone(), e1.t.clone(), info);
            EMut(e0, e1).with(t, info)
        }
        EVal(x) => {
            if let Some(ExprDecl::Val(info, t1)) = ctx.stack.find_expr_decl(&x) {
                unify(ctx, t.clone(), t1, info);
                EVal(x).with(t, info)
            } else {
                unreachable!()
            }
        }
        EVar(x) => {
            if let Some(ExprDecl::Var(info, t1)) = ctx.stack.find_expr_decl(&x) {
                unify(ctx, t.clone(), t1, info);
                EVar(x).with(t, info)
            } else {
                unreachable!()
            }
        }
        EDo(b) => {
            let b = infer_block(ctx, b);
            unify(ctx, t.clone(), b.e.t.clone(), info);
            EDo(b).with(t, info)
        }
        ETuple(es) => {
            let es = es.mapm(ctx, infer_expr);
            let ts = es.clone().map(|e| e.t.clone());
            unify(ctx, t.clone(), TTuple(ts, true).into(), info);
            ETuple(es).with(t, info)
        }
        ETupleAccess(e, i) => {
            let e = infer_expr(ctx, e);
            let ts = ctx.fresh_ts(i + 1);
            unify(ctx, e.t.clone(), TTuple(ts.clone(), false).into(), info);
            unify(ctx, t.clone(), ts[i as usize].clone(), info);
            ETupleAccess(e, i).with(t, info)
        }
        EFor(_, _, _) => todo!(),
        EWhile(e, b) => {
            let e = infer_expr(ctx, e);
            let b = infer_block(ctx, b);
            unify(ctx, e.t.clone(), atom("Bool"), info);
            unify(ctx, b.e.t.clone(), TUnit.into(), info);
            EWhile(e, b).with(t, info)
        }
        EError => EError.with(t, info),
        EVariantAccess(..) | EVariantCheck(..) => {
            unreachable!("Should not occur until after flattening")
        }
    }
}

fn infer_def(ctx: &mut Context, x: Name, ts0: Vector<Type>) -> (Type, Vector<Type>) {
    if let Some(ExprDecl::Def(info, t_fun, gs0, gs1)) = ctx.stack.find_expr_decl(&x) {
        let s0 = gs0.zip(&ts0);
        let s1 = gs1.mapm(ctx, |ctx, x| (x, ctx.fresh_t()));
        let ts1 = s1.clone().map(|(_, t)| t);
        let t_fun = instantiate(compose(s0, s1), t_fun);
        (t_fun, ts0.concat(ts1))
    } else {
        unreachable!()
    }
}

fn infer_pattern(ctx: &mut Context, p: Pattern) -> Pattern {
    let t = p.t.clone();
    let info = p.info;
    match p.kind() {
        PIgnore => PIgnore.with(t, info),
        POr(p0, p1) => {
            let p0 = infer_pattern(ctx, p0);
            let p1 = infer_pattern(ctx, p1);
            unify(ctx, p0.t.clone(), p1.t.clone(), info);
            POr(p0, p1).with(t, info)
        }
        PNoop(p) => {
            let p = infer_pattern(ctx, p);
            unify(ctx, t.clone(), p.t.clone(), info);
            PNoop(p).with(t, info)
        }
        PRecord(xps) => {
            let xps = xps.mapm(ctx, |ctx, (x, p)| (x, infer_pattern(ctx, p)));
            let xts = xps.clone().map(|(x, p)| (x, p.t));
            unify(
                ctx,
                t.clone(),
                TRecord(fields_to_rows(xts, TRowEmpty.into())).into(),
                info,
            );
            PRecord(xps).with(t, info)
        }
        PRecordConcat(p0, p1) => {
            todo!()
            // let p0 = infer_pattern(ctx, p0);
            // let p1 = infer_pattern(ctx, p1);
            // unify(ctx, p0.t.clone(), p1.t.clone(), info);
            // unify(ctx, t.clone(), p0.t.clone(), info);
            // PRecordConcat(p0, p1).with(t, info)
        }
        PArray(ps) => {
            let t1 = ctx.fresh_t();
            let ps = ps.mapm(ctx, |ctx, p| {
                unify(ctx, p.t.clone(), t1.clone(), info);
                infer_pattern(ctx, p)
            });
            // let e = infer_expr(ctx, e);
            unify(
                ctx,
                t.clone(),
                TArray(t1.clone(), Some(ps.len() as i32)).into(),
                info,
            );
            PArray(ps).with(t, info)
        }
        PArrayConcat(p0, p1) => {
            todo!()
            // let p0 = infer_pattern(ctx, p0);
            // let p1 = infer_pattern(ctx, p1);
            // unify(ctx, p0.t.clone(), p1.t.clone(), info);
            // unify(ctx, t.clone(), p0.t.clone(), info);
            // PArrayConcat(p0, p1).with(t, info)
        }
        PConst(c) => {
            let t = const_type(&c);
            unify(ctx, t.clone(), p.t.clone(), info);
            PConst(c).with(t, info)
        }
        PVar(x) => {
            ctx.stack
                .bind_expr_decl(x.clone(), ExprDecl::Var(info, t.clone()));
            PVar(x).with(t, info)
        }
        PVal(x) => {
            ctx.stack
                .bind_expr_decl(x.clone(), ExprDecl::Val(info, t.clone()));
            PVal(x).with(t, info)
        }
        PVariant(x0, ts, x1, p) => {
            let p = infer_pattern(ctx, p);
            let t1 = match ctx.stack.find_type_decl(&x0) {
                Some(TypeDecl::Enum(_, gs, xts)) => {
                    let s = gs.zip(&ts);
                    let t = xts.find_assoc(&x1).unwrap().clone();
                    apply(s, t)
                }
                _ => unreachable!(),
            };
            unify(
                ctx,
                t.clone(),
                TNominal(x0.clone(), ts.clone()).into(),
                info,
            );
            unify(ctx, t1.clone(), p.t.clone(), info);
            PVariant(x0, ts, x1, p).with(t, info)
        }
        PTuple(ps) => {
            let ps = ps.mapm(ctx, infer_pattern);
            let ts = ps.clone().map(|p| p.t);
            unify(ctx, t.clone(), TTuple(ts, true).into(), info);
            PTuple(ps).with(t, info)
        }
        PError => PError.with(t, info),
    }
}

fn const_type(l: &Const) -> Type {
    match l {
        CInt(_) => atom("i32"),
        CFloat(_) => atom("f32"),
        CBool(_) => atom("bool"),
        CString(_) => atom("String"),
        CChar(_) => atom("char"),
        CUnit => TUnit.into(),
    }
}

fn atom(arg: &str) -> Type {
    TNominal(arg.to_string(), vector![]).into()
}

fn unify(ctx: &mut Context, t0: Type, t1: Type, info: Info) {
    let s0 = ctx.get_subst();
    let s1 = mgu(ctx, apply(s0.clone(), t0), apply(s0.clone(), t1), info);
    let s2 = compose(s1, s0);
    let s3 = simplify(s2);
    ctx.set_subst(s3);
}

fn try_unify(ctx: &mut Context, t0: Type, t1: Type, info: Info) -> bool {
    let s0 = ctx.get_subst();
    todo!()
    // try_mgu(ctx, apply(s0.clone(), t0), apply(s0.clone(), t1), info)
}

// Simplify constraints
fn simplify(s: Vector<(Name, Type)>) -> Vector<(Name, Type)> {
    s.map(|(x, t)| match t.kind() {
        TRecordConcat(r0, r1) => {
            let (xts0, t0) = deconstruct_row(r0);
            let (xts1, t1) = deconstruct_row(r1);
            let t = match (t0.kind(), t1.kind()) {
                (TRowEmpty, TRowEmpty) => {
                    TRecord(xts_to_row(xts0.concat(xts1), TRowEmpty.into())).into()
                }
                (TRowEmpty, t) | (t, TRowEmpty) => {
                    TRecord(xts_to_row(xts0.concat(xts1), t.into())).into()
                }
                _ => t,
            };
            (x, t)
        }
        _ => (x, t),
    })
}

fn mgu(ctx: &mut Context, t0: Type, t1: Type, info: Info) -> Vector<(Name, Type)> {
    match (t0.kind(), t1.kind()) {
        (TFun(ts0, t0), TFun(ts1, t1)) if ts0.len() == ts1.len() => {
            let s = mgu_fold(ctx, ts0, ts1, info);
            mgu_acc(ctx, t0, t1, s, info)
        }
        (TVar(x0), t) | (t, TVar(x0)) => match t {
            TVar(x1) if x0 == x1 => vector![],
            t if ftv(t.clone().into()).contains(&x0) => {
                ctx.diagnostics.push_error(Error::InfiniteType {
                    info,
                    t: write_hir::type_to_string(&t.into()),
                });
                vector![]
            }
            t => vector![(x0, t.into())],
        },
        (TRecord(r0), TRecord(r1)) => mgu(ctx, r0, r1, info),
        (TRecordConcat(r0, r1), TRecordConcat(r2, r3)) => mgu_fold(ctx, [r0, r1], [r2, r3], info),
        (TRecordConcat(r0, r1), r2) | (r2, TRecordConcat(r0, r1)) => {
            let (xts0, t0) = deconstruct_row(r0);
            let (xts1, t1) = deconstruct_row(r1);
            let (xts2, t2) = deconstruct_row(r2.into());

            let mut xts = vector![]; // Rows present in r2 but not in r0 or r1
            let mut s = vector![];
            for (x, t) in xts2.into_iter() {
                match (xts0.find_assoc(&x), xts1.find_assoc(&x)) {
                    (Some(t3), _) | (_, Some(t3)) => s = mgu_acc(ctx, t, t3.clone(), s, info),
                    (None, None) => xts.push_back((x, t)),
                }
            }

            match (t0.kind(), t1.kind()) {
                (TVar(x), _) | (_, TVar(x)) => {
                    let r = xts_to_row(xts, t2);
                    mgu_acc(ctx, TVar(x).into(), r.into(), s, info)
                }
                (TGeneric(x), _) | (_, TGeneric(x)) => {
                    let r = xts_to_row(xts, t2);
                    mgu_acc(ctx, TGeneric(x).into(), r.into(), s, info)
                }
                (TRowEmpty, TRowEmpty) => {
                    if xts.is_empty() {
                        vector![]
                    } else {
                        todo!("fail")
                    }
                }
                _ => unreachable!(),
            }
        }
        (TNominal(x0, ts0), TNominal(x1, ts1)) if x0 == x1 => mgu_fold(ctx, ts0, ts1, info),
        (TGeneric(x), TGeneric(y)) if x == y => vector![],
        (TRowEmpty, TRowEmpty) => vector![],
        (TRowExtend(xt0, r0), r1) | (r1, TRowExtend(xt0, r0)) => {
            // println!("t0: {}", write_hir::type_to_string(&t0.into()));
            // println!("t1: {}", write_hir::type_to_string(&t1.into()));
            mgu_row(ctx, (xt0, r0), r1.into(), vector![], info)
        }
        (TAlias(info0, info1, t0), t1) | (t1, TAlias(info0, info1, t0)) => {
            mgu(ctx, t0, t1.into(), info)
        }
        (TTuple(ts0, c0), TTuple(ts1, c1)) if !c0 || !c1 || ts0.len() == ts1.len() => {
            mgu_fold(ctx, ts0, ts1, info)
        }
        (TArray(t0, n0), TArray(t1, n1)) if !n0.is_some() || !n1.is_some() || n0 == n1 => {
            mgu(ctx, t0, t1, info)
        }
        (TUnit, TUnit) => vector![],
        (TNever, _) | (_, TNever) => vector![],
        (TError, _) | (_, TError) => vector![],
        _ => {
            ctx.diagnostics.push_error(Error::TypeMismatch {
                lhs: write_hir::type_to_string(&t0.into()),
                rhs: write_hir::type_to_string(&t1.into()),
                info,
            });
            vector![]
        }
    }
}

fn deconstruct_row(r: Type) -> (Vector<(Name, Type)>, Type) {
    fn rec(r: Type, xts: &mut Vector<(Name, Type)>) -> Type {
        match r.kind() {
            TRowEmpty => TRowEmpty.into(),
            TVar(x) => TVar(x).into(),
            TGeneric(x) => TGeneric(x).into(),
            TRowExtend(xt, r) => {
                xts.push_back(xt);
                rec(r, xts)
            }
            TRecord(r) => rec(r, xts),
            TRecordConcat(r0, r1) => TRecordConcat(r0, r1).into(),
            _ => unreachable!("Expected a row type"),
        }
    }
    let mut xts = vector![];
    let t = rec(r, &mut xts);
    (xts, t)
}

fn mgu_row(
    ctx: &mut Context,
    ((x0, t0), r0): ((Name, Type), Type), // Left-hand side of unification
    r1: Type,                             // Right-hand side of unification
    mut xts1: Vector<(Name, Type)>,       // Rows that are in r1 but not in r0
    info: Info,
) -> Vector<(Name, Type)> {
    // tracing::trace!("mgu_row");
    // for (x, t) in &xts1 {
    //     tracing::trace!("  {}: {}", x, write_hir::type_to_string(&t));
    // }
    match r1.kind() {
        TRowEmpty => {
            // tracing::trace!("TRowEmpty");
            ctx.diagnostics
                .push_error(Error::RowNotFound { info, x: x0 });
            vector![]
        }
        TRowExtend((x1, t1), r2) => {
            if x0 == x1 {
                // tracing::trace!("TRowExtend {} == {}", x0, x1);
                let r3 = xts_to_row(xts1, r2);
                mgu_fold(ctx, [t0, r0], [t1, r3], info)
            } else {
                // tracing::trace!("TRowExtend {} != {}", x0, x1);
                xts1.push_back((x1, t1));
                mgu_row(ctx, ((x0, t0), r0), r2, xts1, info)
            }
        }
        TVar(x) => {
            // tracing::trace!("TVar {}", x);
            let r2 = xts_to_row(xts1, ctx.fresh_r());
            let r3 = TRowExtend((x0, t0), ctx.fresh_r()).into();
            // tracing::trace!("  r0 = r2: {} = {}", write_hir::type_to_string(&r0), write_hir::type_to_string(&r2));
            // tracing::trace!("  r1 = r3: {} = {}", write_hir::type_to_string(&r1), write_hir::type_to_string(&r3));
            mgu_fold(ctx, [r0, r1], [r2, r3], info)
        }
        x => unreachable!("rewrite_row: not a row {:?}", x),
    }
}

fn xts_to_row(xts: impl IntoIterator<Item = (Name, Type)>, r0: Type) -> Type {
    xts.into_iter()
        .fold(r0, |r, (x, t)| TRowExtend((x, t), r).into())
}

fn row_tail(r: Type) -> Option<Type> {
    match r.kind() {
        TRowEmpty => None,
        TRowExtend(_, r) => row_tail(r),
        TVar(x) => Some(TVar(x).into()),
        _ => unreachable!("row_tail: not a row"),
    }
}

fn row_get(x: Name, r: Type) -> Option<Type> {
    match r.kind() {
        TRowEmpty => None,
        TRowExtend((x0, t), r) => {
            if x == x0 {
                Some(t)
            } else {
                row_get(x, r)
            }
        }
        TVar(x) => Some(TVar(x).into()),
        _ => unreachable!("row_contains: not a row"),
    }
}

fn ftv(t: Type) -> OrdSet<Name> {
    fn ftv(acc: &mut OrdSet<Name>, t: Type) {
        match t.kind() {
            TFun(ts, t) => {
                ts.into_iter().for_each(|t| ftv(acc, t));
                ftv(acc, t)
            }
            TRecord(r) => ftv(acc, r),
            TRowEmpty => {}
            TRowExtend((_, t), r) => {
                ftv(acc, t);
                ftv(acc, r);
            }
            TRecordConcat(t0, t1) => {
                ftv(acc, t0);
                ftv(acc, t1);
            }
            TNominal(_, ts) => ts.into_iter().for_each(|t| ftv(acc, t)),
            TGeneric(_) => {}
            TVar(x) => {
                acc.insert(x);
            }
            TTuple(ts, _) => ts.into_iter().for_each(|t| ftv(acc, t)),
            TAlias(_, _, t) => ftv(acc, t),
            TArray(t, n) => ftv(acc, t),
            TArrayConcat(t0, t1) => {
                ftv(acc, t0);
                ftv(acc, t1);
            }
            TUnit => {}
            TNever => {}
            TError => {}
        }
    }
    let mut acc = OrdSet::new();
    ftv(&mut acc, t);
    acc
}

fn mgu_acc(
    ctx: &mut Context,
    t0: Type,
    t1: Type,
    s0: Vector<(Name, Type)>,
    info: Info,
) -> Vector<(Name, Type)> {
    let t0 = apply(s0.clone(), t0);
    let t1 = apply(s0.clone(), t1);
    let s1 = mgu(ctx, t0, t1, info);
    compose(s1, s0)
}

fn mgu_fold(
    ctx: &mut Context,
    ts0: impl IntoIterator<Item = Type>,
    ts1: impl IntoIterator<Item = Type>,
    info: Info,
) -> Vector<(Name, Type)> {
    ts0.into_iter().zip(ts1).fold(vector![], |s0, (t0, t1)| {
        let t0 = apply(s0.clone(), t0);
        let t1 = apply(s0.clone(), t1);
        let s1 = mgu(ctx, t0, t1, info);
        compose(s1, s0)
    })
}

fn compose(s0: Vector<(Name, Type)>, s1: Vector<(Name, Type)>) -> Vector<(Name, Type)> {
    s1.into_iter()
        .map(|(x, t)| (x, apply(s0.clone(), t)))
        .chain(s0.clone())
        .collect()
}

fn apply(s: Vector<(Name, Type)>, t: Type) -> Type {
    let f = |t: Type| apply(s.clone(), t);
    match t.kind() {
        TFun(ts, t) => TFun(ts.map(f), f(t)),
        TRecord(r) => TRecord(f(r)),
        TNominal(x, ts) => TNominal(x, ts.map(f)),
        TRowEmpty => TRowEmpty,
        TRowExtend((x, t), r) => TRowExtend((x, f(t)), f(r)),
        TRecordConcat(t0, t1) => TRecordConcat(f(t0), f(t1)),
        TGeneric(x) => TGeneric(x),
        TVar(x) => {
            return s
                .into_iter()
                .find_map(|(x1, t1)| if x == *x1 { Some(t1) } else { None })
                .unwrap_or(t);
        }
        TTuple(ts, closed) => TTuple(ts.map(f), closed),
        TArray(t, n) => TArray(f(t), n),
        TArrayConcat(t0, t1) => TArrayConcat(f(t0), f(t1)),
        TAlias(info0, info1, t) => TAlias(info0, info1, f(t)),
        TUnit => TUnit,
        TNever => TNever,
        TError => TError,
    }
    .into()
}

fn instantiate(s: Vector<(Name, Type)>, t: Type) -> Type {
    let f = |t: Type| instantiate(s.clone(), t);
    match t.kind() {
        TFun(ts, t) => TFun(ts.map(f), f(t)),
        TRecord(r) => TRecord(f(r)),
        TNominal(x, ts) => TNominal(x, ts.map(f)),
        TRowEmpty => TRowEmpty,
        TRowExtend((x, t), r) => TRowExtend((x, f(t)), f(r)),
        TRecordConcat(t0, t1) => TRecordConcat(f(t0), f(t1)),
        TGeneric(x) => {
            return s
                .into_iter()
                .find_map(|(x1, t1)| if x == *x1 { Some(t1) } else { None })
                .expect("Generic should be bound");
        }
        TVar(x) => TVar(x),
        TTuple(ts, closed) => TTuple(ts.map(f), closed),
        TArray(t, n) => TArray(f(t), n),
        TArrayConcat(t0, t1) => TArrayConcat(f(t0), f(t1)),
        TAlias(info0, info1, t1) => TAlias(info0, info1, f(t1)),
        TUnit => TUnit,
        TNever => TNever,
        TError => TError,
    }
    .into()
}

fn generalise(s: Vector<(Name, Type)>, t: Type) -> Type {
    let f = |t: Type| generalise(s.clone(), t);
    match t.kind() {
        TFun(ts, t) => TFun(ts.map(f), f(t)),
        TRecord(r) => TRecord(f(r)),
        TRowEmpty => TRowEmpty,
        TRowExtend((x, t), r) => TRowExtend((x, f(t)), f(r)),
        TRecordConcat(t0, t1) => TRecordConcat(f(t0), f(t1)),
        TNominal(x, ts) => TNominal(x, ts.map(f)),
        TGeneric(x) => TGeneric(x),
        TVar(x) => {
            return s
                .into_iter()
                .find_map(|(x1, t1)| if x == *x1 { Some(t1) } else { None })
                .unwrap_or(t);
        }
        TTuple(ts, closed) => TTuple(ts.map(f), closed),
        TArray(t, n) => TArray(f(t), n),
        TArrayConcat(t0, t1) => TArrayConcat(f(t0), f(t1)),
        TAlias(info0, info1, t) => TAlias(info0, info1, f(t)),
        TUnit => TUnit,
        TNever => TNever,
        TError => TError,
    }
    .into()
}

fn fields_to_rows(xts: Vector<(Name, Type)>, r: Type) -> Type {
    xts.into_iter()
        .rev()
        .fold(r, |r, (x, t)| TRowExtend((x, t), r).into())
}
