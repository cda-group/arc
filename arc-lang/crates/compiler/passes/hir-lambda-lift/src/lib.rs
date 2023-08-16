#![allow(unused)]
//! Lambda lift HIR => HIR
//! * Lifts all function definitions to the top level
//! * Might be easier to do before type inference
//! * After type inference would however produce better errors
//!
//! ----------------------------------------------------------
//!   def f(a:I32): fun(I32):I32 {
//!       def g(b:I32):I32 {
//!           a + b
//!       }
//!       g
//!   }
//!
//!   f(1)(2)
//!
//! =>
//!
//!   def g(b:I32, {a:I32}):I32 {
//!       a + b
//!   }
//!
//!   def f(a:I32, {}:{}): {ptr:fun(I32, {b:I32}):I32, env:{a:I32}} {
//!       val g = {ptr:g, env:{a:a}};
//!       g
//!   }
//!
//!   do {
//!     val x:{ptr:fun(I32,{b:I32}), env:{b:I32}} = f(1);
//!     x.ptr(2, x.env)
//!   }
//! ----------------------------------------------------------
//!   val x = 1;
//!   val f: fun(a:I32): I32 = fun(a:I32): I32 = x+1;
//!   val g: fun(a:I32): I32 = fun(a:I32): I32 = f;
//!   g(1)
//!
//! =>
//!
//!   val x = 1;
//!   def f(a:I32,{x:I32}):I32 = x+1;
//!   val f: {ptr:fun(a:I32,{b:I32}):I32, env:{x:I32}} = {ptr:f, env:{x}};
//!   val g: {ptr:fun(a:I32,{b:I32}):I32, env:{x:I32}} = f;
//!   g.ptr(1, g.env)
//! ----------------------------------------------------------
//!   def f[T](a:T): fun(): T {
//!       def g(): T {
//!           a
//!       }
//!   }
//!
//! =>
//!
//!   def g[T](a:T, {}:{}): T {
//!       a
//!   }
//!
//!   def f[T](a:T, {}:{}): {ptr:fun({},{}):T, env:{a:T}} {
//!       val g = {ptr:g, env:{a:a}};
//!       g
//!   }
//! ----------------------------------------------------------
//! Cases to consider
//! * Function definitions
//!   1. Check which expression and type variables are captured (i.e. defined outside)
//!   2. Create a new function definition which takes an environment as an extra argument that
//!      contains the captured variables
//!      `def f(.., {_:_,...}):.. = ..`
//!   3. Replace the old function definition with a closure that stores a function pointer and environment
//!      `val f = {ptr:_, env:{_:_,...}}`
//!   4. Update the type wherever the function definition is used
//!     * Store type inside environment and propagate it
//!     * Return the type
//!   4. All references to the function definition are replaced with the closure
//!      `f(..) => f.ptr(.., f.env)`

use context::Context;
use context::ExprDecl;
use context::ScopeKind;
use context::TypeDecl;
use hir::*;
use im_rc::Vector;
use utils::VectorUtils;
pub mod context;

pub fn process(ctx: &mut Context, ss: Vector<Stmt>) -> Vector<Stmt> {
    ss.into_iter().map(|s| ll_stmt(ctx, s)).collect()
}

fn ll_stmt(ctx: &mut Context, s: Stmt) -> Stmt {
    let info = s.info;
    match s.kind {
        SDef(m, x, gs, ps, t, b) => {
            let gs = gs.concat(ctx.generics_in_scope());
            // def foo[A](x:A):A {
            //     def bar(y:A):A = y;
            //     bar(x)
            // }
            //
            // def bar(y:A):A = y;
            //
            // def foo[A](x:A):A {
            //     bar[A](x)
            // }
            ctx.stack
                .bind_expr_decl(x.clone(), ExprDecl::Def(gs.clone()));
            ctx.stack.push_scope(ScopeKind::Def(gs.clone()));
            let b = ll_block(ctx, b);
            ctx.stack.pop_scope();
            SDef(m, x, gs, ps, t, b).with(s.info)
        }
        SRecDef(..) => unreachable!(),
        SBif(m, x, gs, ts, t) => SBif(m, x, gs, ts, t).with(s.info),
        SEnum(m, x, gs, xts) => SEnum(m, x, gs, xts).with(s.info),
        SBit(m, x, gs) => SBit(m, x, gs).with(s.info),
        SVal(p, e) => {
            let e = ll_expr(ctx, e);
            SVal(p, e).with(s.info)
        }
        SExpr(e) => {
            let e = ll_expr(ctx, e);
            SExpr(e).with(s.info)
        }
        SNoop => SNoop.with(s.info),
    }
}

fn ll_block(ctx: &mut Context, b: Block) -> Block {
    let info = b.info;
    let ss = b.ss.map(|s| ll_stmt(ctx, s));
    let e = ll_expr(ctx, b.e);
    Block::new(ss, e, info)
}

fn ll_expr(ctx: &mut Context, e: Expr) -> Expr {
    let info = e.info;
    let t = e.t;
    match e.kind.as_ref().clone() {
        EConst(c) => EConst(c).with(t, info),
        EFun(ps, t, b) => {
            todo!()
        }
        EFunCall(e, es) => {
            todo!()
        }
        EFunReturn(e) => {
            todo!()
        }
        ELoop(b) => {
            let b = ll_block(ctx, b);
            ELoop(b).with(t, info)
        }
        ELoopBreak(e) => {
            let e = ll_expr(ctx, e);
            ELoopBreak(e).with(t, info)
        }
        ELoopContinue => ELoopContinue.with(t, info),
        EMatch(e, pbs) => {
            let e = ll_expr(ctx, e);
            let pbs = pbs.map(|(p, b)| (p, ll_block(ctx, b)));
            EMatch(e, pbs).with(t, info)
        }
        EArray(es) => {
            let es = es.map(|e| ll_expr(ctx, e));
            // let e = ll_expr(ctx, e);
            EArray(es).with(t, info)
        }
        EArrayAccess(e0, e1) => {
            let e0 = ll_expr(ctx, e0);
            let e1 = ll_expr(ctx, e1);
            EArrayAccess(e0, e1).with(t, info)
        }
        EIfElse(e, b0, b1) => {
            let e = ll_expr(ctx, e);
            let b0 = ll_block(ctx, b0);
            let b1 = ll_block(ctx, b1);
            EIfElse(e, b0, b1).with(t, info)
        }
        ERecord(xes) => {
            let xes = xes.map(|(x, e)| (x, ll_expr(ctx, e)));
            // let e = ll_expr(ctx, e);
            ERecord(xes).with(t, info)
        }
        ERecordAccess(e, x) => {
            let e = ll_expr(ctx, e);
            ERecordAccess(e, x).with(t, info)
        }
        EMut(e0, e1) => {
            let e0 = ll_expr(ctx, e0);
            let e1 = ll_expr(ctx, e1);
            EMut(e0, e1).with(t, info)
        }
        EVal(x) => EVal(x).with(t, info),
        EVar(x) => EVar(x).with(t, info),
        EDef(x, ts) => {
            todo!()
        }
        EVariant(x0, ts, x1, e) => {
            let e = ll_expr(ctx, e);
            let ts = ts.map(|t| ll_type(ctx, t));
            EVariant(x0, ts, x1, e).with(t, info)
        }
        EDo(b) => {
            let b = ll_block(ctx, b);
            EDo(b).with(t, info)
        }
        ENoop(e) => {
            let e = ll_expr(ctx, e);
            ENoop(e).with(t, info)
        }
        ETuple(es) => {
            let es = es.map(|e| ll_expr(ctx, e));
            ETuple(es).with(t, info)
        }
        ETupleAccess(e, n) => {
            let e = ll_expr(ctx, e);
            ETupleAccess(e, n).with(t, info)
        }
        EFor(p, e, b) => {
            let e = ll_expr(ctx, e);
            let b = ll_block(ctx, b);
            EFor(p, e, b).with(t, info)
        }
        EWhile(e, b) => {
            let e = ll_expr(ctx, e);
            let b = ll_block(ctx, b);
            EWhile(e, b).with(t, info)
        }
        EError => EError.with(t, info),
        EVariantAccess(_, _, _, _) => unreachable!(),
        EVariantCheck(_, _, _, _) => unreachable!(),
        EFunCallDirect(_, _, _) => todo!(),
        ERecordConcat(_, _) => todo!(),
        EArrayConcat(_, _) => todo!(),
    }
}

fn ll_type(ctx: &mut Context, t: Type) -> Type {
    match t.kind() {
        TFun(ts, t) => {
            todo!()
        }
        TTuple(ts, b) => {
            let ts = ts.map(|t| ll_type(ctx, t));
            TTuple(ts, b).into()
        }
        TRecord(t) => TRecord(t).into(),
        TNominal(x, ts) => {
            let ts = ts.map(|t| ll_type(ctx, t));
            TNominal(x, ts).into()
        }
        TAlias(info0, info1, t) => {
            let t = ll_type(ctx, t);
            TAlias(info0, info1, t).into()
        }
        TRowEmpty => TRowEmpty.into(),
        TRowExtend((x, t), r) => {
            let t = ll_type(ctx, t);
            let r = ll_type(ctx, r);
            TRowExtend((x, t), r).into()
        }
        TGeneric(x) => TGeneric(x).into(),
        TArray(t, n) => {
            let t = ll_type(ctx, t);
            TArray(t, n).into()
        }
        TVar(x) => TVar(x).into(),
        TError => TError.into(),
        TUnit => TUnit.into(),
        TNever => TNever.into(),
        TRecordConcat(_, _) => todo!(),
        TArrayConcat(_, _) => todo!(),
    }
}
