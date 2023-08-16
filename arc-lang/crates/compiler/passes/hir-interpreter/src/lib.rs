#![allow(unused)]

pub mod context;
pub mod definitions;

use context::ExprDecl;
use context::TypeDecl;
use diagnostics::Error;
use hir::*;
use im_rc::vector;
use im_rc::OrdMap;
use im_rc::Vector;
use utils::AssocVectorUtils;
use utils::VectorUtils;
use value::dynamic::Array;
use value::dynamic::Function;
use value::dynamic::Record;
use value::dynamic::Tuple;
use value::dynamic::Variant;
use value::Value;
use value::ValueKind::VUnit;

use crate::context::Context;

pub fn process(ctx: &mut Context, ss: Vector<Stmt>) {
    for s in ss {
        eval_top_stmt(ctx, s.clone());
        ctx.ss.push_back(s);
    }
}

fn eval_top_stmt(ctx: &mut Context, s: Stmt) -> Result<Option<(Value, Type)>, Error> {
    match s.kind.clone() {
        SDef(m, x, gs, ps, t, b) => {
            ctx.stack.bind_expr_decl(x, ExprDecl::Def(m, gs, ps, t, b));
        }
        SBif(m, x, gs, ts, t) => {
            ctx.stack.bind_expr_decl(x, ExprDecl::Bif(m, gs, ts, t));
        }
        SEnum(m, x, gs, xts) => {
            ctx.stack.bind_type_decl(x, TypeDecl::Enum(gs, xts));
        }
        SBit(m, x, gs) => {
            ctx.stack.bind_type_decl(x, TypeDecl::Bit(gs));
        }
        SVal(p, e) => {
            let t = e.t.clone();
            match eval_expr(ctx, e) {
                CValue(v) => {
                    bind(ctx, p, v.clone());
                    return Ok(Some((v, t)));
                }
                CException(e) => return Err(e),
                _ => unreachable!(),
            }
        }
        SExpr(e) => {
            let t = e.t.clone();
            match eval_expr(ctx, e) {
                CValue(v) => return Ok(Some((v, t))),
                CException(e) => return Err(e),
                _ => unreachable!(),
            }
        }
        SRecDef(..) => todo!(),
        SNoop => unreachable!(),
    }
    Ok(None)
}

fn eval_expr_stmt(ctx: &mut Context, s: Stmt) -> Control {
    match s.kind {
        SVal(p, e) => {
            let c = eval_expr(ctx, e);
            if let CValue(v) = c {
                bind(ctx, p, v);
                CValue(().into())
            } else {
                c
            }
        }
        SExpr(e) => {
            let c = eval_expr(ctx, e);
            if let CValue(_) = c {
                CValue(().into())
            } else {
                c
            }
        }
        _ => unreachable!(),
    }
}

fn bind(ctx: &mut Context, p: Pattern, v: Value) {
    match p.kind.as_ref().clone() {
        PVal(x) => ctx.stack.bind_expr_decl(x, ExprDecl::Var(v)),
        PIgnore => {}
        _ => unreachable!(),
    }
}

fn eval_func(ctx: &mut Context, x: Name, vs: Vector<Value>) -> Control {
    match ctx.stack.find_expr_decl(&x) {
        Some(ExprDecl::Def(m, gs, ps, t, b)) => {
            ctx.stack.push_scope(());
            for (p, v) in ps.into_iter().zip(vs) {
                bind(ctx, p, v);
            }
            let c = eval_block(ctx, b);
            ctx.stack.pop_scope();
            match c {
                CValue(v) | CFunReturn(v) => CValue(v),
                CException(s) => return CException(s),
                _ => unreachable!(),
            }
        }
        Some(ExprDecl::Bif(m, gs, ts, t)) => {
            let CString(x) = m.get("name").unwrap().as_ref().unwrap() else {
                unreachable!()
            };
            eval_builtin(ctx, x.clone(), ts, vs)
        }
        _ => unreachable!("{:?}", x),
    }
}

fn eval_builtin(ctx: &mut Context, x: Name, ts: Vector<Type>, vs: Vector<Value>) -> Control {
    let x = x.as_str();
    let vs = vs.into_iter().collect::<Vec<_>>();
    let ts = ts.into_iter().collect::<Vec<_>>();
    let v = ctx.bifs.get(x)(ctx, ts.as_slice(), vs.as_slice());
    CValue(v)
}

fn eval_block(ctx: &mut Context, b: Block) -> Control {
    ctx.stack.push_scope(());
    for s in b.ss {
        eval_expr_stmt(ctx, s.clone());
    }
    let v = val(ctx, b.e);
    ctx.stack.pop_scope();
    CValue(v).into()
}

fn val(ctx: &mut Context, e: Expr) -> Value {
    let EVal(x) = e.kind() else { unreachable!() };
    ctx.find_val(&x)
}

fn eval_expr(ctx: &mut Context, e: Expr) -> Control {
    match e.kind() {
        EConst(c) => CValue(constant(c).into()),
        EFunCallDirect(x, ts, es) => {
            let vs = es.into_iter().map(|e| val(ctx, e)).collect();
            eval_func(ctx, x, vs)
        }
        EFunCall(e, es) => {
            let f = val(ctx, e).as_function();
            let vs = es.into_iter().map(|e| val(ctx, e)).collect();
            eval_func(ctx, f.0, vs)
        }
        EDef(x, ts) => CValue(Function(x).into()).into(),
        EFunReturn(e) => CFunReturn(val(ctx, e).into()),
        EIfElse(e, b0, b1) => {
            if val(ctx, e).as_bool() {
                eval_block(ctx, b0)
            } else {
                eval_block(ctx, b1)
            }
        }
        ELoop(b) => loop {
            match eval_block(ctx, b.clone()) {
                CLoopBreak(v) => break CValue(v),
                CLoopContinue => continue,
                CFunReturn(v) => break CFunReturn(v),
                CValue(_) => continue,
                CException(x) => break CException(x),
            }
        },
        ELoopBreak(e) => {
            let v = val(ctx, e);
            CLoopBreak(v).into()
        }
        ELoopContinue => CLoopContinue,
        ERecord(xes) => {
            let xvs = xes.into_iter().map(|(x, e)| (x, val(ctx, e))).collect();
            CValue(Record(xvs).into())
        }
        ERecordAccess(e, x) => {
            let v = val(ctx, e);
            let xvs = v.as_record();
            CValue(xvs.0.get(&x).unwrap().clone().into())
        }
        EVariant(_, _, x, e) => {
            let v = val(ctx, e);
            CValue(Variant { x, v }.into())
        }
        EVariantAccess(_, _, x, e) => {
            let var = val(ctx, e).as_variant();
            assert_eq!(x, var.x);
            CValue(var.v).into()
        }
        EVariantCheck(_, _, x, e) => {
            let var = val(ctx, e).as_variant();
            CValue((var.x == x).into())
        }
        EMatch(_, _) => unreachable!(),
        EArray(es) => {
            let vs = es.map(|e| val(ctx, e));
            CValue(Array(vs).into())
        }
        EArrayAccess(e0, e1) => {
            let v0 = val(ctx, e0).as_array();
            let v1 = val(ctx, e1).as_usize();
            if let Some(v) = v0.0.get(v1) {
                CValue(v.clone().into())
            } else {
                CException(Error::InterpreterError {
                    info: e.info,
                    s: "array index out of bounds".into(),
                })
            }
        }
        ETuple(es) => {
            let vs = es.map(|e| val(ctx, e));
            CValue(Tuple(vs).into())
        }
        ETupleAccess(e, n) => {
            let v = val(ctx, e).as_tuple().0;
            CValue(v[n as usize].clone().into())
        }
        ERecordConcat(_, _) => todo!(),
        EArrayConcat(_, _) => todo!(),
        EMut(..) => unreachable!(),
        EVal(x) => CValue(val(ctx, e).into()),
        EVar(..) => unreachable!(),
        EDo(..) => unreachable!(),
        ENoop(..) => unreachable!(),
        EFor(..) => unreachable!(),
        EWhile(..) => unreachable!(),
        EFun(..) => unreachable!(),
        EError => unreachable!(),
    }
}

pub fn constant(c: Const) -> Value {
    match c {
        CInt(i) => i.into(),
        CBool(b) => b.into(),
        CFloat(f) => (f as f64).into(),
        CString(s) => builtins::string::String::from(s).into(),
        CUnit => ().into(),
        CChar(c) => c.into(),
    }
}

pub use Control::*;
#[derive(Debug, Clone)]
pub enum Control {
    CLoopBreak(Value),
    CLoopContinue,
    CFunReturn(Value),
    CValue(Value),
    CException(Error),
}
