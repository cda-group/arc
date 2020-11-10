/// Module for representing the program stack of the big-step evaluator.
pub(crate) mod stack;
/// Module for representing the possible values an Arc-Script can evaluate into.
pub(crate) mod value;

use crate::compiler::dfg::from::eval::stack::Stack;
use crate::compiler::dfg::from::eval::value::{Value, ValueKind};
use crate::compiler::dfg::{EdgeData, Node, NodeData, Port, DFG};
use crate::compiler::hir::{
    self, BinOp, BinOpKind, BinOpKind::*, Expr, ExprKind, ItemKind, LitKind, ParamKind, TypeKind,
    UnOp, UnOpKind, UnOpKind::*, HIR,
};
use crate::compiler::info::diags::DiagInterner;
use crate::compiler::info::diags::Error;
use crate::compiler::info::files::Loc;
use crate::compiler::info::names::NameId;
use crate::compiler::info::paths::PathId;
use crate::compiler::info::Info;
use crate::compiler::shared::{Map, New};

use std::collections::HashMap;

#[derive(New)]
pub(super) struct Context<'a> {
    stack: &'a mut Stack,
    dfg: &'a mut DFG,
    hir: &'a HIR,
    info: &'a Info,
}

trait Unwind<T> {
    fn unwind(self, loc: Option<Loc>) -> std::result::Result<T, ControlFlowKind>;
}

impl<T> Unwind<T> for Option<T> {
    fn unwind(self, loc: Option<Loc>) -> std::result::Result<T, ControlFlowKind> {
        self.map(Ok).unwrap_or(ControlFlow(Panic(loc)))
    }
}

pub(super) type EvalResult = std::result::Result<Value, ControlFlowKind>;
pub(super) use std::result::Result::Err as ControlFlow;

pub(super) enum ControlFlowKind {
    Return(Value),
    Break(Value),
    Panic(Option<Loc>),
}

use ControlFlowKind::*;

/// A Big-Step evaluator.
pub(super) trait BigStep {
    /// Evaluates an expression `Self` in the context `ctx`.
    fn eval(&self, ctx: &mut Context) -> EvalResult;
}

impl BigStep for Expr {
    fn eval(&self, ctx: &mut Context) -> EvalResult {
        use ValueKind::*;
        let kind = match &self.kind {
            ExprKind::Lit(kind) => match kind {
                LitKind::U8(v) => U8(*v),
                LitKind::U16(v) => U16(*v),
                LitKind::U32(v) => U32(*v),
                LitKind::U64(v) => U64(*v),
                LitKind::I8(v) => I8(*v),
                LitKind::I16(v) => I16(*v),
                LitKind::I32(v) => I32(*v),
                LitKind::I64(v) => I64(*v),
                LitKind::F32(v) => F32(*v),
                LitKind::F64(v) => F64(*v),
                LitKind::Bool(v) => Bool(*v),
                LitKind::Unit => Unit,
                LitKind::Char(v) => Char(*v),
                LitKind::Str(v) => Str(v.clone()),
                LitKind::Time(_) => todo!(),
                LitKind::Err => unreachable!(),
            },
            ExprKind::Var(x) => return Ok(ctx.stack.lookup(x.id).clone()),
            ExprKind::Item(x) => Item(*x),
            ExprKind::Let(p, e0, e1) => match p.kind {
                ParamKind::Var(x) => {
                    let v0 = e0.eval(ctx)?;
                    ctx.stack.insert(x.id, v0);
                    return e1.eval(ctx);
                }
                ParamKind::Ignore => {
                    e0.eval(ctx)?;
                    return e1.eval(ctx);
                }
                ParamKind::Err => unreachable!(),
            },
            ExprKind::Array(es) => todo!(),
            ExprKind::Struct(fs) => Struct(
                fs.iter()
                    .map(|(x, e)| Ok((x.id, e.eval(ctx)?)))
                    .collect::<Result<_, _>>()?,
            ),
            ExprKind::Tuple(es) => Tuple(
                es.iter()
                    .map(|e| e.eval(ctx))
                    .collect::<Result<_, _>>()?,
            ),
            ExprKind::Enwrap(x0, x1, e0) => todo!(),
            ExprKind::Unwrap(x1, e0) => todo!(),
            ExprKind::Is(x1, e0) => todo!(),
            ExprKind::If(e0, e1, e2) => {
                let v0 = e0.eval(ctx)?;
                match v0.kind {
                    Bool(true) => return e1.eval(ctx),
                    Bool(false) => return e2.eval(ctx),
                    _ => unreachable!(),
                }
            }
            ExprKind::Loop(e) => loop {
                let v = e.eval(ctx);
                if let ControlFlow(Break(v)) = v {
                    return Ok(v);
                } else {
                    v?;
                }
            },
            ExprKind::Break => return ControlFlow(Break(Value::new(Unit, self.tv))),
            ExprKind::Return(e) => return ControlFlow(Return(e.eval(ctx)?)),
            ExprKind::UnOp(op, e) => {
                let v = e.eval(ctx)?;
                match &op.kind {
                    Not => match v.kind {
                        Bool(v) => Bool(!v),
                        _ => unreachable!(),
                    },
                    Neg => match v.kind {
                        I8(v) => I8(v.checked_neg().unwind(self.loc)?),
                        I16(v) => I16(v.checked_neg().unwind(self.loc)?),
                        I32(v) => I32(v.checked_neg().unwind(self.loc)?),
                        I64(v) => I64(v.checked_neg().unwind(self.loc)?),
                        F32(v) => F32(-v),
                        F64(v) => F64(-v),
                        _ => unreachable!(),
                    },
                    UnOpKind::Err => unreachable!(),
                }
            }
            ExprKind::Call(e, es) => match e.eval(ctx)?.kind {
                Item(x) => {
                    let item = ctx.hir.defs.get(&x).unwrap();
                    match &item.kind {
                        ItemKind::Fun(item) => {
                            ctx.stack.push_frame(x);
                            for (p, e) in item.params.iter().zip(es.iter()) {
                                let v = e.eval(ctx)?;
                                match p.kind {
                                    ParamKind::Var(x) => ctx.stack.insert(x.id, v),
                                    ParamKind::Ignore => {}
                                    ParamKind::Err => unreachable!(),
                                }
                            }
                            let v = item.body.eval(ctx);
                            if let ControlFlow(Return(v)) = v {
                                ctx.stack.pop_frame();
                                return Ok(v);
                            } else {
                                return v;
                            }
                        }
                        ItemKind::Task(item) => {
                            ctx.stack.push_frame(x);
                            for (p, e) in item.params.iter().zip(es.iter()) {
                                let v = e.eval(ctx)?;
                                match p.kind {
                                    ParamKind::Var(x) => ctx.stack.insert(x.id, v),
                                    ParamKind::Ignore => {}
                                    ParamKind::Err => unreachable!(),
                                }
                            }
                            for x in &item.items {
                                let item = ctx.hir.defs.get(x).unwrap();
                                match &item.kind {
                                    ItemKind::State(item) => {
                                        let v = e.eval(ctx)?;
                                        ctx.stack.insert(item.name.id, v);
                                    }
                                    _ => {}
                                }
                            }
                            let frame = ctx.stack.take_frame();
                            let id = ctx.dfg.add_node(NodeData::new(x, frame));
                            Task(x, Node::new(id))
                        }
                        _ => unreachable!(),
                    }
                }
                // Connect streams to an instantiated task
                Task(x, target) => {
                    for (iport, e) in es.iter().enumerate() {
                        match e.eval(ctx)?.kind {
                            Stream(source, oport) => {
                                let edge = EdgeData::new(iport, oport);
                                ctx.dfg.add_edge(source.id, target.id, edge);
                            }
                            _ => unreachable!(),
                        }
                    }
                    let item = ctx.hir.defs.get(&x).unwrap();
                    match &item.kind {
                        ItemKind::Task(item) => {
                            let streams = item
                                .oports
                                .iter()
                                .enumerate()
                                .map(|(i, oport)| Value::new(Stream(target, i), *oport))
                                .collect();
                            Tuple(streams)
                        }
                        _ => unreachable!(),
                    }
                }
                _ => unreachable!(),
            },
            ExprKind::Access(e, x) => match e.eval(ctx)?.kind {
                Struct(efs) => return Ok(efs.get(&x.id).unwrap().clone()),
                _ => unreachable!(),
            },
            ExprKind::Project(e, i) => match e.eval(ctx)?.kind {
                Tuple(es) => return Ok(es.get(i.id).unwrap().clone()),
                _ => unreachable!(),
            },
            ExprKind::Emit(e) => match e.eval(ctx)?.kind {
                Variant(x0, x1, v) => todo!(),
                _ => unreachable!(),
            },
            ExprKind::Log(e) => todo!(),
            // Short-circuit
            ExprKind::BinOp(e0, op, e1) if matches!(op.kind, And) => match e0.eval(ctx)?.kind {
                v0 @ Bool(false) => v0,
                Bool(true) => match e1.eval(ctx)?.kind {
                    v1 @ Bool(_) => v1,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            },
            // Short-circuit
            ExprKind::BinOp(e0, op, e1) if matches!(op.kind, Or) => match e0.eval(ctx)?.kind {
                v0 @ Bool(true) => v0,
                Bool(false) => match e1.eval(ctx)?.kind {
                    v1 @ Bool(_) => v1,
                    _ => unreachable!(),
                },
                _ => unreachable!(),
            },
            #[rustfmt::skip]
            ExprKind::BinOp(e0, op, e1) => {
                let v0 = e0.eval(ctx)?;
                let v1 = e1.eval(ctx)?;
                match (v0.kind, &op.kind, v1.kind) {
                    // Add
                    (I8(l),  Add, I8(r))  => I8(l.checked_add(r).unwind(self.loc)?),
                    (I16(l), Add, I16(r)) => I16(l.checked_add(r).unwind(self.loc)?),
                    (I32(l), Add, I32(r)) => I32(l.checked_add(r).unwind(self.loc)?),
                    (I64(l), Add, I64(r)) => I64(l.checked_add(r).unwind(self.loc)?),
                    (F32(l), Add, F32(r)) => F32(l + r),
                    (F64(l), Add, F64(r)) => F64(l + r),
                    // Sub
                    (I8(l),  Sub, I8(r))  => I8(l.checked_sub(r).unwind(self.loc)?),
                    (I16(l), Sub, I16(r)) => I16(l.checked_sub(r).unwind(self.loc)?),
                    (I32(l), Sub, I32(r)) => I32(l.checked_sub(r).unwind(self.loc)?),
                    (I64(l), Sub, I64(r)) => I64(l.checked_sub(r).unwind(self.loc)?),
                    (F32(l), Sub, F32(r)) => F32(l - r),
                    (F64(l), Sub, F64(r)) => F64(l - r),
                    // Mul
                    (I8(l),  Mul, I8(r))  => I8(l.checked_mul(r).unwind(self.loc)?),
                    (I16(l), Mul, I16(r)) => I16(l.checked_mul(r).unwind(self.loc)?),
                    (I32(l), Mul, I32(r)) => I32(l.checked_mul(r).unwind(self.loc)?),
                    (I64(l), Mul, I64(r)) => I64(l.checked_mul(r).unwind(self.loc)?),
                    (F32(l), Mul, F32(r)) => F32(l * r),
                    (F64(l), Mul, F64(r)) => F64(l * r),
                    // Div
                    (I8(l),  Div, I8(r))  => I8(l.checked_div(r).unwind(self.loc)?),
                    (I16(l), Div, I16(r)) => I16(l.checked_div(r).unwind(self.loc)?),
                    (I32(l), Div, I32(r)) => I32(l.checked_div(r).unwind(self.loc)?),
                    (I64(l), Div, I64(r)) => I64(l.checked_div(r).unwind(self.loc)?),
                    (F32(l), Div, F32(r)) => F32(l / r),
                    (F64(l), Div, F64(r)) => F64(l / r),
                    // Pow
                    (I8(l),  Pow, I32(r)) => I8(l.checked_pow(r as u32).unwind(self.loc)?),
                    (I16(l), Pow, I32(r)) => I16(l.checked_pow(r as u32).unwind(self.loc)?),
                    (I32(l), Pow, I32(r)) => I32(l.checked_pow(r as u32).unwind(self.loc)?),
                    (I64(l), Pow, I32(r)) => I64(l.checked_pow(r as u32).unwind(self.loc)?),
                    (F32(l), Pow, I32(r)) => F32(l.powi(r)),
                    (F64(l), Pow, I32(r)) => F64(l.powi(r)),
                    (F32(l), Pow, F32(r)) => F32(l.powf(r)),
                    (F64(l), Pow, F64(r)) => F64(l.powf(r)),
                    // Pow
                    (I8(l),  Mod,  I8(r)) => I8(l.checked_rem(r).unwind(self.loc)?),
                    (I16(l), Mod, I16(r)) => I16(l.checked_rem(r).unwind(self.loc)?),
                    (I32(l), Mod, I32(r)) => I32(l.checked_rem(r).unwind(self.loc)?),
                    (I64(l), Mod, I64(r)) => I64(l.checked_rem(r).unwind(self.loc)?),
                    (F32(l), Mod, I32(r)) => F32(l.powi(r)),
                    (F64(l), Mod, I32(r)) => F64(l.powi(r)),
                    (F32(l), Mod, F32(r)) => F32(l.powf(r)),
                    (F64(l), Mod, F64(r)) => F64(l.powf(r)),
                    // Equ
                    (I8(l),  Equ, I8(r))  => Bool(l == r),
                    (I16(l), Equ, I16(r)) => Bool(l == r),
                    (I32(l), Equ, I32(r)) => Bool(l == r),
                    (I64(l), Equ, I64(r)) => Bool(l == r),
                    (Unit,   Equ, Unit)   => Bool(true),
                    // Neq
                    (I8(l),  Neq, I8(r))  => Bool(l == r),
                    (I16(l), Neq, I16(r)) => Bool(l == r),
                    (I32(l), Neq, I32(r)) => Bool(l == r),
                    (I64(l), Neq, I64(r)) => Bool(l == r),
                    (Unit,   Neq, Unit)   => Bool(false),
                    // Gt
                    (I8(l),  Gt,  I8(r))  => Bool(l > r),
                    (I16(l), Gt,  I16(r)) => Bool(l > r),
                    (I32(l), Gt,  I32(r)) => Bool(l > r),
                    (I64(l), Gt,  I64(r)) => Bool(l > r),
                    (F32(l), Gt,  F32(r)) => Bool(l > r),
                    (F64(l), Gt,  F64(r)) => Bool(l > r),
                    // Lt
                    (I8(l),  Lt,  I8(r))  => Bool(l < r),
                    (I16(l), Lt,  I16(r)) => Bool(l < r),
                    (I32(l), Lt,  I32(r)) => Bool(l < r),
                    (I64(l), Lt,  I64(r)) => Bool(l < r),
                    (F32(l), Lt,  F32(r)) => Bool(l < r),
                    (F64(l), Lt,  F64(r)) => Bool(l < r),
                    // Geq
                    (I8(l),  Geq, I8(r))  => Bool(l >= r),
                    (I16(l), Geq, I16(r)) => Bool(l >= r),
                    (I32(l), Geq, I32(r)) => Bool(l >= r),
                    (I64(l), Geq, I64(r)) => Bool(l >= r),
                    (F32(l), Geq, F32(r)) => Bool(l >= r),
                    (F64(l), Geq, F64(r)) => Bool(l >= r),
                    // Leq
                    (I8(l),  Leq, I8(r))  => Bool(l <= r),
                    (I16(l), Leq, I16(r)) => Bool(l <= r),
                    (I32(l), Leq, I32(r)) => Bool(l <= r),
                    (I64(l), Leq, I64(r)) => Bool(l <= r),
                    (F32(l), Leq, F32(r)) => Bool(l <= r),
                    (F64(l), Leq, F64(r)) => Bool(l <= r),
                    // Seq
                    (_, Seq, r) => r,
                    (l, Pipe, r) => todo!(),
                    _ => unreachable!(),
                }
            }
            ExprKind::Err => unreachable!(),
        };
        Ok(Value::new(kind, self.tv))
    }
}
