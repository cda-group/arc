/// Module for representing the program stack of the big-step evaluator.
pub(crate) mod stack;
/// Module for representing the possible values an Arc-Script can evaluate into.
pub(crate) mod value;
/// Module for representing control-flow constructs.
pub(crate) mod control;

use crate::compiler::dfg::from::eval::control::EvalResult;
use crate::compiler::dfg::from::eval::control::{Control, ControlKind, Unwind};
use crate::compiler::dfg::from::eval::stack::Stack;
use crate::compiler::dfg::from::eval::value::{Value, ValueKind};
use crate::compiler::dfg::{EdgeData, Node, NodeData, DFG};
use crate::compiler::hir::{
    self, BinOp, BinOpKind, BinOpKind::*, Expr, ExprKind, ItemKind, LitKind, ParamKind, TypeKind,
    UnOp, UnOpKind, UnOpKind::*, HIR,
};

use crate::compiler::info::Info;
use arc_script_core_shared::get;
use arc_script_core_shared::New;

use half::{bf16, f16};

/// Context needed while evaluating the `HIR`.
#[derive(New)]
pub(crate) struct Context<'a> {
    stack: &'a mut Stack,
    dfg: &'a mut DFG,
    hir: &'a HIR,
    info: &'a Info,
}

impl Expr {
    /// Evaluates an expression `Self` in the context `ctx`.
    pub(crate) fn eval(&self, ctx: &mut Context<'_>) -> EvalResult {
        use ControlKind::*;
        use ValueKind::*;
        let kind = match &self.kind {
            #[rustfmt::skip]
            ExprKind::Lit(kind) => match kind {
                LitKind::U8(v)   => U8(*v),
                LitKind::U16(v)  => U16(*v),
                LitKind::U32(v)  => U32(*v),
                LitKind::U64(v)  => U64(*v),
                LitKind::I8(v)   => I8(*v),
                LitKind::I16(v)  => I16(*v),
                LitKind::I32(v)  => I32(*v),
                LitKind::I64(v)  => I64(*v),
                LitKind::Bf16(v) => Bf16(*v),
                LitKind::F16(v)  => F16(*v),
                LitKind::F32(v)  => F32(*v),
                LitKind::F64(v)  => F64(*v),
                LitKind::Bool(v) => Bool(*v),
                LitKind::Unit    => Unit,
                LitKind::Char(v) => Char(*v),
                LitKind::Str(v)  => Str(v.clone()),
                LitKind::Time(_) => todo!(),
                LitKind::Err     => unreachable!(),
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
            ExprKind::Array(_es) => todo!(),
            ExprKind::Struct(fs) => Struct(
                fs.iter()
                    .map(|(x, e)| Ok((x.id, e.eval(ctx)?)))
                    .collect::<Result<_, _>>()?,
            ),
            ExprKind::Tuple(es) => Tuple(es.iter().map(|e| e.eval(ctx)).collect::<Result<_, _>>()?),
            ExprKind::Enwrap(x0, e0) => Variant(*x0, e0.eval(ctx)?.into()),
            ExprKind::Unwrap(x0, e0) => {
                let v0 = e0.eval(ctx)?;
                match v0.kind {
                    Variant(x1, v) => {
                        if x1 == *x0 {
                            return Ok(*v);
                        } else {
                            return Control(Panic(self.loc))?;
                        }
                    }
                    _ => return Control(Panic(self.loc))?,
                }
            }
            /// TODO: This is UB if e0 has side-effects.
            /// Handle that in the type-checker.
            ExprKind::Is(x0, e0) => {
                let v0 = e0.eval(ctx)?;
                match v0.kind {
                    Variant(x1, _) => {
                        if x1 == *x0 {
                            Bool(true)
                        } else {
                            return Control(Panic(self.loc))?;
                        }
                    }
                    _ => return Control(Panic(self.loc))?,
                }
            }
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
                if let Control(Break(v)) = v {
                    return Ok(v);
                } else {
                    v?;
                }
            },
            ExprKind::Break => return Control(Break(Value::new(Unit, self.tv))),
            ExprKind::Return(e) => return Control(Return(e.eval(ctx)?)),
            ExprKind::UnOp(op, e) => {
                let v = e.eval(ctx)?;
                match &op.kind {
                    Boxed => todo!(),
                    Not => match v.kind {
                        Bool(v) => Bool(!v),
                        _ => unreachable!(),
                    },
                    Neg => match v.kind {
                        I8(v) => I8(v.checked_neg().or_unwind(self.loc)?),
                        I16(v) => I16(v.checked_neg().or_unwind(self.loc)?),
                        I32(v) => I32(v.checked_neg().or_unwind(self.loc)?),
                        I64(v) => I64(v.checked_neg().or_unwind(self.loc)?),
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
                            if let Control(Return(v)) = v {
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
                                if let ItemKind::State(item) = &item.kind {
                                    let v = e.eval(ctx)?;
                                    let x = ctx.info.paths.resolve(item.path.id).name.id;
                                    ctx.stack.insert(x, v);
                                }
                            }
                            let frame = ctx.stack.take_frame();
                            // Streams and tasks are represented as references (ids) to edges and
                            // nodes respectively in the dataflow graph. This means that instances
                            // of both can be re-used in different parts of the dataflow graph. In
                            // consequence, streams can be multiplexed into and out of tasks.
                            // Another option is to represent streams and tasks as values instead
                            // of references. This would prevent the possibility of multiplexing.
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
                    let item = get!(&item.kind, ItemKind::Task(item));
                    match item.ohub.kind {
                        hir::HubKind::Tagged(x) => {
                            let item = ctx.hir.defs.get(&x).unwrap();
                            let item = get!(&item.kind, ItemKind::Enum(item));
                            let streams = item
                                .variants
                                .iter()
                                .enumerate()
                                .map(|(i, x)| {
                                    let item = ctx.hir.defs.get(x).unwrap();
                                    let item = get!(&item.kind, ItemKind::Variant(item));
                                    Value::new(Stream(target, i), item.tv)
                                })
                                .collect::<Vec<_>>();
                            Tuple(streams)
                        }
                        hir::HubKind::Single(tv) => return Ok(Value::new(Stream(target, 0), tv)),
                    }
                }
                _ => unreachable!(),
            },
            ExprKind::Access(e, x) => match e.eval(ctx)?.kind {
                Struct(vfs) => return Ok(vfs.get(&x.id).unwrap().clone()),
                _ => unreachable!(),
            },
            ExprKind::Project(e, i) => match e.eval(ctx)?.kind {
                Tuple(vs) => return Ok(vs.get(i.id).unwrap().clone()),
                _ => unreachable!(),
            },
            ExprKind::Emit(e) => match e.eval(ctx)?.kind {
                Variant(_x0, _v) => todo!(),
                _ => unreachable!(),
            },
            ExprKind::Log(_e) => todo!(),
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
                    (I8(l),   Add, I8(r))   => I8(l.checked_add(r).or_unwind(self.loc)?),
                    (I16(l),  Add, I16(r))  => I16(l.checked_add(r).or_unwind(self.loc)?),
                    (I32(l),  Add, I32(r))  => I32(l.checked_add(r).or_unwind(self.loc)?),
                    (I64(l),  Add, I64(r))  => I64(l.checked_add(r).or_unwind(self.loc)?),
                    (Bf16(l), Add, Bf16(r)) => Bf16(bf16::from_f32(l.to_f32() + r.to_f32())),
                    (F16(l),  Add, F16(r))  => F16(f16::from_f32(l.to_f32() + r.to_f32())),
                    (F32(l),  Add, F32(r))  => F32(l + r),
                    (F64(l),  Add, F64(r))  => F64(l + r),
                    // Sub
                    (I8(l),   Sub, I8(r))   => I8(l.checked_sub(r).or_unwind(self.loc)?),
                    (I16(l),  Sub, I16(r))  => I16(l.checked_sub(r).or_unwind(self.loc)?),
                    (I32(l),  Sub, I32(r))  => I32(l.checked_sub(r).or_unwind(self.loc)?),
                    (I64(l),  Sub, I64(r))  => I64(l.checked_sub(r).or_unwind(self.loc)?),
                    (Bf16(l), Sub, Bf16(r)) => Bf16(bf16::from_f32(l.to_f32() - r.to_f32())),
                    (F16(l),  Sub, F16(r))  => F16(f16::from_f32(l.to_f32() - r.to_f32())),
                    (F32(l),  Sub, F32(r))  => F32(l - r),
                    (F64(l),  Sub, F64(r))  => F64(l - r),
                    // Mul
                    (I8(l),   Mul, I8(r))   => I8(l.checked_mul(r).or_unwind(self.loc)?),
                    (I16(l),  Mul, I16(r))  => I16(l.checked_mul(r).or_unwind(self.loc)?),
                    (I32(l),  Mul, I32(r))  => I32(l.checked_mul(r).or_unwind(self.loc)?),
                    (I64(l),  Mul, I64(r))  => I64(l.checked_mul(r).or_unwind(self.loc)?),
                    (Bf16(l), Mul, Bf16(r)) => Bf16(bf16::from_f32(l.to_f32() * r.to_f32())),
                    (F16(l),  Mul, F16(r))  => F16(f16::from_f32(l.to_f32() * r.to_f32())),
                    (F32(l),  Mul, F32(r))  => F32(l * r),
                    (F64(l),  Mul, F64(r))  => F64(l * r),
                    // Div
                    (I8(l),   Div, I8(r))   => I8(l.checked_div(r).or_unwind(self.loc)?),
                    (I16(l),  Div, I16(r))  => I16(l.checked_div(r).or_unwind(self.loc)?),
                    (I32(l),  Div, I32(r))  => I32(l.checked_div(r).or_unwind(self.loc)?),
                    (I64(l),  Div, I64(r))  => I64(l.checked_div(r).or_unwind(self.loc)?),
                    (Bf16(l), Div, Bf16(r)) => Bf16(bf16::from_f32(l.to_f32() / r.to_f32())),
                    (F16(l),  Div, F16(r))  => F16(f16::from_f32(l.to_f32() / r.to_f32())),
                    (F32(l),  Div, F32(r))  => F32(l / r),
                    (F64(l),  Div, F64(r))  => F64(l / r),
                    // Pow
                    (I8(l),   Pow, I32(r))  => I8(l.checked_pow(r as u32).or_unwind(self.loc)?),
                    (I16(l),  Pow, I32(r))  => I16(l.checked_pow(r as u32).or_unwind(self.loc)?),
                    (I32(l),  Pow, I32(r))  => I32(l.checked_pow(r as u32).or_unwind(self.loc)?),
                    (I64(l),  Pow, I32(r))  => I64(l.checked_pow(r as u32).or_unwind(self.loc)?),
                    (F32(l),  Pow, I32(r))  => F32(l.powi(r)),
                    (F64(l),  Pow, I32(r))  => F64(l.powi(r)),
                    (F32(l),  Pow, F32(r))  => F32(l.powf(r)),
                    (F64(l),  Pow, F64(r))  => F64(l.powf(r)),
                    // Pow
                    (I8(l),   Mod,  I8(r))  => I8(l.checked_rem(r).or_unwind(self.loc)?),
                    (I16(l),  Mod, I16(r))  => I16(l.checked_rem(r).or_unwind(self.loc)?),
                    (I32(l),  Mod, I32(r))  => I32(l.checked_rem(r).or_unwind(self.loc)?),
                    (I64(l),  Mod, I64(r))  => I64(l.checked_rem(r).or_unwind(self.loc)?),
                    (F32(l),  Mod, I32(r))  => F32(l.powi(r)),
                    (F64(l),  Mod, I32(r))  => F64(l.powi(r)),
                    (F32(l),  Mod, F32(r))  => F32(l.powf(r)),
                    (F64(l),  Mod, F64(r))  => F64(l.powf(r)),
                    // Equ
                    (I8(l),   Equ, I8(r))   => Bool(l == r),
                    (I16(l),  Equ, I16(r))  => Bool(l == r),
                    (I32(l),  Equ, I32(r))  => Bool(l == r),
                    (I64(l),  Equ, I64(r))  => Bool(l == r),
                    (Unit,    Equ, Unit)    => Bool(true),
                    // Neq
                    (I8(l),   Neq, I8(r))   => Bool(l == r),
                    (I16(l),  Neq, I16(r))  => Bool(l == r),
                    (I32(l),  Neq, I32(r))  => Bool(l == r),
                    (I64(l),  Neq, I64(r))  => Bool(l == r),
                    (Unit,    Neq, Unit)    => Bool(false),
                    // Gt
                    (I8(l),   Gt,  I8(r))   => Bool(l > r),
                    (I16(l),  Gt,  I16(r))  => Bool(l > r),
                    (I32(l),  Gt,  I32(r))  => Bool(l > r),
                    (I64(l),  Gt,  I64(r))  => Bool(l > r),
                    (Bf16(l), Gt,  Bf16(r)) => Bool(l > r),
                    (F16(l),  Gt,  F16(r))  => Bool(l > r),
                    (F32(l),  Gt,  F32(r))  => Bool(l > r),
                    (F64(l),  Gt,  F64(r))  => Bool(l > r),
                    // Lt
                    (I8(l),   Lt,  I8(r))   => Bool(l < r),
                    (I16(l),  Lt,  I16(r))  => Bool(l < r),
                    (I32(l),  Lt,  I32(r))  => Bool(l < r),
                    (I64(l),  Lt,  I64(r))  => Bool(l < r),
                    (Bf16(l), Lt,  Bf16(r)) => Bool(l < r),
                    (F16(l),  Lt,  F16(r))  => Bool(l < r),
                    (F32(l),  Lt,  F32(r))  => Bool(l < r),
                    (F64(l),  Lt,  F64(r))  => Bool(l < r),
                    // Geq
                    (I8(l),   Geq, I8(r))   => Bool(l >= r),
                    (I16(l),  Geq, I16(r))  => Bool(l >= r),
                    (I32(l),  Geq, I32(r))  => Bool(l >= r),
                    (I64(l),  Geq, I64(r))  => Bool(l >= r),
                    (Bf16(l), Geq, Bf16(r)) => Bool(l >= r),
                    (F16(l),  Geq, F16(r))  => Bool(l >= r),
                    (F32(l),  Geq, F32(r))  => Bool(l >= r),
                    (F64(l),  Geq, F64(r))  => Bool(l >= r),
                    // Leq
                    (I8(l),   Leq, I8(r))   => Bool(l <= r),
                    (I16(l),  Leq, I16(r))  => Bool(l <= r),
                    (I32(l),  Leq, I32(r))  => Bool(l <= r),
                    (I64(l),  Leq, I64(r))  => Bool(l <= r),
                    (Bf16(l), Leq, Bf16(r)) => Bool(l <= r),
                    (F16(l),  Leq, F16(r))  => Bool(l <= r),
                    (F32(l),  Leq, F32(r))  => Bool(l <= r),
                    (F64(l),  Leq, F64(r))  => Bool(l <= r),
                    // Seq
                    (_, Seq, r) => r,
                    (_l, Pipe, _r) => todo!(),
                    _ => unreachable!(),
                }
            }
            ExprKind::Todo => todo!(),
            ExprKind::Err => unreachable!(),
        };
        Ok(Value::new(kind, self.tv))
    }
}
