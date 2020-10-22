use crate::{ast::*, dataflow::*, typer::*};
use derive_more::Constructor;
use std::collections::HashMap;
use BinOpKind::*;
use ExprKind::*;
use LitKind::*;
use UnOpKind::*;
use Value::*;

#[allow(unused)]
#[derive(Clone, Debug)]
enum Value {
    I8(i8),
    I16(i16),
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    Unit,
    Fun,
    Vector(Vec<Value>),
    Tuple(Vec<Value>),
    StagedPipe(Box<Value>, Box<Value>),
    ValueErr,
}

#[derive(Default, Debug)]
struct Stack {
    pub scopes: Vec<HashMap<Ident, Value>>,
}

impl Stack {
    fn new() -> Self {
        Self::default()
    }
    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }
    fn pop_scope(&mut self) {
        self.scopes.pop();
    }
    fn insert(&mut self, k: Ident, v: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(k, v);
        }
    }
    fn lookup(&self, k: Ident) -> Option<&Value> {
        self.scopes.iter().rev().find_map(|scope| scope.get(&k))
    }
}

#[derive(Constructor)]
struct Context<'a> {
    pub stack: &'a mut Stack,
    pub dataflow: &'a mut Dataflow,
    pub typer: &'a mut Typer,
}

impl Script<'_> {
    pub fn eval(&mut self) -> Dataflow {
        let mut dataflow = Dataflow::new();
        let mut stack = Stack::new();
        stack.push_scope();
        let Script { ast, info, .. } = self;
        let mut ctx = Context {
            stack: &mut stack,
            dataflow: &mut dataflow,
            typer: info.typer.get_mut(),
        };
        let _ = ast.body.eval(&mut ctx);
        dataflow
    }
}

impl Expr {
    #[allow(unused)]
    fn eval(&self, ctx: &mut Context) -> Value {
        match &self.kind {
            Lit(kind) => match kind {
                LitI8(v) => I8(*v),
                LitI16(v) => I16(*v),
                LitI32(v) => I32(*v),
                LitI64(v) => I64(*v),
                LitF32(v) => F32(*v),
                LitF64(v) => F64(*v),
                LitBool(v) => Bool(*v),
                LitUnit => Unit,
                LitErr => ValueErr,
                LitTime(_) => todo!(),
            },
            Var(id) => {
                if let Some(val) = ctx.stack.lookup(*id) {
                    val.clone()
                } else {
                    ValueErr
                }
            }
            Let(id, e) => {
                let v = e.eval(ctx);
                ctx.stack.insert(*id, v);
                Unit
            }
            BinOp(lhs, Seq, rhs) => {
                lhs.eval(ctx);
                ctx.stack.push_scope();
                let rhs = rhs.eval(ctx);
                ctx.stack.pop_scope();
                rhs
            }
            ConsArray(exprs) => todo!(),
            ConsStruct(fields) => todo!(),
            ConsTuple(exprs) => todo!(),
            ConsVariant(sym, exprs) => todo!(),
            Closure(id, expr) => todo!(),
            For(..) => todo!(),
            If(c, t, e) => {
                let c = c.eval(ctx);
                match (c, t, e) {
                    (Bool(c), t, e) if c => t.eval(ctx),
                    (Bool(c), t, e) if !c => e.eval(ctx),
                    _ => ValueErr,
                }
            }
            Match(expr, clauses) => todo!(),
            Loop(cond, body) => todo!(),
            UnOp(kind, expr) => {
                let v = expr.eval(ctx);
                match kind {
                    Not => match v {
                        Bool(v) => Bool(!v),
                        _ => ValueErr,
                    },
                    Neg => match v {
                        I8(v) => I8(-v),
                        I16(v) => I16(-v),
                        I32(v) => I32(-v),
                        I64(v) => I64(-v),
                        F32(v) => F32(-v),
                        F64(v) => F64(-v),
                        _ => ValueErr,
                    },
                    Call(exprs) => match v {
                        Fun => todo!(),
                        _ => ValueErr,
                    },
                    Project(index) => match v {
                        Tuple(values) => todo!(),
                        _ => ValueErr,
                    },
                    Access(field) => match v {
                        _ => todo!(),
                    },
                    UnOpErr | Cast(_) | Emit => ValueErr,
                }
            }
            BinOp(lhs, kind, rhs) => {
                let lhs = lhs.eval(ctx);
                let rhs = rhs.eval(ctx);
                match (lhs, kind, rhs) {
                    // Add
                    (I8(l), Add, I8(r)) => I8(l + r),
                    (I16(l), Add, I16(r)) => I16(l + r),
                    (I32(l), Add, I32(r)) => I32(l + r),
                    (I64(l), Add, I64(r)) => I64(l + r),
                    (F32(l), Add, F32(r)) => F32(l + r),
                    (F64(l), Add, F64(r)) => F64(l + r),
                    // Sub
                    (I8(l), Sub, I8(r)) => I8(l - r),
                    (I16(l), Sub, I16(r)) => I16(l - r),
                    (I32(l), Sub, I32(r)) => I32(l - r),
                    (I64(l), Sub, I64(r)) => I64(l - r),
                    (F32(l), Sub, F32(r)) => F32(l - r),
                    (F64(l), Sub, F64(r)) => F64(l - r),
                    // Mul
                    (I8(l), Mul, I8(r)) => I8(l * r),
                    (I16(l), Mul, I16(r)) => I16(l * r),
                    (I32(l), Mul, I32(r)) => I32(l * r),
                    (I64(l), Mul, I64(r)) => I64(l * r),
                    (F32(l), Mul, F32(r)) => F32(l * r),
                    (F64(l), Mul, F64(r)) => F64(l * r),
                    // Div
                    (I8(l), Div, I8(r)) => I8(l / r),
                    (I16(l), Div, I16(r)) => I16(l / r),
                    (I32(l), Div, I32(r)) => I32(l / r),
                    (I64(l), Div, I64(r)) => I64(l / r),
                    (F32(l), Div, F32(r)) => F32(l / r),
                    (F64(l), Div, F64(r)) => F64(l / r),
                    // Pow
                    (I8(l), Pow, I32(r)) => I8(l.pow(r as u32)),
                    (I16(l), Pow, I32(r)) => I16(l.pow(r as u32)),
                    (I32(l), Pow, I32(r)) => I32(l.pow(r as u32)),
                    (I64(l), Pow, I32(r)) => I64(l.pow(r as u32)),
                    (F32(l), Pow, I32(r)) => F32(l.powi(r)),
                    (F64(l), Pow, I32(r)) => F64(l.powi(r)),
                    (F32(l), Pow, F32(r)) => F32(l.powf(r)),
                    (F64(l), Pow, F64(r)) => F64(l.powf(r)),
                    // Equ
                    (I8(l), Equ, I8(r)) => Bool(l == r),
                    (I16(l), Equ, I16(r)) => Bool(l == r),
                    (I32(l), Equ, I32(r)) => Bool(l == r),
                    (I64(l), Equ, I64(r)) => Bool(l == r),
                    (Unit, Equ, Unit) => Bool(true),
                    // Neq
                    (I8(l), Neq, I8(r)) => Bool(l == r),
                    (I16(l), Neq, I16(r)) => Bool(l == r),
                    (I32(l), Neq, I32(r)) => Bool(l == r),
                    (I64(l), Neq, I64(r)) => Bool(l == r),
                    (Unit, Neq, Unit) => Bool(false),
                    // Or
                    (Bool(l), Or, Bool(r)) => Bool(l || r),
                    // And
                    (Bool(l), And, Bool(r)) => Bool(l && r),
                    // Gt
                    (I8(l), Gt, I8(r)) => Bool(l > r),
                    (I16(l), Gt, I16(r)) => Bool(l > r),
                    (I32(l), Gt, I32(r)) => Bool(l > r),
                    (I64(l), Gt, I64(r)) => Bool(l > r),
                    (F32(l), Gt, F32(r)) => Bool(l > r),
                    (F64(l), Gt, F64(r)) => Bool(l > r),
                    // Lt
                    (I8(l), Lt, I8(r)) => Bool(l < r),
                    (I16(l), Lt, I16(r)) => Bool(l < r),
                    (I32(l), Lt, I32(r)) => Bool(l < r),
                    (I64(l), Lt, I64(r)) => Bool(l < r),
                    (F32(l), Lt, F32(r)) => Bool(l < r),
                    (F64(l), Lt, F64(r)) => Bool(l < r),
                    // Geq
                    (I8(l), Geq, I8(r)) => Bool(l >= r),
                    (I16(l), Geq, I16(r)) => Bool(l >= r),
                    (I32(l), Geq, I32(r)) => Bool(l >= r),
                    (I64(l), Geq, I64(r)) => Bool(l >= r),
                    (F32(l), Geq, F32(r)) => Bool(l >= r),
                    (F64(l), Geq, F64(r)) => Bool(l >= r),
                    // Leq
                    (I8(l), Leq, I8(r)) => Bool(l <= r),
                    (I16(l), Leq, I16(r)) => Bool(l <= r),
                    (I32(l), Leq, I32(r)) => Bool(l <= r),
                    (I64(l), Leq, I64(r)) => Bool(l <= r),
                    (F32(l), Leq, F32(r)) => Bool(l <= r),
                    (F64(l), Leq, F64(r)) => Bool(l <= r),
                    // Staged
                    (l, Pipe, r) => StagedPipe(l.into(), r.into()),
                    (_, BinOpErr, _) => ValueErr,
                    _ => unreachable!(),
                }
            }
            ExprErr => ValueErr,
        }
    }
}
