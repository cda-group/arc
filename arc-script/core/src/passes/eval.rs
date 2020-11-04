use crate::prelude::*;
use std::collections::HashMap;
use Value::*;

#[allow(unused)]
#[derive(Clone, Debug)]
enum Value {
    VI8(i8),
    VI16(i16),
    VI32(i32),
    VI64(i64),
    VF32(f32),
    VF64(f64),
    VBool(bool),
    VUnit,
    VFun,
    VVector(Vec<Value>),
    VTuple(Vec<Value>),
    VPipe(Box<Value>, Box<Value>),
    VErr,
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
        // TODO: Currently a hacky solution
        // which just evaluates the last function of the script
        if let Some(fundef) = ast.fundefs.values().last() {
            fundef.body.eval(&mut ctx);
        }
        dataflow
    }
}

impl Expr {
    #[allow(unused)]
    fn eval(&self, ctx: &mut Context) -> Value {
        match &self.kind {
            Lit(kind) => match kind {
                LitI8(v) => VI8(*v),
                LitI16(v) => VI16(*v),
                LitI32(v) => VI32(*v),
                LitI64(v) => VI64(*v),
                LitF32(v) => VF32(*v),
                LitF64(v) => VF64(*v),
                LitBool(v) => VBool(*v),
                LitUnit => VUnit,
                LitErr => VErr,
                LitTime(_) => todo!(),
            },
            Var(id) => {
                if let Some(val) = ctx.stack.lookup(*id) {
                    val.clone()
                } else {
                    VErr
                }
            }
            Let(id, e1, e2) => {
                let v1 = e1.eval(ctx);
                ctx.stack.insert(*id, v1);
                e2.eval(ctx)
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
                    (VBool(c), t, e) if c => t.eval(ctx),
                    (VBool(c), t, e) if !c => e.eval(ctx),
                    _ => VErr,
                }
            }
            Match(expr, clauses) => todo!(),
            Loop(cond, body) => todo!(),
            UnOp(kind, expr) => {
                let v = expr.eval(ctx);
                match kind {
                    Not => match v {
                        VBool(v) => VBool(!v),
                        _ => VErr,
                    },
                    Neg => match v {
                        VI8(v) => VI8(-v),
                        VI16(v) => VI16(-v),
                        VI32(v) => VI32(-v),
                        VI64(v) => VI64(-v),
                        VF32(v) => VF32(-v),
                        VF64(v) => VF64(-v),
                        _ => VErr,
                    },
                    Call(exprs) => match v {
                        VFun => todo!(),
                        _ => VErr,
                    },
                    Project(index) => match v {
                        VTuple(values) => todo!(),
                        _ => VErr,
                    },
                    Access(field) => todo!(),
                    UnOpErr | Cast(_) | Emit => VErr,
                }
            }
            BinOp(lhs, kind, rhs) => {
                let lhs = lhs.eval(ctx);
                let rhs = rhs.eval(ctx);
                match (lhs, kind, rhs) {
                    // Add
                    (VI8(l), Add, VI8(r)) => VI8(l + r),
                    (VI16(l), Add, VI16(r)) => VI16(l + r),
                    (VI32(l), Add, VI32(r)) => VI32(l + r),
                    (VI64(l), Add, VI64(r)) => VI64(l + r),
                    (VF32(l), Add, VF32(r)) => VF32(l + r),
                    (VF64(l), Add, VF64(r)) => VF64(l + r),
                    // Sub
                    (VI8(l), Sub, VI8(r)) => VI8(l - r),
                    (VI16(l), Sub, VI16(r)) => VI16(l - r),
                    (VI32(l), Sub, VI32(r)) => VI32(l - r),
                    (VI64(l), Sub, VI64(r)) => VI64(l - r),
                    (VF32(l), Sub, VF32(r)) => VF32(l - r),
                    (VF64(l), Sub, VF64(r)) => VF64(l - r),
                    // Mul
                    (VI8(l), Mul, VI8(r)) => VI8(l * r),
                    (VI16(l), Mul, VI16(r)) => VI16(l * r),
                    (VI32(l), Mul, VI32(r)) => VI32(l * r),
                    (VI64(l), Mul, VI64(r)) => VI64(l * r),
                    (VF32(l), Mul, VF32(r)) => VF32(l * r),
                    (VF64(l), Mul, VF64(r)) => VF64(l * r),
                    // Div
                    (VI8(l), Div, VI8(r)) => VI8(l / r),
                    (VI16(l), Div, VI16(r)) => VI16(l / r),
                    (VI32(l), Div, VI32(r)) => VI32(l / r),
                    (VI64(l), Div, VI64(r)) => VI64(l / r),
                    (VF32(l), Div, VF32(r)) => VF32(l / r),
                    (VF64(l), Div, VF64(r)) => VF64(l / r),
                    // Pow
                    (VI8(l), Pow, VI32(r)) => VI8(l.pow(r as u32)),
                    (VI16(l), Pow, VI32(r)) => VI16(l.pow(r as u32)),
                    (VI32(l), Pow, VI32(r)) => VI32(l.pow(r as u32)),
                    (VI64(l), Pow, VI32(r)) => VI64(l.pow(r as u32)),
                    (VF32(l), Pow, VI32(r)) => VF32(l.powi(r)),
                    (VF64(l), Pow, VI32(r)) => VF64(l.powi(r)),
                    (VF32(l), Pow, VF32(r)) => VF32(l.powf(r)),
                    (VF64(l), Pow, VF64(r)) => VF64(l.powf(r)),
                    // Equ
                    (VI8(l), Equ, VI8(r)) => VBool(l == r),
                    (VI16(l), Equ, VI16(r)) => VBool(l == r),
                    (VI32(l), Equ, VI32(r)) => VBool(l == r),
                    (VI64(l), Equ, VI64(r)) => VBool(l == r),
                    (VUnit, Equ, VUnit) => VBool(true),
                    // Neq
                    (VI8(l), Neq, VI8(r)) => VBool(l == r),
                    (VI16(l), Neq, VI16(r)) => VBool(l == r),
                    (VI32(l), Neq, VI32(r)) => VBool(l == r),
                    (VI64(l), Neq, VI64(r)) => VBool(l == r),
                    (VUnit, Neq, VUnit) => VBool(false),
                    // Or
                    (VBool(l), Or, VBool(r)) => VBool(l || r),
                    // And
                    (VBool(l), And, VBool(r)) => VBool(l && r),
                    // Gt
                    (VI8(l), Gt, VI8(r)) => VBool(l > r),
                    (VI16(l), Gt, VI16(r)) => VBool(l > r),
                    (VI32(l), Gt, VI32(r)) => VBool(l > r),
                    (VI64(l), Gt, VI64(r)) => VBool(l > r),
                    (VF32(l), Gt, VF32(r)) => VBool(l > r),
                    (VF64(l), Gt, VF64(r)) => VBool(l > r),
                    // Lt
                    (VI8(l), Lt, VI8(r)) => VBool(l < r),
                    (VI16(l), Lt, VI16(r)) => VBool(l < r),
                    (VI32(l), Lt, VI32(r)) => VBool(l < r),
                    (VI64(l), Lt, VI64(r)) => VBool(l < r),
                    (VF32(l), Lt, VF32(r)) => VBool(l < r),
                    (VF64(l), Lt, VF64(r)) => VBool(l < r),
                    // Geq
                    (VI8(l), Geq, VI8(r)) => VBool(l >= r),
                    (VI16(l), Geq, VI16(r)) => VBool(l >= r),
                    (VI32(l), Geq, VI32(r)) => VBool(l >= r),
                    (VI64(l), Geq, VI64(r)) => VBool(l >= r),
                    (VF32(l), Geq, VF32(r)) => VBool(l >= r),
                    (VF64(l), Geq, VF64(r)) => VBool(l >= r),
                    // Leq
                    (VI8(l), Leq, VI8(r)) => VBool(l <= r),
                    (VI16(l), Leq, VI16(r)) => VBool(l <= r),
                    (VI32(l), Leq, VI32(r)) => VBool(l <= r),
                    (VI64(l), Leq, VI64(r)) => VBool(l <= r),
                    (VF32(l), Leq, VF32(r)) => VBool(l <= r),
                    (VF64(l), Leq, VF64(r)) => VBool(l <= r),
                    // Staged
                    (l, Pipe, r) => VPipe(l.into(), r.into()),
                    (_, BinOpErr, _) => VErr,
                    _ => unreachable!(),
                }
            }
            ExprErr => VErr,
        }
    }
}
