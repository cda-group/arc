use crate::{ast::*, error::*};
use z3::{ast::Int, Config, Context, SatResult, Solver};

impl Script {
    pub fn infer_shape(&mut self) {
        let Script {
            ref mut body,
            ref mut errors,
            ..
        } = self;
        let config = Config::new();
        let ref context = Context::new(&config);
        let mut roots: Vec<Int> = Vec::new();
        body.for_each_dim_expr(|expr, _| roots.push(expr.visit(context, errors)));
        let solver = Solver::new(context);
        match solver.check() {
            SatResult::Sat => {
                let ref mut model = solver.get_model();
                let mut iter = roots.into_iter();
                body.for_each_dim_expr(|expr, _| {
                    let root = iter.next().unwrap();
                    (*expr).kind = match model.eval(&root) {
                        Some(val) => ExprKind::Lit(Lit::I64(val.as_i64().unwrap())),
                        None => ExprKind::Error,
                    }
                });
            }
            SatResult::Unsat => errors.push(CompilerError::ShapeUnsat),
            SatResult::Unknown => errors.push(CompilerError::ShapeUnknown),
        }
    }
}

impl Expr {
    pub fn visit<'ctx>(
        &mut self,
        context: &'ctx Context,
        errors: &mut Vec<CompilerError>,
    ) -> Int<'ctx> {
        match &mut self.kind {
            ExprKind::BinOp(l, op, r) => {
                let l = l.visit(context, errors);
                let r = r.visit(context, errors);
                match op {
                    BinOp::Add => l.add(&[&r]),
                    BinOp::Sub => l.sub(&[&r]),
                    BinOp::Mul => l.mul(&[&r]),
                    BinOp::Div => l.div(&r),
                    _ => Int::new_const(context, "unknown"),
                }
            }
            ExprKind::Var(id) => Int::new_const(context, id.uid.unwrap().0.to_string()),
            ExprKind::Lit(Lit::I32(val)) => Int::from_i64(context, *val as i64),
            _ => {
                errors.push(CompilerError::DisallowedDimExpr { span: self.span });
                Int::new_const(context, "unknown")
            }
        }
    }
}
