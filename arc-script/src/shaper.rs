use crate::{ast::*, error::*, info::Info};
use z3::{ast::Int, Config, Context, SatResult, Solver};
use BinOpKind::*;
use ExprKind::*;
use LitKind::*;

impl<'i> Script<'i> {
    pub fn infer_shape(&mut self) {
        let Script {
            ref mut ast,
            info:
                Info {
                    ref mut errors,
                    ref mut table,
                    ..
                },
            ..
        } = self;
        let config = Config::new();
        let ref context = Context::new(&config);
        let mut roots: Vec<Int> = Vec::new();
        ast.for_each_dim_expr(|expr| roots.push(expr.visit(context, errors)), table);
        let solver = Solver::new(context);
        match solver.check() {
            SatResult::Sat => {
                let ref mut model = solver.get_model();
                let mut iter = roots.into_iter();
                ast.for_each_dim_expr(
                    |expr| {
                        let root = iter.next().unwrap();
                        (*expr).kind = match model.eval(&root) {
                            Some(val) => Lit(LitI64(val.as_i64().unwrap())),
                            None => ExprErr,
                        }
                    },
                    table,
                );
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
            BinOp(l, op, r) => {
                let l = l.visit(context, errors);
                let r = r.visit(context, errors);
                match op {
                    Add => Int::add(context, &[&l, &r]),
                    Sub => Int::sub(context, &[&l, &r]),
                    Mul => Int::mul(context, &[&l, &r]),
                    Div => l.div(&r),
                    _ => Int::new_const(context, "unknown"),
                }
            }
            Var(id) => Int::new_const(context, id.0.to_string()),
            Lit(LitI32(val)) => Int::from_i64(context, *val as i64),
            _ => {
                errors.push(CompilerError::DisallowedDimExpr { span: self.span });
                Int::new_const(context, "unknown")
            }
        }
    }
}
