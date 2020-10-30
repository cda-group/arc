use crate::{prelude::*, error::*, info::Info};
use z3::{ast::Int, Config, Context, SatResult, Solver};

impl<'i> Script<'i> {
    pub fn infer_shape(&mut self) {
        let Script {
            ast,
            info:
                Info {
                    errors,
                    table,
                    typer,
                    ..
                },
            ..
        } = self;
        let config = Config::new();
        let ref ctx = Context::new(&config);
        let mut roots = Vec::<Int>::new();
        ast.for_each_dim(|expr| roots.push(expr.constrain(ctx, errors)), table);
        let solver = Solver::new(ctx);
        match solver.check() {
            SatResult::Sat => {
                let model = &mut solver.get_model();
                let mut iter = roots.into_iter();
                ast.for_each_dim(
                    |expr| {
                        let root = iter.next().unwrap();
                        (*expr).kind = model
                            .eval(&root)
                            .map(|v| DimVal(v.as_i64().unwrap() as i32))
                            .unwrap_or(DimErr);
                    },
                    table,
                );
            }
            SatResult::Unsat => errors.push(CompilerError::ShapeUnsat),
            SatResult::Unknown => errors.push(CompilerError::ShapeUnknown),
        }
    }
}

impl Dim {
    fn constrain_shape<'i>(&self, ctx: &'i Context, errors: &mut Vec<CompilerError>) -> Int<'i> {
        match &self.kind {
            DimOp(l, op, r) => {
                let l = l.constrain_shape(ctx, errors);
                let r = r.constrain_shape(ctx, errors);
                match op {
                    DimAdd => Int::add(ctx, &[&l, &r]),
                    DimSub => Int::sub(ctx, &[&l, &r]),
                    DimMul => Int::mul(ctx, &[&l, &r]),
                    DimDiv => l.div(&r),
                }
            }
            DimVar(x) => Int::new_const(ctx, *x as u32),
            DimVal(v) => Int::from_i64(ctx, *v as i64),
            DimErr => Int::new_const(ctx, "unknown"),
        }
    }
}
