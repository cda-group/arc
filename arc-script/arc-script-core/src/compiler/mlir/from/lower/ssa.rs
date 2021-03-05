use super::Context;
use crate::compiler::hir::Name;
use crate::compiler::hir::{
    Expr, ExprKind, ItemKind, Param, ParamKind, ScalarKind, TypeKind, UnOp, UnOpKind, HIR,
};

use crate::compiler::mlir::{self, Block, ConstKind, Op, OpKind, Region, Var};
use arc_script_core_shared::Lower;
use arc_script_core_shared::Map;
use arc_script_core_shared::VecMap;

type Env = Map<Name, Var>;

trait SSA<T> {
    fn ssa(&self, ctx: &mut Context<'_>, env: &mut Env, ops: &mut Vec<Op>) -> T;
}

impl Lower<Region, Context<'_>> for Expr {
    fn lower(&self, ctx: &mut Context<'_>) -> Region {
        let mut env = Map::default();
        self.to_region(ctx, &mut env, OpKind::Return)
    }
}

impl Expr {
    fn to_region(
        &self,
        ctx: &mut Context<'_>,
        env: &mut Env,
        term: impl Fn(Var) -> OpKind,
    ) -> Region {
        let mut ops = Vec::new();
        let x = self.ssa(ctx, env, &mut ops);
        ops.push(Op::new(None, term(x), self.loc));
        Region::new(vec![Block::new(ops)])
    }
}

/// let (a,(b,c)) = (1,(2,3)) in
/// e
///
/// let x0 = (1,2) in
/// let a  = x0.0 in
/// let x1 = x0.1 in
/// let b  = x1.0 in
/// let c  = x1.1 in
/// e

impl SSA<Var> for Expr {
    /// Turns an expression into an SSA operation.
    fn ssa(&self, ctx: &mut Context<'_>, env: &mut Env, ops: &mut Vec<Op>) -> Var {
        let kind = match &self.kind {
            ExprKind::Let(x, e0, e1) => {
                if let ParamKind::Var(x) = x.kind {
                    let x0 = e0.ssa(ctx, env, ops);
                    env.insert(x, x0);
                }
                return e1.ssa(ctx, env, ops);
            }
            ExprKind::Var(mut x, _) => {
                while let Some(next) = env.get(&x) {
                    x = next.name;
                }
                return Var::new(x, self.tv);
            }
            ExprKind::Item(x) => match ctx.hir.defs.get(x).unwrap().kind {
                ItemKind::Fun(_) => OpKind::Const(ConstKind::Fun(*x)),
                _ => unreachable!(),
            },
            ExprKind::Call(e, es) => {
                if let ExprKind::Item(p) = &e.kind {
                    let xs = es.ssa(ctx, env, ops);
                    mlir::OpKind::Call(*p, xs)
                } else {
                    let x = e.ssa(ctx, env, ops);
                    let xs = es.ssa(ctx, env, ops);
                    mlir::OpKind::CallIndirect(x, xs)
                }
            }
            ExprKind::Select(_e, _es) => todo!(),
            ExprKind::Lit(l) => OpKind::Const(l.lower(ctx)),
            ExprKind::BinOp(e0, op, e1) => {
                let x0 = e0.ssa(ctx, env, ops);
                let op = op.lower(ctx);
                let x1 = e1.ssa(ctx, env, ops);
                OpKind::BinOp(e0.tv, x0, op, x1)
            }
            ExprKind::UnOp(op, e0) => {
                let op = op.lower(ctx);
                let x0 = e0.ssa(ctx, env, ops);
                OpKind::UnOp(op, x0)
            }
            ExprKind::Access(e0, x) => {
                let x0 = e0.ssa(ctx, env, ops);
                OpKind::Access(x0, *x)
            }
            ExprKind::Project(e0, i) => {
                let x0 = e0.ssa(ctx, env, ops);
                OpKind::Project(x0, i.id)
            }
            ExprKind::If(e0, e1, e2) => {
                let x0 = e0.ssa(ctx, env, ops);
                let r1 = e1.to_region(ctx, env, OpKind::Res);
                let r2 = e2.to_region(ctx, env, OpKind::Res);
                OpKind::If(x0, r1, r2)
            }
            ExprKind::Array(es) => {
                let xs = es.ssa(ctx, env, ops);
                OpKind::Array(xs)
            }
            ExprKind::Struct(fs) => {
                let fs = fs.ssa(ctx, env, ops);
                OpKind::Struct(fs)
            }
            ExprKind::Enwrap(x0, e1) => {
                let x1 = e1.ssa(ctx, env, ops);
                OpKind::Enwrap(*x0, x1)
            }
            ExprKind::Unwrap(x0, e1) => {
                let x1 = e1.ssa(ctx, env, ops);
                OpKind::Unwrap(*x0, x1)
            }
            ExprKind::Is(x0, e1) => {
                let x1 = e1.ssa(ctx, env, ops);
                OpKind::Is(*x0, x1)
            }
            ExprKind::Tuple(es) => {
                let xs = es.ssa(ctx, env, ops);
                OpKind::Tuple(xs)
            }
            ExprKind::Emit(e) => {
                let x = e.ssa(ctx, env, ops);
                OpKind::Emit(x)
            }
            ExprKind::Log(e) => {
                let x = e.ssa(ctx, env, ops);
                OpKind::Log(x)
            }
            ExprKind::Loop(e) => {
                let x = e.ssa(ctx, env, ops);
                OpKind::Loop(x)
            }
            ExprKind::Break => OpKind::Break,
            ExprKind::Return(_e) => todo!(),
            ExprKind::Todo => todo!(),
            ExprKind::Err => unreachable!(),
        };
        let x = Var::new(ctx.info.names.fresh(), self.tv);
        let ty = ctx.info.types.resolve(self.tv);
        if !matches!(&ty.kind, TypeKind::Scalar(ScalarKind::Unit)) {
            ops.push(Op::new(Some(x), kind, self.loc));
        }
        x
    }
}

impl SSA<Vec<Var>> for Vec<Expr> {
    fn ssa(&self, ctx: &mut Context<'_>, env: &mut Env, ops: &mut Vec<Op>) -> Vec<Var> {
        self.iter().map(|e| e.ssa(ctx, env, ops)).collect()
    }
}

impl SSA<VecMap<Name, Var>> for VecMap<Name, Expr> {
    fn ssa(&self, ctx: &mut Context<'_>, env: &mut Env, ops: &mut Vec<Op>) -> VecMap<Name, Var> {
        self.into_iter()
            .map(|(f, e)| {
                let x = e.ssa(ctx, env, ops);
                (*f, x)
            })
            .collect()
    }
}
