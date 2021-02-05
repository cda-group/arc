use crate::compiler::dfg::DFG;
use crate::compiler::hir;
use crate::compiler::hir::HIR;
use crate::compiler::info::Info;
use crate::compiler::mlir;
use crate::compiler::mlir::MLIR;
use crate::compiler::shared::{Lower, Map, New};

use super::Context;

/// TODO: For now, only lower functions.
impl Lower<Option<mlir::Item>, Context<'_>> for hir::Item {
    fn lower(&self, ctx: &mut Context<'_>) -> Option<mlir::Item> {
        #[rustfmt::skip]
        let kind = match &self.kind {
            hir::ItemKind::Fun(item)     => mlir::ItemKind::Fun(item.lower(ctx)),
            hir::ItemKind::Alias(item)   => todo!(),
            hir::ItemKind::Enum(item)    => None?,
            hir::ItemKind::Task(item)    => todo!(),
            hir::ItemKind::State(item)   => todo!(),
            hir::ItemKind::Extern(item)  => todo!(),
            hir::ItemKind::Variant(item) => None?,
        };
        Some(mlir::Item::new(kind, self.loc))
    }
}

impl Lower<mlir::Fun, Context<'_>> for hir::Fun {
    fn lower(&self, ctx: &mut Context<'_>) -> mlir::Fun {
        mlir::Fun::new(
            self.name,
            self.params.iter().map(|p| p.lower(ctx)).collect::<Vec<_>>(),
            self.body.lower(ctx),
            self.body.tv,
        )
    }
}

impl Lower<mlir::Var, Context<'_>> for hir::Param {
    fn lower(&self, ctx: &mut Context<'_>) -> mlir::Var {
        match &self.kind {
            hir::ParamKind::Ignore => todo!(),
            hir::ParamKind::Var(x) => mlir::Var::new(*x, self.tv),
            hir::ParamKind::Err => unreachable!(),
        }
    }
}

impl Lower<mlir::UnOp, Context<'_>> for hir::UnOp {
    fn lower(&self, ctx: &mut Context<'_>) -> mlir::UnOp {
        #[rustfmt::skip]
        let kind = match &self.kind {
            hir::UnOpKind::Not => mlir::UnOpKind::Not,
            hir::UnOpKind::Neg => mlir::UnOpKind::Neg,
            hir::UnOpKind::Err => unreachable!(),
        };
        mlir::UnOp::new(kind)
    }
}

impl Lower<mlir::BinOp, Context<'_>> for hir::BinOp {
    fn lower(&self, ctx: &mut Context<'_>) -> mlir::BinOp {
        #[rustfmt::skip]
        let kind = match self.kind {
            hir::BinOpKind::Add  => mlir::BinOpKind::Add,
            hir::BinOpKind::Sub  => mlir::BinOpKind::Sub,
            hir::BinOpKind::Mul  => mlir::BinOpKind::Mul,
            hir::BinOpKind::Div  => mlir::BinOpKind::Div,
            hir::BinOpKind::Mod  => mlir::BinOpKind::Mod,
            hir::BinOpKind::Pow  => mlir::BinOpKind::Pow,
            hir::BinOpKind::Equ  => mlir::BinOpKind::Equ,
            hir::BinOpKind::Neq  => mlir::BinOpKind::Neq,
            hir::BinOpKind::Or   => mlir::BinOpKind::Or,
            hir::BinOpKind::And  => mlir::BinOpKind::And,
            hir::BinOpKind::Xor  => mlir::BinOpKind::Xor,
            hir::BinOpKind::Band => mlir::BinOpKind::Band,
            hir::BinOpKind::Bor  => mlir::BinOpKind::Bor,
            hir::BinOpKind::Bxor => mlir::BinOpKind::Bxor,
            hir::BinOpKind::Gt   => mlir::BinOpKind::Gt,
            hir::BinOpKind::Lt   => mlir::BinOpKind::Lt,
            hir::BinOpKind::Geq  => mlir::BinOpKind::Geq,
            hir::BinOpKind::Leq  => mlir::BinOpKind::Leq,
            hir::BinOpKind::Pipe => unreachable!(),
            hir::BinOpKind::Mut  => mlir::BinOpKind::Mut,
            hir::BinOpKind::Seq  => unreachable!(),
            hir::BinOpKind::Err  => unreachable!(),
        };
        mlir::BinOp::new(kind)
    }
}

impl Lower<mlir::ConstKind, Context<'_>> for hir::LitKind {
    #[rustfmt::skip]
    fn lower(&self, ctx: &mut Context<'_>) -> mlir::ConstKind {
        match self {
            hir::LitKind::I8(v)   => mlir::ConstKind::I8(*v),
            hir::LitKind::I16(v)  => mlir::ConstKind::I16(*v),
            hir::LitKind::I32(v)  => mlir::ConstKind::I32(*v),
            hir::LitKind::I64(v)  => mlir::ConstKind::I64(*v),
            hir::LitKind::U8(v)   => mlir::ConstKind::U8(*v),
            hir::LitKind::U16(v)  => mlir::ConstKind::U16(*v),
            hir::LitKind::U32(v)  => mlir::ConstKind::U32(*v),
            hir::LitKind::U64(v)  => mlir::ConstKind::U64(*v),
            hir::LitKind::Bf16(v) => mlir::ConstKind::Bf16(*v),
            hir::LitKind::F16(v)  => mlir::ConstKind::F16(*v),
            hir::LitKind::F32(v)  => mlir::ConstKind::F32(*v),
            hir::LitKind::F64(v)  => mlir::ConstKind::F64(*v),
            hir::LitKind::Bool(v) => mlir::ConstKind::Bool(*v),
            hir::LitKind::Char(v) => mlir::ConstKind::Char(*v),
            hir::LitKind::Str(v)  => todo!(),
            hir::LitKind::Time(v) => mlir::ConstKind::Time(*v),
            hir::LitKind::Unit    => mlir::ConstKind::Unit,
            hir::LitKind::Err     => unreachable!(),
        }
    }
}
