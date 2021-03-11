/// Module for converting the [`crate::repr::hir::HIR`] into SSA form.
pub(crate) mod ssa;

use crate::compiler::hir;
use crate::compiler::hir::HIR;
use crate::compiler::info::Info;
use crate::compiler::mlir;

use arc_script_core_shared::Lower;
use arc_script_core_shared::New;

#[derive(New)]
pub(crate) struct Context<'i> {
    hir: &'i HIR,
    info: &'i mut Info,
}

/// TODO: For now, only lower functions.
impl Lower<Option<mlir::Item>, Context<'_>> for hir::Item {
    fn lower(&self, ctx: &mut Context<'_>) -> Option<mlir::Item> {
        #[rustfmt::skip]
        let kind = match &self.kind {
            hir::ItemKind::Fun(item)     => mlir::ItemKind::Fun(item.lower(ctx)),
            hir::ItemKind::Alias(_item)   => todo!(),
            hir::ItemKind::Enum(_item)    => None?,
            hir::ItemKind::Task(_item)    => todo!(),
            hir::ItemKind::Extern(_item)  => todo!(),
            hir::ItemKind::Variant(_item) => None?,
        };
        Some(mlir::Item::new(kind, self.loc))
    }
}

impl Lower<mlir::Fun, Context<'_>> for hir::Fun {
    fn lower(&self, ctx: &mut Context<'_>) -> mlir::Fun {
        mlir::Fun::new(
            self.path,
            self.params.iter().map(|p| p.lower(ctx)).collect::<Vec<_>>(),
            self.body.lower(ctx),
            self.body.tv,
        )
    }
}

impl Lower<mlir::Var, Context<'_>> for hir::Param {
    fn lower(&self, _ctx: &mut Context<'_>) -> mlir::Var {
        match &self.kind {
            hir::ParamKind::Ignore => todo!(),
            hir::ParamKind::Var(x) => mlir::Var::new(*x, self.tv),
            hir::ParamKind::Err => unreachable!(),
        }
    }
}

impl Lower<mlir::UnOp, Context<'_>> for hir::UnOp {
    fn lower(&self, _ctx: &mut Context<'_>) -> mlir::UnOp {
        #[rustfmt::skip]
        let kind = match &self.kind {
            hir::UnOpKind::Add   => unreachable!(),
            hir::UnOpKind::Boxed => todo!(),
            hir::UnOpKind::Not   => mlir::UnOpKind::Not,
            hir::UnOpKind::Neg   => mlir::UnOpKind::Neg,
            hir::UnOpKind::Del   => unreachable!(),
            hir::UnOpKind::Err   => unreachable!(),
        };
        mlir::UnOp::new(kind)
    }
}

impl Lower<mlir::BinOp, Context<'_>> for hir::BinOp {
    fn lower(&self, _ctx: &mut Context<'_>) -> mlir::BinOp {
        #[rustfmt::skip]
        let kind = match self.kind {
            hir::BinOpKind::Add   => mlir::BinOpKind::Add,
            hir::BinOpKind::Sub   => mlir::BinOpKind::Sub,
            hir::BinOpKind::Mul   => mlir::BinOpKind::Mul,
            hir::BinOpKind::Div   => mlir::BinOpKind::Div,
            hir::BinOpKind::Mod   => mlir::BinOpKind::Mod,
            hir::BinOpKind::Pow   => mlir::BinOpKind::Pow,
            hir::BinOpKind::Equ   => mlir::BinOpKind::Equ,
            hir::BinOpKind::Neq   => mlir::BinOpKind::Neq,
            hir::BinOpKind::Or    => mlir::BinOpKind::Or,
            hir::BinOpKind::And   => mlir::BinOpKind::And,
            hir::BinOpKind::Xor   => mlir::BinOpKind::Xor,
            hir::BinOpKind::Band  => mlir::BinOpKind::Band,
            hir::BinOpKind::Bor   => mlir::BinOpKind::Bor,
            hir::BinOpKind::Bxor  => mlir::BinOpKind::Bxor,
            hir::BinOpKind::Gt    => mlir::BinOpKind::Gt,
            hir::BinOpKind::Lt    => mlir::BinOpKind::Lt,
            hir::BinOpKind::Geq   => mlir::BinOpKind::Geq,
            hir::BinOpKind::Leq   => mlir::BinOpKind::Leq,
            hir::BinOpKind::Pipe  => unreachable!(),
            hir::BinOpKind::Mut   => mlir::BinOpKind::Mut,
            hir::BinOpKind::By    => todo!(),
            hir::BinOpKind::Seq   => unreachable!(),
            hir::BinOpKind::In    => todo!(),
            hir::BinOpKind::NotIn => todo!(),
            hir::BinOpKind::Err   => unreachable!(),
        };
        mlir::BinOp::new(kind)
    }
}

impl Lower<mlir::ConstKind, Context<'_>> for hir::LitKind {
    #[rustfmt::skip]
    fn lower(&self, _ctx: &mut Context<'_>) -> mlir::ConstKind {
        match self {
            Self::I8(v)   => mlir::ConstKind::I8(*v),
            Self::I16(v)  => mlir::ConstKind::I16(*v),
            Self::I32(v)  => mlir::ConstKind::I32(*v),
            Self::I64(v)  => mlir::ConstKind::I64(*v),
            Self::U8(v)   => mlir::ConstKind::U8(*v),
            Self::U16(v)  => mlir::ConstKind::U16(*v),
            Self::U32(v)  => mlir::ConstKind::U32(*v),
            Self::U64(v)  => mlir::ConstKind::U64(*v),
            Self::Bf16(v) => mlir::ConstKind::Bf16(*v),
            Self::F16(v)  => mlir::ConstKind::F16(*v),
            Self::F32(v)  => mlir::ConstKind::F32(*v),
            Self::F64(v)  => mlir::ConstKind::F64(*v),
            Self::Bool(v) => mlir::ConstKind::Bool(*v),
            Self::Char(v) => mlir::ConstKind::Char(*v),
            Self::Str(_)  => todo!(),
            Self::DateTime(_) => todo!(),
            Self::Duration(_) => todo!(),
            Self::Unit    => mlir::ConstKind::Unit,
            Self::Err     => unreachable!(),
        }
    }
}
