/// Module for evaluating the [`crate::repr::hir::HIR`] into a [`crate::repr::dfg::DFG`]
/// using Big-Step operational semantics.
pub(crate) mod eval;

use crate::compiler::dfg::from::eval::control::Control;
use crate::compiler::dfg::from::eval::control::ControlKind::*;
use crate::compiler::dfg::from::eval::stack::Stack;
use crate::compiler::dfg::from::eval::value::ValueKind;
use crate::compiler::dfg::from::eval::Context;
use crate::compiler::dfg::DFG;
use crate::compiler::hir::Path;
use crate::compiler::hir::{
    BinOp, BinOpKind, BinOpKind::*, Expr, ExprKind, ItemKind, LitKind, ScalarKind, TypeKind, UnOp,
    UnOpKind, UnOpKind::*, HIR,
};
use crate::compiler::info::diags::DiagInterner;
use crate::compiler::info::diags::Error;
use crate::compiler::info::diags::Panic;
use crate::compiler::info::names::NameId;
use crate::compiler::info::paths::PathId;
use crate::compiler::info::types::TypeId;
use crate::compiler::info::Info;
use crate::compiler::shared::VecMap;

fn call(path: Path, args: Vec<Expr>, ftv: TypeId, rtv: TypeId) -> Expr {
    Expr::syn(
        ExprKind::Call(Expr::syn(ExprKind::Item(path), ftv).into(), args),
        rtv,
    )
}

/// Constructs an expression for calling the main-function
fn main_call(hir: &HIR, info: &mut Info) -> Option<Expr> {
    let main = info.names.intern("main").into();
    let main_fun = hir.defs.iter().find_map(|(path, item)| match &item.kind {
        ItemKind::Fun(item) if item.name == main => Some((path, item)),
        _ => None,
    });
    if let Some((path, fun)) = main_fun {
        let ty = info.types.resolve(fun.tv);
        if let TypeKind::Fun(tvs, tv) = ty.kind {
            let ty = info.types.resolve(tv);
            if matches!(ty.kind, TypeKind::Scalar(ScalarKind::Unit) if tvs.is_empty()) {
                return Some(call(*path, vec![], fun.tv, tv));
            } else {
                info.diags.intern(Error::MainWrongSign)
            }
        } else {
            info.diags.intern(Error::MainWrongSign);
        }
    } else {
        info.diags.intern(Error::MainNotFound);
    }
    None
}

/// Constructs a dataflow graph by evaluating the HIR. The HIR is not modified
/// in the process.
impl DFG {
    pub(crate) fn from<'i>(hir: &'i HIR, info: &'i mut Info) -> Result<Self, DiagInterner> {
        let mut dfg = Self::default();
        if let Some(expr) = main_call(hir, info) {
            let mut stack = Stack::default();
            let mut ctx = Context::new(&mut stack, &mut dfg, hir, info);
            if let Control(Panic(loc)) = expr.eval(&mut ctx) {
                let mut diags = DiagInterner::default();
                diags.intern(Panic::Unwind {
                    loc,
                    trace: stack.iter().map(|frame| frame.path).collect(),
                });
                return Result::Err(diags);
            }
        }
        Ok(dfg)
    }
}
