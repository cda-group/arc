pub(crate) mod eval;

use crate::compiler::dfg::lower::hir::eval::control::Control;
use crate::compiler::dfg::lower::hir::eval::control::ControlKind::*;
use crate::compiler::dfg::lower::hir::eval::stack::Stack;
use crate::compiler::dfg::lower::hir::eval::Context;
use crate::compiler::dfg::DFG;
use crate::compiler::hir::Path;
use crate::compiler::hir::{
    BinOp, BinOpKind, BinOpKind::*, Expr, ExprKind, ItemKind, LitKind, ScalarKind, TypeKind, UnOp,
    UnOpKind, UnOpKind::*, HIR,
};
use crate::compiler::info::diags::DiagInterner;
use crate::compiler::info::diags::Error;
use crate::compiler::info::diags::Panic;
use crate::compiler::info::types::TypeId;
use crate::compiler::info::Info;

use tracing::instrument;

/// Constructs a call-expression
fn call(path: Path, args: Vec<Expr>, ftv: TypeId, rtv: TypeId) -> Expr {
    Expr::syn(
        ExprKind::Call(Expr::syn(ExprKind::Item(path), ftv).into(), args),
        rtv,
    )
}

/// Constructs an call-expression for calling the main-function
pub(crate) fn main_call(hir: &HIR, info: &mut Info) -> Option<Expr> {
    // Find the main function
    let main = info.names.intern("main").into();
    let main = info.paths.intern_child(info.paths.root, main);
    let main_fun = hir.defs.iter().find_map(|(path, item)| match &item.kind {
        ItemKind::Fun(item) if item.path.id == main => Some((path, item)),
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
