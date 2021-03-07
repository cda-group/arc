use crate::compiler::dfg::lower::hir::eval::control::Control;
use crate::compiler::dfg::lower::hir::eval::control::ControlKind::*;
use crate::compiler::dfg::lower::hir::eval::stack::Stack;
use crate::compiler::dfg::lower::hir::eval::Context;
use crate::compiler::dfg::lower::hir::main_call;
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

/// Constructs a dataflow graph by evaluating the HIR. The HIR is not modified
/// in the process.
impl DFG {
    #[instrument(name = "HIR & Info => DFG", level = "debug", skip(hir, info))]
    pub(crate) fn from(hir: &HIR, info: &mut Info) -> Result<Self, DiagInterner> {
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
