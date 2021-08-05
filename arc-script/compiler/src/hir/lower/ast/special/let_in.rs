//! Lowers a let-expression `let p = e0 in e1`
//!
//! For example:
//! ```txt
//! let (a, (_, c)) = (1, (2, 3));
//! a
//! ```
//! should after SSA become something like:
//! ```txt
//! let x0 = 1;
//! let x1 = 2;
//! let x2 = 3;
//! let x4 = (x1, x2);
//! let x5 = (x0, x4);
//! let (a, (_, c)) = x5;
//! a
//! ```
//! then you need to somehow do
//! ```txt
//! // construct
//! let x0 = 1;
//! let x1 = 2;
//! let x2 = 3;
//! let x4 = (x1, x2);
//! let x5 = (x0, x4);
//! // deconstruct
//! let a = x5.0;
//! let x7 = x5.1;
//! let x8 = x7.1;
//! a
//! ```
//!

use super::Context;
use crate::ast;
use crate::hir::Expr;
use arc_script_compiler_shared::Lower;

pub(crate) fn lower(p: &ast::Param, e0: &ast::Expr, e1: &ast::Expr, ctx: &mut Context<'_>) -> Expr {
    todo!()
//     let e0 = e0.lower(ctx);
//     let clauses = super::pattern::lower_param_expr(p, e0, ctx);
//     let e1 = e1.lower(ctx);
//     super::pattern::fold_cases(e1, None, clauses)
}
