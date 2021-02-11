#![allow(clippy::toplevel_ref_arg)]
#![allow(unused)]

use crate::compiler::hir::{Expr, ExprKind, Fun, Task, UnOpKind, HIR};
use crate::compiler::info::paths::PathId;
use crate::compiler::shared::Map;

/// Macro for generating pre- and post-order visitors of expressions.
macro_rules! for_each_expr {
    { wrapper: $wrapper:ident, name: $name:ident, pre: $pre:literal, post: $post:literal, ($($ref:tt)*), $iter:ident } => {
        impl Expr {
            pub(crate) fn $wrapper<F: FnMut($($ref)* Self)>($($ref)* self, mut f: F) {
                self.$name(&mut f);
            }
            fn $name<F: FnMut($($ref)* Self)>($($ref)* self, f: &mut F) {
                if $pre {
                    f(self);
                }
                match $($ref)* self.kind {
                    ExprKind::If(c, t, e) => {
                        c.$name(f);
                        t.$name(f);
                        e.$name(f);
                    }
                    ExprKind::Let(_, e1, e2) => {
                        e1.$name(f);
                        e2.$name(f);
                    },
                    ExprKind::Array(ps) => ps.$iter().for_each(|p| p.$name(f)),
                    ExprKind::Tuple(ps) => ps.$iter().for_each(|p| p.$name(f)),
                    ExprKind::Struct(fs) => fs.$iter().for_each(|(_, v)| v.$name(f)),
                    ExprKind::Lit(_) => {}
                    ExprKind::Var(_) => {}
                    ExprKind::Item(_) => {}
                    ExprKind::Emit(_) => {}
                    ExprKind::BinOp(l, _, r) => {
                        l.$name(f);
                        r.$name(f);
                    }
                    ExprKind::Call(e, es) => {
                        e.$name(f);
                        es.$iter().for_each(|p| p.$name(f))
                    },
                    ExprKind::Access(e, _) => e.$name(f),
                    ExprKind::Project(e, _) => e.$name(f),
                    ExprKind::UnOp(op, e) => e.$name(f),
                    ExprKind::Loop(e) => e.$name(f),
                    ExprKind::Log(e) => e.$name(f),
                    ExprKind::Break => {}
                    ExprKind::Return(e) => e.$name(f),
                    ExprKind::Err => {}
                    _ => todo!()
                }
                if $post {
                    f(self);
                }
            }
        }
    }
}

for_each_expr! { wrapper: for_each_expr_mut,           name: a, pre: true,  post: false, (&mut), iter_mut }
for_each_expr! { wrapper: for_each_expr_postorder_mut, name: b, pre: false, post: true,  (&mut), iter_mut }
for_each_expr! { wrapper: for_each_expr,               name: c, pre: true,  post: false, (&),    iter }
for_each_expr! { wrapper: for_each_expr_postorder,     name: d, pre: false, post: true,  (&),    iter }
