use crate::ast;
use crate::hir;
use crate::info::files::Loc;
use crate::info::paths::PathId;
use crate::info::Info;

use arc_script_compiler_shared::itertools::Itertools;
use arc_script_compiler_shared::Shrinkwrap;
use arc_script_compiler_shared::VecMap;

use super::Context;

use std::collections::VecDeque;

// /// Replace all free-type variables with quantifiers
// fn generalise(&self, ctx: &mut Context<'_>) {
//     // Assume all top-level functions are already generalised
// }

/// Generalise the item
/// 0. Input: Item
/// 1. Unify all types
/// 2. Substitute free type variables after unification with quantifiers
/// 3. Output: Scheme(Type, Quantifiers)

/// Specialise the item
/// 0. Input: Scheme(Type, Quantifiers)
/// 1. Clone the item
/// 2. Substitute quantifiers in type with fresh type variables
/// 3. Output: Type

/// Clones a syntax node.
pub(crate) trait Specialise {
    fn specialise(&self, ctx: &mut Context<'_>) -> Self;
}

/// Macro for implementing the `Declare` trait.
macro_rules! specialise {
    {
        [$node:ident, $ctx:ident]
        $($ty:path => $expr:expr ,)*
    } => {
        $(
            impl Specialise for $ty {
                fn specialise(&self, $ctx: &mut Context<'_>) -> Self {
                    let $node = self;
                    $expr
                }
            }
        )*
    };
}

specialise! {
    [node, ctx]

    hir::Item => hir::Item {
        kind: node.kind.specialise(ctx),
        ..node.clone()
    },
    hir::ItemKind => match node {
        hir::ItemKind::TypeAlias(i)  => hir::ItemKind::TypeAlias(i.specialise(ctx)),
        hir::ItemKind::Enum(i)       => hir::ItemKind::Enum(i.specialise(ctx)),
        hir::ItemKind::Fun(i)        => hir::ItemKind::Fun(i.specialise(ctx)),
        hir::ItemKind::Task(i)       => hir::ItemKind::Task(i.specialise(ctx)),
        hir::ItemKind::ExternFun(i)  => hir::ItemKind::ExternFun(i.specialise(ctx)),
        hir::ItemKind::ExternType(i) => hir::ItemKind::ExternType(i.specialise(ctx)),
        hir::ItemKind::Variant(i)    => hir::ItemKind::Variant(i.specialise(ctx)),
    },
    hir::TypeAlias  => crate::todo!(),
    hir::Enum       => crate::todo!(),
    hir::Task       => crate::todo!(),
    hir::ExternFun  => crate::todo!(),
    hir::ExternType => crate::todo!(),
    hir::Variant    => crate::todo!(),
    hir::Fun => hir::Fun {
        params: node.params.specialise(ctx),
        body: node.body.specialise(ctx),
        t: node.t.specialise(ctx),
        rt: node.rt.specialise(ctx),
        ..node.clone()
    },
    Vec<hir::Param> => node.iter().map(|p| p.specialise(ctx)).collect(),
    hir::Param => hir::Param {
        t: node.t.specialise(ctx),
        ..*node
    },
    hir::OnStart => hir::OnStart {
        fun: node.fun.specialise(ctx),
        ..node.clone()
    },
    hir::OnEvent => hir::OnEvent {
        fun: node.fun.specialise(ctx),
        ..node.clone()
    },
    hir::Block => hir::Block {
        stmts: node.stmts.specialise(ctx),
        var: node.var.specialise(ctx),
        ..node.clone()
    },
    hir::Stmt => hir::Stmt {
        kind: node.kind.specialise(ctx),
        ..node.clone()
    },
    hir::StmtKind => match node {
        hir::StmtKind::Assign(x)  => hir::StmtKind::Assign(x.specialise(ctx)),
    },
    Vec<hir::Assign> => node.iter().map(|l| l.specialise(ctx)).collect(),
    hir::Assign => hir::Assign {
        expr: node.expr.specialise(ctx),
        ..node.clone()
    },
    // NOTE: Here we do the specialisation
    hir::Type => ctx.types.fresh(),
    // Expressions are deep-cloned
    hir::Expr => hir::Expr {
        id: node.id.specialise(ctx),
        t: node.t.specialise(ctx),
        ..*node
    },
    hir::ExprId => {
        let kind = ctx.hir.exprs.resolve(*node);
        let clone = kind.specialise(ctx);
        ctx.mono.exprs.intern(clone)
    },
    hir::ExprKind => match node {
        hir::ExprKind::Return(v)         => hir::ExprKind::Return(v.specialise(ctx)),
        hir::ExprKind::Break(v)          => hir::ExprKind::Break(v.specialise(ctx)),
        hir::ExprKind::Continue          => hir::ExprKind::Continue,
        hir::ExprKind::Access(v, x)      => hir::ExprKind::Access(v.specialise(ctx), x.specialise(ctx)),
        hir::ExprKind::After(v, b)       => hir::ExprKind::After(v.specialise(ctx), b.specialise(ctx)),
        hir::ExprKind::Array(vs)         => hir::ExprKind::Array(vs.specialise(ctx)),
        hir::ExprKind::BinOp(v0, op, v1) => hir::ExprKind::BinOp(v0.specialise(ctx), op.specialise(ctx), v1.specialise(ctx)),
        hir::ExprKind::Call(v, vs)       => hir::ExprKind::Call(v.specialise(ctx), vs.specialise(ctx)),
        hir::ExprKind::SelfCall(x, vs)   => hir::ExprKind::SelfCall(x.specialise(ctx), vs.specialise(ctx)),
        hir::ExprKind::Invoke(v, x, vs)  => hir::ExprKind::Invoke(v.specialise(ctx), x.specialise(ctx), vs.specialise(ctx)),
        hir::ExprKind::Cast(v, t)        => hir::ExprKind::Cast(v.specialise(ctx), t.specialise(ctx)),
        hir::ExprKind::Emit(v)           => hir::ExprKind::Emit(v.specialise(ctx)),
        hir::ExprKind::Enwrap(x, v)      => hir::ExprKind::Enwrap(x.specialise(ctx), v.specialise(ctx)),
        hir::ExprKind::Every(v, b)       => hir::ExprKind::Every(v.specialise(ctx), b.specialise(ctx)),
        hir::ExprKind::If(v, b0, b1)     => hir::ExprKind::If(v.specialise(ctx), b0.specialise(ctx), b1.specialise(ctx)),
        hir::ExprKind::Is(x, v)          => hir::ExprKind::Is(x.specialise(ctx), v.specialise(ctx)),
        hir::ExprKind::Lit(l)            => hir::ExprKind::Lit(l.specialise(ctx)),
        hir::ExprKind::Log(v)            => hir::ExprKind::Log(v.specialise(ctx)),
        hir::ExprKind::Loop(b)           => hir::ExprKind::Loop(b.specialise(ctx)),
        hir::ExprKind::Project(v, i)     => hir::ExprKind::Project(v.specialise(ctx), i.specialise(ctx)),
        hir::ExprKind::Select(v, vs)     => hir::ExprKind::Select(v.specialise(ctx), vs.specialise(ctx)),
        hir::ExprKind::Struct(vfs)       => hir::ExprKind::Struct(vfs.specialise(ctx)),
        hir::ExprKind::Tuple(vs)         => hir::ExprKind::Tuple(vs.specialise(ctx)),
        hir::ExprKind::UnOp(op, v)       => hir::ExprKind::UnOp(op.specialise(ctx), v.specialise(ctx)),
        hir::ExprKind::Unwrap(x, v)      => hir::ExprKind::Unwrap(x.specialise(ctx), v.specialise(ctx)),
        hir::ExprKind::Item(x)           => hir::ExprKind::Item(x.specialise(ctx)),
        hir::ExprKind::Unreachable       => hir::ExprKind::Unreachable,
        hir::ExprKind::Initialise(x, v)  => hir::ExprKind::Initialise(x.specialise(ctx), v.specialise(ctx)),
        hir::ExprKind::Err               => hir::ExprKind::Err,
    },
    // Everything else is shallow-cloned
    hir::Var                    => *node,
    hir::Name                   => *node,
    hir::Path                   => *node,
    hir::BinOp                  => *node,
    hir::UnOp                   => *node,
    hir::LitKind                => node.clone(),
    hir::ScopeKind              => *node,
    hir::Index                  => *node,
    VecDeque<hir::Stmt>         => node.iter().map(|s| s.specialise(ctx)).collect(),
    Vec<hir::Expr>              => node.iter().map(|v| v.specialise(ctx)).collect(),
    Vec<hir::Var>               => node.iter().map(|v| v.specialise(ctx)).collect(),
    VecMap<hir::Name, hir::Var> => node.iter().map(|(x, v)| (x.specialise(ctx), v.specialise(ctx))).collect(),
}
