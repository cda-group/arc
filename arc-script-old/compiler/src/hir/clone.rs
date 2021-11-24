use crate::ast;
use crate::hir;
use crate::info::files::Loc;
use crate::info::paths::PathId;
use crate::info::Info;

use arc_script_compiler_shared::itertools::Itertools;
use arc_script_compiler_shared::VecMap;

use std::collections::VecDeque;

struct Context<'i> {
    poly: &'i hir::HIR,
    mono: &'i mut hir::HIR,
    info: &'i mut Info,
}

/// Clones a syntax node.
trait DeepClone {
    fn dc(&self, ctx: &mut Context<'_>) -> Self;
}

/// Macro for implementing the `Declare` trait.
macro_rules! deep_clone {
    {
        [$node:ident, $ctx:ident]
        $($ty:path => $expr:expr ,)*
    } => {
        $(
            impl DeepClone for $ty {
                fn dc(&self, $ctx: &mut Context<'_>) -> Self {
                    let $node = self;
                    $expr
                }
            }
        )*
    };
}

deep_clone! {
    [node, ctx]

    hir::Item => hir::Item {
        kind: node.kind.dc(ctx),
        ..node.clone()
    },
    hir::ItemKind => match node {
            hir::ItemKind::TypeAlias(i)      => hir::ItemKind::TypeAlias(i.dc(ctx)),
            hir::ItemKind::Enum(i)       => hir::ItemKind::Enum(i.dc(ctx)),
            hir::ItemKind::Fun(i)        => hir::ItemKind::Fun(i.dc(ctx)),
            hir::ItemKind::Task(i)       => hir::ItemKind::Task(i.dc(ctx)),
            hir::ItemKind::ExternFun(i)  => hir::ItemKind::ExternFun(i.dc(ctx)),
            hir::ItemKind::ExternType(i) => hir::ItemKind::ExternType(i.dc(ctx)),
            hir::ItemKind::Variant(i)    => hir::ItemKind::Variant(i.dc(ctx)),
    },
    hir::TypeAlias => node.clone(),
    hir::Enum => node.clone(),
    hir::Fun => hir::Fun {
        body: node.body.dc(ctx),
        ..node.clone()
    },
    hir::Task => hir::Task {
        fields: node.fields.dc(ctx),
        on_event: node.on_event.dc(ctx),
        ..node.clone()
    },
    Option<hir::OnEvent> => node.as_ref().map(|o| o.dc(ctx)),
    hir::OnEvent => hir::OnEvent {
        fun: node.fun.dc(ctx),
        ..node.clone()
    },
    hir::ExternFun => node.clone(),
    hir::ExternType => node.clone(),
    hir::Variant => node.clone(),
    hir::Block => hir::Block {
        stmts: node.stmts.dc(ctx),
        var: node.var.dc(ctx),
        ..node.clone()
    },
    hir::Stmt => hir::Stmt {
        kind: node.kind.dc(ctx),
        ..node.clone()
    },
    hir::StmtKind => match node {
        hir::StmtKind::Assign(x)  => hir::StmtKind::Assign(x.dc(ctx)),
    },
    Vec<hir::Assign> => node.iter().map(|l| l.dc(ctx)).collect(),
    hir::Assign => hir::Assign {
        expr: node.expr.dc(ctx),
        ..node.clone()
    },
    // Expressions are deep-cloned
    hir::Expr => hir::Expr {
        id: node.id.dc(ctx),
        ..*node
    },
    hir::ExprId => {
        let kind = ctx.poly.exprs.resolve(node);
        let clone = kind.dc(ctx);
        ctx.mono.exprs.intern(clone)
    },
    hir::ExprKind => match node {
        hir::ExprKind::Return(v)         => hir::ExprKind::Return(v.dc(ctx)),
        hir::ExprKind::Break(v)          => hir::ExprKind::Break(v.dc(ctx)),
        hir::ExprKind::Continue          => hir::ExprKind::Continue,
        hir::ExprKind::Access(e, x)      => hir::ExprKind::Access(e.dc(ctx), x.dc(ctx)),
        hir::ExprKind::After(e, b)       => hir::ExprKind::After(e.dc(ctx), b.dc(ctx)),
        hir::ExprKind::Array(es)         => hir::ExprKind::Array(es.dc(ctx)),
        hir::ExprKind::BinOp(e0, op, e1) => hir::ExprKind::BinOp(e0.dc(ctx), op.dc(ctx), e1.dc(ctx)),
        hir::ExprKind::Call(e, es)       => hir::ExprKind::Call(e.dc(ctx), es.dc(ctx)),
        hir::ExprKind::SelfCall(e, es)   => hir::ExprKind::SelfCall(e.dc(ctx), es.dc(ctx)),
        hir::ExprKind::Invoke(e, x, es)  => hir::ExprKind::Invoke(e.dc(ctx), x.dc(ctx), es.dc(ctx)),
        hir::ExprKind::Cast(e, t)        => hir::ExprKind::Cast(e.dc(ctx), t.dc(ctx)),
        hir::ExprKind::Emit(e)           => hir::ExprKind::Emit(e.dc(ctx)),
        hir::ExprKind::Enwrap(x, e)      => hir::ExprKind::Enwrap(x.dc(ctx), e.dc(ctx)),
        hir::ExprKind::Every(e, b)       => hir::ExprKind::Every(e.dc(ctx), b.dc(ctx)),
        hir::ExprKind::If(e, b0, b1)     => hir::ExprKind::If(e.dc(ctx), b0.dc(ctx), b1.dc(ctx)),
        hir::ExprKind::Is(x, e)          => hir::ExprKind::Is(x.dc(ctx), e.dc(ctx)),
        hir::ExprKind::Lit(l)            => hir::ExprKind::Lit(l.dc(ctx)),
        hir::ExprKind::Log(e)            => hir::ExprKind::Log(e.dc(ctx)),
        hir::ExprKind::Loop(b)           => hir::ExprKind::Loop(b.dc(ctx)),
        hir::ExprKind::Project(e, i)     => hir::ExprKind::Project(e.dc(ctx), i.dc(ctx)),
        hir::ExprKind::Select(e, es)     => hir::ExprKind::Select(e.dc(ctx), es.dc(ctx)),
        hir::ExprKind::Struct(fs)        => hir::ExprKind::Struct(fs.dc(ctx)),
        hir::ExprKind::Tuple(es)         => hir::ExprKind::Tuple(es.dc(ctx)),
        hir::ExprKind::UnOp(op, e)       => hir::ExprKind::UnOp(op.dc(ctx), e.dc(ctx)),
        hir::ExprKind::Unwrap(x, e)      => hir::ExprKind::Unwrap(x.dc(ctx), e.dc(ctx)),
        hir::ExprKind::Err               => hir::ExprKind::Err,
        hir::ExprKind::Item(x)           => hir::ExprKind::Item(x.dc(ctx)),
        hir::ExprKind::Unreachable       => hir::ExprKind::Unreachable,
        hir::ExprKind::Initialise(x, v)  => hir::ExprKind::Initialise(x.dc(ctx), v.dc(ctx)),
    },
    // Everything else is shallow-cloned
    hir::Var         => *node,
    hir::Name        => *node,
    hir::Path        => *node,
    hir::Type        => *node,
    hir::BinOp       => *node,
    hir::UnOp        => *node,
    hir::LitKind     => node.clone(),
    hir::ScopeKind => *node,
    hir::Index       => *node,
    VecDeque<hir::Stmt>         => node.iter().map(|s| s.dc(ctx)).collect(),
    Vec<hir::Expr>              => node.iter().map(|e| e.dc(ctx)).collect(),
    Vec<hir::Var>               => node.iter().map(|e| e.dc(ctx)).collect(),
    VecMap<hir::Name, hir::Var> => node.iter().map(|(x,e)| (x.dc(ctx), e.dc(ctx))).collect(),
    VecMap<hir::Name, hir::Type> => node.iter().map(|(x,t)| (x.dc(ctx), t.dc(ctx))).collect(),
}
