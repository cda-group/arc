//! Module for checking if an AST node is `async`, i.e., contains a blocking operation.
//! We do this check at the AST level so that we can do the FSM conversion while going to HIR.

use super::Context;
use crate::compiler::ast;

pub(crate) trait IsAsync {
    fn is_async(&self, ctx: &Context<'_>) -> bool;
}

macro_rules! is_async {
    {
        [$self:ident, $ctx:ident]
        $($from:ty => $expr:expr ,)*
    } => {
        $(
            impl IsAsync for $from {
                fn is_async(&self, $ctx: &Context<'_>) -> bool {
                    let $self = self;
                    $expr
                }
            }
        )*
    }
}

is_async! {
    [node, ctx]

    ast::Block => node.stmts.is_async(ctx) || node.expr.is_async(ctx),
    ast::Stmt => match &node.kind {
        ast::StmtKind::Empty     => false,
        ast::StmtKind::Assign(a) => a.expr.is_async(ctx),
        ast::StmtKind::Expr(e)   => e.is_async(ctx),
    },
    Option<ast::Expr>          => { if let Some(e) = node { e.is_async(ctx) } else { false } },
    Option<ast::Block>         => { if let Some(e) = node { e.is_async(ctx) } else { false } },
    Vec<ast::Stmt>             => node.iter().any(|e| e.is_async(ctx)),
    Vec<ast::Expr>             => node.iter().any(|e| e.is_async(ctx)),
    Vec<ast::Field<ast::Expr>> => node.iter().any(|f| f.val.is_async(ctx)),
    Vec<ast::Case>             => node.iter().any(|c| c.body.is_async(ctx)),
    ast::Expr => {
        match ctx.ast.exprs.resolve(node) {
            ast::ExprKind::Return(e)           => e.is_async(ctx),
            ast::ExprKind::Break(e)            => e.is_async(ctx),
            ast::ExprKind::Continue            => false,
            ast::ExprKind::Path(x, ts)         => false,
            ast::ExprKind::Lambda(_, e)        => e.is_async(ctx),
            ast::ExprKind::Call(e, es)         => e.is_async(ctx) || es.is_async(ctx),
            ast::ExprKind::Invoke(e, x, es)    => e.is_async(ctx) || es.is_async(ctx),
            ast::ExprKind::If(e, b0, b1)       => e.is_async(ctx) || b0.is_async(ctx) || b1.is_async(ctx),
            ast::ExprKind::IfAssign(a, b0, b1) => a.expr.is_async(ctx) || b0.is_async(ctx) || b1.is_async(ctx),
            ast::ExprKind::Lit(kind)           => false,
            ast::ExprKind::Array(es)           => es.is_async(ctx),
            ast::ExprKind::Struct(fs)          => fs.is_async(ctx),
            ast::ExprKind::Tuple(es)           => es.is_async(ctx),
            ast::ExprKind::Select(e, es)       => e.is_async(ctx) || es.is_async(ctx),
            ast::ExprKind::UnOp(_, e)          => e.is_async(ctx),
            ast::ExprKind::BinOp(e0, _, e1)    => e0.is_async(ctx) || e1.is_async(ctx),
            ast::ExprKind::Emit(e)             => e.is_async(ctx),
            ast::ExprKind::Unwrap(x, e)        => e.is_async(ctx),
            ast::ExprKind::Enwrap(x, e)        => e.is_async(ctx),
            ast::ExprKind::Is(x, e)            => e.is_async(ctx),
            ast::ExprKind::On(cs)              => true,
            ast::ExprKind::Log(e)              => e.is_async(ctx),
            ast::ExprKind::For(p, e0, e1)      => e0.is_async(ctx) || e1.is_async(ctx),
            ast::ExprKind::Match(e, cs)        => e.is_async(ctx) || cs.is_async(ctx),
            ast::ExprKind::Loop(b)             => b.is_async(ctx),
            ast::ExprKind::Cast(e, ty)         => e.is_async(ctx),
            ast::ExprKind::Access(e, f)        => e.is_async(ctx),
            ast::ExprKind::Project(e, i)       => e.is_async(ctx),
            ast::ExprKind::After(e0, e1)       => e0.is_async(ctx) || e1.is_async(ctx),
            ast::ExprKind::Every(e0, e1)       => e0.is_async(ctx) || e1.is_async(ctx),
            ast::ExprKind::Err                 => false,
            ast::ExprKind::Block(b)            => b.is_async(ctx),
        }
    },
}
