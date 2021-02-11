use crate::compiler::ast;
use crate::compiler::hir;
use crate::compiler::hir::from::lower::{Context, Lower};
use crate::compiler::info::names::NameId;
use crate::compiler::info::types::TypeId;

impl<'i, AST: Lower<HIR, Context<'i>>, HIR> Lower<Vec<HIR>, Context<'i>> for &'_ [AST] {
    fn lower(&self, ctx: &mut Context<'i>) -> Vec<HIR> {
        self.iter().map(|p| p.lower(ctx)).collect()
    }
}

impl<'i, AST: Lower<HIR, Context<'i>>, HIR> Lower<HIR, Context<'i>> for Box<AST> {
    fn lower(&self, ctx: &mut Context<'i>) -> HIR {
        AST::lower(self, ctx)
    }
}

impl<'i, AST: Lower<HIR, Context<'i>>, HIR> Lower<Option<HIR>, Context<'i>> for Option<AST> {
    fn lower(&self, ctx: &mut Context<'i>) -> Option<HIR> {
        self.as_ref().map(|t| t.lower(ctx))
    }
}
