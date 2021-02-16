/// Trait for lowering `Self` into `T`.
pub trait Lower<T, C> {
    fn lower(&self, ctx: &mut C) -> T;
}

impl<Context, AST: Lower<HIR, Context>, HIR> Lower<Vec<HIR>, Context> for &'_ [AST] {
    fn lower(&self, ctx: &mut Context) -> Vec<HIR> {
        self.iter().map(|p| p.lower(ctx)).collect()
    }
}

impl<Context, AST: Lower<HIR, Context>, HIR> Lower<HIR, Context> for Box<AST> {
    fn lower(&self, ctx: &mut Context) -> HIR {
        AST::lower(self, ctx)
    }
}

impl<Context, AST: Lower<HIR, Context>, HIR> Lower<Option<HIR>, Context> for Option<AST> {
    fn lower(&self, ctx: &mut Context) -> Option<HIR> {
        self.as_ref().map(|t| t.lower(ctx))
    }
}
