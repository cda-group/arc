use crate::ast::AST;
use crate::info::Info;
use crate::query::Query;
use arc_script_compiler_shared::Lower;

struct Context<'a> {
    info: &'a mut Info,
}

impl Lower<AST, Context<'_>> for Query {
    fn lower(&self, ctx: &mut Context<'_>) -> AST {
        crate::todo!()
    }
}
