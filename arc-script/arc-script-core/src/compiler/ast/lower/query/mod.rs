use crate::compiler::ast::AST;
use crate::compiler::info::Info;
use crate::compiler::query::Query;
use arc_script_core_shared::Lower;

struct Context<'a> {
    info: &'a mut Info,
}

impl Lower<AST, Context<'_>> for Query {
    fn lower(&self, ctx: &mut Context<'_>) -> AST {
        todo!()
    }
}
