use std::path::PathBuf;
use std::sync::Arc;

use crate::compiler::ast::AST;
use crate::compiler::dfg::DFG;
use crate::compiler::hir::HIR;
use crate::compiler::info::opt::Opt;
use crate::compiler::mlir::MLIR;

use salsa::{Database, Runtime};

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct ParseTree;

#[salsa::query_group(CompilerStorage)]
trait CompilerDatabase: Database {
    #[salsa::input]
    fn opt(&self, key: ()) -> Arc<Opt>;

    fn ast(&self, key: ()) -> Arc<AST>;

    fn hir(&self, key: ()) -> Arc<HIR>;

    fn dfg(&self, key: ()) -> Arc<DFG>;

    fn mlir(&self, key: ()) -> Arc<MLIR>;
}

fn ast(db: impl CompilerDatabase, _: ()) -> String {
    let opt = db.opt(());
    db.path(crate::read_file(db.path()));
    panic!()
}

fn hir(db: &impl CompilerDatabase, _: ()) -> ParseTree {
    panic!()
}

fn dfg(db: &impl CompilerDatabase, _: ()) -> String {
    db.parse();
    panic!()
}

fn mlir(db: &impl CompilerDatabase, _: ()) -> String {
    db.parse();
    panic!()
}

#[salsa::database(CompilerStorage)]
#[derive(Debug, Default)]
struct Compiler {
    runtime: Runtime,
}

impl salsa::Database for Compiler {
    fn salsa_runtime(&self) -> &salsa::Runtime {
        &self.runtime
    }

    fn salsa_runtime_mut(&mut self) -> &mut salsa::Runtime {
        &mut self.runtime
    }
}

#[test]
fn test() {
    let mut db: Compiler = Compiler::default();
    let path = Arc::new("/home/klas/Workspace/testing-grounds/test.arc-script");
    db.set_path(path);
    println!("{}", db.pretty());
}
