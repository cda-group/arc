use std::path::PathBuf;
use std::sync::Arc;

use salsa::{Database, Runtime};

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct ParseTree;

#[salsa::query_group(CompilerStorage)]
trait CompilerDatabase: Database {
    // Can be called with `.set_path(..)`
    #[salsa::input]
    fn path(&self, path: String) -> Arc<PathBuf>;

    fn source(&self, _: ()) -> String;

    fn parse(&self, _: ()) -> ParseTree;

    fn pretty(&self, _: ()) -> String;
}

fn source(db: &impl CompilerDatabase, _: ()) -> String {
    let path = db.path();
    db.path(crate::read_file(db.path()));
    panic!()
}

fn parse(db: &impl CompilerDatabase, _: ()) -> ParseTree {
    panic!()
}

fn pretty(db: &impl CompilerDatabase, _: ()) -> String {
    db.parse();
    panic!()
}

#[salsa::database(CompilerStorage)]
#[derive(Debug, Default)]
struct Compiler {
    runtime: Runtime<Compiler>,
}

impl salsa::Database for Compiler {
    fn salsa_runtime(&self) -> &salsa::Runtime<Self> {
        &self.runtime
    }

    fn salsa_runtime_mut(&mut self) -> &mut salsa::Runtime<Self> {
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
