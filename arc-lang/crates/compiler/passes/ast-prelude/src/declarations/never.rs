use ast::Type;
use ast::TypeKind::TNever;
use info::Info;

use super::rust;
use super::string::string;
use super::unit::unit;

pub(crate) fn never() -> Type {
    TNever.with(Info::Builtin)
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .f("todo", [], [], never(), [rust(r#"(|| todo!())"#)])
        .f("unreachable", [], [], never(), [rust(r#"(|| unreachable!())"#)])
        .f("panic", [], [string()], never(), [rust(r#"(|msg| panic!("{}", msg))"#)])
        .f("exit", [], [unit()], never(), [rust(r#"(|| std::process::exit(0))"#)]);
}
