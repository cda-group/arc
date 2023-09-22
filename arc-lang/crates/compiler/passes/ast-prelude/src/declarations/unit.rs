use ast::Type;
use ast::TypeKind::TUnit;
use info::Info;

use super::noop;
use super::rust;
use super::string::string;
use super::t;

pub(crate) fn unit() -> Type {
    TUnit.with(Info::Builtin)
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .f("print", [], [string()], unit(), [rust(r#"(|x| println!("{}", x))"#)])
        .f("debug", ["T"], [t("T")], unit(), [rust(r#"(|x| println!("{:?}", x))"#)])
        .f("dataflow", [], [], unit(), [rust(noop())])
        .f("connect", [], [string()], unit(), [rust(noop())])
        .f("topics", [], [], unit(), [rust(noop())])
        .f("bifs", [], [], unit(), [rust(noop())])
        .f("typeof", ["T"], [t("T")], unit(), [rust(noop())]);
}
