use ast::Type;

use super::array;
use super::array::array;
use super::instance::instance;
use super::rust;
use super::t;
use super::unit::unit;

pub(crate) fn dataflow() -> Type {
    t("Dataflow")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("Dataflow", [], [rust("Dataflow")])
        .f("run", [], [dataflow()], instance(), [rust("Dataflow::run")])
        .f("merge", [], [array(dataflow())], dataflow(), [rust("Dataflow::merge")]);
}
