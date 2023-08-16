use ast::Type;

use super::array;
use super::array::array;
use super::path::path;
use super::rust;
use super::t;
use super::unit::unit;

pub(crate) fn instance() -> Type {
    t("Instance")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("Instance", [], [rust("Instance")])
        .f("logpath", [], [instance()], path(), [rust("(|| ())")])
        .f("kill", [], [instance()], unit(), [rust("(|| ())")]);
}
