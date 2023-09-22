use ast::Type;

use super::char::char;
use super::rust;
use super::t;

pub(crate) fn encoding() -> Type {
    t("Encoding")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("Encoding", [], [rust("Encoding")])
        .f("csv", [], [char()], encoding(), [rust("Encoding::csv")])
        .f("json", [], [], encoding(), [rust("Encoding::json")]);
}
