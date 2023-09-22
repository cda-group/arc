use ast::Type;

use super::rust;
use super::t;

pub(crate) fn i16() -> Type {
    t("i16")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder.t("i16", [], [rust("i16")]);
}
