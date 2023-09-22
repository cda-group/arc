use ast::Type;

use super::rust;
use super::t;

pub(crate) fn char() -> Type {
    t("char")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder.t("char", [], [rust("char")]);
}
