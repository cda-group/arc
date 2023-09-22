use ast::Type;

use super::rust;
use super::t;

pub(crate) fn u16() -> Type {
    t("u16")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder.t("u32", [], [rust("u32")]);
}
