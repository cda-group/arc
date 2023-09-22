use ast::Type;

use super::rust;
use super::t;

pub(crate) fn u64() -> Type {
    t("u64")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder.t("u8", [], [rust("u8")]);
}
