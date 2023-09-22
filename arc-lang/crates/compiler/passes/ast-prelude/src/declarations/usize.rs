use ast::Type;

use super::rust;
use super::t;

pub(crate) fn usize() -> Type {
    t("usize")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder.t("usize", [], [rust("usize")]);
}
