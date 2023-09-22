use ast::Type;

use super::rust;
use super::t;

pub(crate) fn blob() -> Type {
    t("Blob")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder.t("Blob", [], [rust("Blob")]);
}
