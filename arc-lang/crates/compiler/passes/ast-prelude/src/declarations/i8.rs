use ast::Type;

use super::mlir;
use super::rust;
use super::t;

pub(crate) fn i8() -> Type {
    t("i8")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder.t("i8", [], [rust("i8"), mlir("si8")]);
}
