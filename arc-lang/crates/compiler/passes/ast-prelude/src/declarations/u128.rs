use ast::Type;

use super::mlir;
use super::rust;
use super::t;

pub(crate) fn u128() -> Type {
    t("u128")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder.t("u128", [], [rust("u128"), mlir("si128")]);
}
