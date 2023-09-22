use ast::Type;
use ast::unop;

use super::mlir;
use super::rust;
use super::t;

pub(crate) fn f64() -> Type {
    t("f64")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("f64", [], [rust("f64"), mlir("f64")]);
}
