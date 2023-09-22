use super::mlir;
use super::rust;
use super::t;

pub(crate) fn i64() -> ast::Type {
    t("i64")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder.t("i64", [], [rust("i64"), mlir("si64")]);
}
