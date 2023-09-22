use super::mlir;
use super::rust;
use super::t;

pub(crate) fn i128() -> ast::Type {
    t("i128")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder.t("i128", [], [rust("i128"), mlir("si128")]);
}
