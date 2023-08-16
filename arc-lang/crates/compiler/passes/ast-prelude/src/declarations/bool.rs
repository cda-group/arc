use ast::binop;
use ast::unop;

use super::mlir;
use super::rust;
use super::t;

pub(crate) fn bool() -> ast::Type {
    t("bool")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("bool", [], [rust("bool"), mlir("si1")])
        .f(unop!(!), [], [bool()], bool(), [rust("(|a| !a)"), mlir("not_i1")])
        .f(binop!(or), [], [bool(), bool()], bool(), [rust("(|a,b| a||b)"), mlir("or_i32")])
        .f(binop!(and), [], [bool(), bool()], bool(), [rust("(|a,b| a&&b)"), mlir("and_i32")]);
}
