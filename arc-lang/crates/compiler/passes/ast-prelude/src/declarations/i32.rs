use ast::binop;
use ast::unop;

use super::bool::bool;
use super::mlir;
use super::rust;
use super::string::string;
use super::t;
use super::usize::usize;

pub(crate) fn i32() -> ast::Type {
    t("i32")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("i32", [], [rust("i32"), mlir("si32")])
        .f(unop!(-), [], [i32()], i32(), [rust("(|a| -a)"), mlir("neq_i32")])
        .f(unop!(+), [], [i32()], i32(), [rust("(|a| a)"), mlir("neq_i32")])
        .f(binop!(+), [], [i32(), i32()], i32(), [rust("(|a,b| a+b)"), mlir("add_i32")])
        .f(binop!(-), [], [i32(), i32()], i32(), [rust("(|a,b| a-b)"), mlir("sub_i32")])
        .f(binop!(*), [], [i32(), i32()], i32(), [rust("(|a,b| a*b)"), mlir("mul_i32")])
        .f(binop!(/), [], [i32(), i32()], i32(), [rust("(|a,b| a/b)"), mlir("div_i32")])
        .f(binop!(>=), [], [i32(), i32()], bool(), [rust("(|a,b| a>=b)"), mlir("geq_i32")])
        .f(binop!(<=), [], [i32(), i32()], bool(), [rust("(|a,b| a<=b)"), mlir("leq_i32")])
        .f(binop!(<), [], [i32(), i32()], bool(), [rust("(|a,b| a<b)"), mlir("lt_i32")])
        .f(binop!(>), [], [i32(), i32()], bool(), [rust("(|a,b| a>b)"), mlir("gt_i32")])
        .f(binop!(==), [], [i32(), i32()], bool(), [rust("(|a,b| a==b)"), mlir("eq_i32")])
        .f(binop!(!=), [], [i32(), i32()], bool(), [rust("(|a,b| a!=b)"), mlir("neq_i32")])
        .f(unop!(-), [], [i32()], i32(), [rust("(|a| -a)")])
        .f(unop!(+), [], [i32()], i32(), [rust("(|a| a)")])
        .f("as_usize", [], [i32()], usize(), [rust("(|a| a as usize)")])
        .f("i32_to_string", [], [i32()], string(), [rust("(|a| a.to_string())")]);
}
