use super::mlir;
use super::rust;
use super::t;

pub(crate) fn f32() -> ast::Type {
    t("f32")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("f32", [], [rust("f32"), mlir("f32")])
        .f("add_f32", [], [f32(), f32()], f32(), [rust("f32::add_f32"), mlir("add_f32")])
        .f("sub_f32", [], [f32(), f32()], f32(), [rust("f32::sub_f32"), mlir("sub_f32")])
        .f("mul_f32", [], [f32(), f32()], f32(), [rust("f32::mul_f32"), mlir("mul_f32")])
        .f("div_f32", [], [f32(), f32()], f32(), [rust("f32::div_f32"), mlir("div_f32")]);
}
