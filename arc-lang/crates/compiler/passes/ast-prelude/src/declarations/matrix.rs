use ast::Type;

use super::rust;
use super::t;
use super::tc;
use super::usize::usize;
use super::vec::vec;

pub(crate) fn matrix(t: Type) -> Type {
    tc("Matrix", [t])
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("Matrix", ["T"], [rust("Matrix")])
        .f("zeros", ["T"], [usize(), usize()], matrix(t("T")), [])
        .f("insert_axis", ["T"], [matrix(t("T")), usize()], matrix(t("T")), [])
        .f("remove_axis", ["T"], [matrix(t("T")), usize()], matrix(t("T")), [])
        .f("into_vec", ["T"], [matrix(t("T"))], vec(t("T")), []);
}
