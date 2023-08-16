use ast::Type;
use im_rc::vector;

use super::bool::bool;
use super::option::option;
use super::rust;
use super::t;
use super::tc;
use super::tuple::tuple;
use super::usize::usize;

pub(crate) fn vec(v: Type) -> Type {
    tc("Vec", [v])
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("Vec", ["T"], [rust("Vec")])
        .f("new", ["T"], [], vec(t("T")), [rust("Vec::new")])
        .f("push", ["T"], [vec(t("T")), t("T")], vec(t("T")), [rust("Vec::push")])
        .f("pop", ["T"], [vec(t("T"))], tuple([t("T"), option(t("T"))]), [rust("Vec::pop")])
        .f("len", ["T"], [vec(t("T"))], usize(), [rust("Vec::len")])
        .f("get", ["T"], [vec(t("T")), usize()], option(t("T")), [rust("Vec::get")])
        .f("insert", ["T"], [vec(t("T")), usize(), t("T")], vec(t("T")), [rust("Vec::insert")])
        .f("is_empty", ["T"], [vec(t("T"))], bool(), [rust("Vec::is_empty")])
        .f("sort", ["T"], [vec(t("T"))], vec(t("T")), [rust("Vec::sort")]);
}
