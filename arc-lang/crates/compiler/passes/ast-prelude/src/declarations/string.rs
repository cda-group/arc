use ast::Type;

use super::bool::bool;
use super::char::char;
use super::encoding::encoding;
use super::rust;
use super::t;
use super::tuple::tuple;
use super::usize::usize;
use super::vec::vec;

pub(crate) fn string() -> Type {
    t("String")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("String", [], [rust("String")])
        .f("new", [], [], string(), [rust("String::new")])
        .f("with_capacity", [], [usize()], string(), [rust("String::with_capacity")])
        .f("push_char", [], [string(), char()], string(), [rust("String::push")])
        .f("push_string", [], [string(), string()], string(), [rust("String::push_string")])
        .f("remove", [], [string(), usize()], tuple([string(), string()]), [rust("String::remove")])
        .f("insert_char", [], [string(), usize(), char()], string(), [rust("String::insert")])
        .f("is_empty", [], [string()], bool(), [rust("String::is_empty")])
        .f("split_off", [], [string(), usize()], tuple([string(), string()]), [rust("String::split_off")])
        .f("clear", [], [string()], string(), [rust("String::clear")])
        .f("len", [], [string()], usize(), [rust("String::len")])
        .f("decode", ["T"], [string(), encoding()], t("T"), [rust("String::decode")])
        .f("encode", ["T"], [t("T"), encoding()], string(), [rust("String::encode")])
        .f("lines", ["T"], [string()], vec(t("T")), [rust("String::lines")]);
}
