use std::io::Result;
use std::io::Write;

use super::usize::usize;
use ast::Type;
use ast::TypeKind::TArray;
use im_rc::vector;
use im_rc::Vector;
use info::Info;

use super::t;

pub(crate) fn array(t: Type) -> Type {
    TArray(t, None).with(Info::Builtin)
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .f("array_get", ["T"], [array(t("T")), usize()], t("T"), [])
        .f("array_set", ["T"], [array(t("T")), usize(), t("T")], array(t("T")), []);
}
