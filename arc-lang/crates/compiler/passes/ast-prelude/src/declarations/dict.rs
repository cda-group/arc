use ast::Type;

use super::bool::bool;
use super::rust;
use super::t;
use super::tc;
use super::tuple::tuple;

pub(crate) fn dict(k: Type, v: Type) -> Type {
    tc("Dict", [k, v])
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("Dict", ["K", "T"], [rust("Dict")])
        .f("new", ["K", "T"], [], dict(t("K"), t("T")), [rust("Dict::new")])
        .f("insert", ["K", "T"], [dict(t("K"), t("T")), t("K"), t("T")], dict(t("K"), t("T")), [rust("Dict::insert")])
        .f("remove", ["K", "T"], [dict(t("K"), t("T")), t("K")], dict(t("K"), t("T")), [rust("Dict::remove")])
        .f("get", ["K", "T"], [dict(t("K"), t("T")), t("K")], tc("Option", [t("T")]), [rust("Dict::get")])
        .f("contains", ["K", "T"], [dict(t("K"), t("T")), t("K")], bool(), [rust("Dict::contains")]);
}
