use ast::Type;

use super::bool::bool;
use super::rust;
use super::t;
use super::tc;

pub(crate) fn option(t: Type) -> Type {
    tc("Option", [t])
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("Option", ["T"], [rust("Option")])
        .f("some", ["T"], [t("T")], option(t("T")), [rust("Option::some")])
        .f("none", ["T"], [], option(t("T")), [rust("Option::none")])
        .f("is_some", ["T"], [option(t("T"))], bool(), [rust("Option::is_some")])
        .f("is_none", ["T"], [option(t("T"))], bool(), [rust("Option::is_none")])
        .f("unwrap", ["T"], [option(t("T"))], t("T"), [rust("Option::unwrap")]);
}
