use ast::Type;

use super::bool::bool;
use super::rust;
use super::t;
use super::tc;

pub(crate) fn result(t: Type) -> Type {
    tc("Result", [t])
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("Result", ["T"], [rust("Result")])
        .f("ok", ["T"], [t("T")], result(t("T")), [rust("Result::ok")])
        .f("error", ["T"], [t("T")], result(t("T")), [rust("Result::error")])
        .f("is_ok", ["T"], [result(t("T"))], bool(), [rust("Result::is_ok")])
        .f("is_error", ["T"], [result(t("T"))], bool(), [rust("Result::is_error")])
        .f("unwrap_ok", ["T"], [result(t("T"))], t("T"), [rust("Result::unwrap_ok")])
        .f("unwrap_error", ["T"], [result(t("T"))], t("T"), [rust("Result::unwrap_error")]);
}
