use ast::Type;

use super::rust;
use super::string::string;
use super::t;

pub(crate) fn url() -> Type {
    t("Url")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("Url", [], [rust("Url")])
        //
        .f("url", [], [string()], url(), [rust("Url::parse")]);
}
