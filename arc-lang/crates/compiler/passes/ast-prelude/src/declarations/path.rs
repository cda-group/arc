use ast::Type;

use super::rust;
use super::string::string;
use super::t;

pub(crate) fn path() -> Type {
    t("Path")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("Path", [], [rust("Path")])
        //
        .f("path", [], [string()], path(), [rust("Path::new")]);
}
