use ast::Type;

use super::blob::blob;
use super::path::path;
use super::rust;
use super::string::string;
use super::t;
use super::unit::unit;

pub(crate) fn file() -> Type {
    t("File")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("File", [], [rust("File")])
        .f("open", [], [path()], file(), [rust("File::open")])
        .f("read_to_string", [], [file()], string(), [rust("File::read_to_string")])
        .f("read_to_bytes", [], [file()], blob(), [rust("File::read_to_bytes")])
        .f("inspect", [], [file()], unit(), [rust("File::inspect")]);
}
