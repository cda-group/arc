use ast::Type;

use super::bool::bool;
use super::path::path;
use super::rust;
use super::socket::socket_addr;
use super::string::string;
use super::t;
use super::url::url;

pub(crate) fn reader() -> Type {
    t("Reader")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("Reader", [], [rust("Reader")])
        .f("stdin_reader", [], [], reader(), [rust("Reader::stdin")])
        .f("file_reader", [], [path(), bool()], reader(), [rust("Reader::file")])
        .f("http_reader", [], [url()], reader(), [rust("Reader::http")])
        .f("tcp_reader", [], [socket_addr()], reader(), [rust("Reader::tcp")])
        .f("kafka_reader", [], [socket_addr(), string()], reader(), [rust("Reader::kafka")]);
}
