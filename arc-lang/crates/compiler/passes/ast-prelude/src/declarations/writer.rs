use ast::Type;

use super::path::path;
use super::rust;
use super::socket::socket_addr;
use super::string::string;
use super::t;
use super::url::url;

pub(crate) fn writer() -> Type {
    t("Writer")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("Writer", [], [rust("Writer")])
        .f("stdout_writer", [], [], writer(), [rust("Writer::stdout")])
        .f("file_writer", [], [path()], writer(), [rust("Writer::file")])
        .f("http_writer", [], [url()], writer(), [rust("Writer::http")])
        .f("tcp_writer", [], [socket_addr()], writer(), [rust("Writer::tcp")])
        .f("kafka_writer", [], [socket_addr(), string()], writer(), [rust("Writer::kafka")]);
}
