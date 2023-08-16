use ast::Type;

use super::rust;
use super::string::string;
use super::t;

pub(crate) fn socket_addr() -> Type {
    t("SocketAddr")
}

pub(crate) fn declare(builder: &mut super::Builder) {
    builder
        .t("SocketAddr", [], [rust("SocketAddr")])
        .f("socket", [], [string()], socket_addr(), [rust("SocketAddr::parse")]);
}
