use super::result::Result;
use super::Value;
use hir::Type;
use serde::Deserialize;
use serde::Serialize;

use builtins::socket::SocketAddr;

pub fn define(builder: &mut super::Bifs) {
    builder.f("socket", |ctx, t, v| {
        let v0 = v[0].as_string();
        SocketAddr::parse(v0).into()
    });
}
