use builtins::writer::Writer;
use hir::Type;

use super::Value;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f("stdout_writer", |ctx, t, v| Writer::stdout().into())
        .f("file_writer", |ctx, t, v| {
            let v0 = v[0].as_path();
            Writer::file(v0).into()
        })
        .f("http_writer", |ctx, t, v| {
            let v0 = v[0].as_url();
            Writer::http(v0).into()
        })
        .f("tcp_writer", |ctx, t, v| {
            let v0 = v[0].as_socket_addr();
            Writer::tcp(v0).into()
        })
        .f("kafka_writer", |ctx, t, v| {
            let v0 = v[0].as_socket_addr();
            let v1 = v[1].as_string();
            Writer::kafka(v0, v1).into()
        });
}
