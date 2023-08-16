use super::Value;
use std::io::Result;
use std::io::Write;

pub use builtins::reader::Reader;
use hir::Type;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f("stdin_reader", |ctx, t, v| Reader::stdin().into())
        .f("file_reader", |ctx, t, v| {
            let path = v[0].as_path();
            let watch = v[1].as_bool();
            Reader::file(path, watch).into()
        })
        .f("http_reader", |ctx, t, v| {
            let url = v[0].as_url();
            Reader::http(url).into()
        })
        .f("tcp_reader", |ctx, t, v| {
            let addr = v[0].as_socket_addr();
            Reader::tcp(addr).into()
        })
        .f("kafka_reader", |ctx, t, v| {
            let addr = v[0].as_socket_addr();
            let topic = v[1].as_string();
            Reader::kafka(addr, topic).into()
        });
}
