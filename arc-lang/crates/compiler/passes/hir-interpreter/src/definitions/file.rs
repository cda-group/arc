use super::Value;

pub use builtins::file::File;
use hir::Type;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f("open", |ctx, t, v| {
            let v0 = v[0].as_path();
            File::open(v0).into()
        })
        .f("read_to_string", |ctx, t, v| {
            let v0 = v[0].as_file();
            v0.read_to_string().into()
        })
        .f("read_to_bytes", |ctx, t, v| {
            let v0 = v[0].as_file();
            v0.read_to_bytes().into()
        })
        .f("inspect", |ctx, t, v| {
            let v0 = v[0].as_file();
            v0.inspect();
            ().into()
        });
}
