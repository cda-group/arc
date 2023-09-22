use super::result::Result;
use hir::Type;

use builtins::url::Url;

use super::Value;

pub fn define(builder: &mut super::Bifs) {
    builder.f("url", |ctx, t, v| {
        let v0 = v[0].as_string();
        Url::parse(v0).map(Into::into).into()
    });
}
