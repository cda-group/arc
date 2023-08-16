pub use builtins::encoding::Encoding;

use hir::Type;
use serde::Deserialize;
use serde::Serialize;

use super::Value;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f("csv", |ctx, t, v| {
            let v0 = v[0].as_char();
            Encoding::csv(v0).into()
        })
        .f("json", |ctx, t, v| Encoding::Json.into());
}
