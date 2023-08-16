use super::Value;
use crate::Function;
use builtins::duration::Duration;
use builtins::time_source::TimeSource;
use hir::Type;
use serde::Deserialize;
use serde::Serialize;

pub fn define(builder: &mut super::Bifs) {
    builder
        .f("ingestion", |ctx, t, v| {
            let v0 = v[0].as_duration();
            TimeSource::Ingestion {
                watermark_interval: v0,
            }
            .into()
        })
        .f("event", |ctx, t, v| {
            let v0 = v[0].as_function();
            let v1 = v[1].as_duration();
            let v2 = v[2].as_duration();
            TimeSource::Event {
                extractor: v0,
                watermark_interval: v1,
                slack: v2,
            }
            .into()
        });
}
