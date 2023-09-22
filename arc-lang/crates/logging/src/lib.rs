use std::path::Path;
use std::str::FromStr;

use tracing_subscriber::prelude::*;
use tracing_subscriber::reload::Handle;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::Registry;

pub struct Logger {
    handle: Handle<EnvFilter, Registry>,
}

pub const FILTER: &str = "info,librdkafka=off,rdkafka::client=off";

impl Logger {
    pub fn file(path: &Path) -> Self {
        let (layer, handle) =
            tracing_subscriber::reload::Layer::new(EnvFilter::from_str(FILTER).unwrap());
        let file = std::fs::File::create(path).expect("failed to create log file");
        let fmt = tracing_subscriber::fmt::layer()
            .with_file(false)
            .with_line_number(false)
            .with_ansi(false)
            .compact()
            .with_writer(file)
            .with_filter(layer);
        tracing_subscriber::registry().with(fmt).init();

        Self { handle }
    }

    pub fn stderr() -> Self {
        let (layer, handle) =
            tracing_subscriber::reload::Layer::new(EnvFilter::from_str(FILTER).unwrap());
        let fmt = tracing_subscriber::fmt::layer()
            .with_file(false)
            .with_line_number(false)
            .compact()
            .with_writer(std::io::stderr)
            .with_filter(layer);
        tracing_subscriber::registry().with(fmt).init();
        Self { handle }
    }

    pub fn reload(&self, filter: &str) {
        self.handle
            .reload(EnvFilter::from_str(filter).unwrap())
            .unwrap();
    }

    pub fn enable_kafka(&self) {
        self.handle
            .reload(EnvFilter::from_str("info,librdkafka=on,rdkafka::client=on").unwrap())
            .unwrap();
    }

    pub fn disable_kafka(&self) {
        self.handle
            .reload(EnvFilter::from_str("info,librdkafka=off,rdkafka::client=off").unwrap())
            .unwrap();
    }
}
