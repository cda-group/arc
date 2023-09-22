use std::time::Duration;

use tracing::log::LevelFilter;
use tracing::Level;
use tracing_subscriber::filter::Directive;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;

pub fn init() {
    // let console_layer = console_subscriber::ConsoleLayer::builder()
    //     .server_addr(([127, 0, 0, 1], 5555))
    //     .spawn();

    let fmt_layer = tracing_subscriber::fmt::layer()
        .compact()
        .with_filter(EnvFilter::from_default_env().add_directive(Directive::from(Level::INFO)));
    // .with_target(true)
    // .with_thread_ids(true)
    // .with_line_number(true)
    // .with_file(true)
    // .with_thread_names(true)
    // .with_ansi(true);

    tracing_subscriber::registry()
        // .with(console_layer)
        .with(fmt_layer)
        .init();
}
