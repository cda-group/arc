use crate::opts::Opt;
use arc_script_compiler::prelude::Result;

use tracing::Level;
use tracing_flame::FlameSubscriber;
use tracing_flame::FlushGuard;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::prelude::*;
use tracing_subscriber::EnvFilter;
use tracing_subscriber::FmtSubscriber;
use tracing_subscriber::Registry;

/// Initializes a logger for debugging or profiling.
///
/// # Errors
///
/// Will return `Err` if `tracing` reports an initialisation error.
pub fn init(opt: &Opt) -> Result<Option<impl Drop>> {
    match opt {
        _ if opt.profile => {
            // Create flamegraphs
            let (sub_flame, flush_guard) = FlameSubscriber::with_file("./tracing.folded").unwrap();

            Registry::default().with(sub_flame).init();

            Ok(Some(flush_guard))
        }
        _ if opt.debug => {
            // Filter output based on options
            let sub_filter = EnvFilter::from_default_env()
                .add_directive(
                    match opt.verbosity {
                        0 => Level::INFO,
                        1 => Level::DEBUG,
                        _ => Level::TRACE,
                    }
                    .into(),
                )
                .add_directive("ena=error".parse()?);

            // Print debugging info
            match opt.verbosity {
                0 | 1 | 2 => {
                    let sub_fmt = FmtSubscriber::new().without_time().with_target(false);
                    Registry::default().with(sub_fmt).with(sub_filter).init();
                }
                _ => {
                    let sub_fmt = FmtSubscriber::new()
                        .without_time()
                        .with_target(true)
                        .pretty();
                    Registry::default().with(sub_fmt).with(sub_filter).init();
                }
            }

            Ok(None)
        }
        _ => Ok(None),
    }
}
