use arc_script_core_shared::Result;

/// Initializes a logger for debugging.
///
/// # Errors
///
/// Will return `Err` if `tracing` reports an initialisation error.
pub fn init(verbosity: i32) -> Result<()> {
    let sub = tracing_subscriber::FmtSubscriber::builder()
        .compact()
        .without_time();
    let sub = match verbosity {
        0 => sub.with_max_level(tracing::Level::INFO),
        1 => sub.with_max_level(tracing::Level::DEBUG),
        _ => sub.with_max_level(tracing::Level::TRACE),
    };
    tracing::subscriber::set_global_default(sub.finish())?;
    Ok(())
}
