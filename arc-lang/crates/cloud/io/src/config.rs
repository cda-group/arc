use const_format::formatcp;

pub const DEFAULT_COORDINATOR_TCP_PORT: u16 = 8000;
pub const DEFAULT_COORDAINTOR_REST_PORT: u16 = 8001;
pub const DEFAULT_BROKER_PORT: u16 = 9092;

pub const DEFAULT_COORDINATOR_TCP_ADDR: &str =
    formatcp!("localhost:{}", DEFAULT_COORDINATOR_TCP_PORT);
pub const DEFAULT_COORDINATOR_BROKER_ADDR: &str = formatcp!("localhost:{}", DEFAULT_BROKER_PORT);
