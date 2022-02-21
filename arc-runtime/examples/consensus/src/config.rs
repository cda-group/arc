use hocon::Error;
use hocon::Hocon;
use hocon::HoconLoader;

use std::time::Duration;

pub struct Config {
    pub wait_timeout: Duration,
    pub num_threads: usize,
    pub num_nodes: usize,
    pub ble_hb_delay: u64,
    pub ble_initial_delay_factor: Option<u64>,
    pub num_proposals: u64,
    pub num_elections: u64,
    pub gc_idx: u64,
}

impl Config {
    pub fn load(path: &str) -> Result<Config, Error> {
        let cfg = HoconLoader::new().load_file(path)?.hocon()?;

        Ok(Config {
            wait_timeout: cfg["wait_timeout"].as_duration().unwrap_or_default(),
            num_threads: cfg["num_threads"].as_i64().unwrap_or_default() as usize,
            num_nodes: cfg["num_nodes"].as_i64().unwrap_or_default() as usize,
            ble_hb_delay: cfg["ble_hb_delay"].as_i64().unwrap_or_default() as u64,
            num_proposals: cfg["num_proposals"].as_i64().unwrap_or_default() as u64,
            num_elections: cfg["num_elections"].as_i64().unwrap_or_default() as u64,
            gc_idx: cfg["gc_idx"].as_i64().unwrap_or_default() as u64,
            ble_initial_delay_factor: cfg["ble_initial_delay_factor"].as_i64().map(|x| x as u64),
        })
    }
}
