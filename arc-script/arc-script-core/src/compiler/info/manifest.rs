use http_serde;
use serde_derive::Deserialize;
use std::str::FromStr;

#[derive(Deserialize)]
pub struct Config {
    pub sources: Vec<SourceConfig>,
    pub sinks: Vec<SinkConfig>,
}

#[derive(Deserialize)]
pub struct SourceConfig {
    pub schema: Schema,
    pub provider: Provider,
}

#[derive(Deserialize)]
pub struct SinkConfig {
    pub schema: Schema,
    pub provider: Provider,
}

#[derive(Deserialize)]
pub enum Schema {
    Json(Json),
    Proto(Proto),
    Csv(Csv),
}

#[derive(Deserialize)]
pub struct Json {}

#[derive(Deserialize)]
pub struct Proto {
    #[serde(with = "http_serde::uri")]
    pub uri: http::Uri,
}

#[derive(Deserialize)]
pub enum Provider {
    Kafka(Kafka),
    File(File),
}

#[derive(Deserialize)]
pub struct Kafka {
    #[serde(with = "http_serde::uri")]
    pub uri: http::Uri,
}

#[derive(Deserialize)]
pub struct File {
    #[serde(with = "http_serde::uri")]
    pub uri: http::Uri,
}

#[derive(Deserialize)]
pub struct Csv {
    pub records: Vec<CsvColumn>,
}

#[derive(Deserialize)]
pub struct CsvColumn {
    pub label: String,
    pub ty: CsvType,
}

#[derive(Deserialize)]
pub enum CsvType {
    Int,
    Float,
    String,
}

impl FromStr for Config {
    type Err = toml::de::Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        toml::from_str(s)
    }
}
