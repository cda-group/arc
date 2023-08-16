#![allow(unused)]

use shared::api::CoordinatorAPI;
use shared::api::QueryConfig;
use shared::api::RestAPI;
use shared::api::StateBackend;

async fn rest_call(parallelism: usize, source: String) {
    let client = reqwest::Client::new();
    let res = client
        .post(format!(
            "https://{}/api",
            shared::config::DEFAULT_COORDINATOR_REST_ADDR
        ))
        .json(&CoordinatorAPI::Query {
            source,
            config: QueryConfig {
                parallelism,
                state_backend: StateBackend::Sled,
            },
        })
        .send()
        .await
        .expect("Failed to send request");
    match res.status() {
        reqwest::StatusCode::OK => match res.json::<RestAPI>().await {
            Ok(ClientAPI::QueryResponse { data }) => println!("{}", data),
            Err(e) => eprintln!("Failed to parse response: {}", e),
        },
        _ => {
            println!("Error: {}", res.status());
        }
    }
}
