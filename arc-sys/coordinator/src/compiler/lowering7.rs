use std::fs::create_dir_all;
use std::path::Path;
use std::path::PathBuf;
use std::process::Stdio;

use itertools::Itertools;
use tokio::io::AsyncBufReadExt;
use tokio::io::BufReader;
use tokio::process::Command;

use crate::compiler::lowering6::Graph6;
use crate::server::ServerConfig;
use crate::server::WorkerId;

pub struct Graph7 {
    pub deployment: Vec<(Vec<WorkerId>, PathBuf)>,
}

const WORKSPACE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../workspace");
const DATAFLOW: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../dataflow");

pub async fn lower(package_name: &str, graph: Graph6, server_config: &mut ServerConfig) -> Graph7 {
    let workspace = Path::new(WORKSPACE);
    create_crate(package_name, workspace, &graph).await;
    let target_triples = graph.shards.iter().group_by(|shard| {
        &server_config
            .workers
            .get(&shard.worker_id)
            .unwrap()
            .arch
            .target_triple
    });
    let mut deployment = Vec::new();
    for (target_triple, shards) in target_triples.into_iter() {
        let path = build_crate(target_triple, package_name, workspace).await;
        let worker_ids = shards.map(|shard| shard.worker_id).collect::<Vec<_>>();
        deployment.push((worker_ids, path));
    }
    Graph7 { deployment }
}

async fn create_crate(package_name: &str, workspace: &Path, graph: &Graph6) {
    let workspace_toml = workspace.join("Cargo.toml");
    let crates = workspace.join("crates");
    let package = crates.join(package_name);
    let package_toml = package.join("Cargo.toml");
    let src = package.join("src");
    let main = src.join("main.rs");
    create_dir_all(src).expect("Failed to create src directory");
    std::fs::write(
        &workspace_toml,
        indoc::indoc!(
            r#"
                [workspace]
                members = [ "crates/*" ]
            "#
        ),
    )
    .expect("Unable to write file");
    std::fs::write(&main, &graph.code).expect("Unable to write file");
    std::fs::write(
        &package_toml,
        indoc::formatdoc!(
            r#"
                [package]
                name = "{package_name}"
                version = "0.0.0"
                edition = "2021"

                [dependencies]
                dataflow = {{ path = "{DATAFLOW}" }}
                "#,
        ),
    )
    .expect("Unable to write file");
}

async fn build_crate(
    target_triple: &str,
    package_name: &str,
    workspace: &Path,
) -> PathBuf {
    let mut cmd = Command::new("cargo")
        .arg("build")
        .arg("--package")
        .arg(package_name)
        .arg("--target")
        .arg(target_triple)
        .arg("--release")
        .current_dir(workspace)
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to execute process");
    let mut stderr = BufReader::new(cmd.stderr.as_mut().unwrap()).lines();
    while let Ok(Some(line)) = stderr.next_line().await {
        tracing::info!("{}", line);
    }
    cmd.wait().await.expect("Failed to wait on `cross build`");
    workspace
        .join("target")
        .join(target_triple)
        .join("release")
        .join(package_name)
}
