use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;

use api::Cluster;
use itertools::Itertools;

use crate::lowering6::Graph6;
use api::WorkerId;

pub struct Graph7 {
    pub deployment: Vec<(Vec<WorkerId>, PathBuf)>,
}

const WORKSPACE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../workspace");
const RUNTIME: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../runtime");

pub fn lower(package_name: &str, graph: Graph6, cluster: &mut Cluster) -> Graph7 {
    let workspace = Path::new(WORKSPACE);
    create_crate(package_name, workspace, &graph);
    let target_triples = graph.shards.iter().group_by(|shard| {
        &cluster
            .workers
            .get(&shard.worker_id)
            .unwrap()
            .arch
            .target_triple
    });
    let mut deployment = Vec::new();
    for (target_triple, shards) in target_triples.into_iter() {
        let path = build_crate(target_triple, package_name, workspace);
        let worker_ids = shards.map(|shard| shard.worker_id).collect::<Vec<_>>();
        deployment.push((worker_ids, path));
    }
    Graph7 { deployment }
}

fn create_crate(package_name: &str, workspace: &Path, graph: &Graph6) {
    let workspace_toml = workspace.join("Cargo.toml");
    let crates = workspace.join("crates");
    let package = crates.join(package_name);
    let package_toml = package.join("Cargo.toml");
    let src = package.join("src");
    let main = src.join("main.rs");
    std::fs::create_dir_all(src).expect("Failed to create src directory");
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
                runtime = {{ path = "{RUNTIME}" }}
            "#,
        ),
    )
    .expect("Unable to write file");
}

fn build_crate(target_triple: &str, package_name: &str, workspace: &Path) -> PathBuf {
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
    while let Some(Ok(line)) = stderr.next() {
        tracing::info!("{}", line);
    }
    cmd.wait().expect("Failed to wait on `cross build`");
    workspace
        .join("target")
        .join(target_triple)
        .join("release")
        .join(package_name)
}
