use std::path::PathBuf;

use anyhow::Result;
use diagnostics::Diagnostics;
use names::Generator;

use crate::Package;

const RUNTIME: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../../runtime");

pub struct Context {
    pub(crate) name_generator: Generator<'static>,
    pub(crate) workspace: PathBuf,
    pub(crate) crates: PathBuf,
    pub(crate) target: PathBuf,
    pub diagnostics: Diagnostics,
    pub pids: Vec<u32>,
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context")
            .field("workspace", &self.workspace)
            .field("crates", &self.crates)
            .field("diagnostics", &self.diagnostics)
            .finish()
    }
}

impl Default for Context {
    fn default() -> Self {
        let dir = directories::ProjectDirs::from("org", "cda-group", "arc-lang")
            .expect("Unable to find project directories");
        let cache = dir.cache_dir();
        tracing::info!("Cache directory: {}", cache.display());
        let workspace = cache.join("workspace");
        let workspace_toml = workspace.join("Cargo.toml");
        let crates = workspace.join("crates");
        let target = workspace.join("target");
        std::fs::create_dir_all(&crates).expect("Failed to create crates directory");
        std::fs::write(
            &workspace_toml,
            indoc::formatdoc!(
                r#"[workspace]
                   members = [ "crates/*" ]
                   resolver = "2"

                   [workspace.package]
                   version = "0.0.0"
                   edition = "2021"

                   [workspace.dependencies]
                   runtime = {{ path = "{RUNTIME}" }}"#,
            ),
        )
        .expect("Unable to write file");
        Self {
            name_generator: Generator::new(names::ADJECTIVES, names::NOUNS, names::Name::Plain),
            workspace,
            crates,
            target,
            diagnostics: Diagnostics::default(),
            pids: Vec::new(),
        }
    }
}

impl Context {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear_caches(&mut self) {
        std::fs::remove_dir_all(&self.crates).expect("Failed to remove crates directory");
        std::fs::create_dir_all(&self.crates).expect("Failed to create crates directory");
        std::fs::remove_dir_all(&self.target).expect("Failed to remove target directory");
    }

    pub fn show_caches(&self) {
        for entry in std::fs::read_dir(&self.crates).expect("Failed to read crates directory") {
            println!("{}", entry.expect("Failed to read entry").path().display());
        }
    }

    pub fn new_package(&mut self) -> Result<Package> {
        let base = self.name_generator.next().unwrap();
        let time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs();
        let workspace = self.workspace.clone();
        let name = format!("{base}-{time}");
        let root = self.crates.join(&name);
        let src = root.join("src");
        let main = src.join("main.rs");
        let toml = root.join("Cargo.toml");
        std::fs::create_dir_all(&src)?;
        std::fs::write(
            &toml,
            indoc::formatdoc!(
                r#"[package]
                   name = "{name}"
                   version.workspace = true
                   edition.workspace = true

                   [dependencies]
                   runtime.workspace = true"#,
                name = name
            ),
        )?;
        Ok(Package {
            workspace,
            target: self.target.clone(),
            root,
            main,
            toml,
            name,
        })
    }
}
