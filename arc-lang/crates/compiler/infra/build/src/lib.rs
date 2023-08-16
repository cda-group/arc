#![allow(unused)]
pub mod context;

use anyhow::Result;
use std::error::Error;
use std::io::BufRead;
use std::io::BufReader;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;

use context::Context;

pub struct Package {
    pub workspace: PathBuf,
    pub target: PathBuf,
    pub name: String,
    pub root: PathBuf,
    pub main: PathBuf,
    pub toml: PathBuf,
}

pub struct Executable {
    pub path: PathBuf,
}

impl Package {
    pub fn build(&self) -> Result<Executable> {
        tracing::info!(
            "Building {}",
            self.workspace.join("crates").join(&self.name).display()
        );
        let mut cmd = Command::new("cargo")
            .arg("build")
            .arg("--package")
            .arg(&self.name)
            .arg("--release")
            .arg("--color")
            .arg("always")
            .arg("--target-dir")
            .arg(&self.workspace.join("target"))
            .current_dir(&self.workspace)
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to execute process");
        for line in BufReader::new(cmd.stderr.as_mut().unwrap()).lines() {
            tracing::info!("{}", line?);
        }
        if cmd.wait()?.success() {
            tracing::info!("Succeeded building crate {}", self.name);
            let path = self.workspace.join("target/release").join(&self.name);
            Ok(Executable { path })
        } else {
            tracing::error!("Failed building crate {}", self.name);
            return Err(anyhow::anyhow!("Build failed"));
        }
    }
}

impl Executable {
    pub fn run(&self) -> Result<()> {
        let dir = std::env::current_dir()?;
        tracing::info!("Running `{}` from `{}`", self.path.display(), dir.display());
        let mut cmd = Command::new(&self.path)
            .current_dir(std::env::current_dir()?)
            .stderr(Stdio::piped())
            .spawn()
            .expect("Failed to execute process");
        for line in BufReader::new(cmd.stderr.as_mut().unwrap()).lines() {
            tracing::info!("{}", line?);
        }
        cmd.wait()
            .expect(&format!("Failed to wait on `{}`", self.path.display()));
        Ok(())
    }
}
