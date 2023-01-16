use std::io::BufRead;
use std::path::Path;
use std::process::exit;
use std::process::Command;
use std::process::Stdio;

fn main() {
    if std::env::var("ARC_LANG_CMD").is_ok() {
        return;
    } else if let Ok(path) = which::which("arc-mlir") {
        println!("cargo:rustc-env=ARC_LANG_CMD={}", path.display());
    } else {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let arc_lang_cmd = Path::new(&out_dir).join("arc-lang/install/default/bin/main");
        if !arc_lang_cmd.exists() {
            which::which("dune").expect("dune not found in PATH");
            let mut child = Command::new("dune")
                .arg("build")
                .arg("--root")
                .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../arc-lang"))
                .stderr(Stdio::piped())
                .arg("--build-dir")
                .arg(Path::new(&out_dir).join("arc-lang"))
                .spawn()
                .expect("dune failed");
            if std::env::var("ARC_CLIENT_DEBUG").is_ok() {
                println!("cargo:warning=Building Arc-Lang ...");
                for line in std::io::BufReader::new(child.stderr.as_mut().unwrap()).lines() {
                    println!("cargo:warning={}", line.unwrap());
                }
            }
            let status = child.wait().expect("CMake failed");
            if !status.success() {
                exit(status.code().expect("CMake was terminated"));
            }
        }
        println!("cargo:rustc-env=ARC_LANG_CMD={}", arc_lang_cmd.display());
    }
}
