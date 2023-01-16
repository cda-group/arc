use std::io::BufRead;
use std::path::Path;
use std::process::exit;
use std::process::Command;
use std::process::Stdio;

fn main() {
    if std::env::var("ARC_MLIR_CMD").is_ok() {
        return;
    } else if let Ok(path) = which::which("arc-mlir") {
        println!("cargo:rustc-env=ARC_MLIR_CMD={}", path.display());
    } else {
        let out_dir = std::env::var("OUT_DIR").unwrap();
        let arc_mlir_cmd = Path::new(&out_dir).join("arc-mlir/llvm-build/bin/arc-mlir");
        if !arc_mlir_cmd.exists() {
            which::which("cmake").expect("dune not found in PATH");
            let mut child = Command::new("../../build")
                .env("ARC_MLIR_BUILD", Path::new(&out_dir).join("arc-mlir"))
                .stderr(Stdio::piped())
                .spawn()
                .expect("CMake failed");
            if std::env::var("ARC_COORDINATOR_DEBUG").is_ok() {
                println!("cargo:warning=Building Arc-MLIR ...");
                for line in std::io::BufReader::new(child.stderr.as_mut().unwrap()).lines() {
                    println!("cargo:warning={}", line.unwrap());
                }
            }
            let status = child.wait().expect("CMake failed");
            if !status.success() {
                exit(status.code().expect("CMake was terminated"));
            }
        }
        println!("cargo:rustc-env=ARC_MLIR_CMD={}", arc_mlir_cmd.display());
    }
}
