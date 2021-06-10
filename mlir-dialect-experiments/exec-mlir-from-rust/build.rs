use std::process::Command;

fn main() {
    Command::new("make").output().unwrap();
    println!("cargo:rustc-link-search={}/mlir/target", env!("CARGO_MANIFEST_DIR"));
}
