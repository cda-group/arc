//! Macro implementation.

use arc_script_core::compiler::compile;
use arc_script_core::compiler::info::diags::sink::Buffer;
use arc_script_core::prelude::modes::{Input, Mode, Output};

use std::env;
use std::fs;
use std::io::BufWriter;
use std::io::Write;
use std::path::{Path, PathBuf};
use walkdir::WalkDir;

/// See [`super::process_root`].
pub fn process_root() {
    let root = &std::env::var("CARGO_MANIFEST_DIR").expect("$CARGO_MANIFEST_DIR was not set");
    let out = &env::var("OUT_DIR").expect("$OUT_DIR env-var was not set");

    for entry in WalkDir::new(root).into_iter().filter_map(|e| e.ok()) {
        let mut path = entry.path();
        if let Some(name) = path.file_name() {
            if name == "main.arc" {
                // Create /a/b/c/my-project/src/x/y/z/main.arc
                let arc_path = PathBuf::from(path);
                println!("cargo-warning={}", arc_path.display());

                // Create /a/b/c/my-project/target/build/my-project-xxx/out/src/x/y/z/main.rs
                let mut rust_path = arc_path.clone();
                rust_path.set_extension("rs");
                println!("cargo-warning=intial: {}", rust_path.display());
                println!("cargo-warning=root: {:?}", root);
                let rust_path = rust_path.strip_prefix(root).expect("Failed stripping root");
                let out_dir = PathBuf::from(out);
                println!("cargo-warning=out_dir: {}", out_dir.display());
                let rust_path = out_dir.join(rust_path);
                println!("cargo-warning=rust_path: {}", rust_path.display());

                // Compile file
                let mut sink = Buffer::no_color();
                let mode = Mode {
                    input: Input::File(Some(arc_path.clone())),
                    output: Output::Rust,
                    ..Default::default()
                };

                // If successful, write it to the target directory
                if let Ok(report) = compile(mode, &mut sink) {
                    let output = sink.as_slice();
                    dbg!(&output);
                    if report.is_ok() {
                        let parent = rust_path.parent().expect("Path has no parent");
                        fs::create_dir_all(parent).expect("Failed creating directories");
                        let mut file = fs::File::create(rust_path).expect("Failed creating file");
                        file.write_all(output).expect("Failed writing file");

                        println!("cargo:rerun-if-changed={}", arc_path.display());
                        println!("cargo:rerun-if-changed=build.rs");
                    } else {
                        let message = std::str::from_utf8(output).expect("Internal Error");
                        println!("cargo:warning={}", message);
                    }
                } else {
                    println!("cargo:warning=Internal compiler error");
                }
            }
        }
    }
}
