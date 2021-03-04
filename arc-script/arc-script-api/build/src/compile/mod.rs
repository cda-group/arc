//! Macro implementation.
use super::Builder;

use arc_script_core::compiler::compile;
use arc_script_core::compiler::info::diags::sink::Buffer;
use arc_script_core::prelude::modes::Input;
use arc_script_core::prelude::modes::Mode;
use arc_script_core::prelude::modes::Output;

use std::env;
use std::fs;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use walkdir::WalkDir;

/// See [`super::process_root`].
pub fn process_root(builder: Builder) {
    let cargo_dir = &std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = &env::var("OUT_DIR").unwrap();

    for entry in WalkDir::new(cargo_dir).into_iter().filter_map(Result::ok) {
        let mut path = entry.path();
        if let Some(name) = path.file_name() {
            if name == "main.arc" || builder.compile_all {
                // Path to /a/b/c/my-project/src/x/y/z/main.arc
                let input_path = PathBuf::from(path);

                // Compile file
                let mut sink = Buffer::no_color();
                let mode = Mode {
                    input: Input::File(Some(input_path.clone())),
                    output: Output::Rust,
                    ..Default::default()
                };

                let report = compile(mode, &mut sink).expect("Internal Compiler Error");

                if report.is_ok() {
                    // Path to /a/b/c/my-project/target/build/my-project-xxx/out/src/x/y/z/main.rs
                    let mut output_path = input_path.clone();
                    output_path.set_extension("rs");
                    let output_path = output_path.strip_prefix(cargo_dir).unwrap();
                    let output_path = PathBuf::from(out_dir).join(output_path);

                    let output_dir = output_path.parent().unwrap();
                    fs::create_dir_all(output_dir).unwrap();

                    fs::File::create(output_path)
                        .unwrap()
                        .write_all(sink.as_slice())
                        .unwrap();

                    println!("cargo:rerun-if-changed={}", input_path.display());
                    println!("cargo:rerun-if-changed=build.rs");
                } else {
                    panic!("{}", std::str::from_utf8(sink.as_slice()).unwrap());
                }
            }
        }
    }
}
