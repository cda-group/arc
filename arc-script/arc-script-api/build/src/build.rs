//! Macro implementation.
use super::Builder;

use arc_script_core::compiler::compile;
use arc_script_core::compiler::info::diags::sink::Buffer;
use arc_script_core::prelude::modes::get_rust_backend;
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

fn has_extension(path: &Path, s: &str) -> bool {
    path.extension().filter(|ext| *ext == s).is_some()
}

impl Builder {
    /// Finds all main.arc files in the crate and compiles them. Currently, there are two caveats:
    /// * All scripts are compiled if one is changed.
    /// * All main.arc files are compiled, even unused ones. However, other files ending with .arc
    /// are only compiled if they are depended on directly or transitively by a main.arc file.
    pub fn build(mut self) {
        println!("cargo:rerun-if-changed=build.rs");

        let cargo_dir = &std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let out_dir = &env::var("OUT_DIR").unwrap();

        if self.source_dirs.is_empty() {
            self.source_dirs.push("".to_owned());
        }

        for source_dir in &self.source_dirs {
            let source_dir = Path::new(cargo_dir).join(source_dir);
            for entry in WalkDir::new(source_dir).into_iter().filter_map(Result::ok) {
                let mut path = entry.path();
                if let Some(name) = path.file_name() {
                    if name == "main.arc" || has_extension(path, "arc") && self.no_exclude {
                        if let Err(e) =
                            std::panic::catch_unwind(|| self.build_file(path, out_dir, cargo_dir))
                        {
                            eprintln!("Internal compiler error when compiling : {:?}", path);
                            eprintln!("{:?}", e);
                        }
                    }
                }
            }
        }
    }

    fn build_file(&self, path: &Path, out_dir: &str, cargo_dir: &str) {
        // Path to /a/b/c/my-project/src/x/y/z/main.arc
        let input_path = PathBuf::from(path);
        println!("cargo:rerun-if-changed={}", input_path.display());

        // Compile file
        let mut sink = Buffer::no_color();
        let mode = Mode {
            input: Input::File(Some(input_path.clone())),
            output: get_rust_backend(),
            ..Default::default()
        };

        match compile(mode, &mut sink) {
            Err(_) => {
                eprintln!(
                    "Internal compiler error when compiling : {:?}",
                    input_path.into_os_string()
                );
                panic!("{}", std::str::from_utf8(sink.as_slice()).unwrap())
            }
            Ok(val) => {
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
            }
        }
    }
}
