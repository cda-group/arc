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
use std::panic;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::Mutex;
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
            self.source_dirs.push(String::new());
        }

        let msg = Arc::new(Mutex::new(String::new()));
        set_hook(msg.clone());

        for source_dir in &self.source_dirs {
            let source_dir = Path::new(cargo_dir).join(source_dir);
            for entry in WalkDir::new(source_dir).into_iter().filter_map(Result::ok) {
                let mut path = entry.path();
                if let Some(name) = path.file_name() {
                    if name == "main.arc"
                        || has_extension(path, "arc") && self.compile_unused_sources
                    {
                        let res = panic::catch_unwind(|| self.build_file(path, out_dir, cargo_dir));
                        if let Err(e) = res {
                            let mut file =
                                self.create_output_file(PathBuf::from(path), cargo_dir, out_dir);
                            writeln!(
                                file,
                                r#"compile_error!("Internal Compiler Error: {}");"#,
                                msg.lock().unwrap()
                            );
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
        let mut mode = Mode {
            input: Input::File(Some(input_path.clone())),
            output: if self.enable_optimisations {
                Output::RustMLIR
            } else {
                get_rust_backend()
            },
            ..Default::default()
        };

        let mut file = self.create_output_file(input_path, cargo_dir, out_dir);
        match compile(mode, &mut sink) {
            Err(_) => {
                writeln!(
                    file,
                    r#"compile_error!("Internal Compiler Error: {}");"#,
                    std::str::from_utf8(sink.as_slice()).unwrap()
                );
            }
            Ok(report) => {
                if report.is_ok() {
                    file.write_all(sink.as_slice()).unwrap();
                } else {
                    file.write_all(r##"compile_error!(r#"Compilation Failed: "##.as_bytes())
                        .unwrap();
                    file.write_all(sink.as_slice()).unwrap();
                    file.write_all(r##""#);"##.as_bytes()).unwrap();
                }
            }
        }
    }

    fn create_output_file(
        &self,
        mut path: PathBuf,
        cargo_dir: &str,
        out_dir: &str,
    ) -> std::fs::File {
        if let Some(prefix) = &self.prefix {
            let mut filename = path.file_name().unwrap().to_str().unwrap().to_owned();
            path.set_file_name(format!("{}{}", prefix, filename))
        }
        path.set_extension("rs");
        let output_path = path.strip_prefix(cargo_dir).unwrap();
        let output_path = PathBuf::from(out_dir).join(output_path);
        let output_dir = output_path.parent().unwrap();
        fs::create_dir_all(output_dir).unwrap();
        fs::File::create(output_path).unwrap()
    }
}

fn set_hook(msg: Arc<Mutex<String>>) {
    std::panic::set_hook(Box::new(move |panic_info| {
        let location = panic_info
            .location()
            .map(|location| format!(" in '{}' at line {}", location.file(), location.line()))
            .unwrap_or_else(String::new);

        let payload = panic_info
            .payload()
            .downcast_ref::<&str>()
            .map(|s| format!("'{}'", s))
            .unwrap_or_else(String::new);

        *msg.lock().unwrap() = format!("{}{}", payload, location);
    }))
}
