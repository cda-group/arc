//! Script builder implementation.

use arc_script_core::prelude::compiler;
use arc_script_core::prelude::diags::sink::Buffer;
use arc_script_core::prelude::modes::Input;
use arc_script_core::prelude::modes::Mode;
use arc_script_core::prelude::modes::Output;

use crate::stage::fun::Fun;

use std::env;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

#[derive(Debug, Default, Clone)]
pub struct Script {
    path: PathBuf,
    funs: Vec<Fun>,
}

impl Script {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            funs: Vec::new(),
        }
    }
    pub fn stage(mut self, fun: Fun) -> Self {
        self.funs.push(fun);
        self
    }
    /// This function is assumed to be called from inside a `build.rs` build-script file.
    /// It compiles a `Script` into an `.rs` file and places it in the target directory.
    /// The file can later be included as a module.
    pub fn compile(self) {
        let mode = Mode {
            input: Input::File(Some(self.path.clone())),
            output: Output::Rust,
            ..Default::default()
        };
        let mut sink = Buffer::no_color();
        if let Ok(report) = compiler::compile(mode, &mut sink) {
            let output = sink.as_slice();
            if report.is_ok() {
                self.generate(output);
            } else {
                let message = std::str::from_utf8(output).expect("Internal Error");
                println!("cargo:warning={}", message);
            }
        } else {
            println!("cargo:warning=Internal compiler error");
        }
    }
    /// Saves a string of rust-source to the crate's `target/` directory.
    fn generate(self, source: &[u8]) {
        let out_dir = PathBuf::from(env::var("OUT_DIR").expect("$OUT_DIR env-var was not set"));

        let mut path = out_dir.join(self.path.clone());
        path.set_extension("rs");
        let parent = path.parent().expect("Path has no parent");
        fs::create_dir_all(parent).expect("Failed creating directories");
        let mut file = fs::File::create(path).expect("Failed creating file");
        file.write_all(source).expect("Failed writing file");

        println!("cargo:rerun-if-changed={:?}", self.path);
        println!("cargo:rerun-if-changed=build.rs");
    }
}
