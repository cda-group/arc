use crate::ast::repr::{Module, AST};
use crate::info::diags::Error;

use crate::info::Info;

use arc_script_compiler_shared::Result;
use std::fs;
use std::fs::File;
use std::io::prelude::*;
use std::path::{Path, PathBuf};

/// Name of an arc-script configuration file, in the same sense as a `Cargo.toml`.
const MANIFEST_FILENAME: &str = "Arcon.toml";

/// Name of an arc-script main-file.
const MAIN_FILENAME: &str = "main.arc";

/// Name of an arc-script source directory.
const SRC_DIRNAME: &str = "src";

/// Name of an arc-script module file.
const MOD_FILENAME: &str = "mod.arc";

macro_rules! prelude {
    ($($s:literal),+) => { concat!($(include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/../stdlib/", $s))),+) }
}

/// Prelude prepended to all arc-scripts.
const PRELUDE: &str = prelude!("strings.arc", "tasks.arc");

/// Reads a file and returns its contents.
pub(crate) fn read_file(path: &impl AsRef<Path>) -> Result<String> {
    let mut file = File::open(path)?;
    let mut source = String::new();
    file.read_to_string(&mut source)?;
    Ok(source)
}

/// Traverses upwards to find the project root.
fn find_root(path: PathBuf) -> Option<PathBuf> {
    let mut root = Some(path);
    while let Some(dir) = root {
        for entry in fs::read_dir(&dir).ok()? {
            if entry.ok()?.file_name() == MANIFEST_FILENAME {
                return Some(dir);
            }
        }
        root = Some(dir.parent()?.to_path_buf());
    }
    None
}

/// Find the src/main.arc source file.
fn find_main(path: PathBuf) -> Option<PathBuf> {
    for entry in fs::read_dir(path).ok()? {
        let path = entry.ok()?.path();
        if path.is_dir() && path.file_name()? == SRC_DIRNAME {
            for entry in fs::read_dir(path).ok()? {
                let path = entry.ok()?.path();
                if path.file_name()? == MAIN_FILENAME && path.is_file() {
                    return Some(path);
                }
            }
        }
    }
    None
}

impl AST {
    /// Parses a single source string and gives error if it contains imports.
    pub(crate) fn parse_source(&mut self, source: String, info: &mut Info) {
        // Read the file, parse it, and construct the module.
        let name = info.names.resolve(info.names.common.root.id).to_string();
        let module = Module::parse(name, source, self, info);

        if !module.imports(info).is_empty() {
            panic!();
        }

        self.modules.insert(info.paths.root, module);
    }
    /// Parses main and all its dependencies modules from the project root.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn parse_path(&mut self, main: Option<PathBuf>, info: &mut Info) {
        let path = main
            .filter(|path| path.is_file())
            .and_then(|path| path.canonicalize().ok())
            .or_else(|| {
                std::env::current_dir()
                    .ok()
                    .and_then(find_root)
                    .and_then(find_main)
            });
        if let Some(path) = path {
            let root = path.parent().unwrap().to_path_buf();
            let main = PathBuf::from(path.file_name().unwrap());

            tracing::debug!("Found project root {:?} with main module {:?}", root, main);

            self.import(&root, main, info);
        } else {
            info.diags.intern(Error::FileNotFound);
        }
    }
    /// Imports a module into the AST.
    /// NB: For now only supports the following directory structure:
    /// src/         <-- root
    ///   main.rs    <-- path (`::`)
    ///   foo/
    ///     mod.rs   <-- path (`::foo`)
    ///     bar/
    ///       mod.rs <-- path (`::foo::bar`)
    /// NB: Assumes that `use` is the only way to refer to items in other files.
    #[cfg(not(target_arch = "wasm32"))]
    fn import(&mut self, root_path: &Path, mut module_path: PathBuf, info: &mut Info) {
        tracing::debug!("Importing {:?} ...", module_path);

        // Construct the full path of the module.
        let mut full_path = root_path.to_path_buf();
        full_path.push(&module_path);

        if !full_path.is_file() {
            info.diags.intern(Error::FileNotFound);
            return;
        }

        if !matches!(full_path.extension(), Some(ext) if ext == "arc") {
            info.diags.intern(Error::BadExtension);
            return;
        }

        // Read the file, parse it, and construct the module.
        let name = full_path.to_str().unwrap().to_owned();
        let mut source = read_file(&full_path).unwrap();

        // Append prelude
        if !info.mode.no_prelude {
            source.push_str(PRELUDE);
        }

        let module = Module::parse(name, source, self, info);

        // Import any dependencies the module might have, if they have not already been imported.
        let dependency_paths = module.imports(info);
        if !dependency_paths.is_empty() {
            tracing::debug!("Found dependencies {:#?}", dependency_paths);
        }
        module_path.pop();

        let module_path_id = info.intern_ospath(&module_path);
        self.modules.insert(module_path_id, module);

        for import_path in dependency_paths {
            let mut import_path = import_path.path_buf(info);
            let import_path_id = info.intern_ospath(&import_path);
            if !self.modules.contains_key(&import_path_id) {
                import_path.push(MOD_FILENAME);
                self.import(root_path, import_path, info);
            }
        }
    }
}
