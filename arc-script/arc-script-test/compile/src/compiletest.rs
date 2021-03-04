use cargo_toml::Manifest;
use compiletest::common::ConfigWithTemp;
use compiletest::common::Mode;
use compiletest::Config;

use std::collections::HashMap;
use std::path::PathBuf;

const DEPS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/debug/deps");
const CARGO_TOML: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
// const DEPS: &[&str] = &["prost", "arcon", "arc_script"];

#[test]
fn compiletest() {
    let config = &mut default_config();
    config.clean_rmeta();

    run(config, Mode::RunPass, "src/tests/expect_pass");
    run(config, Mode::Ui, "src/tests/expect_fail");
    run(config, Mode::RunPass, "src/tests/expect_mlir_fail_todo");
}

/// Executes a `compiletest`.
fn run(config: &mut ConfigWithTemp, mode: Mode, src_base: impl Into<PathBuf>) {
    config.config.mode = mode;
    config.config.src_base = src_base.into();
    compiletest::run_tests(&config);
}

/// Returns a default configuration for `compiletest`
fn default_config() -> ConfigWithTemp {
    // Create links to dependencies which are needed by the test
    let dev_deps = get_dev_deps();
    // let dev_deps = DEPS.iter().map(|d| d.to_string()).collect::<Vec<_>>();
    let libs = get_libs();
    let dev_libs = filter_libs(libs, &dev_deps);

    let externs = dev_deps
        .iter()
        .zip(&dev_libs)
        .map(|(dep, path)| format!("--extern {}={}", dep, path.display()))
        .collect::<Vec<_>>()
        .join(" ");

    // Must use --edition=2018 or else rustc might complain
    let target_rustcflags = format!("--edition=2018 {} -L {}", externs, DEPS_DIR).into();

    // If export BLESS=1 is set, then accept failing outputs in UI-tests
    let bless = std::env::var("BLESS").is_ok();

    Config {
        bless,
        target_rustcflags,
        quiet: false,
        verbose: false,
        ..Default::default()
    }
    .tempdir()
}

/// Returns all dev-dependencies of this crate
fn get_dev_deps() -> Vec<String> {
    Manifest::from_path(CARGO_TOML)
        .unwrap()
        .dev_dependencies
        .keys()
        .map(|package| package.replace("-", "_").clone())
        .collect()
}

/// Returns a HashMap which maps library-names to library-paths of all libraries in the crate.
fn get_libs() -> HashMap<String, PathBuf> {
    std::fs::read_dir(DEPS_DIR)
        .unwrap()
        .filter_map(|dependency| {
            let dependency = dependency.unwrap();
            let path = dependency.path();
            let filename = path.file_name().unwrap().to_str().unwrap();
            match filename.split('-').collect::<Vec<_>>().as_slice() {
                [lib_name, disambiguator] if disambiguator.ends_with(".rlib") => {
                    Some(((*lib_name).to_string(), dependency.path()))
                }
                _ => None,
            }
        })
        .collect::<std::collections::HashMap<String, PathBuf>>()
}

/// Filters out library paths to specific specific dependencies.
fn filter_libs(libs: HashMap<String, PathBuf>, deps: &[String]) -> Vec<PathBuf> {
    deps.iter()
        .map(|c| {
            libs.get(&format!("lib{}", c.replace("-", "_")))
                .cloned()
                .unwrap_or_else(|| PathBuf::from(c))
        })
        .collect()
}
