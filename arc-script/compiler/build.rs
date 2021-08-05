//! Build script related to internal compiler functions.

extern crate lalrpop;

/// A build-script. Here we generate the `arc-script` parser.
fn main() {
    lalrpop::Configuration::new()
        .emit_whitespace(false)
        .use_cargo_dir_conventions()
        .process()
        .unwrap();
}
