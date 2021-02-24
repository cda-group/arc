#![cfg_attr(not(test), allow(dead_code, unused_imports))]

use arc_script_core::prelude::compiler::compile;
use arc_script_core::prelude::diags::Buffer;
use arc_script_core::prelude::modes::{Input, Mode, Output};

use std::env;

fn snapshot(paths: &str, output: Output) {
    env::set_var("INSTA_OUTPUT", "summary");
    env::set_var("INSTA_FORCE_PASS", "1");
    let mut settings = insta::Settings::clone_current();
    settings.set_prepend_module_to_snapshot(false);
    settings.remove_snapshot_suffix();
    let mut sink = Buffer::no_color();
    settings.bind(|| {
        insta::glob!(paths, |path| {
            println!("Testing {}", path.display());
            let mode = Mode {
                input: Input::File(Some(path.into())),
                output,
                ..Default::default()
            };
            compile(mode, &mut sink).unwrap();
            let s = std::str::from_utf8(sink.as_slice()).unwrap();
            insta::assert_snapshot!(s);
            sink.clear();
        })
    });
}

#[test]
fn test_ast() {
    snapshot(
        "{expect-pass, expect-fail, expect-fail-todo, expect-mlir-fail-todo}/*",
        Output::AST,
    );
}

#[test]
fn test_hir() {
    snapshot(
        "{expect-pass, expect-fail, expect-fail-todo, expect-mlir-fail-todo}/*",
        Output::HIR,
    );
}

#[test]
fn test_dfg() {
    snapshot(
        "{expect-pass, expect-fail, expect-fail-todo, expect-mlir-fail-todo}/*",
        Output::DFG,
    );
}

#[test]
fn test_rust() {
    snapshot(
        "{expect-pass, expect-fail, expect-fail-todo, expect-mlir-fail-todo}/*",
        Output::Rust,
    );
}

#[test]
fn test_mlir() {
    snapshot(
        "{expect-pass, expect-fail, expect-fail-todo}/*",
        Output::MLIR,
    );
}
