#![cfg_attr(not(test), allow(dead_code, unused_imports))]

use arc_script_core::prelude::compiler::compile;
use arc_script_core::prelude::diags::Buffer;
use arc_script_core::prelude::modes::{Input, Mode, Output};

use std::env;

fn snapshot(output: Output, paths: &str) {
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
    snapshot(Output::AST, "expect-mlir-fail-todo/*");
    snapshot(Output::AST, "expect-pass/*");
    snapshot(Output::AST, "expect-fail/*");
    snapshot(Output::AST, "expect-fail-todo/*");
}

#[test]
fn test_hir() {
    snapshot(Output::HIR, "expect-mlir-fail-todo/*");
    snapshot(Output::HIR, "expect-pass/*");
    snapshot(Output::HIR, "expect-fail/*");
    snapshot(Output::HIR, "expect-fail-todo/*");
}

#[test]
fn test_dfg() {
    snapshot(Output::DFG, "expect-mlir-fail-todo/*");
    snapshot(Output::DFG, "expect-pass/*");
    snapshot(Output::DFG, "expect-fail/*");
    snapshot(Output::DFG, "expect-fail-todo/*");
}

#[test]
fn test_rust() {
    snapshot(Output::Rust, "expect-mlir-fail-todo/*");
    snapshot(Output::Rust, "expect-pass/*");
    snapshot(Output::Rust, "expect-fail/*");
    snapshot(Output::Rust, "expect-fail-todo/*");
}

#[test]
fn test_mlir() {
    snapshot(Output::MLIR, "expect-pass/*");
    snapshot(Output::MLIR, "expect-fail/*");
    snapshot(Output::MLIR, "expect-fail-todo/*");
}
