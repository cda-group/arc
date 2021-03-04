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
    snapshot(Output::AST, "tests/expect_mlir_fail_todo/*.arc");
    snapshot(Output::AST, "tests/expect_pass/*.arc");
    snapshot(Output::AST, "tests/expect_fail/*.arc");
    snapshot(Output::AST, "tests/expect_fail_todo/*.arc");
}

#[test]
fn test_hir() {
    snapshot(Output::HIR, "tests/expect_mlir_fail_todo/*.arc");
    snapshot(Output::HIR, "tests/expect_pass/*.arc");
    snapshot(Output::HIR, "tests/expect_fail/*.arc");
    snapshot(Output::HIR, "tests/expect_fail_todo/*.arc");
}

#[test]
fn test_dfg() {
    snapshot(Output::DFG, "tests/expect_mlir_fail_todo/*.arc");
    snapshot(Output::DFG, "tests/expect_pass/*.arc");
    snapshot(Output::DFG, "tests/expect_fail/*.arc");
    snapshot(Output::DFG, "tests/expect_fail_todo/*.arc");
}

#[test]
fn test_rust() {
    snapshot(Output::Rust, "tests/expect_mlir_fail_todo/*.arc");
    snapshot(Output::Rust, "tests/expect_pass/*.arc");
    snapshot(Output::Rust, "tests/expect_fail/*.arc");
    snapshot(Output::Rust, "tests/expect_fail_todo/*.arc");
}

#[test]
fn test_mlir() {
    snapshot(Output::MLIR, "tests/expect_pass/*.arc");
    snapshot(Output::MLIR, "tests/expect_fail/*.arc");
    snapshot(Output::MLIR, "tests/expect_fail_todo/*.arc");
}
