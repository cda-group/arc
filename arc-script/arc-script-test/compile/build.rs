fn main() {
    arc_script_build::Builder::default()
        .no_exclude(true)
        .source_dirs(["src/tests/expect_pass", "src/tests/expect_mlir_fail_todo"])
        .build();
}
