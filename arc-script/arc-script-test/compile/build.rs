fn main() {
    arc_script_build::Builder::default()
        .compile_unused_sources(true)
        .enable_optimisations(false)
        .source_dirs(["src/tests/expect_pass", "src/tests/expect_mlir_fail_todo"])
        .build();

    arc_script_build::Builder::default()
        .compile_unused_sources(true)
        .enable_optimisations(true)
        .prefix_output_filename_with("opt_")
        .source_dirs(["src/tests/expect_pass"])
        .build();
}
