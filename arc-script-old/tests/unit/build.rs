fn main() {
    arc_script_api_build::Builder::default()
        .library_name("unit_tests")
        .compile_unused_sources(true)
        .enable_optimisations(false)
        .disable_prelude(true)
        .source_dirs(["src/tests/expect_pass", "src/tests/expect_mlir_fail_todo"])
        .build();

    arc_script_api_build::Builder::default()
        .compile_unused_sources(true)
        .enable_optimisations(false)
        .disable_prelude(true)
        .source_dirs(["src/tests/expect_pass_integration"])
        .build();

    arc_script_api_build::Builder::default()
        .library_name("opt_unit_tests")
        .compile_unused_sources(true)
        .enable_optimisations(true)
        .disable_prelude(true)
        .prefix_output_filename_with("opt_")
        .source_dirs(["src/tests/expect_pass"])
        .build();
}
