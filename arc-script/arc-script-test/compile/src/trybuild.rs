/// This is a test-suite for testing if compilation fails or succeeds
///
/// NOTE: For now disabled, but who knows what the future holds.
#[test]
fn trybuild() {
    let t = trybuild::TestCases::new();
    t.compile_fail("src/tests/expect-fail/*.rs");
    t.pass("src/tests/expect-pass/*.rs");
    t.pass("src/tests/expect-mlir-fail-todo/*.rs");
}
