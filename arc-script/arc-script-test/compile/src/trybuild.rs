/// This is a test-suite for testing if compilation fails or succeeds
#[test]
fn trybuild() {
    let t = trybuild::TestCases::new();
    t.compile_fail("src/expect-fail/**/*.rs");
    t.pass("src/expect-pass/**/*.rs");
}
