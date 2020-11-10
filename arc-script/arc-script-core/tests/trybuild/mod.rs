
/// This is a test-suite for testing if compilation fails or succeeds
#[test]
fn trybuild() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/trybuild/expect-fail/*.rs");
    t.pass("tests/trybuild/expect-pass/*.rs");
}
