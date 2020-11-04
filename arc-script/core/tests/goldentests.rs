use goldentests::{TestConfig, TestResult};

#[test]
fn run_golden_tests() -> TestResult<()> {
    // Pretty Printer tests are prefixed with `--*`
    TestConfig::new("../target/debug/arc-script", "tests/goldentests", "--[ARC] ")?.run_tests()?;
    // MLIR tests are prefixed with `--*`
    TestConfig::new("../target/debug/arc-script", "tests/goldentests", "--[MLIR] ")?.run_tests()?;
    Ok(())
}
