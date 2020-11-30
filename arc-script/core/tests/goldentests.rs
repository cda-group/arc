use goldentests::{TestConfig, TestResult};
use std::env;
use std::path::Path;

#[test]
fn run_golden_tests() -> TestResult<()> {
    let ctd = env::var("OUT_DIR").unwrap();
    let path = Path::new(&ctd);
    let arc_script = path.join("..").join("..").join("..").join("arc-script");
    // Pretty Printer tests are prefixed with `--*`
    TestConfig::new(arc_script, "tests/goldentests", "--[ARC] ")?.run_tests()?;
    Ok(())
}
