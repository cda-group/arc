# XFAIL: *
# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
# String interpolation is supported using the $ and ${} syntax.

def main() {
  val hello = "hello";
  val world = "world";
  val result = "$hello $world, 1+2 = ${1+2}";
}
# ANCHOR_END: example
