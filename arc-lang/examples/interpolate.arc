# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
# String interpolation is supported using the $ and ${} syntax.

def main() {
  val hello = "hello";
  val world = "world";
  val result = "$hello $world, 1+2 = ${i32_to_string(1+2)}";
}
# ANCHOR_END: example
