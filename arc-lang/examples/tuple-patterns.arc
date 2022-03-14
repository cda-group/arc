# XFAIL: *
# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

def main() {
# ANCHOR: example
# Pattern matching on tuples:
val a = (1, 2, 3);
val (x0, x1, x2) = a;
# ANCHOR_END: example
}
