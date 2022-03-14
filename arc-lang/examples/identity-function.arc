# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
def id[A](x: A): A = x

def main() {
    val x = id(1);
    val y = id(1.0);
}
# ANCHOR_END: example
