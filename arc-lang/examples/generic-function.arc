# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
def id[A](x: A): A = x

def main() {
    id(1);
    id(1.0);
    id::[i32](1);
    id::[f32](1.0);
}
# ANCHOR_END: example
