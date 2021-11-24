# XFAIL: *
# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

def test() {
# ANCHOR: example
# Pattern matching on vectors
val a = [1,2,3];

match a {
    [1, ..] => 1,
    [.., 3] => 3,
    _ => 0
};
# ANCHOR_END: example
}
