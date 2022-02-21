# XFAIL: *
# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

def test() {
# ANCHOR: example
# Pattern matching on enums
val a = Option::Some(3);

match a {
    Option::Some(2) => 2,
    Option::Some(x) => x,
    Option::None(_) => 0
};
# ANCHOR_END: example
}
