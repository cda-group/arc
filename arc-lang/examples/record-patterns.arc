# XFAIL: *
# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

def test() {
# ANCHOR: example
# Pattern matching on records
val a = {x0:0, x1:2, x2:3};
val {x0, x1, x2} = a;   # Extract `x0`, `x1`, and `x2`
val {x0} = a;           # Extract only `x0`
val {x0, x1, x2:y} = a; # Extract `x0` and `x1`, and alias `x2` to `y`
val {x0, x1|b} = a;     # Extract `x0` and `x1`, and bind `{x1}` to `b`
# ANCHOR_END: example
}
