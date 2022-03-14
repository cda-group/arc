# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
def test(typed_param: i32, untyped_param) = typed_param + untyped_param
# ANCHOR_END: example

def main() {}
