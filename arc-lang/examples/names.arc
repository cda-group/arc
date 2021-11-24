# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
def this_is_a_name(this_is_also_a_name) {
    val this_is_yet_another_name = 1;
}
# ANCHOR_END: example
