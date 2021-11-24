# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
# `(_ + _)` desugars into a lambda function `fun(x0, x1): x0 + x1`
def test() = foo(_ + _)

def foo(bar) = bar(1, 2)
# ANCHOR_END: example
