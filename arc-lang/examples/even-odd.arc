# XFAIL: *
# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
# The declaration order of top-level definitions is insignificant. In other words,
# functions can reference other functions declared farther down in the code.

def is_even(n) = if n == 0 { true } else { is_odd(n-1) }

def is_odd(n) = if n == 0 { false } else { is_even(n-1) }
# ANCHOR_END: example
