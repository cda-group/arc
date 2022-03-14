# XFAIL: *
# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
# The following code shows how to define the Fibonacci function functionally.

def fib(n) = match n {
    0 => 0,
    1 => 1,
    n => fib(n-2) + fib(n-1)
}
# ANCHOR_END: example

def main() = {
    fib(10);
}
