# XFAIL: *
# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
# The following code shows how to define the Fibonacci function imperatively.

def fib(n) {
    var a = 0;
    var b = 1;
    while a < n {
        a = b;
        b = a + b;
    };
    return a;
}
# ANCHOR_END: example

def main() = {
    fib(10);
}
