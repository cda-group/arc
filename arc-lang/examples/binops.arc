# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

def test() {
# ANCHOR: example
val n = 1;
val b = true;

# Arithmetic
# - n;
n + n;
n - n;
n * n;
n / n;
# n ** n;
n % n;

# Equality
n == n;
# n != n;

# Logical
b and b;
b or b;
# n band n;
# n bor n;
# n bxor n;
# not b;

# Containers
# n in [n, n, n];
# n not in [];
# ANCHOR_END: example
}
