# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
# Binary operators can be lifted into functions.
def apply(binop, l, r) = binop(l, r)

def main() = apply((+), 1, 3)
# ANCHOR_END: example
