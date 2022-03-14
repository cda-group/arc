# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
# Binary operators can be lifted into functions.
def apply(binop, l, r) = binop(l, r)

def main() = apply((+), 1, 3)
# ANCHOR_END: example
