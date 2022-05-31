# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
def add(i32, i32): i32;

def add(a, b) = a + b
# ANCHOR_END: example

def main() {}
