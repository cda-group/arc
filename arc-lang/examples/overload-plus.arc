# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
def test1() = 1 + 2
def test2() = 1.0 + 2.0
# ANCHOR_END: example

def main() {}
