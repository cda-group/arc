# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

def main() {
# ANCHOR: example
# Pattern matching on tuples:
val a = (1, 2, 3);
val (x0, x1, x2) = a;
# ANCHOR_END: example
}
