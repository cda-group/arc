# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
val pi = 3.14;

def main() {
    val x = pi * 2.0;
}
# ANCHOR_END: example
