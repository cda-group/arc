# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
def id(x) = x

def main() {
    id(1);
    id(1.0);
}
# ANCHOR_END: example
