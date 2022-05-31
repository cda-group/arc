# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
def main() {
    match Bar#(1) {
        Bar#(x) => x,
        Baz#(y) => y,
    }
}
# ANCHOR_END: example
