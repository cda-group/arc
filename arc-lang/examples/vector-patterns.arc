# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

def main() {
# ANCHOR: example
# Pattern matching on vectors
val a = [1,2,3];

match a {
    [1,2|t] => 1,
    _ => 0
};
# ANCHOR_END: example
}
