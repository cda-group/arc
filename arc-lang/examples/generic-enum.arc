# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
def main() {

    match Some(5) {
        Some(x) => assert(x == 5),
        None => assert(false),
    };

    match Some(5.0) {
        Some(x) => assert(x == 5.0),
        None => assert(false),
    };

}
# ANCHOR_END: example

