# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
enum Option[T] {
    Some(T),
    None
}

def main() {

    match Option::Some(5) {
        Option::Some(x) => assert(x == 5),
        Option::None => assert(false),
    };

    match Option::Some(5.0) {
        Option::Some(x) => assert(x == 5.0),
        Option::None => assert(false),
    };

}
# ANCHOR_END: example

