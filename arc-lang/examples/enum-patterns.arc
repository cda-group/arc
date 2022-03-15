# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

def main() {
# ANCHOR: example
# Pattern matching on enums
val a = Option::Some(3);

match a {
    Option::Some(2) => 2,
    Option::Some(x) => x,
    Option::None(_) => 0
};
# ANCHOR_END: example
}
