# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

def main() {
# ANCHOR: example
# Pattern matching on enums
val a = Some(Some(#{a:3}));

match a {
    Some(Some(#{a:3})) => 3,
    None => 1+1,
};
# ANCHOR_END: example
}
