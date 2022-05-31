# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
task map(i, f): (o) {
    for x in i {
        o.push(f(x));
    }
}
# ANCHOR_END: example

def main() {}
