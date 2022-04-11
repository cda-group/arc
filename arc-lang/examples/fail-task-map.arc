# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
task map(i, f): (o) {
    loop {
        o ! f(receive i);
    }
}

def main() {
    val c = map(c, fun(x) = x + 1);
}
# ANCHOR_END: example
