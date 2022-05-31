# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
task map(i, f): (o) {
    loop {
        o.push(f(i.pull()));
    }
}

def main() {
    val c = map(c, fun(x) = x + 1);
}
# ANCHOR_END: example
