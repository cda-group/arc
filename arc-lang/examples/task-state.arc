# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
task reduce(i, f, init): (o) {
    var state = init;
    for x in i {
        state = f(state, x);
        o ! state;
    }
}

def main() {
    let sum = reduce(0..100, fun(a, b): a + b, 0);
    let count = reduce(0..100, fun(a, _): a + 1, 0);
    let max = reduce(0..100, fun(a, b): if a > b { a } else { b }, 0);
}
# ANCHOR_END: example
