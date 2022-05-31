# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

@{rust:"source"}
extern def source(): PullChan[i32];

# ANCHOR: example
task fold(i, f, init): (o) {
    var state = init;
    loop {
        val x = i.pull();
        state = f(state, x);
        o.push(state);
    }
}

def main() {
    val sum = fold(source(), fun(a, b) = a + b, 0);
    val count = fold(source(), fun(a, _) = a + 1, 0);
    val max = fold(source(), fun(a, b) = if a > b { a } else { b }, 0);
}
# ANCHOR_END: example
