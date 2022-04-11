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
    val stream0 = source(0..);
    val stream1 = map(stream0, fun(x) = x + 1);
}
# ANCHOR_END: example

# ANCHOR: for-loop
task map(i, f): (o) {
    for x in i {
        o ! f(x);
    }
}
# ANCHOR_END: for-loop

# ANCHOR: annotated
task map(i: Stream[i32], f: fun(i32):i32): (o: Drain[i32]) {
    loop {
        o ! f(receive i);
    }
}

def main() {
    val stream0: Stream[i32] = source(0..);
    val stream1: Stream[i32] = map(stream0, fun(x) = x + 1);
}
# ANCHOR_END: annotated
