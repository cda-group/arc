# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

@{rust: "source"}
extern def source(): PullChan[i32];
# ANCHOR: example
task merge(s0, s1): (s2) {
    loop {
        s2.push(s0.pull());
        s2.push(s1.pull());
    }
}

def main() {
    val stream0 = source();
    val stream1 = source();
    val stream2 = merge(stream0, stream1);
}
# ANCHOR_END: example
