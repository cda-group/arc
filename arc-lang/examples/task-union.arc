# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

@{rust:"source"}
extern def source(): PullChan[i32];
# ANCHOR: example
task union(s0, s1): (s2) {
    loop {
        on {
            x in s0 => s2.push(x),
            x in s1 => s2.push(x),
        }
    }
}

def main() {
    val stream0 = source();
    val stream1 = source();
    val stream2 = union(stream0, stream1);
}
# ANCHOR_END: example
