# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
task merge(s0, s1): (s2) {
    loop {
        s2 ! (receive s0);
        s2 ! (receive s1);
    }
}

def main() {
    val stream0 = 0..100;
    val stream1 = 0..100;
    val stream2 = merge(stream0, stream1);
}
# ANCHOR_END: example
