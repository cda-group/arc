# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
task union(s0, s1): (s2) {
    loop {
        on {
            x in s0 => s2 ! x,
            x in s1 => s2 ! x,
        }
    }
}

def main() {
    val stream0 = 0..100;
    val stream1 = 0..100;
    val stream2 = union(stream0, stream1);
}
# ANCHOR_END: example
