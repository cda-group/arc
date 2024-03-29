# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

extern def read_numbers_stream(): Stream[i32];

# ANCHOR: example
task split(s0, f): (s1, s2) {
    loop {
        val x = receive s0;
        s1 ! x;
        s2 ! x;
    }
}

def main() {
    val stream0 = read_numbers_stream();
    val (stream1, stream2) = split(stream0, stream1);
}
# ANCHOR_END: example
