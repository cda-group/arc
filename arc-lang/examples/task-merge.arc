# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

extern def read_numbers_stream(): Stream[i32];

# ANCHOR: example
task merge(s0, s1, f): (s2) {
    loop {
        val x = receive s0;
        val y = receive s1;
        s2 emit f(x, y);
    }
}

def main() {
    val stream0 = read_numbers_stream();
    val stream1 = read_numbers_stream();
    val stream2 = merge(stream0, stream1, (+));
}
# ANCHOR_END: example
