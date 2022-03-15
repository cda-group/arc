# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

task union(s0, s1): (s2) {
    loop {
        on {
            x in s0 => s2 ! x,
            x in s1 => s2 ! x,
        }
    }
}

extern def read_numbers_stream(): Stream[i32];

def main() {
    val stream0 = read_numbers_stream();
    val stream1 = read_numbers_stream();
    val stream2 = union(stream0, stream1);
}
