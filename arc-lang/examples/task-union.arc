# XFAIL: *
# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

task union(s0, s1): (s2) {
    loop {
        on {
            x in s0 => emit x in s2,
            x in s1 => emit x in s2
        }
    }
}

val stream0 = read_numbers_stream();
val stream1 = read_numbers_stream();
val stream2 = union(stream0, stream1);
