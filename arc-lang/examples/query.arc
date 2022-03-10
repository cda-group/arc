# XFAIL: *
# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

# ANCHOR: implicit
def test0(s) =
    from x in s {
        where x.k != 1
        group x.k
        reduce
            sum of x.v,
            count
    }
# ANCHOR_END: implicit

# ANCHOR: explicit
def test1(s: Stream[#{k:i32,v:i32}]) =
    from x in s {
        where x.k != 1
        group k = x.k
        reduce
            sum = sum of x.v,
            count = count
    }
# ANCHOR_END: explicit

def main() {}
