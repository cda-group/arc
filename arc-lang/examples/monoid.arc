# XFAIL: *
# RUN: arc-lang %s | arc-mlir-rust-test %t - -rustinclude %s.rust-tests
# RUN: arc-lang %s | arc-mlir-rust-test %t-canon - -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
class Monoid {
    def identity(): Self;
    def merge(Self, Self): Self;
}
impl Monoid for {sum:i32} {
    def identity() {
        #{sum: 0}
    }
    def merge(a, b) {
        #{sum: a.sum + b.sum}
    }
}
def test() = x.merge(y.merge(identity()))
# ANCHOR_END: example
