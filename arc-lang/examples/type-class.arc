# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: class
class Add<T> {
    def add(T, T): T;
}
# ANCHOR_END: class

# ANCHOR: instance
instance Add<#{sum:i32}> {
    def add(l, r) = #{sum: l.sum + r.sum}
}
# ANCHOR_END: instance
