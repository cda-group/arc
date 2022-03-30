# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

def main() {
# ANCHOR: example
var a = 1;
val b = #{x:1, y:2, c:3};
val c = (1, 2, 3);
val d = [1, 2, 3];

a = 4;
b.x = 4;
c.0 = 4;
d[0u32] = 4;
# ANCHOR_END: example

assert(a == 4);
assert(b.x == 4);
assert(c.0 == 4);
assert(d[0u32] == 4);
}
