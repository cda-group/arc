# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

def main() {
# ANCHOR: example
# Pattern matching on records
    val x = #{a:1};
    val y = #{b:2|x};
# val #{x0, x1, x2} = a;   # Extract `x0`, `x1`, and `x2`
# val #{x0} = a;           # Extract only `x0`
# val #{x0, x1, x2:y} = a; # Extract `x0` and `x1`, and alias `x2` to `y`
# val #{x0, x1|b} = a;     # Extract `x0` and `x1`, and bind `{x1}` to `b`
# ANCHOR_END: example
}
