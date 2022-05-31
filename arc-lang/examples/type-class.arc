# XFAIL: *
# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: class
class Foo[T] {
    def bar(T): T;
}
# ANCHOR_END: class

# ANCHOR: instance
instance[T] Foo[T] {
    def bar(x) = x
}
# ANCHOR_END: instance

def main() {
    val x = bar(1);
}
