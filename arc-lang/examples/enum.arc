# RUN: arc -o %t run %s -- -rustinclude %s.rust-tests
# RUN: arc -o %t-canon run %s -- -rustinclude %s.rust-tests -canonicalize

# ANCHOR: example
enum Foo[T] {
    Bar(T)
}

def main() {
    Foo::Bar(1);
    Foo::Bar(1.0);
}
# ANCHOR_END: example
